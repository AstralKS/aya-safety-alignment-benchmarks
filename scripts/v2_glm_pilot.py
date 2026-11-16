"""Prepare/verify offline, or explicitly run the initial 30-request GLM pilot.

Run is deliberately one-shot: no resume, retry, redirects, or bulk scoring.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import string
import threading
import time
import uuid

import requests
from requests.adapters import HTTPAdapter

from v2_glm_budget import (
    ATTEMPT_BOUND, INPUT_BOUND, OUTPUT_BOUND, Ledger,
    json_bytes, usage_price, write_once,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "zai-org/GLM-5.3-Flash:deepinfra"
RESPONSE_MODELS = frozenset(("zai-org/GLM-5.3-Flash", MODEL))
TIMEOUT = (10.0, 120.0)  # Explicit connect/read bounds, NOT a total deadline.
SEED = "v2-glm-initial-20261111"
TASK_COUNTS = {"simpleqa_verified": 8, "abstention_bench": 7}
AYA_MODELS = {"tiny": "CohereLabs/tiny-aya-global", "8b": "CohereLabs/aya-expanse-8b"}
CHECKPOINTS = {
    "tiny": ("runs/tiny-baseline/checkpoint/responses.jsonl",
             "f702cd1c09170bfc64ca85d4821ba688025f71ac15b03917f466fa578498ffda"),
    "8b": ("runs/8b-baseline/checkpoint/responses.jsonl",
           "46b92a3471d0751be99c999e94eeb3c6739e2e190a77685cefcc435f488be156"),
}
RUBRICS = {
    "simpleqa_verified": {
        "path": "data/raw/simpleqa-starter/simpleqa-verified-benchmark-starter-code.ipynb",
        "source_sha256": "14d0c0513efefdfe7936e05c6fc09b4b4a191cc31273ca8bfbcdeaea0c6fdb1b",
        "name": "GRADER_TEMPLATE",
        "rubric_sha256": "84c004ec4fcf8f0703bb0d734544036a72e847bfa7116429e5aee5e53ccc8cf3",
        "fields": ("question", "target", "predicted_answer"),
    },
    "abstention_bench": {
        "path": "data/raw/AbstentionBench/recipe/evaluation_judge_prompts.py",
        "source_sha256": "94e751db02712e4abac63336b253fa92bf7cf9b6098a1dab690d655c72c337c9",
        "name": "LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT",
        "rubric_sha256": "10c86ba21852d6a0e3b3250a4827bb858cdcb9c96f24746e4813112c77877bdf",
        "fields": ("question", "ref_answer", "abstention_label", "model_answer"),
    },
}
DIAGNOSTIC_HEADERS = frozenset((
    "x-request-id", "x-amzn-trace-id", "x-compute-type", "x-compute-time",
    "x-inference-provider", "x-ratelimit-limit", "x-ratelimit-remaining",
    "x-ratelimit-reset", "retry-after", "content-type", "date",
))
WAVES = tuple((level, wave) for level in (1, 2, 4, 8) for wave in (1, 2))


class PilotError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PilotError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def read_json(data):
    def reject(value):
        raise PilotError(f"Nonfinite JSON: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(data, parse_float=Decimal, parse_constant=reject, object_pairs_hook=unique)


def literal_rubric(source, name):
    """Read an assignment without executing a single byte of upstream code."""
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            require(node in tree.body and len(targets) == 1, "Rubric assignment must be unique/top-level")
            value = node.value
            strip = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                     and value.func.attr == "strip" and not value.args and not value.keywords)
            if strip:
                value = value.func.value
            require(isinstance(value, ast.Constant) and isinstance(value.value, str),
                    "Rubric must be a string literal or literal.strip() with no arguments")
            matches.append(value.value.strip() if strip else value.value)
    require(len(matches) == 1, f"Expected exactly one {name} assignment")
    return matches[0]


def extract_rubric(root, contract):
    raw = (root / contract["path"]).read_bytes()
    require(sha(raw) == contract["source_sha256"], "Official rubric source hash mismatch")
    if contract["path"].endswith(".ipynb"):
        notebook = read_json(raw)
        cells = []
        pattern = re.compile(r"^\s*" + re.escape(contract["name"]) + r"\s*(?::[^=\n]+)?=", re.MULTILINE)
        for cell in notebook["cells"]:
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            if cell.get("cell_type") == "code" and pattern.search(source):
                cells.append(source)
        require(len(cells) == 1, "Expected one matching rubric cell (unrelated !pip cells are not parsed)")
        source = cells[0]
    else:
        source = raw.decode("utf-8")
    rubric = literal_rubric(source, contract["name"])
    require(sha(rubric.encode()) == contract["rubric_sha256"], "Official rubric text hash mismatch")
    fields = []
    for _, field, spec, conversion in string.Formatter().parse(rubric):
        if field is not None:
            require(not spec and conversion is None, "Unexpected rubric formatting")
            fields.append(field)
    require(set(fields) == set(contract["fields"]), "Unexpected rubric fields")
    return rubric


def jsonl_index(raw, label):
    rows = {}
    for line in raw.splitlines(keepends=True):
        row = read_json(line)
        require(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"], f"Invalid ID in {label}")
        require(row["id"] not in rows, f"Duplicate ID in {label}")
        rows[row["id"]] = (row, line)
    return rows


def rank(short, dataset, identifier):
    return sha(f"{SEED}:{short}:{dataset}:{identifier}".encode())


def payload_for(row, answer, rubric):
    refs = row.get("reference_answers")
    require(refs is None or (isinstance(refs, list) and all(isinstance(x, str) for x in refs)), "Invalid references")
    require(isinstance(row.get("question"), str) and row["question"], "Empty question")
    require(type(row.get("should_abstain")) is bool, "Invalid abstention label")
    if row["dataset"] == "simpleqa_verified":
        require(isinstance(refs, list) and len(refs) == 1 and refs[0], "SimpleQA needs exactly one target")
        prompt = rubric.format(question=row["question"], target=refs[0], predicted_answer=answer)
    else:
        # Exact upstream reference formatting, including None/empty list -> [].
        reference = "[" + "; ".join(refs) + "]" if refs else "[]"
        prompt = rubric.format(question=row["question"], ref_answer=reference,
                               abstention_label=row["should_abstain"], model_answer=answer)
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "n": 1, "max_tokens": OUTPUT_BOUND, "reasoning_effort": "low"}


def derive(root):
    # Lazy offline-only verifier. No scorer imports, HTTP, embeddings, or model loading.
    from prepare_v2_data import verify as verify_v2
    verified_manifest = verify_v2(root)
    inputs = {}
    for name in ("manifest.json", "evaluation.jsonl", "reuse.json"):
        path = f"data/v2/{name}"
        inputs[path] = (root / path).read_bytes()
    manifest = read_json(inputs["data/v2/manifest.json"])
    require(manifest == verified_manifest, "v2 manifest changed after offline verification")
    for name in ("evaluation.jsonl", "reuse.json"):
        raw = inputs[f"data/v2/{name}"]
        entry = manifest["outputs"][name]
        require(entry["path"] == f"data/v2/{name}" and sha(raw) == entry["sha256"]
                and len(raw) == entry["bytes"], "v2 output hash/size mismatch")
    rows = jsonl_index(inputs["data/v2/evaluation.jsonl"], "v2 evaluation")
    require(len(rows) == manifest["total_evaluation_rows"], "v2 evaluation count mismatch")
    reuse = read_json(inputs["data/v2/reuse.json"])
    require(set(reuse["models"]) == set(AYA_MODELS), "Unexpected Aya model set")
    rubrics = {task: extract_rubric(root, contract) for task, contract in RUBRICS.items()}
    for contract in RUBRICS.values():
        inputs[contract["path"]] = (root / contract["path"]).read_bytes()
        require(sha(inputs[contract["path"]]) == contract["source_sha256"], "Rubric source changed during prepare")
    selected = []
    for short, model in AYA_MODELS.items():
        record = reuse["models"][short]
        path, anchor = CHECKPOINTS[short]
        require(record["model_id"] == model and record["checkpoint"]["path"] == path
                and record["checkpoint"]["sha256"] == anchor, "Anchored reuse checkpoint identity mismatch")
        raw = (root / path).read_bytes()
        require(sha(raw) == anchor, "Anchored checkpoint hash mismatch")
        inputs[path] = raw
        responses = jsonl_index(raw, path)
        require(len(responses) == record["checkpoint"]["rows"], "Checkpoint row count mismatch")
        anchored = {}
        for item in record["responses"]:
            identifier = item["id"]
            require(identifier not in anchored and identifier in rows and identifier in responses,
                    "Missing/duplicate reusable ID")
            row, line = rows[identifier]
            response, response_line = responses[identifier]
            dataset = row["dataset"]
            require(dataset in TASK_COUNTS and item["dataset"] == dataset and response.get("dataset") == dataset,
                    "Reuse dataset mismatch")
            require(item["model_id"] == model and response.get("model_id") == model, "Reuse response model mismatch")
            require(response.get("status") == "ok" and isinstance(response.get("response"), str)
                    and response["response"].strip(), "Reusable response must be ok and nonempty")
            answer_sha = sha(response["response"].encode())
            require(answer_sha == item["response_sha256"], "Reusable answer hash mismatch")
            anchored[identifier] = (row, line, response, response_line, answer_sha)
        require(len(anchored) == reuse["count_per_model"], "Reuse count mismatch")
        require(dict(Counter(item[0]["dataset"] for item in anchored.values())) == reuse["counts_per_model"],
                "Reuse task counts mismatch")
        for task, count in TASK_COUNTS.items():
            candidates = [identifier for identifier, item in anchored.items() if item[0]["dataset"] == task]
            chosen = sorted(candidates, key=lambda identifier: (rank(short, task, identifier), identifier))[:count]
            require(len(chosen) == count, "Insufficient reusable pilot responses")
            for identifier in chosen:
                row, line, response, response_line, answer_sha = anchored[identifier]
                payload = payload_for(row, response["response"], rubrics[task])
                selected.append(({
                    "id": identifier, "aya_model": short, "aya_model_id": model, "dataset": task,
                    "ranking_sha256": rank(short, task, identifier),
                    "input_path": "data/v2/evaluation.jsonl", "input_row_sha256": sha(line),
                    "checkpoint_path": path, "checkpoint_sha256": anchor,
                    "checkpoint_row_sha256": sha(response_line), "answer_sha256": answer_sha,
                    "saved_aya_prompt_sha256": response.get("prompt_sha256"),
                    "rubric_source_path": RUBRICS[task]["path"],
                    "rubric_source_sha256": RUBRICS[task]["source_sha256"],
                    "rubric_sha256": RUBRICS[task]["rubric_sha256"],
                    "request_sha256": sha(canonical(payload)),
                }, payload))
    # Mix tasks/models across waves, using only identity-based ranking.
    selected.sort(key=lambda item: (item[0]["ranking_sha256"], item[0]["aya_model"], item[0]["id"]))
    require(len(selected) == 30 and len({(x[0]["aya_model"], x[0]["id"]) for x in selected}) == 30,
            "Pilot must have 30 distinct response identities")
    for index, (record, _) in enumerate(selected, 1):
        record["request_file"] = f"request-{index:03d}.json"
    return {"input_files": {path: {"sha256": sha(raw), "bytes": len(raw)} for path, raw in sorted(inputs.items())},
            "samples": [record for record, _ in selected]}, [payload for _, payload in selected]


def frozen_manifest(pilot_id, derived):
    return {
        "schema_version": 1, "pilot_id": pilot_id, "selection_seed": SEED,
        "model": MODEL, "url": URL, "response_model_allowlist": sorted(RESPONSE_MODELS),
        "timeout_connect_read_seconds": list(TIMEOUT), "automatic_retries": 0, "allow_redirects": False,
        "input_token_bound": INPUT_BOUND, "output_token_bound": OUTPUT_BOUND,
        "reserve_per_attempt_usd": str(ATTEMPT_BOUND), "full_30_attempt_bound_usd": str(30 * ATTEMPT_BOUND),
        "waves": [[level, wave] for level, wave in WAVES],
        "disclosure": "Initial pilot only: not sustained throughput, human validation, or bulk grades. Forced thinking is not disabled.",
        **derived,
    }


def pilot_directory(root, pilot_id):
    require(isinstance(pilot_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", pilot_id), "Invalid pilot ID")
    parent = root / "scoring/v2-glm/pilots"
    directory = parent / pilot_id
    require(directory.resolve().is_relative_to(root.resolve()), "Pilot directory escapes root")
    return directory


def prepare(root=DEFAULT_ROOT, pilot_id=None):
    root = Path(root).resolve()
    pilot_id = pilot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    directory = pilot_directory(root, pilot_id)
    derived, payloads = derive(root)
    directory.mkdir(parents=True, exist_ok=False)
    for record, payload in zip(derived["samples"], payloads):
        write_once(directory / record["request_file"], canonical(payload))
    write_once(directory / "manifest.json", json_bytes(frozen_manifest(pilot_id, derived)))
    return directory


def verify(directory, root=DEFAULT_ROOT):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    require(directory == pilot_directory(root, directory.name).resolve(), "Unexpected prepared pilot location")
    actual = read_json((directory / "manifest.json").read_bytes())
    derived, payloads = derive(root)
    expected = frozen_manifest(directory.name, derived)
    require(actual == expected, "Prepared manifest differs from recomputed frozen inputs")
    require({p.name for p in directory.glob("request-*.json")} == {x["request_file"] for x in derived["samples"]},
            "Prepared request file set mismatch")
    for record, payload in zip(derived["samples"], payloads):
        require((directory / record["request_file"]).read_bytes() == canonical(payload), "Prepared payload tamper")
    return expected


def inspect_response(body, dataset, final_parser=None):
    """Trust envelope/usage independently of strict final-only label parsing."""
    result = {"ok": False, "label": None, "accounting": None, "error": None, "reasoning": {}}
    try:
        envelope = read_json(body)
        require(isinstance(envelope, dict), "Response is not an object")
        require(envelope.get("model") in RESPONSE_MODELS, "Wrong/missing response model")
        choices = envelope.get("choices")
        require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), "Expected exactly one choice")
        choice = choices[0]
        require(choice.get("index") == 0 and type(choice.get("index")) is int, "Invalid choice index")
        message = choice.get("message")
        require(isinstance(message, dict) and message.get("role") == "assistant", "Invalid assistant envelope")
        content = message.get("content")
        require(content is None or isinstance(content, str), "Non-text final content")
        require(not message.get("tool_calls") and not message.get("function_call"), "Unexpected tool envelope")
        result["reasoning"] = {key: value for key, value in message.items() if key in ("reasoning", "reasoning_content", "reasoning_details")}
        result["finish_reason"] = choice.get("finish_reason")
        result["response_model"] = envelope["model"]
        result["usage"] = envelope.get("usage")
        nonempty = bool(content) or any(bool(x) for x in result["reasoning"].values())
        result["accounting"] = usage_price(envelope.get("usage"), nonempty)
        require(not result["accounting"]["bound_failure"], "Token/monetary bound exceeded; charge was NOT retroactively prevented")
        require(choice.get("finish_reason") == "stop", "Truncated/non-stop final response")
        require(not message.get("refusal"), "Unexpected refusal envelope")
        require(isinstance(content, str), "Missing final content")
        if final_parser is None:
            labels = ("A", "B", "C") if dataset == "simpleqa_verified" else ("Yes", "No")
            require(content.strip() in labels, "Invalid final-only label")
            result["label"] = content.strip()
        else:
            result["label"] = final_parser(content.strip())
        result["ok"] = True
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def token():
    value = os.environ.get("HF_TOKEN")
    if not value:
        try:
            from huggingface_hub import get_token
            value = get_token()
        except ImportError:
            value = None
    require(isinstance(value, str) and value.strip(), "HF token unavailable (HF_TOKEN or cached huggingface_hub token required)")
    return value.strip()


def http_request(payload, hf_token):
    # A session per call avoids sharing mutable requests.Session state across threads.
    with requests.Session() as session:
        session.trust_env = False
        session.mount("https://", HTTPAdapter(max_retries=0))
        return session.post(URL, data=canonical(payload), headers={
            "Authorization": "Bearer " + hf_token, "Content-Type": "application/json",
        }, timeout=TIMEOUT, allow_redirects=False)


class InFlight:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            require(self.active <= 8, "In-flight limiter exceeded")

    def leave(self):
        with self.lock:
            self.active -= 1


def execute_attempt(directory, pilot_id, record, payload, ledger, hf_token, inflight, attempt_id, final_parser=None):
    outcome = {"attempt_id": attempt_id, "request_file": record["request_file"],
               "aya_model": record["aya_model"], "id": record["id"], "dataset": record["dataset"],
               "ok": False, "label": None, "accounting": None, "error": None,
               "dispatched": False, "latency_seconds": None}
    journal_path = directory / "attempts" / f"{attempt_id}.response.json"
    try:
        if "cache_key" in record:
            ledger.reserve(attempt_id, pilot_id, record["request_sha256"], job_cache_key=record["cache_key"])
        else:
            ledger.reserve(attempt_id, pilot_id, record["request_sha256"])
        write_once(directory / "attempts" / f"{attempt_id}.dispatch.json", json_bytes({
            "attempt_id": attempt_id, "request_sha256": record["request_sha256"],
            "reservation_committed": True,
        }))
        inflight.enter()
        start = time.perf_counter()
        try:
            outcome["dispatched"] = True
            response = http_request(payload, hf_token)
            raw = response.content
        finally:
            outcome["latency_seconds"] = time.perf_counter() - start
            inflight.leave()
        outcome["http_status"] = response.status_code
        journal = {
            "kind": "http_response", "attempt_id": attempt_id,
            "request_sha256": record["request_sha256"], "http_status": response.status_code,
            "raw_body_base64": base64.b64encode(raw).decode("ascii"), "raw_body_sha256": sha(raw),
            "diagnostic_headers": {key.lower(): value for key, value in response.headers.items()
                                   if key.lower() in DIAGNOSTIC_HEADERS},
            "latency_seconds": outcome["latency_seconds"],
        }
        # Persist exact raw body before parsing or any budget settlement.
        write_once(journal_path, json_bytes(journal))
        require(response.status_code == 200, f"HTTP status {response.status_code}; unknown charge reserve retained")
        parsed = inspect_response(raw, record["dataset"], final_parser)
        outcome.update(parsed)
        if parsed["accounting"] is not None:
            ledger.settle(attempt_id, parsed["accounting"], journal_path)
        return outcome
    except requests.RequestException as error:
        outcome["error"] = "Transport ambiguity: " + type(error).__name__
        try:
            write_once(journal_path, json_bytes({"kind": "transport_error", "attempt_id": attempt_id,
                                               "request_sha256": record["request_sha256"],
                                               "error_type": type(error).__name__,
                                               "latency_seconds": outcome["latency_seconds"]}))
        except OSError as journal_error:
            outcome["error"] += "; journal failure: " + type(journal_error).__name__
        return outcome
    except (OSError, ValueError, TypeError, KeyError) as error:
        outcome["ok"] = False
        outcome["label"] = None
        outcome["error"] = "Accounting/journal/checkpoint failure: " + str(error)
        return outcome


def recover_saved(journal_path, root=DEFAULT_ROOT, receipt_dir=None, settle=False):
    """Reparse the original pilot journal offline; never replay or alter old logs."""
    root, journal_path = Path(root).resolve(), Path(journal_path).resolve()
    directory = journal_path.parent.parent
    manifest = verify(directory, root)
    raw = journal_path.read_bytes()
    journal = read_json(raw)
    require(journal.get("kind") == "http_response" and journal.get("http_status") == 200,
            "Recovery requires a saved HTTP 200 journal")
    body = base64.b64decode(journal["raw_body_base64"], validate=True)
    require(sha(body) == journal["raw_body_sha256"], "Recovery raw-body hash mismatch")
    samples = [record for record in manifest["samples"] if record["request_sha256"] == journal["request_sha256"]]
    require(len(samples) == 1, "Recovery request is not a unique frozen pilot sample")
    sample = samples[0]
    parsed = inspect_response(body, sample["dataset"])
    receipt = {"repair_version": "optional-usage-details-null-v1", "attempt_id": journal["attempt_id"],
               "original_journal_sha256": sha(raw), "request_sha256": journal["request_sha256"],
               "sample": sample, "reparsed": parsed, "http_dispatched": False}
    receipt_dir = Path(receipt_dir) if receipt_dir else root / "scoring/v2-glm/results" / ("recover-" + journal["attempt_id"])
    require(receipt_dir.resolve().is_relative_to((root / "scoring/v2-glm/results").resolve()), "Unexpected repair receipt destination")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / ("repair-" + sha(raw) + ".json")
    content = json_bytes(receipt)
    if path.exists():
        require(path.read_bytes() == content, "Conflicting immutable repair receipt")
    else:
        write_once(path, content)
    if settle:
        require(parsed["accounting"] is not None, "Saved attempt has no trusted usage to settle")
        Ledger(root / "scoring/budget.json").settle(journal["attempt_id"], parsed["accounting"], journal_path)
    return receipt


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def level_summaries(waves):
    levels = []
    for level in (1, 2, 4, 8):
        entries = [wave for wave in waves if wave["concurrency"] == level]
        if not entries:
            continue
        outcomes = [item for wave in entries for item in wave["outcomes"]]
        elapsed = sum(wave["elapsed_seconds"] for wave in entries)
        latencies = [item["latency_seconds"] for item in outcomes if item["latency_seconds"] is not None]
        calls = sum(item["dispatched"] for item in outcomes)
        levels.append({"concurrency": level, "waves_completed": len(entries), "elapsed_seconds": elapsed,
                       "calls_dispatched": calls, "successful_labels": sum(item["ok"] for item in outcomes),
                       "calls_per_second": calls / elapsed if elapsed else None,
                       "p50_latency_seconds": percentile(latencies, 0.50),
                       "p95_latency_seconds": percentile(latencies, 0.95),
                       "error_count": sum(not item["ok"] for item in outcomes),
                       "errors": dict(Counter(item["error"] for item in outcomes if item["error"]))})
    return levels


def run(directory, root=DEFAULT_ROOT):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = verify(directory, root)
    require(not (directory / "run-start.json").exists(), "Pilot replay/restart refused; no automatic resume")
    hf_token = token()
    guard = directory / "run.guard"
    # One process; a stuck guard is intentionally not removed automatically.
    write_once(guard, json_bytes({"pid": os.getpid()}))
    try:
        write_once(directory / "run-start.json", json_bytes({
            "pilot_id": manifest["pilot_id"], "run_id": uuid.uuid4().hex,
            "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "manifest_sha256": sha((directory / "manifest.json").read_bytes()),
        }))
        ledger = Ledger(root / "scoring/budget.json")
        ledger.start(directory)
        (directory / "attempts").mkdir(exist_ok=False)
        waves, cursor, stop_error = [], 0, None
        inflight = InFlight()
        overall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for level, number in WAVES:
                records = manifest["samples"][cursor:cursor + level]
                require(len(records) == level, "Incomplete wave")
                # Freeze/recheck all payloads before launching exact C calls.
                payloads = []
                try:
                    for record in records:
                        raw = (directory / record["request_file"]).read_bytes()
                        require(sha(raw) == record["request_sha256"], "Request changed after verify")
                        payloads.append(read_json(raw))
                except (OSError, ValueError) as error:
                    stop_error = "Pre-wave request checkpoint failed: " + str(error)
                    break
                start = time.perf_counter()
                futures = []
                for record, payload in zip(records, payloads):
                    attempt_id = uuid.uuid4().hex
                    future = pool.submit(execute_attempt, directory, manifest["pilot_id"], record, payload,
                                         ledger, hf_token, inflight, attempt_id)
                    futures.append((future, attempt_id, record))
                # Drain ALL launched calls before stopping/escalating, including failures.
                outcomes = []
                for future, attempt_id, record in futures:
                    try:
                        outcomes.append(future.result())
                    except Exception as error:
                        outcomes.append({"attempt_id": attempt_id, "request_file": record["request_file"],
                                         "aya_model": record["aya_model"], "id": record["id"], "dataset": record["dataset"],
                                         "ok": False, "label": None, "error": "Worker failure: " + type(error).__name__,
                                         "dispatched": True, "latency_seconds": None})
                wave = {"concurrency": level, "wave": number,
                        "elapsed_seconds": time.perf_counter() - start, "outcomes": outcomes}
                waves.append(wave)
                try:
                    write_once(directory / f"wave-{level}-{number}.json", json_bytes(wave))
                except OSError as error:
                    stop_error = "Wave journal failure: " + type(error).__name__
                cursor += level
                if stop_error or any(not item["ok"] for item in outcomes):
                    stop_error = stop_error or "Failed wave checkpoint; later waves/escalation stopped"
                    break
        summary = {
            "pilot_id": manifest["pilot_id"], "ok": stop_error is None and cursor == 30,
            "stop_error": stop_error, "samples_attempted": cursor,
            "elapsed_seconds": time.perf_counter() - overall_start,
            "peak_actual_in_flight": inflight.peak, "levels": level_summaries(waves),
            "accounting": ledger.totals(manifest["pilot_id"]),
            "disclosure": manifest["disclosure"],
            "timeout_is_connect_read_not_total_deadline": True,
        }
        write_once(directory / "summary.json", json_bytes(summary))
        return summary
    finally:
        guard.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "recover-saved"))
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot-id")
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--settle", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            require(args.directory is None, "prepare takes --pilot-id, not a directory")
            result = {"prepared_directory": str(prepare(args.root, args.pilot_id)), "requests": 30}
        elif args.command == "verify":
            require(args.directory is not None and args.pilot_id is None, "verify requires a prepared directory")
            result = {"verified_pilot": verify(args.directory, args.root)["pilot_id"], "requests": 30}
        elif args.command == "recover-saved":
            require(args.directory is not None and args.pilot_id is None, "recover-saved requires the original journal path")
            result = recover_saved(args.directory, args.root, args.receipt_dir, args.settle)
        else:
            require(args.directory is not None and args.pilot_id is None, "run requires a prepared directory")
            result = run(args.directory, args.root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"{args.command} failed: {error}\n")
    print(json_bytes(result).decode("utf-8"))
    if args.command == "run" and not result["ok"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
