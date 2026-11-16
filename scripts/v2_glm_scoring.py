"""GLM-only fixed-benchmark scoring and staged FActScore, with durable no-replay jobs."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import re
import string
import threading
import uuid
import unicodedata
import zipfile

import v2_benchmark_metrics as arithmetic
import v2_glm_pilot as client
from v2_glm_budget import BudgetError, Ledger, file_lock, json_bytes, write_once


DEFAULT_ROOT = client.DEFAULT_ROOT
FIXED_TASKS = ("simpleqa_verified", "abstention_bench", "hallulens_precisewikiqa", "hallulens_nonexistentrefusal")
EXPECTED = {"simpleqa_verified": 1000, "abstention_bench": 978, "hallulens_precisewikiqa": 1000,
            "hallulens_nonexistentrefusal": 3000, "factscore_bios": 180}
PARSER_VERSION = "glm-bulk-strict-v2-required-totals-optional-null-details"
SOURCE_PINS = {
    "data/raw/AbstentionBench/recipe/evaluation_judge_prompts.py": "94e751db02712e4abac63336b253fa92bf7cf9b6098a1dab690d655c72c337c9",
    "data/v2/raw/HalluLens/tasks/shortform/precise_wikiqa.py": "b758ae75dc9a15473e96b966a8361c72362a454246a25d1ad92f14c52a9f3627",
    "data/v2/raw/HalluLens/tasks/refusal_test/prompt.py": "d0ce8cd06b587d6b7048c02e9c802f6e25a4b730bc7097b15aae15ac4954f052",
    "data/v2/raw/HalluLens/tasks/refusal_test/nonsense_mixed_entities.py": "fb63f49b6339564c56f8c7c8758805325955b0bf01f26df7695f5854a20085f3",
    "data/v2/raw/HalluLens/tasks/refusal_test/nonsense_name.py": "d25432ef27a693261b99aa36f44ae05e0fe988277f707a36633b92979e1c3b01",
    "data/v2/raw/FActScore/factscore/atomic_facts.py": "16d095911fe3a372e34c9abd917f0223503cef586db172c6ab20e8436ccd6737",
    "data/v2/raw/FActScore/factscore/factscorer.py": "d6f61090ad3e2d0cdb84c46c48d27278c2be3c2ae5f3fec2268771a8413e4cdb",
}
DEMOS_PATH = "scoring/v2-glm/preflight/pricing/factscore-demos.zip"
DEMOS_SHA = "5ab4ba73f8c2483867535ae314e692c0c4c8b6343006f1cf2378235153b7fae0"
GTR_MODEL = "sentence-transformers/gtr-t5-large"
GTR_REVISION = "51aab2cd10b205ba0f3f1ebaf1d4d5460bbf728d"
TOKENIZER_MODEL = "FacebookAI/roberta-large"
TOKENIZER_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"


def require(condition, message):
    client.require(condition, message)


def digest(value):
    return client.sha(client.canonical(value))


def exact_json(left, right):
    """Compare parsed JSON values without bool/int or decimal/string coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_json(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_json(a, b) for a, b in zip(left, right))
    return left == right


def immutable(path, value):
    immutable_content(path, json_bytes(value))


def immutable_content(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, f"Immutable artifact differs: {path}")
    else:
        write_once(path, raw)


@dataclass(frozen=True)
class Template:
    original: str
    effective: str
    sources: dict
    deviation: str | None = None


class SourceCatalog:
    """Pinned upstream data only: source text is AST-read, never imported/executed."""
    def __init__(self, root):
        self.root = Path(root)
        self.source_text = {}
        for path, expected in SOURCE_PINS.items():
            raw = (self.root / path).read_bytes()
            require(client.sha(raw) == expected, f"Official source hash mismatch: {path}")
            self.source_text[path] = raw.decode("utf-8")
        self.templates = {}
        simple = client.RUBRICS["simpleqa_verified"]
        text = client.extract_rubric(self.root, simple)
        self.templates["simpleqa"] = Template(text, text, {simple["path"]: simple["source_sha256"]})
        path = "data/raw/AbstentionBench/recipe/evaluation_judge_prompts.py"
        for key, name in (("abstention", "LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT"),
                          ("abstention_correctness", "LLM_JUDGE_CORRECTNESS_PROMPT")):
            self.templates[key] = self.literal(path, name)
        path = "data/v2/raw/HalluLens/tasks/shortform/precise_wikiqa.py"
        self.templates["precise_abstention"] = self.literal(path, "ABSTAIN_PROMPT_UPDATED")
        original = self.literal(path, "IS_HALLUCINATION_RESPONSE")
        contradictory = "Please answer with Yes or No without any explanation."
        require(original.original.count(contradictory) == 1, "PreciseWikiQA repair target changed")
        self.templates["precise_correctness"] = Template(original.original,
            original.original.replace(contradictory, "Please answer with exactly CORRECT, INCORRECT, or UNVERIFIABLE without any explanation."),
            original.sources, "Only the contradictory final Yes/No instruction is repaired; three-way labels retained.")
        path = "data/v2/raw/HalluLens/tasks/refusal_test/prompt.py"
        for key, name in (("mixed_species", "ABSTAIN_PROMPT_ANIMAL_NONSENSE"),
                          ("mixed_medicine", "ABSTAIN_PROMPT_NONSENSE_MEDICINE"),
                          ("generated", "ABSTAIN_PROMPT_PLACE_NONSENSE")):
            self.templates[key] = self.literal(path, name)
        raw = (self.root / DEMOS_PATH).read_bytes()
        require(client.sha(raw) == DEMOS_SHA, "FActScore demonstration archive hash mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            demos_raw = archive.read("demos/demons.json")
        self.demos = client.read_json(demos_raw)
        require(isinstance(self.demos, dict) and len(self.demos) >= 7 and all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(f, str) and f for f in v)
            for k, v in self.demos.items()), "Invalid upstream biography demonstrations")
        self.demos_member_sha = client.sha(demos_raw)

    def literal(self, path, name):
        text = client.literal_rubric(self.source_text[path], name)
        return Template(text, text, {path: SOURCE_PINS[path]})


def parse_final(text, schema):
    require(isinstance(text, str), "Final output must be text")
    text = text.strip()
    if schema["kind"] == "enum":
        require(text in schema["values"], "Invalid exact final label")
        return text
    value = client.read_json(text)
    if schema["kind"] == "boolean-object":
        key = schema["key"]
        require(isinstance(value, dict) and set(value) == {key} and type(value[key]) is bool,
                "Expected exact one-key boolean JSON object")
        return value[key]
    if schema["kind"] == "atomic":
        require(isinstance(value, dict) and set(value) == {"sentence", "facts"}, "Invalid atomic extraction JSON schema")
        require(value["sentence"] == schema["sentence"], "Extraction changed/omitted original sentence")
        facts = value["facts"]
        require(isinstance(facts, list) and all(isinstance(f, str) and f.strip() and f == f.strip() for f in facts),
                "Atomic facts must be nonempty exact strings")
        require(len(set(facts)) == len(facts), "Duplicate extracted atomic facts (never silently deduplicated)")
        return value
    raise client.PilotError("Unknown strict parser schema")


def payload(prompt):
    return {"model": client.MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "n": 1, "max_tokens": 2048, "reasoning_effort": "low"}


def cache_identity(source_files, answer_sha256, request, parser):
    decoding = {k: v for k, v in request.items() if k != "messages"}
    return {"source_files": source_files, "answer_sha256": answer_sha256,
            "prompt_sha256": client.sha(request["messages"][0]["content"].encode("utf-8")),
            "request_sha256": client.sha(client.canonical(request)), "model": request["model"],
            "decoding_sha256": digest(decoding), "parser_sha256": digest(parser)}


def make_job(row, answer, short, stage, template, fields, parser, extra=None):
    request = payload(template.effective.format(**fields))
    identity = cache_identity(template.sources, client.sha(answer.encode("utf-8")), request, parser)
    lineage = {"id": row["id"], "dataset": row["dataset"], "aya_model": short,
               "aya_model_id": client.AYA_MODELS[short], "answer_sha256": identity["answer_sha256"],
               "input_row_sha256": row.get("_input_row_sha256", digest({k: v for k, v in row.items() if not k.startswith("_")})),
               "v2_manifest_sha256": row.get("_v2_manifest_sha256"),
               "response_file": row.get("_response_file"), "response_row_sha256": row.get("_response_row_sha256"),
               "source_files": template.sources, "original_rubric_sha256": client.sha(template.original.encode()),
               "effective_rubric_sha256": client.sha(template.effective.encode()), "rubric_deviation": template.deviation,
               **(extra or {})}
    return {"cache_key": digest(identity), "identity": identity, "stage": stage, "payload": request,
            "request_sha256": identity["request_sha256"], "parser": parser, "lineage": lineage}


def fixed_jobs(row, answer, short, catalog, abstention=None):
    task = row["dataset"]
    require(task in FIXED_TASKS, "Unsupported fixed benchmark")
    require(isinstance(answer, str) and answer.strip(), "Empty response is not a usable scored answer")
    refs = row.get("reference_answers")
    require(refs is None or (isinstance(refs, list) and all(isinstance(x, str) for x in refs)), "Invalid source references")
    question = row["question"]
    templates = catalog.templates
    def job(stage, key, fields, schema):
        return make_job(row, answer, short, stage, templates[key], fields, schema)
    if task == "simpleqa_verified":
        require(isinstance(refs, list) and len(refs) == 1 and refs[0], "SimpleQA needs one exact reference")
        return [job("simpleqa", "simpleqa", {"question": question, "target": refs[0], "predicted_answer": answer},
                    {"kind": "enum", "values": ["A", "B", "C"]})]
    if task == "abstention_bench":
        require(type(row.get("should_abstain")) is bool, "Invalid released abstention label")
        jobs = [job("abstention", "abstention", {"question": question,
                    "ref_answer": "[" + "; ".join(refs) + "]" if refs else "[]",
                    "abstention_label": row["should_abstain"], "model_answer": answer},
                    {"kind": "enum", "values": ["Yes", "No"]})]
        if row["should_abstain"] is False and abstention == "No":
            require(refs is not None, "Answerable AbstentionBench row lacks references")
            jobs.append(job("correctness", "abstention_correctness", {"question": question,
                "ref_answer": "\n".join(refs), "model_answer": answer}, {"kind": "enum", "values": ["correct", "incorrect"]}))
        return jobs
    if task == "hallulens_precisewikiqa":
        require(isinstance(refs, list) and len(refs) == 1, "PreciseWikiQA needs exactly one source answer")
        fields = {"prompt": question, "generation": answer}
        return [job("abstention", "precise_abstention", fields, {"kind": "boolean-object", "key": "is_abstaining"}),
                job("correctness", "precise_correctness", {**fields, "gold_answer": refs[0]},
                    {"kind": "enum", "values": ["CORRECT", "INCORRECT", "UNVERIFIABLE"]})]
    meta = row["metadata"]
    require(isinstance(meta.get("name"), str) and meta["name"], "NonExistentRefusal missing entity name")
    fields = {"prompt": question, "name": meta["name"], "generation": answer}
    if row["component"] == "Mixed":
        require(meta.get("type") in ("animal", "plant", "bacteria", "medicine"), "Unknown Mixed domain; no rubric fallback")
        key = "mixed_medicine" if meta["type"] == "medicine" else "mixed_species"
    else:
        require(row["component"] == "Generated" and isinstance(meta.get("type_"), str) and meta["type_"], "Unknown Generated domain")
        place = meta.get("place")
        require(place is None or isinstance(place, str), "Invalid Generated place")
        fields.update(TYPE=meta["type_"], PLACE=" in " + place if place else "")
        key = "generated"
    return [job("does_believe", key, fields, {"kind": "boolean-object", "key": "does_believe"})]


def sentence_spans(answer):
    """Explicit stdlib segmentation variant; exact offsets, no NLP/model download."""
    spans, start = [], 0
    abbreviations = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.", "St."}
    for match in re.finditer(r"[.!?](?:[\"')\]]*)\s+|\n+", answer):
        end = match.start() if match.group().startswith("\n") else match.start() + len(match.group().rstrip())
        fragment = answer[start:end]
        last = fragment.split()[-1] if fragment.split() else ""
        if not match.group().startswith("\n") and (last in abbreviations or re.fullmatch(r"(?:[A-Z]\.)+", last)):
            continue
        left = start + len(fragment) - len(fragment.lstrip())
        right = end - (len(fragment) - len(fragment.rstrip()))
        if left < right:
            spans.append((left, right))
        start = match.end()
    fragment = answer[start:]
    left = start + len(fragment) - len(fragment.lstrip())
    right = len(answer.rstrip())
    if left < right:
        spans.append((left, right))
    # No non-whitespace input may disappear in segmentation.
    covered = "".join(answer[a:b] for a, b in spans)
    require(re.sub(r"\s", "", covered) == re.sub(r"\s", "", answer), "Sentence segmentation dropped source content")
    return spans


def nearest_demo(sentence, demos):
    """BM25Okapi defaults on upstream whitespace tokens; explicit lexical tie policy."""
    documents = [key.split(" ") for key in demos]
    n = len(documents)
    lengths = [len(doc) for doc in documents]
    average_length = sum(lengths) / n
    frequencies = Counter(token for doc in documents for token in set(doc))
    idf = {token: math.log(n - count + 0.5) - math.log(count + 0.5) for token, count in frequencies.items()}
    floor = 0.25 * sum(idf.values()) / len(idf)
    idf = {token: floor if score < 0 else score for token, score in idf.items()}
    scores = []
    for index, doc in enumerate(documents):
        counts = Counter(doc)
        score = sum(idf.get(token, 0) * counts[token] * 2.5 /
                    (counts[token] + 1.5 * (0.25 + 0.75 * lengths[index] / average_length))
                    for token in sentence.split(" "))
        scores.append((score, index))
    return list(demos)[max(scores)[1]]


def extraction_jobs(row, answer, short, catalog):
    require(row["dataset"] == "factscore_bios", "Atomic extraction requires biography input")
    jobs = []
    source = "data/v2/raw/FActScore/factscore/atomic_facts.py"
    for number, (start, end) in enumerate(sentence_spans(answer)):
        sentence = answer[start:end]
        demo_keys = list(catalog.demos)[:7] + [nearest_demo(sentence, catalog.demos)]
        original = ""
        for key in demo_keys:
            original += f"Please breakdown the following sentence into independent facts: {key}\n"
            original += "".join(f"- {fact}\n" for fact in catalog.demos[key]) + "\n"
        original += f"Please breakdown the following sentence into independent facts: {sentence}\n"
        # Escape literal demo braces before format; effective rubric is frozen per sentence.
        effective = original + ("\nReturn only a JSON object with exactly two keys: \"sentence\" (copy the input sentence exactly) "
                                "and \"facts\" (an array of independent fact strings). Use an empty array only when there are no factual claims. "
                                "Do not add facts not asserted in the sentence; preserve names, numbers and qualifications.\n")
        template = Template(original.replace("{", "{{").replace("}", "}}"),
                            effective.replace("{", "{{").replace("}", "}}"), {source: SOURCE_PINS[source], DEMOS_PATH: DEMOS_SHA},
                            "GLM strict JSON instead of upstream bullet parsing; stdlib exact-span sentence segmentation; seven fixed demos + one BM25 demo (highest-index tie). No spaCy postprocessing or silent fact fallback.")
        jobs.append(make_job(row, answer, short, "atomic_extraction", template, {},
                    {"kind": "atomic", "sentence": sentence}, {"topic": row["metadata"]["topic"],
                    "sentence_index": number, "sentence_start": start, "sentence_end": end,
                    "segmentation_version": "stdlib-exact-spans-v1", "demos_archive_sha256": DEMOS_SHA,
                    "demos_member_sha256": catalog.demos_member_sha, "demo_sentences": demo_keys}))
    return jobs


def request_contract(job):
    return {key: job[key] for key in ("cache_key", "identity", "payload", "request_sha256", "parser")}


class JobStore:
    def __init__(self, root, run_id, concurrency=8, dispatch=False):
        self.root = Path(root).resolve()
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id), "Invalid scoring run ID")
        require(type(concurrency) is int and 1 <= concurrency <= 8, "Concurrency must be 1..8")
        self.run_id, self.concurrency, self.dispatch = run_id, concurrency, dispatch
        self.directory = self.root / "scoring/v2-glm/results" / run_id
        self.journal_root = self.root / "scoring/v2-glm/request-journals"
        require(self.directory.resolve().is_relative_to(self.root) and self.journal_root.resolve().is_relative_to(self.root), "Scoring path escapes root")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal_root.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.root / "scoring/budget.json")
        self.inflight = client.InFlight()
        self.pilot_cache = {}
        self.pilot_contracts = {}
        self.pilot_markers = {}
        self._token = None
        self._token_lock = threading.Lock()
        self.halted = False
        self.replacement_authorizations = {}
        for path in sorted((self.directory / "recovery").glob("transport-replacement-*.json")):
            receipt = client.read_json(path.read_bytes())
            require(receipt.get("kind") == "glm-transport-replacement-v1" and receipt.get("run_id") == run_id
                    and isinstance(receipt.get("abandoned_attempt_ids"), list)
                    and len(receipt["abandoned_attempt_ids"]) == 1, "Invalid transport replacement authorization")
            self.replacement_authorizations[receipt["cache_key"]] = (
                receipt["request_sha256"], set(receipt["abandoned_attempt_ids"]))

    def index_pilots(self):
        for manifest_path in (self.root / "scoring/v2-glm/pilots").glob("*/manifest.json"):
            manifest = client.verify(manifest_path.parent, self.root)
            keys = {}
            for record in manifest["samples"]:
                request = client.read_json((manifest_path.parent / record["request_file"]).read_bytes())
                parser = {"kind": "enum", "values": ["A", "B", "C"] if record["dataset"] == "simpleqa_verified" else ["Yes", "No"]}
                identity = cache_identity({record["rubric_source_path"]: record["rubric_source_sha256"]}, record["answer_sha256"], request, parser)
                key = digest(identity)
                keys[record["request_sha256"]] = key
                self.pilot_contracts.setdefault(key, set()).add(manifest["pilot_id"])
            for path in (manifest_path.parent / "attempts").glob("*.response.json"):
                journal = client.read_json(path.read_bytes())
                key = keys.get(journal.get("request_sha256"))
                if key is not None:
                    self.pilot_cache.setdefault(key, []).append(path)
            for path in (manifest_path.parent / "attempts").glob("*.dispatch.json"):
                marker = client.read_json(path.read_bytes())
                key = keys.get(marker.get("request_sha256"))
                if key is not None:
                    self.pilot_markers.setdefault(key, set()).add(marker["attempt_id"])

    def reparse(self, job, path):
        raw_journal = path.read_bytes()
        journal = client.read_json(raw_journal)
        require(journal.get("request_sha256") == job["request_sha256"], "Journal request hash mismatch")
        result = {"cache_key": job["cache_key"], "attempt_id": journal["attempt_id"], "ok": False,
                  "label": None, "accounting": None, "state": "error", "journal_path": str(path),
                  "journal_sha256": client.sha(raw_journal), "error": None, "dispatched": False}
        if journal.get("kind") != "http_response" or journal.get("http_status") != 200:
            result["error"] = "Saved transport/HTTP failure; charge reserve retained"
            return result
        body = base64.b64decode(journal["raw_body_base64"], validate=True)
        require(client.sha(body) == journal["raw_body_sha256"], "Saved raw response hash mismatch")
        parsed = client.inspect_response(body, job["lineage"]["dataset"], lambda text: parse_final(text, job["parser"]))
        result.update(parsed)
        if parsed["accounting"] is not None:
            self.ledger.settle(journal["attempt_id"], parsed["accounting"], path)
        result["state"] = "success" if result["ok"] else "terminal_invalid" if parsed["accounting"] is not None else "unknown"
        if (parsed.get("accounting") or {}).get("bound_failure"):
            result["state"] = "bound_failure"
            self.halted = True
        immutable(self.directory / "repairs" / (journal["attempt_id"] + "-" + digest({"journal": result["journal_sha256"], "parser": job["parser"], "version": PARSER_VERSION}) + ".json"),
                  {"parser_version": PARSER_VERSION, "result": result})
        return result

    def _one(self, job, retry_invalid):
        key = job["cache_key"]
        identity = cache_identity(job["lineage"]["source_files"], job["lineage"]["answer_sha256"], job["payload"], job["parser"])
        require(job["payload"] == payload(job["payload"]["messages"][0]["content"]), "Scoring request violates fixed model/decoding contract")
        require(identity == job["identity"] and key == digest(identity) and job["request_sha256"] == identity["request_sha256"],
                "Scoring request/cache contract changed before dispatch or reuse")
        immutable(self.directory / "requests" / (digest({"key": key, "lineage": job["lineage"]}) + ".json"), job)
        with file_lock(self.journal_root / key):
            folder = self.journal_root / key
            folder.mkdir(exist_ok=True)
            immutable(folder / "job.json", request_contract(job))
            attempts_dir = folder / "attempts"
            attempts_dir.mkdir(exist_ok=True)
            paths = list(attempts_dir.glob("*.response.json")) + self.pilot_cache.get(key, [])
            parsed = [self.reparse(job, path) for path in paths]
            success = next((result for result in parsed if result["ok"]), None)
            if success is not None:
                return success
            ledger_attempts = self.ledger.snapshot().get("v2_glm", {}).get("attempts", {})
            known = {attempt_id for attempt_id, record in ledger_attempts.items() if record.get("job_cache_key") == key
                     or (record.get("pilot_id") in self.pilot_contracts.get(key, set())
                         and record.get("request_sha256") == job["request_sha256"])}
            known |= self.pilot_markers.get(key, set())
            known |= {result["attempt_id"] for result in parsed}
            known |= {path.name.removesuffix(".dispatch.json") for path in attempts_dir.glob("*.dispatch.json")}
            responded = {result["attempt_id"] for result in parsed}
            if known - responded:
                return {"cache_key": key, "attempt_id": sorted(known - responded)[0], "ok": False, "label": None,
                        "state": "unknown", "error": "Durable reserved/dispatched attempt has no raw response; replay refused", "dispatched": False}
            if parsed and (key not in retry_invalid or any(result["state"] != "terminal_invalid" for result in parsed)):
                authorization = self.replacement_authorizations.get(key)
                replaced = (authorization is not None and job["request_sha256"] == authorization[0]
                            and all(result["ok"] is not True for result in parsed)
                            and {result["attempt_id"] for result in parsed} <= authorization[1])
                if not replaced:
                    return parsed[-1]
            if not self.dispatch or self.halted:
                return {"cache_key": key, "ok": False, "label": None, "state": "pending", "error": None, "dispatched": False}
            attempt_id = uuid.uuid4().hex
            record = {"request_file": key + ".json", "request_sha256": job["request_sha256"],
                      "cache_key": key,
                      "aya_model": job["lineage"]["aya_model"], "id": job["lineage"]["id"], "dataset": job["lineage"]["dataset"]}
            with self._token_lock:
                if self._token is None:
                    self._token = client.token()
            result = client.execute_attempt(folder, "bulk:" + self.run_id, record, job["payload"],
                        self.ledger, self._token, self.inflight, attempt_id, lambda text: parse_final(text, job["parser"]))
            result["cache_key"] = key
            journal_path = attempts_dir / (attempt_id + ".response.json")
            result["journal_path"] = str(journal_path)
            result["journal_sha256"] = client.sha(journal_path.read_bytes()) if journal_path.exists() else None
            result["state"] = "success" if result["ok"] else "terminal_invalid" if result.get("accounting") is not None else "unknown"
            if "Insufficient shared headroom" in (result.get("error") or ""):
                result["state"] = "budget_exhausted"
                self.halted = True
            if (result.get("accounting") or {}).get("bound_failure") or "journal/checkpoint failure" in (result.get("error") or ""):
                self.halted = True
            immutable(attempts_dir / (attempt_id + ".outcome.json"), result)
            return result

    def evaluate(self, jobs, retry_invalid=()):
        jobs = list(jobs)
        retry_invalid = set(retry_invalid)
        require(retry_invalid <= {job["cache_key"] for job in jobs}, "Explicit retry key is not in these scoring jobs")
        results = []
        with file_lock(self.directory / "run"):
            # Every invocation has its own immutable baseline copy, never a full-run cost gate.
            session = self.directory / "sessions" / uuid.uuid4().hex
            session.mkdir(parents=True)
            self.ledger.start(session, required=None)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                for offset in range(0, len(jobs), self.concurrency):
                    batch = jobs[offset:offset + self.concurrency]
                    futures = [pool.submit(self._one, job, retry_invalid) for job in batch]
                    for job, future in zip(batch, futures):
                        try:
                            results.append({**future.result(), "lineage": job["lineage"], "stage": job["stage"]})
                        except (OSError, ValueError, TypeError, KeyError) as error:
                            self.halted = True
                            results.append({"cache_key": job["cache_key"], "stage": job["stage"], "lineage": job["lineage"],
                                            "ok": False, "label": None, "state": "accounting_or_journal_error",
                                            "error": str(error), "dispatched": False})
                    if self.halted:
                        break
            immutable(session / "outcomes.json", {"results": results, "halted": self.halted,
                      "peak_actual_in_flight": self.inflight.peak, "accounting": self.ledger.totals()})
        return results


def collected_generation(root, short, manifest, manifest_sha, generation, generation_sha):
    """Validate already-collected completed jobs only; never collect an unmarked job."""
    import launch_v2
    import v2_generation_queue as queue
    root = Path(root).resolve()
    class CollectedOnlyPath(type(root)):
        def exists(self, **kwargs):
            exists = super().exists(**kwargs)
            if self.name == "collected.json":
                require(exists, "Collected receipt disappeared; offline loader cannot download")
            return exists

        def open(self, mode="r", *args, **kwargs):
            require(not any(flag in mode for flag in "wax+"), "Offline collection cannot write artifacts")
            return super().open(mode, *args, **kwargs)

    contract = manifest["generation_contract"]["models"][short]
    expected_config = {key: contract[key] for key in ("model_id", "revision", "eos_token_id", "pad_token_id")}
    expected_config.update(chat_template_sha256=contract["chat_template"]["sha256"],
                           selected_template_sha256=contract["chat_template"]["selected_template_sha256"],
                           v2_manifest_sha256=manifest_sha, generation_sha256=generation_sha)
    records = {}
    for directory in sorted((root / f"runs/v2/{short}/jobs").glob("*")):
        marker = directory / "collected.json"
        if not marker.is_file():
            continue  # Uncollected/failed/running outputs cannot authorize an aggregate.
        require(directory.resolve().is_relative_to(root), "Collected job escapes scoring root")
        job = launch_v2.verify_job(directory)
        require(all(type(job["config"].get(key)) is type(value) and job["config"][key] == value
                    for key, value in expected_config.items()), "Collected job differs from verified generation contract")
        for example in job["examples"]:
            require(example["id"] in generation and example == generation[example["id"]][0],
                    "Collected job example differs from verified generation input")
        # collect's existing-marker branch revalidates runtime, rows and immutable hashes.
        require(marker.is_file(), "Collected receipt disappeared; offline loader cannot download")
        completed = queue.collect(CollectedOnlyPath(directory), job)
        for response in completed:
            identifier = response["id"]
            require(identifier not in records, "Duplicate ID across collected completed jobs")
            # The queue uses stdlib float parsing; align with our strict Decimal JSON
            # reader before comparison (real downloads include batch_seconds).
            records[identifier] = client.read_json(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return records


def load_available(root, models):
    from prepare_v2_data import verify
    root = Path(root)
    verified = verify(root)
    raw_manifest = (root / "data/v2/manifest.json").read_bytes()
    require(client.read_json(raw_manifest) == verified, "v2 inputs changed after verification")
    evaluation = (root / "data/v2/evaluation.jsonl").read_bytes()
    generation_raw = (root / "data/v2/generation.jsonl").read_bytes()
    reuse_raw = (root / "data/v2/reuse.json").read_bytes()
    for name, raw in (("evaluation.jsonl", evaluation), ("generation.jsonl", generation_raw), ("reuse.json", reuse_raw)):
        require(client.sha(raw) == verified["outputs"][name]["sha256"], "Current v2 input hash mismatch")
    indexed = client.jsonl_index(evaluation, "v2 evaluation")
    generation = client.jsonl_index(generation_raw, "v2 generation")
    reuse = client.read_json(reuse_raw)
    result = []
    for short in models:
        model = client.AYA_MODELS[short]
        record = reuse["models"][short]
        path, anchor = client.CHECKPOINTS[short]
        require(record["model_id"] == model and record["checkpoint"]["path"] == path and record["checkpoint"]["sha256"] == anchor, "Reuse checkpoint contract changed")
        raw = (root / path).read_bytes()
        require(client.sha(raw) == anchor, "Original response checkpoint hash mismatch")
        saved = client.jsonl_index(raw, path)
        responses = {}
        for entry in record["responses"]:
            identifier = entry["id"]
            require(identifier in saved and identifier in indexed and identifier not in responses, "Missing/duplicate anchored reusable response")
            row, _ = indexed[identifier]
            response, line = saved[identifier]
            require(entry["model_id"] == model and response.get("model_id") == model and response.get("status") == "ok"
                    and entry["dataset"] == row["dataset"] == response.get("dataset") and isinstance(response.get("response"), str)
                    and response["response"].strip() and client.sha(response["response"].encode()) == entry["response_sha256"], "Anchored reusable response mismatch")
            responses[identifier] = (response, line, path)
        require(len(responses) == 1978, "Expected all 1,978 anchored reusable responses")
        new_path = f"runs/v2/{short}/responses.jsonl"
        if (root / new_path).exists():
            completed = collected_generation(root, short, verified, client.sha(raw_manifest), generation, client.sha(generation_raw))
            for identifier, (response, line) in client.jsonl_index((root / new_path).read_bytes(), new_path).items():
                require(identifier in indexed and identifier not in responses, "New aggregate has unknown/reused ID")
                require(identifier in completed and exact_json(response, completed[identifier]),
                        "New aggregate record differs from verified collected completed output")
                if "input_row_sha256" in response:
                    require(response["input_row_sha256"] == client.sha(indexed[identifier][1]), "New aggregate input-row hash mismatch")
                if "v2_manifest_sha256" in response:
                    require(response["v2_manifest_sha256"] == client.sha(raw_manifest), "New aggregate v2 manifest mismatch")
                responses[identifier] = (response, line, new_path)
        for identifier, (response, line, path) in responses.items():
            row, input_line = indexed[identifier]
            result.append(({**row, "_input_row_sha256": client.sha(input_line), "_v2_manifest_sha256": client.sha(raw_manifest),
                            "_response_file": path, "_response_row_sha256": client.sha(line)}, response["response"], short))
    return result, {"manifest_sha256": client.sha(raw_manifest), "evaluation_sha256": client.sha(evaluation),
                    "reuse_sha256": client.sha(reuse_raw), "expected_counts_per_model": EXPECTED}


def factscore_metrics(rows, expected=180):
    complete = [row for row in rows if isinstance(row.get("facts"), list)]
    no_facts = [row for row in complete if not row["facts"]]
    verified = [row for row in complete if row["facts"] and all(type(fact.get("is_supported")) is bool for fact in row["facts"])]
    scores = [sum(fact["is_supported"] for fact in row["facts"]) / len(row["facts"]) for row in verified]
    gamma = [score * (1 if len(row["facts"]) > 10 else math.exp(1 - 10 / len(row["facts"]))) for row, score in zip(verified, scores)]
    return {"macro_precision": sum(scores) / len(scores) if scores else None,
            "gamma10_macro_precision": sum(gamma) / len(gamma) if gamma else None,
            "responding_fraction_of_completed_extractions": (len(complete) - len(no_facts)) / len(complete) if complete else None,
            "responding_fraction_full": (len(complete) - len(no_facts)) / expected if len(complete) == expected else None,
            "no_fact_excluded_from_precision": len(no_facts), "completed_extractions": len(complete),
            "fully_verified_responses": len(verified), "expected_responses": expected,
            "precision_response_coverage": len(verified) / expected,
            "total_facts_verified": sum(len(row["facts"]) for row in verified),
            "mean_facts_per_verified_response": sum(len(row["facts"]) for row in verified) / len(verified) if verified else None,
            "partial_metrics_are_not_full_benchmark_scores": len(complete) != expected or len(verified) + len(no_facts) != expected}


class EvidenceCatalog:
    def __init__(self, root):
        from freeze_factscore_evidence import verify
        self.root = Path(root)
        verify(self.root)  # Offline frozen-oldid integrity verification, never freeze/HTTP.
        raw = (self.root / "data/v2/factscore-evidence-manifest.json").read_bytes()
        manifest = client.read_json(raw)
        self.manifest_sha256 = client.sha(raw)
        self.subjects = {}
        for record in manifest["records"]:
            require(record["topic"] not in self.subjects and record["status"] == "ok", "Duplicate/invalid frozen evidence subject")
            text_entry = next(entry for entry in record["files"] if entry["path"].endswith("/text.txt"))
            path = self.root / text_entry["path"]
            require(path.resolve().is_relative_to((self.root / "data/v2/factscore_evidence").resolve()), "Evidence text path escapes frozen source")
            raw_text = path.read_bytes()
            require(client.sha(raw_text) == text_entry["sha256"], "Frozen evidence text hash mismatch")
            self.subjects[record["topic"]] = {**record, "text": raw_text.decode("utf-8"), "text_file": text_entry["path"], "text_sha256": text_entry["sha256"]}
        require(len(self.subjects) == 180, "FActScore requires exactly 180 frozen subjects")


def isolated_source(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Extraction source must be a root-relative isolated GLM path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to((root / "scoring/v2-glm").resolve()), "Extraction source escapes isolated GLM namespace")
    return path


def read_extraction_journal(root, relative):
    """Read provenance only: no settlement, dispatch, fallback or fact rewriting."""
    path = isolated_source(root, relative)
    raw = path.read_bytes()
    journal = client.read_json(raw)
    require(isinstance(journal, dict) and journal.get("kind") == "http_response"
            and type(journal.get("http_status")) is int and journal["http_status"] == 200,
            "Extraction provenance requires the original HTTP 200 raw receipt")
    body = base64.b64decode(journal["raw_body_base64"], validate=True)
    require(client.sha(body) == journal.get("raw_body_sha256"), "Extraction raw response hash mismatch")
    parsed = client.inspect_response(body, "factscore_bios", client.read_json)
    require(parsed["ok"], "Extraction receipt is invalid/truncated/untrusted: " + str(parsed["error"]))
    extracted = parsed["label"]
    facts = extracted.get("facts") if isinstance(extracted, dict) else extracted
    require(isinstance(facts, list) and all(isinstance(text, str) and text.strip() for text in facts),
            "Extraction receipt must contain a JSON facts array of exact strings")
    return journal, parsed, {"extraction_journal": path.relative_to(Path(root).resolve()).as_posix(),
                            "extraction_journal_sha256": client.sha(raw),
                            "extraction_response_sha256": client.sha(body)}


def extraction_packet(available, jobs, outcomes, binding, *, root):
    by_key = {outcome["cache_key"]: outcome for outcome in outcomes}
    by_subject = {}
    for job in jobs:
        identity = (job["lineage"]["aya_model"], job["lineage"]["id"])
        by_subject.setdefault(identity, []).append(job)
    responses = []
    for row, answer, short in available:
        if row["dataset"] != "factscore_bios":
            continue
        subject_jobs = by_subject.get((short, row["id"]), [])
        complete = bool(subject_jobs) and all(by_key.get(job["cache_key"], {}).get("ok") for job in subject_jobs)
        sentences, facts = [], []
        for job in subject_jobs:
            outcome = by_key.get(job["cache_key"], {})
            atoms = outcome["label"]["facts"] if outcome.get("ok") else None
            receipt = {}
            if atoms is not None:
                path = Path(outcome["journal_path"]).resolve()
                require(path.is_relative_to(Path(root).resolve()), "Extraction journal is outside the scoring root")
                journal, parsed, receipt = read_extraction_journal(root, path.relative_to(Path(root).resolve()).as_posix())
                require(journal.get("request_sha256") == job["request_sha256"] and parsed["label"] == outcome["label"]
                        and receipt["extraction_journal_sha256"] == outcome.get("journal_sha256"),
                        "Extraction outcome differs from the original GLM journal")
            sentences.append({"index": job["lineage"]["sentence_index"], "start": job["lineage"]["sentence_start"],
                              "end": job["lineage"]["sentence_end"], "sentence": job["parser"]["sentence"],
                              "cache_key": job["cache_key"], "facts": atoms,
                              "journal_sha256": outcome.get("journal_sha256")})
            if atoms is not None:
                for index, text in enumerate(atoms):
                    fact = {"aya_model": short, "id": row["id"], "topic": row["metadata"]["topic"],
                            "answer_sha256": client.sha(answer.encode()), "sentence_index": job["lineage"]["sentence_index"],
                            "fact_index": index, "text": text, "fact_sha256": client.sha(text.encode()),
                            "extraction_cache_key": job["cache_key"], "extraction_journal_sha256": outcome.get("journal_sha256")}
                    fact["fact_id"] = "fact-" + digest(fact)
                    fact.update(receipt)  # Add provenance without changing the existing fact identity/content.
                    facts.append(fact)
        responses.append({"aya_model": short, "id": row["id"], "topic": row["metadata"]["topic"],
                          "answer_sha256": client.sha(answer.encode()), "completed": complete,
                          "sentences": sentences, "facts": facts if complete else None})
    return {"kind": "glm-factscore-extractions-v1", "v2_inputs": binding, "responses": responses,
            "deviation": "Fresh GLM sentence-level extraction with upstream demos; stdlib segmentation and strict JSON, not old human annotations or upstream spaCy postprocessing."}


def validate_extraction_packet(packet, available, catalog, store, binding):
    require(packet.get("kind") == "glm-factscore-extractions-v1" and packet.get("v2_inputs") == binding, "Extraction input binding changed")
    identities = {(response["aya_model"], response["id"]) for response in packet["responses"]}
    require(len(identities) == len(packet["responses"]), "Duplicate extracted biography identity")
    by_identity = {(entry[2], entry[0]["id"]): entry for entry in available if entry[0]["dataset"] == "factscore_bios"}
    require(identities <= by_identity.keys(), "Saved extraction does not match available Aya responses")
    selected = [by_identity[(response["aya_model"], response["id"])] for response in packet["responses"]]
    jobs = [job for row, answer, short in selected for job in extraction_jobs(row, answer, short, catalog)]
    dispatch = store.dispatch
    store.dispatch = False
    try:
        outcomes = store.evaluate(jobs)
    finally:
        store.dispatch = dispatch
    expected = extraction_packet(selected, jobs, outcomes, binding, root=store.root)
    require(expected == packet, "Extraction fact identities/content differ from durable GLM journals")
    return selected


def retrieval_ready(packet, evidence, *, root):
    facts = []
    for response in packet["responses"]:
        if not response["completed"]:
            continue
        source = evidence.subjects[response["topic"]]
        for fact in response["facts"]:
            _, parsed, receipt = read_extraction_journal(root, fact.get("extraction_journal"))
            require(all(fact.get(key) == value for key, value in receipt.items()), "Fact extraction receipt provenance changed")
            extracted = parsed["label"]
            atoms = extracted.get("facts") if isinstance(extracted, dict) else extracted
            index = fact.get("fact_index")
            require(type(index) is int and 0 <= index < len(atoms) and atoms[index] == fact["text"]
                    and client.sha(fact["text"].encode()) == fact.get("fact_sha256"),
                    "Exported fact content/index differs from fresh GLM extraction")
            facts.append({**fact, "fact": fact["text"], "response_sha256": fact["answer_sha256"],
                          "source_text_sha256": source["text_sha256"], "source_text_file": source["text_file"],
                          "title": source["title"], "revision": source["revision"], "oldid_url": source["url"]})
    topics = {fact["topic"] for fact in facts}
    exported_evidence = [{"topic": topic, "text": source["text"], "text_sha256": source["text_sha256"],
                          "revision": source["revision"]} for topic, source in evidence.subjects.items() if topic in topics]
    facts_raw = b"".join(client.canonical(fact) for fact in facts)
    facts_sha = client.sha(facts_raw)
    relative = "scoring/v2-glm/facts-exports/facts-" + facts_sha + ".jsonl"
    immutable_content(isolated_source(root, relative), facts_raw)
    return {"kind": "glm-factscore-retrieval-ready-v1", "extraction_packet_sha256": client.sha(json_bytes(packet)),
            "evidence_manifest_sha256": evidence.manifest_sha256, "retrieval_model": GTR_MODEL,
            "provenance": {"evidence_manifest_sha256": evidence.manifest_sha256,
                           "facts_file": {"path": relative, "sha256": facts_sha}},
            "retrieval_revision": GTR_REVISION, "tokenizer_model": TOKENIZER_MODEL, "tokenizer_revision": TOKENIZER_REVISION,
            "passage_tokens": 256, "passages_per_fact": 5, "evidence": exported_evidence, "facts": facts,
            "required_retrieval_input": "Controller v2_factscore_retrieval.run(payload): retrieval.jsonl plus completed manifest.json, pinned remote GTR/title+fact/inner product and RoBERTa 256-token subject-local chunks. No local model/tokenizer loading."}


def verification_jobs(ready, retrieval, evidence, available, catalog):
    require(isinstance(retrieval, list), "Legacy retrieval dictionaries/character spans are not approved; controller JSONL plus completed manifest required")
    return remote_verification_jobs(ready, retrieval, evidence, available)


def read_remote_retrieval(path, ready, manifest_path=None):
    """Consume the controller's completed remote JSONL and input-hash receipt."""
    require(Path(path).suffix == ".jsonl", "Controller retrieval must be JSONL with a completed manifest; no legacy JSON path")
    records = [client.read_json(line) for line in Path(path).read_bytes().splitlines()]
    manifest_path = Path(manifest_path) if manifest_path else Path(path).with_name("manifest.json")
    manifest = client.read_json(manifest_path.read_bytes())
    require(isinstance(manifest, dict) and all(isinstance(record, dict) for record in records),
            "Remote retrieval receipt/records must be JSON objects")
    require(manifest.get("stage") == "completed" and type(manifest.get("facts")) is int
            and manifest["facts"] == len(records) == len(ready["facts"]), "Remote retrieval is incomplete/missing fact outputs")
    for key, expected in (("model", GTR_MODEL), ("revision", GTR_REVISION), ("tokenizer", TOKENIZER_MODEL),
                          ("tokenizer_revision", TOKENIZER_REVISION), ("passage_tokens", 256),
                          ("passages_per_fact", 5), ("passage_scope", "original subject only"),
                          ("similarity", "np.inner; model default pooling/normalization")):
        require(type(manifest.get(key)) is type(expected) and manifest[key] == expected,
                f"Remote retrieval receipt {key} mismatch")
    # Match the exact serialization used by the controller, not our JSONL encoder.
    expected_input_sha = client.sha(json.dumps(ready, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())
    require(manifest.get("input_sha256") == expected_input_sha, "Remote retrieval input receipt does not bind this frozen extraction/evidence export")
    return records


def remote_verification_jobs(ready, records, evidence, available):
    expected_facts = {(fact["aya_model"], fact["fact_id"]): fact for fact in ready["facts"]}
    require(len(expected_facts) == len(ready["facts"]), "Duplicate exported fact identity")
    observed = set()
    by_identity = {(short, row["id"]): (row, answer) for row, answer, short in available}
    jobs = []
    source_path = "data/v2/raw/FActScore/factscore/factscorer.py"
    def normalized(text):
        return " ".join(unicodedata.normalize("NFC", text).split())
    for record in records:
        require(isinstance(record, dict), "Remote retrieval record must be a JSON object")
        short = record.get("aya_model")
        if short is None:
            short = next((name for name, model in client.AYA_MODELS.items() if record.get("model_id") == model), None)
        identity = (short, record.get("fact_id"))
        require(identity in expected_facts and identity not in observed, "Unknown/duplicate remote fact identity")
        observed.add(identity)
        fact = expected_facts[identity]
        require(all(key in record and type(record[key]) is type(value) and record[key] == value
                    for key, value in fact.items()), "Remote output changed original fact content/answer/extraction lineage")
        source = evidence.subjects[fact["topic"]]
        require(record.get("status") == "ok" and record.get("evidence_revision") == source["revision"]
                and type(record.get("evidence_revision")) is int
                and record.get("evidence_text_sha256") == source["text_sha256"], "Remote retrieval frozen-oldid/text binding mismatch")
        for key, expected in (("retrieval_model", GTR_MODEL), ("retrieval_revision", GTR_REVISION),
                              ("tokenizer_model", TOKENIZER_MODEL), ("tokenizer_revision", TOKENIZER_REVISION)):
            require(record.get(key) == expected, f"Remote output {key} mismatch")
        require(record.get("query") == fact["topic"] + " " + fact["fact"].strip(), "Remote GTR query changed official title+fact formulation")
        passages = record.get("passages")
        # Official top-k returns min(5, available chunks); never fabricate/pad short pages.
        require(isinstance(passages, list) and 1 <= len(passages) <= 5, "Remote top-five passages missing/invalid")
        indices, previous_score, context, selected = set(), None, "", []
        for passage in passages:
            require(isinstance(passage, dict), "Remote passage must be a JSON object")
            index, score, text = passage.get("passage_index"), passage.get("score"), passage.get("text")
            require(type(index) is int and index >= 0 and index not in indices, "Invalid/duplicate remote passage index")
            indices.add(index)
            require(isinstance(score, (int, float, Decimal)) and not isinstance(score, bool), "Invalid remote inner-product score")
            score = Decimal(str(score))
            require(score.is_finite() and (previous_score is None or score <= previous_score), "Nonfinite/unsorted remote passage scores")
            previous_score = score
            require(passage.get("title") == fact["topic"] and isinstance(text, str) and text.strip()
                    and client.sha(text.encode()) == passage.get("text_sha256"), "Remote passage title/text hash mismatch")
            cleaned = text.replace("<s>", "").replace("</s>", "")
            require(normalized(cleaned) and normalized(cleaned) in normalized(source["text"]),
                    "Remote passage is not present in this subject's frozen text (no local tokenizer/fallback)")
            selected.append({"title": passage["title"], "text": text, "passage_index": index,
                             "text_sha256": passage["text_sha256"]})
        require(len(selected) == 5 or indices == set(range(len(selected))),
                "Short-page top-k must contain every available chunk index, without padding or gaps")
        for passage in reversed(selected):
            context += "Title: {}\nText: {}\n\n".format(passage["title"], passage["text"].replace("<s>", "").replace("</s>", ""))
        definition = f"Answer the question about {fact['topic']} based on the given context.\n\n" + context.strip()
        if definition[-1] not in string.punctuation:
            definition += "."
        prompt = f"{definition.strip()}\n\nInput: {fact['fact'].strip()} True or False?\nOutput:"
        escaped = prompt.replace("{", "{{").replace("}", "}}")
        selection_sha = digest({"retrieval_model": GTR_MODEL, "retrieval_revision": GTR_REVISION,
                               "tokenizer_model": TOKENIZER_MODEL, "tokenizer_revision": TOKENIZER_REVISION,
                               "passage_tokens": 256, "passages": selected})
        template = Template(escaped, escaped, {source_path: SOURCE_PINS[source_path],
            "data/v2/factscore-evidence-manifest.json": evidence.manifest_sha256,
            source["text_file"]: source["text_sha256"], "controller-gtr-selection": selection_sha},
            "Pinned frozen oldid source; controller remote GTR/RoBERTa256 top min(5,N) subject-local chunks; strict True/False, no upstream permissive fallback. Passage text is checked against frozen text without local tokenization.")
        row, answer = by_identity[(short, fact["id"])]
        jobs.append(make_job(row, answer, short, "fact_verification", template, {},
            {"kind": "enum", "values": ["True", "False"]}, {"fact": fact, "retrieval_record_sha256": client.sha(json_bytes(record)),
            "passage_indices": [p["passage_index"] for p in selected], "passages_returned": len(selected),
            "retrieval_model": GTR_MODEL, "retrieval_revision": GTR_REVISION,
            "tokenizer_model": TOKENIZER_MODEL, "tokenizer_revision": TOKENIZER_REVISION,
            "evidence_manifest_sha256": evidence.manifest_sha256, "selection_sha256": selection_sha}))
    require(observed == set(expected_facts), "Remote retrieval omits extracted fact identities")
    return jobs


def fixed_metrics(outcomes, available, models, tasks):
    output = []
    source_rows = {(short, row["id"]): row for row, _, short in available}
    require(len(source_rows) == len(available), "Duplicate available metric identity")
    stages = {"simpleqa_verified": "simpleqa", "abstention_bench": "abstention",
              "hallulens_precisewikiqa": "abstention", "hallulens_nonexistentrefusal": "does_believe"}
    labels = {("simpleqa_verified", "simpleqa"): (str, {"A", "B", "C"}),
              ("abstention_bench", "abstention"): (str, {"Yes", "No"}),
              ("abstention_bench", "correctness"): (str, {"correct", "incorrect"}),
              ("hallulens_precisewikiqa", "abstention"): (bool, {True, False}),
              ("hallulens_precisewikiqa", "correctness"): (str, {"CORRECT", "INCORRECT", "UNVERIFIABLE"}),
              ("hallulens_nonexistentrefusal", "does_believe"): (bool, {True, False})}
    seen = set()
    for outcome in outcomes:
        lineage, stage = outcome["lineage"], outcome["stage"]
        short, task, identifier = lineage["aya_model"], lineage["dataset"], lineage["id"]
        identity = (short, task, identifier, stage)
        require(short in models and task in tasks and (short, identifier) in source_rows
                and source_rows[(short, identifier)]["dataset"] == task, "Metric outcome has unknown source identity")
        require((task, stage) in labels and identity not in seen, "Unknown/duplicate metric stage identity")
        seen.add(identity)
        require(type(outcome.get("ok")) is bool, "Metric outcome validity must be boolean")
        if outcome["ok"]:
            kind, values = labels[(task, stage)]
            require(type(outcome.get("label")) is kind and outcome["label"] in values, "Invalid exact metric label")
    for short in models:
        for task in tasks:
            expected = EXPECTED[task]
            results = [o for o in outcomes if o["lineage"]["aya_model"] == short and o["lineage"]["dataset"] == task]
            valid = [o for o in results if o.get("ok")]
            primary_outcomes = [o for o in results if o["stage"] == stages[task]]
            primary = [o for o in valid if o["stage"] == stages[task]]
            correctness = {o["lineage"]["id"]: o["label"] for o in valid if o["stage"] == "correctness"}
            available_count = sum(row["dataset"] == task and model == short for row, _, model in available)
            require(available_count <= expected, "Too many available benchmark metric rows")
            record = {"aya_model": short, "dataset": task, "expected_responses": expected,
                      "usable_responses_available": available_count, "responses_missing": expected - available_count,
                      "primary_valid": len(primary), "primary_coverage": len(primary) / expected,
                      "primary_missing_or_error": expected - len(primary),
                      "primary_missing": expected - len(primary_outcomes),
                      "primary_errors_or_pending": len(primary_outcomes) - len(primary),
                      "errors_or_pending": sum(not o.get("ok") for o in results),
                      "full_primary_complete": len(primary) == expected,
                      "full_conditional_complete": len(primary) == expected}
            if task == "simpleqa_verified":
                counts = Counter(o["label"] for o in primary)
                full = arithmetic.simpleqa([o["label"] for o in primary], expected)
                observed = arithmetic.simpleqa([o["label"] for o in primary], len(primary))
                record.update(label_counts=dict(counts), observed_accuracy=observed["correct_answer_yield"],
                              full_accuracy=full["correct_answer_yield"] if record["full_primary_complete"] else None)
                for key in ("correct_answer_yield", "answer_coverage", "accuracy_when_attempted"):
                    record["observed_" + key] = observed[key]
                    record["full_" + key] = full[key] if record["full_primary_complete"] else None
            elif task == "abstention_bench":
                paired, eligible_ids = [], set()
                for outcome in primary:
                    identifier = outcome["lineage"]["id"]
                    desired = source_rows[(short, identifier)]["should_abstain"]
                    require(type(desired) is bool, "Invalid released AbstentionBench metric label")
                    row = {"should_abstain": desired, "abstained": outcome["label"] == "Yes"}
                    if not desired and not row["abstained"]:
                        eligible_ids.add(identifier)
                        if identifier in correctness:
                            row["correct"] = correctness[identifier] == "correct"
                    paired.append(row)
                conditional_ids = {o["lineage"]["id"] for o in results if o["stage"] == "correctness"}
                require(conditional_ids <= eligible_ids, "Conditional correctness has no valid eligible primary pair")
                full = arithmetic.abstention(paired, expected)
                observed = arithmetic.abstention(paired, len(paired))
                resolved = [r for r in paired if r["should_abstain"] or r["abstained"] or "correct" in r]
                joint = arithmetic.abstention(resolved, len(resolved))
                conditional_missing = full["conditional_correctness_missing_or_error"]
                record["full_conditional_complete"] = record["full_primary_complete"] and conditional_missing == 0
                record.update(observed_abstention_agreement=observed["behavior_accuracy"],
                              observed_behavior_accuracy=observed["behavior_accuracy"],
                              full_behavior_accuracy=full["behavior_accuracy"] if record["full_primary_complete"] else None,
                              correctness_eligible_nonabstained_answerable=len(eligible_ids), correctness_valid=len(correctness),
                              correctness_missing_or_error=conditional_missing,
                              joint_valid=len(resolved), joint_missing_or_error=expected - len(resolved),
                              joint_successes=full["joint_successes"], observed_joint_task_success=joint["joint_task_success"],
                              full_joint_task_success=full["joint_task_success"] if record["full_conditional_complete"] else None,
                              observed_correctness=arithmetic.ratio(sum(x == "correct" for x in correctness.values()), len(correctness)),
                              full_correctness=arithmetic.ratio(sum(x == "correct" for x in correctness.values()), len(eligible_ids))
                                  if record["full_conditional_complete"] else None)
            elif task == "hallulens_nonexistentrefusal":
                full = arithmetic.nonexistent([o["label"] for o in primary], expected)
                observed = arithmetic.nonexistent([o["label"] for o in primary], len(primary))
                record.update(false_acceptances=full["false_acceptances"],
                              observed_false_acceptance_rate=observed["false_acceptance_rate"],
                              full_false_acceptance_rate=full["false_acceptance_rate"] if record["full_primary_complete"] else None,
                              observed_refusal_rate=arithmetic.ratio(len(primary) - full["false_acceptances"], len(primary)),
                              full_refusal_rate=arithmetic.ratio(expected - full["false_acceptances"], expected) if record["full_primary_complete"] else None)
            else:
                abstentions = {o["lineage"]["id"]: o["label"] for o in primary}
                paired = [{"abstained": value, "correctness": correctness[identifier]}
                          for identifier, value in abstentions.items() if identifier in correctness]
                full = arithmetic.precise_wiki(paired, expected)
                observed = arithmetic.precise_wiki(paired, len(paired))
                record["full_conditional_complete"] = record["full_primary_complete"] and len(paired) == expected
                record.update(correctness_valid=len(correctness), correctness_missing_or_error=expected - len(correctness),
                    paired_valid=len(paired), paired_missing_or_error=expected - len(paired), nonabstained_paired=full["non_abstained"],
                    observed_hallucination_rate_nonabstained=observed["hallucination_rate_when_not_abstained"],
                    full_hallucination_rate_nonabstained=full["hallucination_rate_when_not_abstained"] if record["full_conditional_complete"] else None,
                    observed_refusal_rate=arithmetic.ratio(sum(abstentions.values()), len(abstentions)),
                    full_refusal_rate=arithmetic.ratio(sum(abstentions.values()), expected) if record["full_primary_complete"] else None,
                    observed_correct_rate=observed["correct_answer_rate"],
                    full_correct_rate=full["correct_answer_rate"] if record["full_conditional_complete"] else None)
            record["partial_metrics_are_not_full_benchmark_scores"] = not (record["full_primary_complete"] and record["full_conditional_complete"])
            output.append(record)
    return output


def score_fixed(root, run_id, models=("tiny", "8b"), tasks=FIXED_TASKS, concurrency=8, dispatch=False,
                retry_invalid=(), only_keys=None):
    available, binding = load_available(root, models)
    catalog = SourceCatalog(root)
    store = JobStore(root, run_id, concurrency, dispatch)
    store.index_pilots()
    selected = [entry for entry in available if entry[0]["dataset"] in tasks]
    retry_invalid = set(retry_invalid)
    only = None if only_keys is None else set(only_keys)
    outcomes, all_keys = [], set()
    for task in tasks:
        task_entries = [entry for entry in selected if entry[0]["dataset"] == task]
        jobs = [job for row, answer, short in task_entries for job in fixed_jobs(row, answer, short, catalog)]
        keys = {job["cache_key"] for job in jobs}
        all_keys.update(keys)
        if only is not None:
            jobs = [job for job in jobs if job["cache_key"] in only]
        if not jobs:
            continue
        outcomes.extend(store.evaluate(jobs, retry_invalid & keys))
        if task == "abstention_bench" and not store.halted:
            primary = {(o["lineage"]["aya_model"], o["lineage"]["id"]): o for o in outcomes
                       if o["stage"] == "abstention" and o.get("ok")}
            correctness_jobs = []
            for row, answer, short in task_entries:
                judgment = primary.get((short, row["id"]))
                correctness_jobs.extend(job for job in fixed_jobs(row, answer, short, catalog,
                    abstention=judgment["label"] if judgment else None) if job["stage"] == "correctness")
            correctness_keys = {job["cache_key"] for job in correctness_jobs}
            all_keys.update(correctness_keys)
            if correctness_jobs:
                outcomes.extend(store.evaluate(correctness_jobs, retry_invalid & correctness_keys))
        if store.halted:
            break
    if not store.halted:
        require(retry_invalid <= all_keys, "Retry key not in current available scoring jobs")
        if only is not None:
            require(only <= all_keys, "Requested key is not in current available scoring jobs")
    result = {"kind": "glm-fixed-benchmark-results-v1", "v2_inputs": binding, "results": outcomes,
              "metrics": fixed_metrics(outcomes, selected, models, tasks), "halted": store.halted,
              "peak_actual_in_flight": store.inflight.peak, "accounting": store.ledger.totals(),
              "no_human_adjudication_or_old_grades": True}
    path = store.directory / "snapshots" / (uuid.uuid4().hex + ".json")
    immutable(path, result)
    return {"snapshot": str(path), "metrics": result["metrics"], "halted": store.halted, "accounting": result["accounting"]}


def factscore_stage(root, run_id, stage, models=("tiny", "8b"), concurrency=8, dispatch=False,
                     extraction_path=None, retrieval_path=None, retry_invalid=(), retrieval_manifest_path=None):
    if stage == "verify":
        require(retrieval_path is not None and Path(retrieval_path).suffix == ".jsonl",
                "Verification requires controller JSONL plus completed manifest; legacy JSON is rejected before execution")
    available, binding = load_available(root, models)
    available = [entry for entry in available if entry[0]["dataset"] == "factscore_bios"]
    catalog = SourceCatalog(root)
    store = JobStore(root, run_id, concurrency, dispatch)
    if stage == "extract":
        jobs = [job for row, answer, short in available for job in extraction_jobs(row, answer, short, catalog)]
        outcomes = store.evaluate(jobs, retry_invalid)
        packet = extraction_packet(available, jobs, outcomes, binding, root=store.root)
    else:
        require(extraction_path is not None, "This stage requires the immutable GLM extraction snapshot")
        packet = client.read_json(Path(extraction_path).read_bytes())
        available = validate_extraction_packet(packet, available, catalog, store, binding)
        evidence = EvidenceCatalog(root)
        ready = retrieval_ready(packet, evidence, root=store.root)
        if stage == "retrieval-ready":
            packet = ready
        else:
            require(stage == "verify" and retrieval_path is not None, "Verification requires explicit remote GTR passage selections")
            retrieval = read_remote_retrieval(retrieval_path, ready, retrieval_manifest_path)
            jobs = verification_jobs(ready, retrieval, evidence, available, catalog)
            outcomes = store.evaluate(jobs, retry_invalid)
            by_fact = {o["lineage"]["fact"]["fact_id"]: o for o in outcomes}
            responses = []
            for response in packet["responses"]:
                facts = None if response["facts"] is None else [{**fact,
                        "is_supported": (by_fact[fact["fact_id"]]["label"] == "True") if by_fact.get(fact["fact_id"], {}).get("ok") else None,
                        "verification_cache_key": by_fact.get(fact["fact_id"], {}).get("cache_key")} for fact in response["facts"]]
                responses.append({**response, "facts": facts})
            packet = {"kind": "glm-factscore-verification-results-v1", "v2_inputs": binding,
                      "retrieval_ready_sha256": client.sha(json_bytes(ready)), "results": outcomes, "responses": responses,
                      "metrics": {short: factscore_metrics([r for r in responses if r["aya_model"] == short]) for short in models},
                      "accounting": store.ledger.totals(), "halted": store.halted}
    path = store.directory / "snapshots" / (stage + "-" + uuid.uuid4().hex + ".json")
    immutable(path, packet)
    return {"snapshot": str(path), "kind": packet["kind"], "metrics": packet.get("metrics"),
            "facts_ready": len(packet.get("facts", [])), "responses": len(packet.get("responses", [])), "halted": store.halted}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "factscore-extract", "factscore-retrieval-ready", "factscore-verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(client.AYA_MODELS), default=["tiny", "8b"])
    parser.add_argument("--tasks", nargs="+", choices=FIXED_TASKS, default=list(FIXED_TASKS))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Explicitly dispatch never-attempted jobs; no implicit retries")
    parser.add_argument("--retry-invalid", nargs="*", default=[], metavar="EXACT_CACHE_KEY")
    parser.add_argument("--only-keys", nargs="+", default=None, metavar="EXACT_CACHE_KEY",
                        help="Restrict scoring to these cache keys (recovery and diagnostics)")
    parser.add_argument("--extractions", type=Path)
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        require(len(args.models) == len(set(args.models)) and len(args.tasks) == len(set(args.tasks)), "Duplicate model/task selection")
        if args.command == "score":
            result = score_fixed(args.root, args.run_id, args.models, args.tasks, args.concurrency,
                                 args.execute, args.retry_invalid, args.only_keys)
        else:
            stage = args.command.removeprefix("factscore-")
            require(stage != "retrieval-ready" or not args.execute, "Retrieval-ready is offline only; controller supplies GTR ranking")
            result = factscore_stage(args.root, args.run_id, stage, args.models, args.concurrency,
                                    args.execute, args.extractions, args.retrieval, args.retry_invalid, args.retrieval_manifest)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"{args.command} failed: {error}\n")
    print(json_bytes(result).decode("utf-8"))
    if result.get("halted"):
        parser.exit(1)


if __name__ == "__main__":
    main()
