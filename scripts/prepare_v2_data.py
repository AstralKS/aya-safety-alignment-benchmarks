"""Assemble or verify the approved v2 data split, entirely offline.

Only the four named data/v2 outputs are writable. Sources are verified against
download receipts before parsing; legacy rows are copied byte-for-byte. This
module never imports a scorer, loads a model, or performs inference.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from factscore_selection import FACTSCORE_PATH, FACTSCORE_MEMBER, SELECTED_COUNT, select_archive


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20261111
PINS = {
    "alexanderstern/hallulens": "290730dd150bbc784c9abb111890b996b7d72b0e",
    "facebookresearch/HalluLens": "80307ac6bc9fd396a38b7a0de4196b931611b728",
    "yinzhangyue/SelfAware": "6b15164a3e11a2be714afde0615879dcde5c01f8",
    "shmsw25/FActScore": "f28272deffcf33efc1f1117d5479c10bb75221a9",
}
SELFAWARE_SCORER_MODEL = "princeton-nlp/sup-simcse-roberta-large"
MODELS = {"tiny": "CohereLabs/tiny-aya-global", "8b": "CohereLabs/aya-expanse-8b"}
MODEL_REVISIONS = {
    "tiny": "af89d219b53ed9b13b8a4645f9c8028973510324",
    "8b": "5062468bf9bc0c6035fd64e06274333ec127d980",
}
REUSE_COUNTS = {"simpleqa_verified": 1000, "abstention_bench": 978}
NEW_COUNTS = {
    "hallulens_precisewikiqa": 1000,
    "hallulens_nonexistentrefusal": 3000,
    "selfaware": 3369,
    "factscore_bios": SELECTED_COUNT,
}
GENERATION_FIELDS = ("id", "dataset", "component", "question", "system_prompt", "max_new_tokens")
RECEIPTS = ("data/v2/selfaware-source-receipt.json", "data/v2/new-sources-receipt.json")
# Trust anchors from the untouched source receipts and original preflight audit.
# Never derive expected hashes from a mutable file's own checksum declaration.
RECEIPT_SHA256 = {
    RECEIPTS[0]: "e7839a8830a6090ee45372628fc60d581a241d566af9d705ac772764def91228",
    RECEIPTS[1]: "fdda39a0053bcab5988c64dae768e5004c3021d9ae93b594e9a44510420dcf65",
}
PREFLIGHT_PATH = "scoring/v2-glm/preflight-input-integrity.json"
PREFLIGHT_SHA256 = "9bedb8765581c17e94d474fc2f1b355280548b181f30a0dc59af7bf699db1e51"
FROZEN_INPUT_SHA256 = {
    "data/evaluation.jsonl": "b4aaadab39cc65ee5356ce25528afaea48e992bbc3da6d61450f317c7292c549",
    "data/manifest.json": "3f80a7a2bcb976b55c8025342a453a735429b549f142160086d5c26e6ff206a3",
    "runs/tiny-baseline/checkpoint/responses.jsonl": "f702cd1c09170bfc64ca85d4821ba688025f71ac15b03917f466fa578498ffda",
    "runs/8b-baseline/checkpoint/responses.jsonl": "46b92a3471d0751be99c999e94eeb3c6739e2e190a77685cefcc435f488be156",
}
# Saved run metadata/notebooks were inspected read-only. Notebooks are hashed,
# never imported or executed. Their hashes bind the statically audited policies.
RUNTIME_SHA256 = {
    "runs/tiny-baseline/launch.json": "7ad780e4896de6419a4ed8b2a9bbc504cbdd0ae051767b0871075bcd68cf6386",
    "runs/8b-baseline/launch.json": "ef1b5702d5c79940096db341a3c11cbf8906aa2388b6203a66126b56fcbef3a4",
    "runs/tiny-baseline/checkpoint/manifest.json": "9947816fa75b2d178b72b2b6b956d37b27f60a57c212073bec31a1d8a8f34532",
    "runs/8b-baseline/checkpoint/manifest.json": "5ab95ac14ee817ab5fe9f09fd48141abe70916fa4f295a85ba33f1c1a4dcd6b4",
    "runs/tiny-baseline/checkpoint/chat_template.txt": "3d5e180dfaa3aacd0eab5c71db4d381d7131a99c5c1a5891fcf1092b9a4be799",
    "runs/8b-baseline/checkpoint/chat_template.txt": "e03da95a536bb1e3a5c5cd77cbec1e78e146d432058ea17c9ce33b772cdffd52",
    "notebooks/tiny-baseline.ipynb": "7bc03622e68809b4fe41034bb23f0747974e43e59b66c366a9ceec839510cc50",
    "notebooks/8b-baseline.ipynb": "8876806ad1d157076f92a3221bc9a176e3ca96edce9dfb93059788f662ce3ab5",
}
HALLULENS_ROOT = "data/v2/raw/hallulens_snapshot/"
PARQUET_PATHS = {
    name: f"{HALLULENS_ROOT}{name}/test-00000-of-00001.parquet"
    for name in ("precise_wiki", "mixed_entities", "generated_entities")
}
SELFAWARE_PATH = "data/v2/raw/SelfAware/data/SelfAware.json"
FACTSCORE_URL = "https://drive.usercontent.google.com/uc?id=155exEdKs7R21gZF4G-x54-XN3qswBcPo&export=download"
KNOWN_404 = (
    "https://raw.githubusercontent.com/facebookresearch/HalluLens/"
    f"{PINS['facebookresearch/HalluLens']}/tasks/shortform/prompt.py"
)
OUTPUT_NAMES = ("evaluation.jsonl", "generation.jsonl", "reuse.json", "manifest.json")


class PreparationError(ValueError):
    """A source, checkpoint, or destination violates the assembly contract."""


def _require(condition, message):
    if not condition:
        raise PreparationError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _json_line(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _jsonl(data, label):
    result = []
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        _require(line.strip(), f"{label}:{number}: blank JSONL row")
        value = json.loads(line)
        _require(isinstance(value, dict), f"{label}:{number}: expected object")
        result.append((value, line))
    return result


def _safe_path(root, relative, prefix):
    _require(isinstance(relative, str) and relative.startswith(prefix), f"Unexpected path: {relative}")
    path = root / relative
    _require(path.resolve().is_relative_to((root / prefix).resolve()), f"Path escapes {prefix}: {relative}")
    _require(path.resolve().is_relative_to(root), f"Path escapes root: {relative}")
    return path


def _anchored_bytes(root, relative, expected_sha256, label):
    content = _safe_path(root, relative, relative.split("/", 1)[0] + "/").read_bytes()
    _require(_sha(content) == expected_sha256, f"{label} hash mismatch: {relative}")
    return content


def _verified_preflight(root):
    content = _anchored_bytes(root, PREFLIGHT_PATH, PREFLIGHT_SHA256, "Preflight integrity receipt")
    receipt = json.loads(content)
    entries = receipt.get("files", [])
    records = {entry["path"]: entry for entry in entries}
    _require(len(records) == len(entries), "Duplicate preflight integrity path")
    verified = {}
    for relative, expected in FROZEN_INPUT_SHA256.items():
        _require(relative in records, f"Missing frozen input record: {relative}")
        record = records[relative]
        _require(record.get("sha256") == expected, f"Frozen input record disagrees with source audit: {relative}")
        data = _anchored_bytes(root, relative, expected, "Frozen input")
        _require(len(data) == record.get("bytes"), f"Frozen input size mismatch: {relative}")
        verified[relative] = data
    return verified, {
        "receipt_path": PREFLIGHT_PATH, "receipt_sha256": _sha(content),
        "files": [records[relative] for relative in sorted(verified)],
    }


def _expected_source_url(path):
    if path == FACTSCORE_PATH:
        return FACTSCORE_URL
    if path.startswith(HALLULENS_ROOT):
        return (
            "https://huggingface.co/datasets/alexanderstern/hallulens/resolve/"
            f"{PINS['alexanderstern/hallulens']}/{path[len(HALLULENS_ROOT):]}"
        )
    for folder, repo in (("HalluLens", "facebookresearch/HalluLens"),
                         ("SelfAware", "yinzhangyue/SelfAware"),
                         ("FActScore", "shmsw25/FActScore")):
        prefix = f"data/v2/raw/{folder}/"
        if path.startswith(prefix):
            return f"https://raw.githubusercontent.com/{repo}/{PINS[repo]}/{path[len(prefix):]}"
    raise PreparationError(f"No pinned source for {path}")


def _verified_sources(root):
    sources, receipts = {}, []
    for relative in RECEIPTS:
        content = _anchored_bytes(root, relative, RECEIPT_SHA256[relative], "Source receipt")
        receipt = json.loads(content)
        if relative == RECEIPTS[0]:
            _require(receipt.get("repo") == "yinzhangyue/SelfAware", "SelfAware receipt repository mismatch")
            _require(receipt.get("revision") == PINS["yinzhangyue/SelfAware"], "SelfAware receipt pin mismatch")
        errors = receipt.get("errors", [])
        for error in errors:
            _require(error.get("url") == KNOWN_404 and "404" in error.get("error", ""),
                     f"Unapproved download error: {error}")
        files = receipt.get("files")
        _require(isinstance(files, list) and files, f"Empty source receipt: {relative}")
        for entry in files:
            path = entry.get("path")
            local = _safe_path(root, path, "data/v2/raw/")
            _require(path not in sources, f"Duplicate receipt path: {path}")
            _require(entry.get("url") == _expected_source_url(path), f"Source pin/URL mismatch: {path}")
            data = local.read_bytes()
            _require(_sha(data) == entry.get("sha256"), f"Source hash mismatch: {path}")
            _require(len(data) == entry.get("bytes"), f"Source size mismatch: {path}")
            sources[path] = data
        receipts.append({
            "path": relative,
            "sha256": _sha(content),
            "files": sorted(files, key=lambda entry: entry["path"]),
            "accepted_download_errors": errors,
        })
    required = {*PARQUET_PATHS.values(), SELFAWARE_PATH, FACTSCORE_PATH,
                "data/v2/raw/HalluLens/tasks/shortform/precise_wikiqa.py",
                "data/v2/raw/SelfAware/code/eval_model.py"}
    _require(required <= sources.keys(), f"Missing receipted sources: {sorted(required - sources.keys())}")
    return sources, receipts


def _legacy(frozen):
    """Parse the same in-memory bytes already checked against the frozen audit."""
    manifest_bytes = frozen["data/manifest.json"]
    manifest = json.loads(manifest_bytes)
    original = frozen["data/evaluation.jsonl"]
    _require(_sha(original) == manifest.get("evaluation_sha256"), "Original evaluation hash mismatch")
    _require(original.endswith(b"\n"), "Original evaluation must end with a newline for exact row reuse")
    rows, kept, lines = {}, [], []
    for row, line in _jsonl(original, "data/evaluation.jsonl"):
        identifier = row.get("id")
        _require(isinstance(identifier, str) and identifier and identifier not in rows,
                 f"Invalid/duplicate original ID: {identifier}")
        _require(isinstance(row.get("dataset"), str), f"Invalid original dataset: {identifier}")
        rows[identifier] = row
        if row["dataset"] in REUSE_COUNTS:
            _require(isinstance(row.get("question"), str), f"Invalid original question: {identifier}")
            kept.append(row)
            lines.append(line)
    _require(dict(Counter(row["dataset"] for row in kept)) == REUSE_COUNTS, "Original reuse counts mismatch")
    for key, count in REUSE_COUNTS.items():
        _require(manifest.get("counts", {}).get(key) == count, f"Original manifest count mismatch: {key}")
    reuse = {"schema_version": 2, "count_per_model": len(kept), "counts_per_model": REUSE_COUNTS, "models": {}}
    for short_name, model in MODELS.items():
        path = f"runs/{short_name}-baseline/checkpoint/responses.jsonl"
        checkpoint = frozen[path]
        responses = {}
        for response, _ in _jsonl(checkpoint, path):
            identifier = response.get("id")
            _require(isinstance(identifier, str) and identifier in rows, f"Unknown checkpoint ID in {path}: {identifier}")
            _require(identifier not in responses, f"Duplicate checkpoint ID in {path}: {identifier}")
            _require(response.get("model_id") == model, f"Checkpoint model mismatch in {path}: {identifier}")
            _require(response.get("status") == "ok", f"Checkpoint status is not ok in {path}: {identifier}")
            _require(response.get("dataset") == rows[identifier]["dataset"], f"Checkpoint dataset mismatch in {path}: {identifier}")
            _require(isinstance(response.get("response"), str), f"Checkpoint response is not a string in {path}: {identifier}")
            responses[identifier] = response
        missing = [row["id"] for row in kept if row["id"] not in responses]
        _require(not missing, f"Missing reusable checkpoint IDs in {path}: {missing[:10]}")
        reuse["models"][short_name] = {
            "model_id": model,
            "checkpoint": {"path": path, "sha256": _sha(checkpoint), "rows": len(responses)},
            "responses": [{
                "id": row["id"], "dataset": row["dataset"], "model_id": model,
                "response_sha256": _sha(responses[row["id"]]["response"].encode("utf-8")),
            } for row in kept],
        }
    provenance = {
        "evaluation_path": "data/evaluation.jsonl", "evaluation_sha256": _sha(original),
        "manifest_path": "data/manifest.json", "manifest_sha256": _sha(manifest_bytes),
        "revisions": manifest.get("revisions", {}),
        "abstention_prompt_deviation": manifest.get("abstention_prompt_deviation"),
        "selected_ids": [row["id"] for row in kept],
        "rows_copied_byte_for_byte": True,
    }
    return kept, b"".join(lines), reuse, provenance


def _generation_contract(root, frozen, reuse):
    metadata = {path: _anchored_bytes(root, path, expected, "Runtime metadata")
                for path, expected in RUNTIME_SHA256.items()}
    decoding = json.loads(frozen["data/manifest.json"])["decoding"]
    _require(decoding.get("do_sample") is False and decoding.get("num_beams") == 1,
             "Saved decoding contract is not greedy")
    models = {}
    for short, model_id in MODELS.items():
        prefix = f"runs/{short}-baseline"
        runtime_path = f"{prefix}/checkpoint/manifest.json"
        launch_path = f"{prefix}/launch.json"
        template_path = f"{prefix}/checkpoint/chat_template.txt"
        notebook_path = f"notebooks/{short}-baseline.ipynb"
        runtime = json.loads(metadata[runtime_path])
        launch = json.loads(metadata[launch_path])
        revision = MODEL_REVISIONS[short]
        _require(runtime["model_id"] == model_id and runtime["resolved_revision"] == revision,
                 f"Saved runtime model/revision mismatch: {short}")
        for config in (runtime["config"], launch["config"]):
            _require(config["model_id"] == model_id and config["revision"] == revision,
                     f"Saved launch model/revision mismatch: {short}")
        _require(launch["saved_notebook"].replace("\\", "/") == notebook_path,
                 f"Saved notebook identity mismatch: {short}")
        _require(runtime["weight_dtype"] == "float16" and runtime["quantization"] is None,
                 f"Saved precision/quantization mismatch: {short}")
        template_hash = _sha(metadata[template_path])
        _require(template_hash == runtime["chat_template_sha256"], f"Saved chat template hash mismatch: {short}")
        # The saved file is str(tokenizer.chat_template), a dict literal; it is
        # NOT the Jinja source alone. literal_eval reads data without executing it.
        templates = ast.literal_eval(metadata[template_path].decode("utf-8"))
        _require(isinstance(templates, dict) and isinstance(templates.get("default"), str),
                 f"Missing native default template: {short}")
        models[short] = {
            "model_id": model_id, "revision": revision,
            "weight_dtype": runtime["weight_dtype"], "quantization": runtime["quantization"],
            "eos_token_id": runtime["eos_token_id"], "pad_token_id": runtime["pad_token_id"],
            "saved_runtime_versions": {key: runtime[key] for key in ("python", "torch", "transformers", "accelerate")},
            "saved_checkpoint": reuse["models"][short]["checkpoint"],
            "chat_template": {
                "identity": f"{model_id}@{revision}:tokenizer.chat_template/default",
                "selection": "default", "path": template_path, "sha256": template_hash,
                "hash_representation": "UTF-8 str(tokenizer.chat_template), as saved by the original runtime",
                "selected_template_sha256": _sha(templates["default"].encode("utf-8")),
            },
            "runtime_metadata": [{"path": path, "sha256": _sha(metadata[path])}
                                 for path in (runtime_path, launch_path, template_path, notebook_path)],
        }
    return {
        "scope": "new generation.jsonl rows; original rows and saved responses are reused exactly",
        "models": models,
        "decoding": {
            "do_sample": decoding["do_sample"], "num_beams": decoding["num_beams"],
            "use_cache": True,
            "temperature_override": None,
            "temperature_policy": "not passed; retain model generation_config value, inactive during greedy decoding",
            "unspecified_parameters": "retain the pinned model generation_config defaults without additional overrides",
        },
        "chat_template_policy": {
            "method": "tokenizer.apply_chat_template", "selection": "native default from the pinned model revision",
            "tokenize": False, "add_generation_prompt": True, "padding_side": "left",
            "native_preambles": "preserve all preambles embedded in each model's native template",
        },
        "prompting": {
            "new_system_prompt": "", "system_role_policy": "omit the system message when system_prompt is empty",
            "user_message": "question text supplied exactly to the native template",
            "reflection": False, "generation_passes": 1,
        },
        "max_new_tokens": {"hallulens_precisewikiqa": 256, "hallulens_nonexistentrefusal": 512,
                           "selfaware": 256, "factscore_bios": 1536},
        "policy_provenance": "explicit greedy fields from the anchored original data manifest; chat/temperature/cache policy statically audited from the hash-anchored saved notebooks, never executed",
    }


def _read_parquet(data):
    # PyArrow is the only non-stdlib dependency; no upstream source is executed.
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pq.read_table(pa.BufferReader(data)).to_pylist()


def _rank(source, source_index, question):
    return _sha(f"{SEED}:{source}:{source_index}:{question}".encode("utf-8"))


def _largest_remainder(counts, target):
    total = sum(counts.values())
    _require(0 <= target <= total and total > 0, "Insufficient unique rows for proportional allocation")
    allocations = {key: target * count // total for key, count in sorted(counts.items())}
    order = sorted(counts, key=lambda key: (-(target * counts[key] % total), key))
    for key in order[:target - sum(allocations.values())]:
        allocations[key] += 1
    return allocations


def _select(rows, source, stratum, raw_counts, allocations=None, target=None):
    _require(all(isinstance(row.get("prompt"), str) and row["prompt"] for row in rows),
             f"Invalid prompt in {source}")
    _require(all(stratum in row for row in rows), f"Missing stratum in {source}")
    actual_counts = dict(Counter(str(row[stratum]) for row in rows))
    if raw_counts is not None:
        _require(actual_counts == raw_counts, f"Raw strata counts mismatch: {source}")
    seen, groups, duplicates = {}, defaultdict(list), []
    for index, row in enumerate(rows):
        question = row["prompt"]
        if question in seen:
            kept_index = seen[question]
            duplicates.append({
                "source_index": index, "kept_source_index": kept_index,
                "stratum": str(row[stratum]), "kept_stratum": str(rows[kept_index][stratum]),
                "question_sha256": _sha(question.encode("utf-8")),
            })
        else:
            seen[question] = index
            groups[str(row[stratum])].append((index, row))
    unique_counts = {key: len(group) for key, group in sorted(groups.items())}
    if allocations is None:
        allocations = _largest_remainder(unique_counts, target)
    selected = []
    selected_indices = {}
    for key, count in sorted(allocations.items()):
        _require(len(groups[key]) >= count, f"Insufficient unique rows in {source}/{key}")
        ranked = sorted(groups[key], key=lambda item: (_rank(source, item[0], item[1]["prompt"]), item[0]))
        chosen = ranked[:count]
        selected.extend(chosen)
        selected_indices[key] = [index for index, _ in chosen]
    details = {
        "hash_source": source, "stratum_field": stratum, "raw_count": len(rows),
        "raw_strata_counts": actual_counts, "unique_question_count": len(seen),
        "unique_strata_counts": unique_counts, "allocations": allocations,
        "selected_source_indices_by_stratum": selected_indices,
        "duplicate_exclusions": duplicates,
    }
    return selected, details


def _new_row(key, source_index, dataset, component, question, answers, abstain, limit, metadata):
    return {
        "id": f"v2-{key}-{source_index}", "dataset": dataset, "component": component,
        "question": question, "reference_answers": answers, "should_abstain": abstain,
        "system_prompt": "", "max_new_tokens": limit,
        "metadata": {**metadata, "source_index": source_index},
    }


def _hallulens(sources):
    new, selections = [], {}
    specs = (
        ("precise_wiki", "h_score_cat", {str(i): 500 for i in range(10)},
         {str(i): 100 for i in range(10)}, None, "hallulens_precisewikiqa", "PreciseWikiQA", 256),
        ("mixed_entities", "type", {key: 2000 for key in ("animal", "plant", "bacteria", "medicine")},
         {key: 500 for key in ("animal", "plant", "bacteria", "medicine")}, None,
         "hallulens_nonexistentrefusal", "Mixed", 512),
        ("generated_entities", "type_", None, None, 1000,
         "hallulens_nonexistentrefusal", "Generated", 512),
    )
    for source, stratum, raw_counts, allocations, target, dataset, component, limit in specs:
        rows = _read_parquet(sources[PARQUET_PATHS[source]])
        if source == "generated_entities":
            _require(len(rows) == 1300 and len({row.get("type_") for row in rows}) == 35,
                     "Generated raw count/type count mismatch")
            _require(all(isinstance(row.get("type_"), str) and row["type_"] for row in rows),
                     "Invalid Generated type")
        selected, details = _select(rows, source, stratum, raw_counts, allocations, target)
        id_key = dataset if source == "precise_wiki" else f"{dataset}_{component.lower()}"
        records = []
        for index, raw in selected:
            if source == "precise_wiki":
                _require(all(field in raw for field in ("answer", "reference", "pageid", "revid", "title")),
                         f"Missing PreciseWikiQA evidence: {index}")
                _require(isinstance(raw["answer"], str) and isinstance(raw["reference"], str),
                         f"Invalid PreciseWikiQA gold: {index}")
            records.append(_new_row(
                id_key, index, dataset, component, raw["prompt"],
                [raw["answer"]] if source == "precise_wiki" else None,
                source != "precise_wiki", limit, raw,
            ))
        details.update({
            "source_path": PARQUET_PATHS[source], "id_key": id_key, "max_new_tokens": limit,
            "selected_ids": [row["id"] for row in records],
            "allocation_method": "proportional largest remainder after exact-prompt deduplication; remainder descending, lexical tie"
            if target is not None else "fixed per stratum after exact-prompt deduplication",
        })
        selections[source] = details
        new.extend(records)
    return new, selections


def _selfaware(sources):
    document = json.loads(sources[SELFAWARE_PATH])
    rows = document.get("example")
    _require(isinstance(rows, list) and len(rows) == 3369, "SelfAware row count mismatch")
    identifiers, questions = set(), set()
    records = []
    for index, raw in enumerate(rows):
        identifier = raw.get("question_id")
        _require(type(identifier) is int and identifier not in identifiers, f"Invalid/duplicate SelfAware ID: {identifier}")
        _require(isinstance(raw.get("question"), str) and raw["question"], f"Invalid SelfAware question: {identifier}")
        _require(raw["question"] not in questions, f"Duplicate SelfAware question (never silently dropped): {identifier}")
        _require(type(raw.get("answerable")) is bool, f"Invalid SelfAware answerable flag: {identifier}")
        answer = raw.get("answer")
        valid_answer = (isinstance(answer, list) and all(isinstance(item, str) for item in answer)) if raw["answerable"] else ("answer" in raw and answer is None)
        _require(valid_answer,
                 f"Invalid SelfAware answer: {identifier}")
        _require(isinstance(raw.get("source"), str), f"Invalid SelfAware source: {identifier}")
        identifiers.add(identifier)
        questions.add(raw["question"])
        records.append(_new_row("selfaware", index, "selfaware", raw["source"], raw["question"],
                                raw["answer"], not raw["answerable"], 256, raw))
    labels = {"true": sum(row["answerable"] for row in rows), "false": sum(not row["answerable"] for row in rows)}
    _require(labels == {"true": 2338, "false": 1031}, "SelfAware released-label counts mismatch")
    source_counts = dict(Counter(row["source"] for row in rows))
    _require(source_counts.get("SelfAware") == 1032, "SelfAware source count mismatch")
    anomalies = [row["question_id"] for row in rows if row["source"] == "SelfAware" and row["answerable"]]
    _require(anomalies == [3149], "SelfAware released anomaly mismatch")
    return records, {
        "source_path": SELFAWARE_PATH, "raw_count": len(rows), "selected_ids": [row["id"] for row in records],
        "max_new_tokens": 256, "sampling": "all source rows in original order; no relabeling or deduplication",
        "released_label_counts": labels, "source_counts": source_counts,
        "upstream_quantity_field": document.get("quantity"),
        "paper_claimed_counts": {"answerable": 2337, "unanswerable": 1032},
        "anomalies": [{"question_id": 3149, "source": "SelfAware", "answerable": True,
                       "action": "preserved exactly as released"}],
        "official_scorer_model_id": SELFAWARE_SCORER_MODEL,
        "scoring_performed": False,
    }


def _factscore(sources):
    try:
        selected, details = select_archive(sources[FACTSCORE_PATH])
    except ValueError as error:
        raise PreparationError(str(error)) from error
    records = [_new_row(
        "factscore_bios", index, "factscore_bios", "labeled", details["prompt_template"].format(topic=topic),
        None, None, 1536, {"topic": topic, "source_path": FACTSCORE_PATH, "zip_member": FACTSCORE_MEMBER},
    ) for index, topic in selected]
    return records, details


def _overlaps(rows):
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question"]].append({key: row[key] for key in ("id", "dataset", "component")})
    return [{"question": question, "question_sha256": _sha(question.encode("utf-8")), "items": items}
            for question, items in sorted(by_question.items()) if len({row["dataset"] for row in items}) > 1]


def _derive(root):
    frozen, integrity = _verified_preflight(root)
    sources, receipts = _verified_sources(root)
    old, old_bytes, reuse, original = _legacy(frozen)
    generation_contract = _generation_contract(root, frozen, reuse)
    new, selections = _hallulens(sources)
    rows, selections["selfaware"] = _selfaware(sources)
    new.extend(rows)
    rows, selections["factscore_bios"] = _factscore(sources)
    new.extend(rows)
    _require(dict(Counter(row["dataset"] for row in new)) == NEW_COUNTS, "New evaluation counts mismatch")
    _require(all(row["max_new_tokens"] == generation_contract["max_new_tokens"][row["dataset"]]
                 and row["system_prompt"] == generation_contract["prompting"]["new_system_prompt"] for row in new),
             "New prompt/limit disagrees with generation contract")
    all_rows = old + new
    _require(len(new) == 7549 and len(all_rows) == 9527, "Evaluation total mismatch")
    _require(len({row["id"] for row in all_rows}) == len(all_rows), "Duplicate assembled ID")
    for dataset in NEW_COUNTS:
        questions = [row["question"] for row in new if row["dataset"] == dataset]
        _require(len(questions) == len(set(questions)), f"Duplicate assembled question within {dataset}")
    generation = [{key: row[key] for key in GENERATION_FIELDS} for row in new]
    outputs = {
        "evaluation.jsonl": old_bytes + b"".join(_json_line(row) for row in new),
        "generation.jsonl": b"".join(_json_line(row) for row in generation),
        "reuse.json": _json_bytes(reuse),
    }
    manifest = {
        "schema_version": 2, "selection_seed": SEED, "counts": {**REUSE_COUNTS, **NEW_COUNTS},
        "total_evaluation_rows": len(all_rows), "new_generation_rows": len(new),
        "reused_rows_per_model": len(old), "source_pins": PINS, "source_receipts": receipts,
        "original_data": original, "selection": selections,
        "input_integrity": integrity, "generation_contract": generation_contract,
        "selection_policy": {
            "source_index": "zero-based physical row index before deduplication",
            "ranking": "sha256(f'20261111:{source}:{source_index}:{question}'.encode('utf-8')) ascending",
            "hash_tie_break": "source_index ascending",
            "duplicate_policy": "HalluLens exact prompt: keep lowest source_index before strata/allocation; SelfAware: preserve all rows",
            "ordering": "reused original order; PreciseWikiQA, Mixed, Generated strata lexical then hash order; SelfAware and FActScore source order",
        },
        "generation_fields": list(GENERATION_FIELDS),
        "cross_dataset_exact_question_overlaps": _overlaps(all_rows),
        "disclosures": [
            "Approved six-benchmark local-variant stack; only GLM-5.3-Flash is the factual judge. This script performs no scoring or inference.",
            "HalluLens is a pinned third-party snapshot of LLM-generated sources with sample human validation, not individually human-keyed gold.",
            "HalluLens duplicate prompts are removed before stratified sampling, keeping the first source row even across strata; all exclusions are recorded.",
            "NonExistentRefusal uses 2,000 Mixed (500/domain) and 1,000 Generated; Generated proportional quotas use deduplicated type counts.",
            "SelfAware uses all 3,369 released rows; 2,338 true / 1,031 false and question 3149 are preserved despite the paper's 2,337 / 1,032 counts.",
            "SelfAware's official scorer model is recorded for downstream use; no embedding model is loaded or executed here.",
            "FActScore's official released ZIP supplies 183 original biography topics; the scored selection retains 180 after explicit user-approved local exclusions of William Post, Carlos Alfonso and Jagan, preserving original indices, IDs and prompts. These are not labels for new Aya text; fresh extraction and Wikipedia evidence are required downstream.",
            "The FActScore code is git-pinned; the separately hosted data ZIP is pinned by its receipt SHA-256, not by the code revision.",
            "The recorded shortform/prompt.py 404 is accepted; actual PreciseWikiQA prompts are in the verified precise_wikiqa.py artifact.",
            "Exact cross-dataset question overlaps are reported without removing or changing source rows.",
            "Generation is new rows only, with a strict six-field allowlist; references and provenance stay in evaluation/manifest artifacts.",
        ],
        "outputs": {name: {"path": f"data/v2/{name}", "sha256": _sha(data), "bytes": len(data)}
                    for name, data in outputs.items()},
    }
    outputs["manifest.json"] = _json_bytes(manifest)
    return outputs, manifest


def _destinations(root, outputs, require_existing=False):
    missing = []
    # Complete this preflight for EVERY destination before opening any for write.
    for name, expected in outputs.items():
        _require(name in OUTPUT_NAMES, f"Unapproved output name: {name}")
        path = _safe_path(root, f"data/v2/{name}", "data/v2/")
        _require(not path.is_symlink(), f"Refusing symlink destination: {path}")
        if path.exists():
            _require(path.is_file() and path.read_bytes() == expected, f"Destination differs; refusing overwrite: {path}")
        elif require_existing:
            raise PreparationError(f"Missing assembled destination: {path}")
        else:
            missing.append((path, expected))
    return missing


def assemble(root=DEFAULT_ROOT):
    """Derive all bytes, reject any differing destination, write only missing files."""
    root = Path(root).resolve()
    outputs, manifest = _derive(root)
    missing = _destinations(root, outputs)
    for path, content in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwriting a file created after preflight.
        with path.open("xb") as stream:
            stream.write(content)
    return manifest


def verify(root=DEFAULT_ROOT):
    """Independently rederive all four artifacts from sources and compare full bytes."""
    root = Path(root).resolve()
    outputs, manifest = _derive(root)
    _destinations(root, outputs, require_existing=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("assemble", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = {"assemble": assemble, "verify": verify}[args.command](args.root)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{args.command} failed: {error}\n")
    print(json.dumps({
        "command": args.command, "evaluation_rows": manifest["total_evaluation_rows"],
        "generation_rows": manifest["new_generation_rows"],
        "reused_per_model": manifest["reused_rows_per_model"],
        "exact_cross_dataset_overlaps": len(manifest["cross_dataset_exact_question_overlaps"]),
        "duplicate_exclusions": {key: len(value.get("duplicate_exclusions", []))
                                 for key, value in manifest["selection"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
