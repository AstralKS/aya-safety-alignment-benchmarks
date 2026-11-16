"""Offline-only pilot tests; every writable fixture stays in the approved temp root."""

from __future__ import annotations

import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import requests

import v2_glm_budget as budget
import v2_glm_pilot as pilot


TEMP_ROOT = Path(r"C:\Users\AstralK\AppData\Local\Temp\opencode")


def process_reserve(path, barrier, queue, attempt_id):
    barrier.wait(timeout=10)
    try:
        budget.Ledger(path).reserve(attempt_id, "process-race", "request")
        queue.put("reserved")
    except budget.BudgetError:
        queue.put("refused")


def envelope(label="A", **changes):
    value = {"model": pilot.MODEL, "choices": [{"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": label, "reasoning_content": "Yes No C are reasoning only"}}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                       "completion_tokens_details": {"reasoning_tokens": 15}}}
    value.update(changes)
    return value


def response(label="A", status=200, value=None):
    result = MagicMock()
    result.status_code = status
    result.content = pilot.canonical(value or envelope(label))
    result.headers = {"X-Request-Id": "diagnostic", "Authorization": "must-not-save"}
    return result


class TempCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glm-pilot-tests-", dir=TEMP_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_ledger(self):
        path = self.root / "scoring/budget.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {key: str(val) if isinstance(val, Decimal) else val for key, val in budget.LEGACY.items()}
        value["preserved_extra"] = {"old": "untouched"}
        budget.write_once(path, budget.json_bytes(value))
        return budget.Ledger(path)

    def raw_journal(self, attempt_id, value=None):
        path = self.root / (attempt_id + ".response.json")
        raw = pilot.canonical(value or envelope())
        budget.write_once(path, budget.json_bytes({
            "kind": "http_response", "attempt_id": attempt_id, "http_status": 200,
            "request_sha256": "request", "raw_body_base64": base64.b64encode(raw).decode(),
            "raw_body_sha256": pilot.sha(raw),
        }))
        return path


class RubricTests(TempCase):
    def test_literals_strip_and_reject_arbitrary_calls(self):
        self.assertEqual(pilot.literal_rubric("X = '  abc  '", "X"), "  abc  ")
        self.assertEqual(pilot.literal_rubric("X = '  abc  '.strip()", "X"), "abc")
        self.assertEqual(pilot.literal_rubric("X: str = 'abc'", "X"), "abc")
        for source in ("X = str('abc')", "X = 'abc'.strip('a')", "X = 'abc'.strip(x=1)",
                       "X = 'abc'.lower()", "X = 'abc' + 'def'", "X = f'abc'",
                       "X = Y = 'abc'", "X = 'abc'\nX = 'def'", "if True:\n X = 'abc'"):
            with self.subTest(source=source), self.assertRaises(pilot.PilotError):
                pilot.literal_rubric(source, "X")

    def test_notebook_only_matching_cells_and_hashes(self):
        content = pilot.canonical({"cells": [{"cell_type": "code", "source": "!pip install imaginary"},
            {"cell_type": "code", "source": ["X = '''  {question}\n", "{target}\n{predicted_answer}  '''.strip()"]}]})
        (self.root / "official.ipynb").write_bytes(content)
        rubric = "{question}\n{target}\n{predicted_answer}"
        contract = {"path": "official.ipynb", "name": "X", "source_sha256": pilot.sha(content),
                    "rubric_sha256": pilot.sha(rubric.encode()), "fields": ("question", "target", "predicted_answer")}
        self.assertEqual(pilot.extract_rubric(self.root, contract), rubric)
        with self.assertRaises(pilot.PilotError):
            pilot.extract_rubric(self.root, {**contract, "source_sha256": "0" * 64})
        with self.assertRaises(pilot.PilotError):
            pilot.extract_rubric(self.root, {**contract, "rubric_sha256": "0" * 64})

    def test_official_rubrics_read_only(self):
        # Explicit named sources only; no production writes, no scorer imports.
        for contract in pilot.RUBRICS.values():
            text = pilot.extract_rubric(pilot.DEFAULT_ROOT, contract)
            self.assertEqual(pilot.sha(text.encode()), contract["rubric_sha256"])

    def test_abstention_reference_format_no_extra_instruction(self):
        row = {"dataset": "abstention_bench", "question": "q", "reference_answers": ["one", "two"], "should_abstain": True}
        rubric = "{question}|{ref_answer}|{abstention_label}|{model_answer}\n"
        payload = pilot.payload_for(row, "answer", rubric)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "q|[one; two]|True|answer\n"}])
        for refs in (None, []):
            self.assertEqual(pilot.payload_for({**row, "reference_answers": refs}, "answer", rubric)
                             ["messages"][0]["content"], "q|[]|True|answer\n")


class PreparationTests(TempCase):
    def setUp(self):
        super().setUp()
        self.anchors = {}
        self.rows = []
        self.reuse = {"count_per_model": 22, "counts_per_model": {"simpleqa_verified": 12, "abstention_bench": 10}, "models": {}}
        for dataset, count in self.reuse["counts_per_model"].items():
            for index in range(count):
                self.rows.append({"id": f"{dataset}-{index}", "dataset": dataset,
                                  "question": f"Question {dataset} {index}", "reference_answers": ["target"], "should_abstain": False})
        for short, model in pilot.AYA_MODELS.items():
            records = [{"id": row["id"], "dataset": row["dataset"], "model_id": model, "status": "ok",
                        "response": f"Saved {short} {row['id']}", "prompt_sha256": "opaque-native-template-prompt"}
                       for row in self.rows]
            path = f"runs/{short}-baseline/checkpoint/responses.jsonl"
            raw = b"".join(pilot.canonical(row) for row in records)
            (self.root / path).parent.mkdir(parents=True)
            (self.root / path).write_bytes(raw)
            self.anchors[short] = (path, pilot.sha(raw))
            self.reuse["models"][short] = {"model_id": model,
                "checkpoint": {"path": path, "sha256": pilot.sha(raw), "rows": len(records)},
                "responses": [{"id": record["id"], "dataset": record["dataset"], "model_id": model,
                               "response_sha256": pilot.sha(record["response"].encode())} for record in records]}
        self.write_inputs()
        self.addCleanup(patch.stopall)
        patch("prepare_v2_data.verify", side_effect=lambda root: pilot.read_json(
            (root / "data/v2/manifest.json").read_bytes())).start()
        patch.object(pilot, "CHECKPOINTS", self.anchors).start()
        patch.object(pilot, "RUBRICS", {key: {**val, "source_sha256": pilot.sha(b"mocked official source")}
                                       for key, val in pilot.RUBRICS.items()}).start()
        patch.object(pilot, "extract_rubric", side_effect=lambda root, c: (
            "{question}\n{target}\n{predicted_answer}" if c["name"] == "GRADER_TEMPLATE" else
            "{question}\n{ref_answer}\n{abstention_label}\n{model_answer}")).start()
        for contract in pilot.RUBRICS.values():
            path = self.root / contract["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mocked official source")

    def write_inputs(self):
        folder = self.root / "data/v2"
        folder.mkdir(parents=True, exist_ok=True)
        outputs = {"evaluation.jsonl": b"".join(pilot.canonical(row) for row in self.rows),
                   "reuse.json": pilot.canonical(self.reuse)}
        for name, raw in outputs.items():
            (folder / name).write_bytes(raw)
        (folder / "manifest.json").write_bytes(pilot.canonical({"total_evaluation_rows": len(self.rows),
            "outputs": {name: {"path": f"data/v2/{name}", "sha256": pilot.sha(raw), "bytes": len(raw)}
                        for name, raw in outputs.items()}}))

    def test_deterministic_30_counts_frozen_verify_and_no_identity_leak(self):
        directory = pilot.prepare(self.root, "test-one")
        manifest = pilot.verify(directory, self.root)
        self.assertEqual(len(manifest["samples"]), 30)
        self.assertEqual(Counter((x["aya_model"], x["dataset"]) for x in manifest["samples"]),
                         Counter({(short, task): count for short in pilot.AYA_MODELS for task, count in pilot.TASK_COUNTS.items()}))
        again = pilot.prepare(self.root, "test-two")
        self.assertEqual(manifest["samples"], pilot.verify(again, self.root)["samples"])
        for sample in manifest["samples"]:
            payload = pilot.read_json((directory / sample["request_file"]).read_bytes())
            self.assertEqual(set(payload), {"model", "messages", "temperature", "n", "max_tokens", "reasoning_effort"})
            self.assertNotIn(sample["aya_model_id"], payload["messages"][0]["content"])
        with self.assertRaises(FileExistsError):
            pilot.prepare(self.root, "test-one")

    def test_ranking_independent_of_answers(self):
        first, _ = pilot.derive(self.root)
        for short, (path, _) in self.anchors.items():
            records = [pilot.read_json(line) for line in (self.root / path).read_bytes().splitlines()]
            for row in records:
                row["response"] = "Changed ungraded answer"
            raw = b"".join(pilot.canonical(row) for row in records)
            (self.root / path).write_bytes(raw)
            self.anchors[short] = (path, pilot.sha(raw))
            self.reuse["models"][short]["checkpoint"]["sha256"] = pilot.sha(raw)
            for row in self.reuse["models"][short]["responses"]:
                row["response_sha256"] = pilot.sha(b"Changed ungraded answer")
        self.write_inputs()
        second, _ = pilot.derive(self.root)
        self.assertEqual([(x["aya_model"], x["id"]) for x in first["samples"]],
                         [(x["aya_model"], x["id"]) for x in second["samples"]])

    def test_payload_manifest_checkpoint_input_tamper(self):
        directory = pilot.prepare(self.root, "test-tamper")
        path = directory / "request-001.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(pilot.PilotError):
            pilot.verify(directory, self.root)
        path.write_bytes(original)
        path = directory / "manifest.json"
        raw = path.read_bytes()
        value = json.loads(raw)
        value["samples"][0]["answer_sha256"] = "0" * 64
        path.write_bytes(pilot.canonical(value))
        with self.assertRaises(pilot.PilotError):
            pilot.verify(directory, self.root)
        path.write_bytes(raw)
        path = self.root / self.anchors["tiny"][0]
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        with self.assertRaises(pilot.PilotError):
            pilot.verify(directory, self.root)
        path.write_bytes(raw)
        path = self.root / "data/v2/evaluation.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(pilot.PilotError):
            pilot.verify(directory, self.root)

    def test_reuse_response_checks(self):
        for field, invalid in (("status", "error"), ("model_id", "wrong"), ("dataset", "wrong"), ("response", " ")):
            with self.subTest(field=field):
                path, _ = self.anchors["tiny"]
                original = (self.root / path).read_bytes()
                records = [pilot.read_json(line) for line in original.splitlines()]
                records[0][field] = invalid
                raw = b"".join(pilot.canonical(row) for row in records)
                (self.root / path).write_bytes(raw)
                self.anchors["tiny"] = (path, pilot.sha(raw))
                self.reuse["models"]["tiny"]["checkpoint"]["sha256"] = pilot.sha(raw)
                self.write_inputs()
                with self.assertRaises(pilot.PilotError):
                    pilot.derive(self.root)
                (self.root / path).write_bytes(original)
                self.anchors["tiny"] = (path, pilot.sha(original))
                self.reuse["models"]["tiny"]["checkpoint"]["sha256"] = pilot.sha(original)
                self.write_inputs()


class ParsingTests(unittest.TestCase):
    def test_required_usage_counts_missing_null_or_noninteger_are_not_accounted(self):
        usage = {"prompt_tokens": 1378, "completion_tokens": 30, "total_tokens": 1408}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            for value in (None, True, -1, 1.0, "1"):
                with self.subTest(field=field, value=value):
                    self.assertIsNone(self.inspect(envelope(usage={**usage, field: value}))["accounting"])
            missing = {key: value for key, value in usage.items() if key != field}
            with self.subTest(field=field, missing=True):
                self.assertIsNone(self.inspect(envelope(usage=missing))["accounting"])

    def test_optional_usage_details_null_missing_are_unpriced(self):
        usage = {"prompt_tokens": 1378, "completion_tokens": 30, "total_tokens": 1408,
                 "estimated_cost": "0.00022170000000000002",
                 "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": None},
                 "completion_tokens_details": {"reasoning_tokens": 28}}
        parsed = self.inspect(envelope("No", usage=usage), "abstention_bench")
        self.assertTrue(parsed["ok"], parsed)
        self.assertEqual(Decimal(parsed["accounting"]["token_priced_usd"]), Decimal("0.0002217"))
        for details in ({}, {"prompt_tokens_details": None, "completion_tokens_details": None},
                        {"prompt_tokens_details": {"cached_tokens": None}, "completion_tokens_details": {"reasoning_tokens": None}}):
            result = self.inspect(envelope("No", usage={"prompt_tokens": 1378, "completion_tokens": 30,
                                                         "total_tokens": 1408, **details}), "abstention_bench")
            self.assertTrue(result["ok"], result)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsNone(self.inspect(envelope(usage={**usage, field: None}))["accounting"])

    def inspect(self, value, dataset="simpleqa_verified"):
        return pilot.inspect_response(pilot.canonical(value), dataset)

    def test_exact_final_labels_ignore_reasoning(self):
        for label in ("A", " B\n", "C"):
            self.assertTrue(self.inspect(envelope(label))["ok"])
        for label in ("Yes", "No", " Yes\n"):
            self.assertTrue(self.inspect(envelope(label), "abstention_bench")["ok"])
        for label in ("a", "A.", "**A**", "A because...", "", None, "Yes", "<think>A</think>A"):
            with self.subTest(label=label):
                parsed = self.inspect(envelope(label))
                self.assertFalse(parsed["ok"])
                self.assertIsNone(parsed["label"])
                self.assertIsNotNone(parsed["accounting"])

    def test_truncation_settles_but_bad_envelope_does_not(self):
        value = envelope()
        value["choices"][0]["finish_reason"] = "length"
        parsed = self.inspect(value)
        self.assertFalse(parsed["ok"])
        self.assertIsNotNone(parsed["accounting"])
        for model in ("other/GLM-5.3-Flash", pilot.MODEL + "-evil", "ZAI-org/GLM-5.3-Flash", None):
            self.assertIsNone(self.inspect(envelope(model=model))["accounting"])
        self.assertTrue(self.inspect(envelope(model="zai-org/GLM-5.3-Flash"))["ok"])
        for choices in ([], [envelope()["choices"][0]] * 2, "invalid"):
            self.assertIsNone(self.inspect(envelope(choices=choices))["accounting"])
        value = envelope()
        value["choices"][0]["message"]["role"] = "user"
        self.assertIsNone(self.inspect(value)["accounting"])
        self.assertFalse(pilot.inspect_response(b'{"model": NaN}', "simpleqa_verified")["ok"])
        self.assertFalse(pilot.inspect_response(b'{"model": "a", "model": "b"}', "simpleqa_verified")["ok"])

    def test_usage_strict_counts_totals_details(self):
        bad = [None, {}, {"prompt_tokens": True, "completion_tokens": 1},
               {"prompt_tokens": 1.0, "completion_tokens": 1},
               {"prompt_tokens": -1, "completion_tokens": 1},
               {"prompt_tokens": 1, "completion_tokens": 0},
               {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
               {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": True},
               {"prompt_tokens": 1, "completion_tokens": 1, "completion_tokens_details": "invalid"},
               {"prompt_tokens": 1, "completion_tokens": 1, "completion_tokens_details": {"reasoning_tokens": 2}},
               {"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": True}},
               {"prompt_tokens": 1, "completion_tokens": 1, "estimated_cost": "NaN"},
               {"prompt_tokens": 1, "completion_tokens": 1, "estimated_cost": -1}]
        for usage in bad:
            with self.subTest(usage=usage):
                self.assertIsNone(self.inspect(envelope(usage=usage))["accounting"])
        accounting = self.inspect(envelope())["accounting"]
        self.assertEqual(Decimal(accounting["token_priced_usd"]), Decimal("0.000025"))
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "estimated_cost": "0.01"}
        self.assertEqual(self.inspect(envelope(usage=usage))["accounting"]["accounted_usd"], "0.01")

    def test_bounds_record_actual_not_fake_cap(self):
        for usage in ({"prompt_tokens": budget.INPUT_BOUND + 1, "completion_tokens": 1, "total_tokens": budget.INPUT_BOUND + 2},
                      {"prompt_tokens": 1, "completion_tokens": 2049, "total_tokens": 2050},
                      {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "estimated_cost": "1"}):
            parsed = self.inspect(envelope(usage=usage))
            self.assertFalse(parsed["ok"])
            self.assertTrue(parsed["accounting"]["bound_failure"])
            self.assertIn("NOT retroactively prevented", parsed["error"])


class LedgerTests(TempCase):
    def leave_one_attempt_room(self, ledger):
        ledger.reserve("prior", "prior", "request")
        charge = budget.CEILING - budget.LEGACY["reserved_upper_usd"] - budget.ATTEMPT_BOUND
        value = envelope(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "estimated_cost": str(charge)})
        ledger.settle("prior", budget.usage_price(value["usage"], True), self.raw_journal("prior", value))

    def test_full_bound_and_legacy_preservation_idempotent_settle(self):
        self.assertEqual(budget.ATTEMPT_BOUND, Decimal("0.1583104"))
        self.assertEqual(30 * budget.ATTEMPT_BOUND, Decimal("4.7493120"))
        ledger = self.make_ledger()
        original = ledger.path.read_bytes()
        ledger.start(self.root)
        self.assertEqual((self.root / "ledger-before.json").read_bytes(), original)
        self.assertEqual(pilot.read_json((self.root / "ledger-before-sha256.json").read_bytes())["sha256"], pilot.sha(original))
        ledger.reserve("one", "pilot", "request")
        snap = ledger.snapshot()
        self.assertEqual(budget.money(snap["reserved_upper_usd"]), budget.LEGACY["reserved_upper_usd"] + budget.ATTEMPT_BOUND)
        self.assertEqual(snap["requests"], 8050)
        self.assertEqual(snap["preserved_extra"], {"old": "untouched"})
        accounting = budget.usage_price(envelope()["usage"], True)
        journal = self.raw_journal("one")
        ledger.settle("one", accounting, journal)
        settled = ledger.path.read_bytes()
        ledger.settle("one", accounting, journal)
        self.assertEqual(ledger.path.read_bytes(), settled)
        self.assertEqual(budget.money(ledger.snapshot()["reported_cost_usd"]), budget.LEGACY["reported_cost_usd"])
        self.assertEqual(ledger.snapshot()["settled_requests"], 7447)
        with self.assertRaises(budget.BudgetError):
            ledger.settle("one", {**accounting, "accounted_usd": "0.1"}, journal)
        self.assertEqual(Decimal(ledger.totals("pilot")["unknown_reserves_usd"]), 0)

    def test_thread_race_only_one_attempt_fits(self):
        ledger = self.make_ledger()
        # A prior trusted, over-bound charge leaves room for exactly one attempt.
        self.leave_one_attempt_room(ledger)
        barrier = threading.Barrier(2)
        def reserve(attempt):
            barrier.wait()
            try:
                ledger.reserve(attempt, "race", "request")
                return True
            except budget.BudgetError:
                return False
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(reserve, ("a", "b")))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(budget.money(ledger.snapshot()["reserved_upper_usd"]), budget.CEILING)

    def test_process_race_only_one_attempt_fits(self):
        ledger = self.make_ledger()
        self.leave_one_attempt_room(ledger)
        context = multiprocessing.get_context("spawn")
        barrier, queue = context.Barrier(2), context.Queue()
        processes = [context.Process(target=process_reserve, args=(str(ledger.path), barrier, queue, name))
                     for name in ("process-one", "process-two")]
        try:
            for process in processes:
                process.start()
            results = [queue.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sorted(results), ["refused", "reserved"])
            self.assertEqual(budget.money(ledger.snapshot()["reserved_upper_usd"]), budget.CEILING)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            queue.close()

    def test_journal_usage_hash_request_tampering_cannot_release_reserve(self):
        ledger = self.make_ledger()
        ledger.reserve("one", "pilot", "request")
        accounting = budget.usage_price(envelope()["usage"], True)
        path = self.raw_journal("one")
        original = path.read_bytes()
        for field, invalid in (("raw_body_sha256", "0" * 64), ("request_sha256", "wrong"), ("http_status", 302)):
            data = json.loads(original)
            data[field] = invalid
            path.write_bytes(pilot.canonical(data))
            with self.assertRaises(budget.BudgetError):
                ledger.settle("one", accounting, path)
            self.assertEqual(ledger.snapshot()["v2_glm"]["attempts"]["one"]["state"], "reserved")
        path.write_bytes(original)
        with self.assertRaises(budget.BudgetError):
            ledger.settle("one", {**accounting, "accounted_usd": "0"}, path)

    def test_atomic_failure_retains_reserve_and_unknown_charge(self):
        ledger = self.make_ledger()
        ledger.reserve("one", "pilot", "request")
        raw = ledger.path.read_bytes()
        with self.assertRaises(FileNotFoundError):
            ledger.settle("one", budget.usage_price(envelope()["usage"], True), self.root / "missing")
        with patch.object(budget.os, "replace", side_effect=OSError("simulated replace failure")):
            with self.assertRaises(OSError):
                ledger.settle("one", budget.usage_price(envelope()["usage"], True), self.raw_journal("one"))
        self.assertEqual(ledger.path.read_bytes(), raw)
        self.assertEqual(ledger.totals("pilot")["unknown_reserves_usd"], str(budget.ATTEMPT_BOUND))

    def test_lock_stuck_fails_closed_and_invalid_ledger(self):
        ledger = self.make_ledger()
        lock = Path(str(ledger.path) + ".lock")
        lock.write_bytes(b"prior owner")
        with self.assertRaises(budget.BudgetError):
            with budget.file_lock(ledger.path, timeout=0):
                pass
        self.assertEqual(lock.read_bytes(), b"prior owner")
        lock.unlink()
        for key, value in (("ceiling_usd", "19"), ("reserved_upper_usd", "NaN"),
                           ("reported_cost_usd", True), ("requests", 8048), ("settled_requests", False)):
            raw = ledger.path.read_bytes()
            data = pilot.read_json(raw)
            data[key] = value
            ledger.path.write_bytes(budget.json_bytes(data))
            with self.assertRaises(budget.BudgetError):
                ledger.snapshot()
            ledger.path.write_bytes(raw)


class RecoveryTests(TempCase):
    def test_original_optional_null_journal_repair_without_http_or_log_rewrite(self):
        ledger = self.make_ledger()
        directory = pilot.pilot_directory(self.root, "saved-pilot")
        (directory / "attempts").mkdir(parents=True)
        request_sha = "request"
        ledger.reserve("original", "saved-pilot", request_sha)
        value = envelope("No", model="zai-org/GLM-5.3-Flash", usage={
            "prompt_tokens": 1378, "completion_tokens": 30, "total_tokens": 1408,
            "estimated_cost": "0.00022170000000000002", "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": None},
            "completion_tokens_details": {"reasoning_tokens": 28}})
        body = pilot.canonical(value)
        path = directory / "attempts/original.response.json"
        budget.write_once(path, budget.json_bytes({"kind": "http_response", "http_status": 200,
            "attempt_id": "original", "request_sha256": request_sha, "raw_body_base64": base64.b64encode(body).decode(),
            "raw_body_sha256": pilot.sha(body)}))
        original = path.read_bytes()
        manifest = {"samples": [{"request_sha256": request_sha, "dataset": "abstention_bench"}]}
        with patch.object(pilot, "verify", return_value=manifest), patch.object(pilot, "http_request") as http:
            receipt = pilot.recover_saved(path, self.root)
            self.assertEqual(receipt["reparsed"]["label"], "No")
            self.assertEqual(ledger.snapshot()["v2_glm"]["attempts"]["original"]["state"], "reserved")
            pilot.recover_saved(path, self.root, settle=True)
            settled = ledger.path.read_bytes()
            pilot.recover_saved(path, self.root, settle=True)
            self.assertEqual(ledger.path.read_bytes(), settled)
            http.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(ledger.totals()["attempts_settled"], 1)


class ExecutionTests(TempCase):
    def setUp(self):
        super().setUp()
        self.ledger = self.make_ledger()
        self.directory = pilot.pilot_directory(self.root, "mock-run")
        self.directory.mkdir(parents=True)
        self.samples = []
        for index in range(30):
            task = "simpleqa_verified" if index % 2 == 0 else "abstention_bench"
            payload = {"model": pilot.MODEL, "messages": [{"role": "user", "content": "mock prompt"}],
                       "temperature": 0, "n": 1, "max_tokens": 2048, "reasoning_effort": "low"}
            raw = pilot.canonical(payload)
            filename = f"request-{index + 1:03d}.json"
            (self.directory / filename).write_bytes(raw)
            self.samples.append({"request_file": filename, "request_sha256": pilot.sha(raw),
                                 "id": str(index), "aya_model": "tiny", "dataset": task})
        self.manifest = {"pilot_id": "mock-run", "samples": self.samples, "disclosure": "Initial pilot only"}
        (self.directory / "manifest.json").write_bytes(pilot.canonical(self.manifest))
        self.addCleanup(patch.stopall)
        patch.object(pilot, "verify", return_value=self.manifest).start()
        patch.object(pilot, "token", return_value="secret-token-never-journal").start()

    def test_http_request_explicit_transport_contract(self):
        session = MagicMock()
        session.__enter__.return_value = session
        with patch.object(pilot.requests, "Session", return_value=session), patch.object(pilot, "HTTPAdapter") as adapter:
            pilot.http_request({"message": "hello"}, "secret")
        adapter.assert_called_once_with(max_retries=0)
        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        self.assertEqual(args, (pilot.URL,))
        self.assertEqual(kwargs["timeout"], (10.0, 120.0))
        self.assertIs(kwargs["allow_redirects"], False)
        self.assertIs(session.trust_env, False)

    def mocked_success(self, payload, hf_token):
        # Request sequence is known; rubric/payload isn't artificially modified for tests.
        return response("A" if payload["messages"][0]["content"] == "mock simpleqa" else "Yes")

    def set_task_prompts(self):
        for record in self.samples:
            path = self.directory / record["request_file"]
            payload = pilot.read_json(path.read_bytes())
            payload["messages"][0]["content"] = "mock simpleqa" if record["dataset"] == "simpleqa_verified" else "mock abstention"
            raw = pilot.canonical(payload)
            path.write_bytes(raw)
            record["request_sha256"] = pilot.sha(raw)

    def test_all_waves_30_once_replay_refused_journals_no_secrets(self):
        self.set_task_prompts()
        lock, eight = threading.Lock(), threading.Barrier(8, timeout=10)
        count = 0
        def success(payload, hf_token):
            nonlocal count
            with lock:
                count += 1
                index = count
            if index >= 15:
                eight.wait()  # Two exact 8-call waves, not simulated peak counters.
            return self.mocked_success(payload, hf_token)
        with patch.object(pilot, "http_request", side_effect=success) as http:
            summary = pilot.run(self.directory, self.root)
        self.assertTrue(summary["ok"], summary)
        self.assertEqual(http.call_count, 30)
        self.assertEqual([x["waves_completed"] for x in summary["levels"]], [2, 2, 2, 2])
        self.assertEqual(summary["peak_actual_in_flight"], 8)
        self.assertEqual(summary["accounting"]["attempts_settled"], 30)
        self.assertEqual(len(list((self.directory / "attempts").glob("*.response.json"))), 30)
        self.assertNotIn(b"secret-token", b"".join(p.read_bytes() for p in self.directory.rglob("*.json")))
        self.assertNotIn(b"must-not-save", b"".join(p.read_bytes() for p in self.directory.rglob("*.json")))
        with patch.object(pilot, "http_request") as http:
            with self.assertRaises(pilot.PilotError):
                pilot.run(self.directory, self.root)
            http.assert_not_called()

    def test_failed_wave_drains_exact_four_and_stops(self):
        self.set_task_prompts()
        lock = threading.Lock()
        four = threading.Barrier(4, timeout=5)
        count = 0
        def http(payload, hf_token):
            nonlocal count
            with lock:
                count += 1
                index = count
            if 7 <= index <= 10:
                four.wait()  # Must dispatch/drain all 4 even when one fails.
            return response("A" if "simpleqa" in payload["messages"][0]["content"] else "Yes", status=429 if index == 7 else 200)
        with patch.object(pilot, "http_request", side_effect=http):
            summary = pilot.run(self.directory, self.root)
        self.assertFalse(summary["ok"])
        self.assertEqual(count, 10)
        self.assertEqual(summary["peak_actual_in_flight"], 4)
        self.assertEqual(summary["accounting"]["attempts_reserved"], 10)
        self.assertEqual(summary["accounting"]["attempts_settled"], 9)
        self.assertEqual(summary["accounting"]["unknown_reserves_usd"], str(budget.ATTEMPT_BOUND))

    def test_timeout_one_call_no_retry_unknown_charge(self):
        with patch.object(pilot, "http_request", side_effect=requests.ReadTimeout("may contain secrets")) as http:
            summary = pilot.run(self.directory, self.root)
        self.assertFalse(summary["ok"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(summary["accounting"]["attempts_settled"], 0)
        self.assertEqual(summary["accounting"]["unknown_reserves_usd"], str(budget.ATTEMPT_BOUND))

    def test_journal_failure_keeps_charge_unknown_and_stops(self):
        original = pilot.write_once
        def failing(path, content):
            if str(path).endswith(".response.json"):
                raise OSError("simulated journal failure")
            return original(path, content)
        with patch.object(pilot, "http_request", return_value=response()) as http, patch.object(pilot, "write_once", side_effect=failing):
            summary = pilot.run(self.directory, self.root)
        self.assertFalse(summary["ok"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(summary["accounting"]["unknown_reserves_usd"], str(budget.ATTEMPT_BOUND))

    def test_wave_journal_accounting_failure_and_guard_fail_closed(self):
        (self.directory / "run.guard").write_bytes(b"other process")
        with patch.object(pilot, "http_request") as http:
            with self.assertRaises(FileExistsError):
                pilot.run(self.directory, self.root)
            http.assert_not_called()
        self.assertEqual((self.directory / "run.guard").read_bytes(), b"other process")
        (self.directory / "run.guard").unlink()
        original = pilot.write_once
        def failing_wave(path, content):
            if Path(path).name.startswith("wave-"):
                raise OSError("mock wave journal failure")
            return original(path, content)
        with patch.object(pilot, "http_request", return_value=response()) as http, patch.object(pilot, "write_once", side_effect=failing_wave):
            summary = pilot.run(self.directory, self.root)
        self.assertFalse(summary["ok"])
        self.assertEqual(http.call_count, 1)
        self.assertIn("Wave journal failure", summary["stop_error"])

    def test_settlement_transaction_failure_stops_and_retains_unknown(self):
        with patch.object(pilot, "http_request", return_value=response()) as http, patch.object(budget.Ledger, "settle", side_effect=OSError("mock accounting failure")):
            summary = pilot.run(self.directory, self.root)
        self.assertFalse(summary["ok"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(summary["accounting"]["attempts_settled"], 0)

    def test_every_failed_checkpoint_stops_later_waves(self):
        replies = [response(status=status) for status in (302, 400, 500, 429)]
        replies += [response(value=envelope(model="wrong")), response(value=envelope(usage=None)), response("A.")]
        for index, reply in enumerate(replies):
            directory = pilot.pilot_directory(self.root, f"failure-run-{index}")
            directory.mkdir()
            manifest = {**self.manifest, "pilot_id": directory.name}
            (directory / "manifest.json").write_bytes(pilot.canonical(manifest))
            for sample in self.samples:
                (directory / sample["request_file"]).write_bytes((self.directory / sample["request_file"]).read_bytes())
            with self.subTest(index=index), patch.object(pilot, "verify", return_value=manifest), patch.object(pilot, "http_request", return_value=reply) as http:
                summary = pilot.run(directory, self.root)
                self.assertFalse(summary["ok"])
                self.assertEqual(http.call_count, 1)
                self.assertEqual(summary["samples_attempted"], 1)
                self.assertEqual(len(summary["levels"]), 1)

    def test_wrong_model_missing_usage_bad_label_bound_http_stop(self):
        # Run execute_attempt in isolation to avoid resetting/reusing any run marker.
        (self.directory / "attempts").mkdir()
        cases = [response(status=status) for status in (302, 400, 500, 429)]
        cases += [response(value=envelope(model="wrong")), response(value=envelope(usage=None)),
                  response("A."), response(value=envelope(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "estimated_cost": "1"}))]
        for index, reply in enumerate(cases):
            attempt_id = f"failure-{index}"
            record = self.samples[0]
            payload = pilot.read_json((self.directory / record["request_file"]).read_bytes())
            with patch.object(pilot, "http_request", return_value=reply) as http:
                outcome = pilot.execute_attempt(self.directory, "mock-run", record, payload,
                                                self.ledger, "token", pilot.InFlight(), attempt_id)
            self.assertFalse(outcome["ok"])
            self.assertIsNone(outcome["label"])
            self.assertEqual(http.call_count, 1)
            attempt = self.ledger.snapshot()["v2_glm"]["attempts"][attempt_id]
            self.assertEqual(attempt["state"], "settled" if index >= 6 else "reserved")


if __name__ == "__main__":
    unittest.main()
