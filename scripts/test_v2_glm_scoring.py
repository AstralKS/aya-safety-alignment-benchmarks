"""Mocked-only bulk scoring tests, with approved temporary writable fixtures."""

import base64
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import v2_glm_budget as budget
import v2_glm_pilot as pilot
import v2_glm_scoring as scoring
from test_v2_glm_pilot import TempCase, envelope, response


def remote_outputs(ready, evidence):
    records = []
    for fact in ready["facts"]:
        source = evidence.subjects[fact["topic"]]
        # Mock the controller output for a short page: one complete chunk, no padding.
        text = "<s>" + source["text"] + "</s>"
        records.append({**fact, "status": "ok", "evidence_revision": source["revision"],
            "evidence_text_sha256": source["text_sha256"], "retrieval_model": scoring.GTR_MODEL,
            "retrieval_revision": scoring.GTR_REVISION, "tokenizer_model": scoring.TOKENIZER_MODEL,
            "tokenizer_revision": scoring.TOKENIZER_REVISION, "query": fact["topic"] + " " + fact["fact"].strip(),
            "passages": [{"title": fact["topic"], "text": text, "passage_index": 0, "score": 0.875,
                          "text_sha256": pilot.sha(text.encode())}]})
    return records


def remote_manifest(ready, **changes):
    return {"stage": "completed", "facts": len(ready["facts"]), "model": scoring.GTR_MODEL,
        "revision": scoring.GTR_REVISION, "tokenizer": scoring.TOKENIZER_MODEL,
        "tokenizer_revision": scoring.TOKENIZER_REVISION, "passage_tokens": 256, "passages_per_fact": 5,
        "passage_scope": "original subject only", "similarity": "np.inner; model default pooling/normalization",
        "input_sha256": pilot.sha(json.dumps(ready, ensure_ascii=False, sort_keys=True).encode()), **changes}


class FixedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = scoring.SourceCatalog(pilot.DEFAULT_ROOT)

    def row(self, dataset, **changes):
        return {"id": "sample", "dataset": dataset, "question": "Question?",
                "reference_answers": ["one", "two"], "should_abstain": False,
                "component": "Mixed", "metadata": {"type": "plant", "name": "Invented name"}, **changes}

    def test_simpleqa_exact_official_template(self):
        row = self.row("simpleqa_verified", reference_answers=["target"])
        jobs = scoring.fixed_jobs(row, "answer", "tiny", self.catalog)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["payload"], pilot.payload_for(row, "answer", pilot.extract_rubric(pilot.DEFAULT_ROOT, pilot.RUBRICS["simpleqa_verified"])))
        self.assertEqual(jobs[0]["parser"], {"kind": "enum", "values": ["A", "B", "C"]})

    def test_abstention_correctness_gated_and_reference_format(self):
        row = self.row("abstention_bench")
        jobs = scoring.fixed_jobs(row, "answer", "8b", self.catalog)
        self.assertEqual([x["stage"] for x in jobs], ["abstention"])
        self.assertIn("[REFERENCE ANSWERS]: [one; two]", jobs[0]["payload"]["messages"][0]["content"])
        jobs = scoring.fixed_jobs(row, "answer", "8b", self.catalog, abstention="No")
        self.assertEqual([x["stage"] for x in jobs], ["abstention", "correctness"])
        self.assertIn("References:\none\ntwo", jobs[1]["payload"]["messages"][0]["content"])
        for label, should_abstain in (("Yes", False), ("No", True), (None, False)):
            self.assertEqual(len(scoring.fixed_jobs({**row, "should_abstain": should_abstain}, "answer", "tiny", self.catalog, abstention=label)), 1)

    def test_precisewiki_independent_and_narrow_rubric_repair(self):
        jobs = scoring.fixed_jobs(self.row("hallulens_precisewikiqa", reference_answers=["gold"]), "answer", "tiny", self.catalog)
        self.assertEqual([x["stage"] for x in jobs], ["abstention", "correctness"])
        job = jobs[1]
        prompt = job["payload"]["messages"][0]["content"]
        self.assertNotIn("Please answer with Yes or No", prompt)
        self.assertIn("CORRECT", prompt)
        self.assertNotEqual(job["lineage"]["original_rubric_sha256"], job["lineage"]["effective_rubric_sha256"])
        self.assertIs(scoring.parse_final('{"is_abstaining":false}', jobs[0]["parser"]), False)
        with self.assertRaises(ValueError):
            scoring.parse_final("Yes", job["parser"])

    def test_nonexistent_domains_and_generated_place(self):
        for domain in ("animal", "plant", "bacteria", "medicine"):
            job = scoring.fixed_jobs(self.row("hallulens_nonexistentrefusal", metadata={"type": domain, "name": "Fake"}), "answer", "tiny", self.catalog)[0]
            self.assertIn("does_believe", job["payload"]["messages"][0]["content"])
            self.assertEqual(job["parser"], {"kind": "boolean-object", "key": "does_believe"})
        row = self.row("hallulens_nonexistentrefusal", component="Generated", metadata={"type_": "restaurant", "name": "Fake", "place": "Paris"})
        prompt = scoring.fixed_jobs(row, "answer", "tiny", self.catalog)[0]["payload"]["messages"][0]["content"]
        self.assertIn("Fake in Paris", prompt)
        with self.assertRaises(ValueError):
            scoring.fixed_jobs(self.row("hallulens_nonexistentrefusal", metadata={"type": "unknown", "name": "Fake"}), "answer", "tiny", self.catalog)

    def test_strict_parsers_no_reasoning_default_or_json_fallback(self):
        schema = {"kind": "boolean-object", "key": "does_believe"}
        for bad in ('{"does_believe":"false"}', '{"does_believe":0}', '{"does_believe":null}',
                    '{"does_believe":true,"extra":1}', '```json\n{"does_believe":false}\n```',
                    '{"does_believe":true,"does_believe":false}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scoring.parse_final(bad, schema)
        parsed = pilot.inspect_response(pilot.canonical(envelope('{"does_believe":false}')), "unused",
                                        lambda text: scoring.parse_final(text, schema))
        self.assertTrue(parsed["ok"], parsed)
        self.assertIs(parsed["label"], False)
        self.assertIsNotNone(parsed["accounting"])

    def test_cache_exact_answer_source_prompt_model_decoding(self):
        row = self.row("simpleqa_verified", reference_answers=["target"])
        one = scoring.fixed_jobs(row, "answer", "tiny", self.catalog)[0]
        two = scoring.fixed_jobs({**row, "id": "other"}, "answer", "8b", self.catalog)[0]
        self.assertEqual(one["cache_key"], two["cache_key"])
        self.assertNotEqual(one["cache_key"], scoring.fixed_jobs(row, "different", "tiny", self.catalog)[0]["cache_key"])


class AvailableContractTests(TempCase):
    """Real packaged jobs and completed downloads, copied only into approved temp."""
    def setUp(self):
        super().setUp()
        self.manifest = json.loads((pilot.DEFAULT_ROOT / "data/v2/manifest.json").read_bytes())
        for relative in ("data/v2/manifest.json", "data/v2/evaluation.jsonl", "data/v2/generation.jsonl", "data/v2/reuse.json"):
            self.copy(relative)
        self.jobs = {}
        for short, names in {
            "tiny": ["tiny-selfaware-000-770142ed", "tiny-selfaware-001-cf58230b"],
            "8b": ["8b-selfaware-000-770142ed"],
        }.items():
            self.copy(pilot.CHECKPOINTS[short][0])
            self.jobs[short] = []
            records = []
            for name in names:
                relative = f"runs/v2/{short}/jobs/{name}"
                for member in ("job.json", "runner.py", "run.ipynb", "kernel-metadata.json", "collected.json",
                               "output/baseline/responses.jsonl", "output/baseline/manifest.json"):
                    self.copy(relative + "/" + member)
                self.jobs[short].append(self.root / relative)
                records.extend(json.loads(line) for line in (self.root / relative / "output/baseline/responses.jsonl").read_bytes().splitlines())
            self.write_records(short, records)

    def copy(self, relative):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((pilot.DEFAULT_ROOT / relative).read_bytes())

    def records(self, short="tiny"):
        return [json.loads(line) for line in (self.root / f"runs/v2/{short}/responses.jsonl").read_bytes().splitlines()]

    def write_records(self, short, records):
        path = self.root / f"runs/v2/{short}/responses.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(pilot.canonical(row) for row in records))

    def load(self, models=("tiny", "8b")):
        # Only the data assembly verifier is substituted; generation validators are real.
        with patch("prepare_v2_data.verify", return_value=self.manifest), patch("subprocess.run", side_effect=AssertionError("No downloads or subprocesses")) as external:
            value = scoring.load_available(self.root, models)
        external.assert_not_called()
        return value

    def test_real_tiny1500_8b750_collected_generation_contracts(self):
        available, _ = self.load()
        for short, expected in (("tiny", 1500), ("8b", 750)):
            generated = [(row, text) for row, text, model in available if model == short and row["_response_file"].startswith("runs/v2/")]
            self.assertEqual(len(generated), expected)
            disk = {row["id"]: row["response"] for row in self.records(short)}
            self.assertEqual({row["id"]: text for row, text in generated}, disk)

    def test_partial_aggregate_can_use_only_collected_completed_outputs(self):
        records = self.records()
        self.write_records("tiny", records[:7])
        self.assertEqual(len(self.load(("tiny",))[0]), 1978 + 7)
        self.jobs["tiny"][0].joinpath("collected.json").unlink()
        with self.assertRaises(ValueError):
            self.load(("tiny",))

    def test_receipt_disappearing_inside_collector_cannot_trigger_download(self):
        import v2_generation_queue as queue
        collect = queue.collect
        def disappearing(path, job):
            (self.jobs["tiny"][0] / "collected.json").unlink()
            return collect(path, job)
        with patch.object(queue, "collect", side_effect=disappearing), self.assertRaises(ValueError):
            self.load(("tiny",))

    def test_aggregate_requires_exact_record_provenance_text_and_status(self):
        original = self.records()
        mutations = [(key, None) for key in ("revision", "chat_template_sha256", "selected_template_sha256",
                     "example_sha256", "input_sha256", "prompt_sha256", "component", "max_new_tokens")]
        mutations += [("revision", "wrong"), ("response", "Injected text."), ("status", "error"),
                      ("status", None), ("input_tokens", 1), ("output_tokens", True), ("finish_reason", "unknown"),
                      ("unexpected", "not in downloaded record"), ("v2_manifest_sha256", "wrong")]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                records = deepcopy(original)
                if value is None:
                    records[0].pop(key, None)
                else:
                    records[0][key] = value
                self.write_records("tiny", records)
                with self.assertRaises(ValueError):
                    self.load(("tiny",))

    def test_duplicate_aggregate_and_completed_shard_ids_rejected(self):
        records = self.records()
        self.write_records("tiny", records + records[:1])
        with self.assertRaises(ValueError):
            self.load(("tiny",))
        self.write_records("tiny", records)
        shutil.copytree(self.jobs["tiny"][0], self.jobs["tiny"][0].with_name("duplicate-collected-job"))
        with self.assertRaises(ValueError):
            self.load(("tiny",))

    def test_completed_download_receipt_runtime_and_packaged_job_tamper_rejected(self):
        directory = self.jobs["tiny"][0]
        for member in ("output/baseline/responses.jsonl", "output/baseline/manifest.json", "collected.json", "run.ipynb", "runner.py", "kernel-metadata.json"):
            with self.subTest(member=member):
                path = directory / member
                raw = path.read_bytes()
                path.write_bytes(json.dumps({**json.loads(raw), "answers": 1}).encode() if member == "collected.json" else raw + b" ")
                try:
                    with self.assertRaises(ValueError):
                        self.load(("tiny",))
                finally:
                    path.write_bytes(raw)

    def test_completed_runtime_errors_and_invalid_rows_rejected_even_with_matching_receipt(self):
        directory = self.jobs["tiny"][0]
        marker_path = directory / "collected.json"
        marker_raw = marker_path.read_bytes()
        for member, changes in (("manifest.json", {"stage": "completed_with_errors", "errors": 1}),
                                ("manifest.json", {"resolved_revision": "wrong"}),
                                ("manifest.json", {"transformers": "wrong"}),
                                ("manifest.json", {"device_map": {"": "cpu"}}),
                                ("responses.jsonl", {"status": "error"}),
                                ("responses.jsonl", {"prompt_sha256": None}),
                                ("responses.jsonl", {"example_sha256": "wrong"}),
                                ("responses.jsonl", {"input_tokens": True}),
                                ("responses.jsonl", {"finish_reason": "unknown"})):
            with self.subTest(member=member, changes=changes):
                path = directory / "output/baseline" / member
                raw = path.read_bytes()
                if member == "manifest.json":
                    updated = json.dumps({**json.loads(raw), **changes}).encode()
                    hash_key = "runtime_manifest_sha256"
                else:
                    rows = [json.loads(line) for line in raw.splitlines()]
                    rows[0].update(changes)
                    updated = b"".join(pilot.canonical(row) for row in rows)
                    hash_key = "responses_sha256"
                path.write_bytes(updated)
                marker_path.write_bytes(json.dumps({**json.loads(marker_raw), hash_key: pilot.sha(updated)}).encode())
                try:
                    with self.assertRaises(ValueError):
                        self.load(("tiny",))
                finally:
                    path.write_bytes(raw)
                    marker_path.write_bytes(marker_raw)

    def test_collected_job_bound_to_verified_generation_and_native_templates(self):
        import launch_v2
        original = launch_v2.verify_job(self.jobs["tiny"][0])
        for key in ("v2_manifest_sha256", "generation_sha256", "chat_template_sha256", "selected_template_sha256", "eos_token_id", "pad_token_id"):
            with self.subTest(key=key):
                job = deepcopy(original)
                job["config"][key] = "wrong"
                with patch.object(launch_v2, "verify_job", return_value=job), self.assertRaises(ValueError):
                    self.load(("tiny",))
        job = deepcopy(original)
        job["examples"][0]["question"] += " altered"
        with patch.object(launch_v2, "verify_job", return_value=job), self.assertRaises(ValueError):
            self.load(("tiny",))

    def test_readonly_current_production_loader_uses_no_downloads(self):
        with patch("subprocess.run", side_effect=AssertionError("No downloads")) as external:
            available, _ = scoring.load_available(pilot.DEFAULT_ROOT, ("tiny", "8b"))
        external.assert_not_called()
        self.assertGreaterEqual(sum(short == "tiny" for _, _, short in available), 1978 + 1500)
        self.assertGreaterEqual(sum(short == "8b" for _, _, short in available), 1978 + 750)


class FixedMetricTests(unittest.TestCase):
    def metrics(self, task, labels, expected=None):
        available, outcomes = [], []
        for index, (desired, primary, correctness) in enumerate(labels):
            row = {"id": str(index), "dataset": task, "should_abstain": desired}
            available.append((row, "answer", "tiny"))
            primary_stage = {"simpleqa_verified": "simpleqa", "hallulens_nonexistentrefusal": "does_believe"}.get(task, "abstention")
            for stage, label in ((primary_stage, primary), ("correctness", correctness)):
                if label is not None:
                    outcomes.append({"lineage": {"aya_model": "tiny", "dataset": task, "id": str(index)},
                                     "stage": stage, "ok": label != "pending", "label": label})
        with patch.dict(scoring.EXPECTED, {task: expected or len(labels)}):
            return scoring.fixed_metrics(outcomes, available, ["tiny"], [task])[0], outcomes, available

    def test_simpleqa_full_and_observed_yield_coverage_attempted_accuracy(self):
        metric, _, _ = self.metrics("simpleqa_verified", [(False, x, None) for x in ("A", "B", "C")], expected=4)
        self.assertEqual(metric["observed_answer_coverage"], 2 / 3)
        self.assertEqual(metric["observed_accuracy_when_attempted"], 1 / 2)
        self.assertEqual(metric["observed_correct_answer_yield"], 1 / 3)
        self.assertIsNone(metric["full_correct_answer_yield"])
        self.assertIsNone(metric["full_answer_coverage"])
        self.assertIsNone(metric["full_accuracy_when_attempted"])
        self.assertEqual(metric["primary_missing_or_error"], 1)
        self.assertFalse(metric["full_conditional_complete"])
        metric, _, _ = self.metrics("simpleqa_verified", [(False, x, None) for x in ("A", "B", "C")])
        self.assertTrue(metric["full_primary_complete"] and metric["full_conditional_complete"])
        self.assertEqual(metric["full_answer_coverage"], 2 / 3)
        self.assertEqual(metric["full_accuracy_when_attempted"], 1 / 2)

    def test_abstention_joint_success_requires_complete_eligible_correctness(self):
        labels = [(True, "Yes", None), (True, "No", None), (False, "Yes", None),
                  (False, "No", "correct"), (False, "No", "pending")]
        metric, _, _ = self.metrics("abstention_bench", labels)
        self.assertTrue(metric["full_primary_complete"])
        self.assertFalse(metric["full_conditional_complete"])
        self.assertEqual(metric["observed_joint_task_success"], 2 / 4)
        self.assertEqual(metric["joint_valid"], 4)
        self.assertEqual(metric["correctness_missing_or_error"], 1)
        self.assertEqual(metric["full_behavior_accuracy"], 3 / 5)
        self.assertIsNone(metric["full_joint_task_success"])
        labels[-1] = (False, "No", "incorrect")
        metric, _, _ = self.metrics("abstention_bench", labels)
        self.assertTrue(metric["full_conditional_complete"])
        self.assertEqual(metric["full_joint_task_success"], 2 / 5)

    def test_precise_pairs_and_nonexistent_full_metrics(self):
        metric, _, _ = self.metrics("hallulens_precisewikiqa", [(False, False, "INCORRECT"), (False, True, "pending")])
        self.assertTrue(metric["full_primary_complete"])
        self.assertFalse(metric["full_conditional_complete"])
        self.assertIsNone(metric["full_hallucination_rate_nonabstained"])
        self.assertEqual(metric["paired_missing_or_error"], 1)
        metric, _, _ = self.metrics("hallulens_nonexistentrefusal", [(False, False, None), (False, True, None)])
        self.assertEqual(metric["full_false_acceptance_rate"], 1 / 2)

    def test_empty_and_unattempted_metrics_never_invent_denominators(self):
        for task in scoring.FIXED_TASKS:
            with self.subTest(task=task):
                metric = scoring.fixed_metrics([], [], ["tiny"], [task])[0]
                self.assertFalse(metric["full_primary_complete"])
                self.assertFalse(metric["full_conditional_complete"])
                self.assertEqual(metric["primary_missing_or_error"], scoring.EXPECTED[task])
        metric, _, _ = self.metrics("simpleqa_verified", [(False, "C", None)])
        self.assertEqual(metric["full_answer_coverage"], 0)
        self.assertIsNone(metric["full_accuracy_when_attempted"])
        metric, _, _ = self.metrics("abstention_bench", [(False, "Yes", None)])
        self.assertTrue(metric["full_conditional_complete"])
        self.assertEqual(metric["full_joint_task_success"], 0)
        self.assertIsNone(metric["full_correctness"])

    def test_precise_unpaired_judgments_are_not_correctness_or_joint_scores(self):
        metric, _, _ = self.metrics("hallulens_precisewikiqa", [(False, "pending", "CORRECT")])
        self.assertEqual(metric["correctness_valid"], 1)
        self.assertEqual(metric["paired_valid"], 0)
        self.assertIsNone(metric["observed_correct_rate"])
        self.assertIsNone(metric["full_correct_rate"])
        self.assertFalse(metric["full_conditional_complete"])

    def test_metrics_missing_generation_and_pending_are_separately_counted(self):
        metric, _, _ = self.metrics("simpleqa_verified", [(False, "A", None), (False, "pending", None)], expected=4)
        self.assertEqual(metric["responses_missing"], 2)
        self.assertEqual(metric["primary_missing"], 2)
        self.assertEqual(metric["primary_errors_or_pending"], 1)
        self.assertEqual(metric["primary_missing_or_error"], 3)
        self.assertTrue(metric["partial_metrics_are_not_full_benchmark_scores"])

    def test_metrics_reject_duplicate_unknown_stages_labels_and_ineligible_pairs(self):
        for task, labels in (("simpleqa_verified", [(False, "A", None)]),
                             ("abstention_bench", [(False, "No", "correct")]),
                             ("hallulens_precisewikiqa", [(False, False, "CORRECT")]),
                             ("hallulens_nonexistentrefusal", [(False, False, None)])):
            _, outcomes, available = self.metrics(task, labels)
            bad = [outcomes + outcomes[:1]]
            for changes in ({"stage": "unknown"}, {"label": None}, {"label": "wrong"}, {"label": 0},
                            {"lineage": {"aya_model": "tiny", "dataset": task, "id": "unknown"}}):
                bad.append([{**outcomes[0], **changes}] + outcomes[1:])
            for rows in bad:
                with self.subTest(task=task, rows=rows), self.assertRaises(ValueError):
                    scoring.fixed_metrics(rows, available, ["tiny"], [task])
        for desired, primary in ((True, "No"), (False, "Yes"), (False, "pending")):
            with self.subTest(desired=desired, primary=primary), self.assertRaises(ValueError):
                self.metrics("abstention_bench", [(desired, primary, "correct")])


class StoreTests(TempCase):
    def setUp(self):
        super().setUp()
        self.ledger = self.make_ledger()
        catalog = scoring.SourceCatalog(pilot.DEFAULT_ROOT)
        row = {"id": "id", "dataset": "simpleqa_verified", "question": "q", "reference_answers": ["target"], "should_abstain": False}
        self.job = scoring.fixed_jobs(row, "answer", "tiny", catalog)[0]

    def store(self, dispatch=True):
        return scoring.JobStore(self.root, "test-run", concurrency=2, dispatch=dispatch)

    def test_durable_success_resume_never_repeats_http(self):
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response()) as http:
            first = self.store().evaluate([self.job])
            second = self.store().evaluate([self.job])
        self.assertTrue(first[0]["ok"], first)
        self.assertTrue(second[0]["ok"], second)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(self.ledger.totals()["attempts_reserved"], 1)

    def test_same_payload_two_aya_identities_share_one_attempt(self):
        other = {**self.job, "lineage": {**self.job["lineage"], "id": "other-id", "aya_model": "8b"}}
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response()) as http:
            outcomes = self.store().evaluate([self.job, other])
        self.assertTrue(all(o["ok"] for o in outcomes), outcomes)
        self.assertEqual(http.call_count, 1)
        self.assertEqual({o["lineage"]["aya_model"] for o in outcomes}, {"tiny", "8b"})

    def test_transport_failure_saved_and_restart_never_retried(self):
        import requests
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", side_effect=requests.ReadTimeout("secret")) as http:
            first = self.store().evaluate([self.job])
            second = self.store().evaluate([self.job])
        self.assertFalse(first[0]["ok"])
        self.assertFalse(second[0]["ok"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(self.ledger.totals()["attempts_settled"], 0)

    def test_limited_headroom_no_full_run_gate_and_ceiling_stop(self):
        self.ledger.reserve("prior", "prior", "request")
        charge = budget.CEILING - budget.LEGACY["reserved_upper_usd"] - budget.ATTEMPT_BOUND
        prior = envelope(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "estimated_cost": str(charge)})
        self.ledger.settle("prior", budget.usage_price(prior["usage"], True), self.raw_journal("prior", prior))
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response()) as http:
            result = self.store().evaluate([self.job])
        self.assertTrue(result[0]["ok"], result)
        self.assertEqual(http.call_count, 1)

    def test_pilot_exact_request_cache_reused_without_http(self):
        store = self.store(dispatch=False)
        self.ledger.reserve("pilot-saved", "pilot", self.job["request_sha256"])
        raw = pilot.canonical(envelope())
        path = self.root / "pilot-original.response.json"
        budget.write_once(path, budget.json_bytes({"kind": "http_response", "http_status": 200,
            "attempt_id": "pilot-saved", "request_sha256": self.job["request_sha256"],
            "raw_body_base64": base64.b64encode(raw).decode(), "raw_body_sha256": pilot.sha(raw)}))
        store.pilot_cache[self.job["cache_key"]] = [path]
        before = path.read_bytes()
        with patch.object(pilot, "http_request") as http:
            result = store.evaluate([self.job])
        self.assertTrue(result[0]["ok"], result)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.ledger.totals()["attempts_settled"], 1)
        http.assert_not_called()

    def test_transport_ambiguity_and_invalid_do_not_automatically_retry(self):
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response("A.")) as http:
            first = self.store().evaluate([self.job])
            again = self.store().evaluate([self.job])
            explicit = self.store().evaluate([self.job], retry_invalid=[self.job["cache_key"]])
        self.assertFalse(first[0]["ok"])
        self.assertFalse(again[0]["ok"])
        self.assertFalse(explicit[0]["ok"])
        self.assertEqual(http.call_count, 2)
        self.assertNotEqual(first[0]["attempt_id"], explicit[0]["attempt_id"])

    def test_offline_pending_no_token_or_dispatch(self):
        with patch.object(pilot, "token") as token, patch.object(pilot, "http_request") as http:
            result = self.store(dispatch=False).evaluate([self.job])
        self.assertEqual(result[0]["state"], "pending")
        token.assert_not_called()
        http.assert_not_called()

    def test_unknown_dispatch_marker_restart_no_replay(self):
        store = self.store()
        folder = store.journal_root / self.job["cache_key"]
        folder.mkdir(parents=True)
        budget.write_once(folder / "job.json", budget.json_bytes(scoring.request_contract(self.job)))
        (folder / "attempts").mkdir()
        self.ledger.reserve("unknown", "bulk:test-run", self.job["request_sha256"])
        budget.write_once(folder / "attempts/unknown.dispatch.json", budget.json_bytes({"attempt_id": "unknown", "request_sha256": self.job["request_sha256"]}))
        with patch.object(pilot, "http_request") as http:
            result = store.evaluate([self.job])
        self.assertFalse(result[0]["ok"])
        self.assertEqual(result[0]["state"], "unknown")
        http.assert_not_called()

    def test_reserved_pilot_without_response_cannot_be_replayed_as_bulk(self):
        directory = self.root / "scoring/v2-glm/pilots/frozen-pilot"
        (directory / "attempts").mkdir(parents=True)
        budget.write_once(directory / "manifest.json", b"{}\n")
        budget.write_once(directory / "request-001.json", pilot.canonical(self.job["payload"]))
        source, source_sha = next(iter(self.job["identity"]["source_files"].items()))
        sample = {"request_sha256": self.job["request_sha256"], "request_file": "request-001.json",
                  "dataset": "simpleqa_verified", "rubric_source_path": source, "rubric_source_sha256": source_sha,
                  "answer_sha256": self.job["identity"]["answer_sha256"]}
        self.ledger.reserve("pilot-unknown", "frozen-pilot", self.job["request_sha256"])
        store = self.store()
        with patch.object(pilot, "verify", return_value={"pilot_id": "frozen-pilot", "samples": [sample]}), patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response()) as http:
            store.index_pilots()
            result = store.evaluate([self.job], retry_invalid=[self.job["cache_key"]])
        self.assertFalse(result[0]["ok"])
        self.assertEqual(result[0]["state"], "unknown")
        http.assert_not_called()
        self.assertEqual(self.ledger.totals()["attempts_reserved"], 1)

    def test_resumed_bound_failure_stops_before_new_job_or_explicit_retry(self):
        store = scoring.JobStore(self.root, "bounds", concurrency=1, dispatch=True)
        bad = envelope(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "estimated_cost": "1"})
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response(value=bad)) as http:
            first = store.evaluate([self.job])
        self.assertTrue(store.halted)
        row = {"id": "other", "dataset": "simpleqa_verified", "question": "different", "reference_answers": ["target"], "should_abstain": False}
        other = scoring.fixed_jobs(row, "answer", "tiny", scoring.SourceCatalog(pilot.DEFAULT_ROOT))[0]
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response()) as http:
            restarted = scoring.JobStore(self.root, "bounds", concurrency=1, dispatch=True)
            outcomes = restarted.evaluate([self.job, other], retry_invalid=[self.job["cache_key"]])
        http.assert_not_called()
        self.assertTrue(restarted.halted)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(self.ledger.totals()["attempts_reserved"], 1)

    def test_job_cache_contract_tamper_refused_before_reservation(self):
        job = deepcopy(self.job)
        job["payload"]["messages"][0]["content"] = "Injected different scoring request."
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response()) as http:
            result = self.store().evaluate([job])
        self.assertFalse(result[0]["ok"])
        http.assert_not_called()
        self.assertEqual(self.ledger.totals()["attempts_reserved"], 0)

    def test_saved_usage_null_reparsed_settled_without_http(self):
        store = self.store(dispatch=False)
        folder = store.journal_root / self.job["cache_key"]
        folder.mkdir(parents=True)
        budget.write_once(folder / "job.json", budget.json_bytes(scoring.request_contract(self.job)))
        (folder / "attempts").mkdir()
        self.ledger.reserve("saved", "bulk:test-run", self.job["request_sha256"])
        raw = pilot.canonical(envelope(usage={"prompt_tokens": 1378, "completion_tokens": 30,
                                           "total_tokens": 1408, "prompt_tokens_details": {"cache_write_tokens": None}}))
        budget.write_once(folder / "attempts/saved.response.json", budget.json_bytes({
            "kind": "http_response", "http_status": 200, "attempt_id": "saved",
            "request_sha256": self.job["request_sha256"], "raw_body_base64": base64.b64encode(raw).decode(),
            "raw_body_sha256": pilot.sha(raw)}))
        with patch.object(pilot, "http_request") as http:
            result = store.evaluate([self.job])
        self.assertTrue(result[0]["ok"], result)
        self.assertEqual(self.ledger.totals()["attempts_settled"], 1)
        http.assert_not_called()


class FactScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = scoring.SourceCatalog(pilot.DEFAULT_ROOT)

    def test_atomic_extraction_strict_sentence_fact_content_preserved(self):
        sentence = "Alice was born in 1990 and is a writer."
        schema = {"kind": "atomic", "sentence": sentence}
        value = {"sentence": sentence, "facts": ["Alice was born in 1990.", "Alice is a writer."]}
        self.assertEqual(scoring.parse_final(pilot.canonical(value).decode(), schema), value)
        for bad in ({"sentence": "different", "facts": []}, {"sentence": sentence, "facts": [""]},
                    {"sentence": sentence, "facts": ["same", "same"]}, {"sentence": sentence, "facts": [True]},
                    {"sentence": sentence}, {**value, "extra": 1}):
            with self.assertRaises(ValueError):
                scoring.parse_final(pilot.canonical(bad).decode(), schema)

    def test_extraction_jobs_bind_exact_spans_upstream_demos(self):
        row = {"id": "bio", "dataset": "factscore_bios", "metadata": {"topic": "Alice"}, "question": "Tell me about Alice."}
        answer = "Alice was born in 1990. She is a writer."
        jobs = scoring.extraction_jobs(row, answer, "tiny", self.catalog)
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertIn("Please breakdown the following sentence into independent facts:", job["payload"]["messages"][0]["content"])
            self.assertEqual(answer[job["lineage"]["sentence_start"]:job["lineage"]["sentence_end"]], job["parser"]["sentence"])
            self.assertIn("demos_archive_sha256", job["lineage"])

    def test_fact_metrics_macro_precision_gamma_and_nofact_coverage(self):
        rows = [{"id": "one", "facts": [{"is_supported": True}, {"is_supported": False}]},
                {"id": "two", "facts": [{"is_supported": True}]}, {"id": "no-facts", "facts": []},
                {"id": "error", "facts": None}]
        metrics = scoring.factscore_metrics(rows, expected=180)
        self.assertEqual(metrics["macro_precision"], 0.75)
        self.assertEqual(metrics["responding_fraction_of_completed_extractions"], 2 / 3)
        self.assertEqual(metrics["no_fact_excluded_from_precision"], 1)
        self.assertEqual(metrics["fully_verified_responses"], 2)
        self.assertEqual(metrics["total_facts_verified"], 3)
        self.assertLess(metrics["gamma10_macro_precision"], metrics["macro_precision"])

    def retrieval_fixture(self):
        text = "Alice is a writer. Alice was born in 1990. Alice lives in Rome. Alice wrote books. Alice won an award."
        source = {"topic": "Alice", "title": "Alice", "revision": 123, "url": "https://en.wikipedia.org/w/index.php?oldid=123",
                  "text": text, "text_sha256": pilot.sha(text.encode()), "text_file": "data/v2/factscore_evidence/alice/text.txt"}
        evidence = SimpleNamespace(subjects={"Alice": source}, manifest_sha256="frozen-manifest")
        row = {"id": "bio", "dataset": "factscore_bios", "question": "Tell me about Alice.", "metadata": {"topic": "Alice"}}
        fact = {"fact_id": "fact-one", "fact_sha256": pilot.sha(b"Alice is a writer."), "text": "Alice is a writer.",
                "fact": "Alice is a writer.", "aya_model": "tiny", "id": "bio", "topic": "Alice", "answer_sha256": pilot.sha(text.encode())}
        ready = {"facts": [fact], "evidence_manifest_sha256": "frozen-manifest", "retrieval_model": scoring.GTR_MODEL, "retrieval_revision": scoring.GTR_REVISION}
        return ready, remote_outputs(ready, evidence), evidence, [(row, text, "tiny")]

    def test_frozen_remote_short_page_verification_and_strict_support_labels(self):
        ready, retrieval, evidence, available = self.retrieval_fixture()
        jobs = scoring.verification_jobs(ready, retrieval, evidence, available, self.catalog)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        prompt = job["payload"]["messages"][0]["content"]
        self.assertIn("Alice won an award", prompt)
        self.assertTrue(prompt.endswith("Input: Alice is a writer. True or False?\nOutput:"))
        self.assertEqual(scoring.parse_final("True", job["parser"]), "True")
        for bad in ("Probably true", "False.", "true false", "UNVERIFIABLE"):
            with self.assertRaises(ValueError):
                scoring.parse_final(bad, job["parser"])
        for key, bad in (("retrieval_revision", "wrong"), ("tokenizer_revision", "wrong"), ("evidence_revision", 124)):
            with self.assertRaises(ValueError):
                scoring.verification_jobs(ready, [{**retrieval[0], key: bad}], evidence, available, self.catalog)
        broken = [{**retrieval[0], "passages": []}]
        with self.assertRaises(ValueError):
            scoring.verification_jobs(ready, broken, evidence, available, self.catalog)

    def test_legacy_character_span_retrieval_hard_rejected(self):
        ready, _, evidence, available = self.retrieval_fixture()
        retrieval = {"kind": "glm-factscore-gtr-retrieval-v1", "passages": [{"start": 0, "end": 5}], "queries": []}
        with self.assertRaises(ValueError):
            scoring.verification_jobs(ready, retrieval, evidence, available, self.catalog)


class PipelineTests(TempCase):
    def setUp(self):
        super().setUp()
        self.make_ledger()
        self.catalog = scoring.SourceCatalog(pilot.DEFAULT_ROOT)

    def test_score_available_abstention_then_eligible_correctness(self):
        available = [({"id": "abstention", "dataset": "abstention_bench", "question": "q", "reference_answers": ["gold"], "should_abstain": False}, "answer", "tiny")]
        def http(request, token):
            text = request["messages"][0]["content"]
            return response("No" if "[GROUND TRUTH ABSTENTION LABEL]" in text else "correct")
        with patch.object(scoring, "load_available", return_value=(available, {"binding": "fixture"})), patch.object(scoring, "SourceCatalog", return_value=self.catalog), patch.object(scoring.JobStore, "index_pilots"), patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", side_effect=http) as http_mock:
            result = scoring.score_fixed(self.root, "pipeline", models=["tiny"], tasks=["abstention_bench"], dispatch=True)
        self.assertEqual(http_mock.call_count, 2)
        self.assertEqual(result["metrics"][0]["correctness_valid"], 1)
        self.assertEqual(result["metrics"][0]["observed_correctness"], 1)
        self.assertFalse(result["metrics"][0]["full_primary_complete"])

    def test_extraction_packet_revalidated_against_raw_journal_and_tamper(self):
        row = {"id": "bio", "dataset": "factscore_bios", "question": "Tell me about Alice.", "metadata": {"topic": "Alice"}}
        answer = "Alice is a writer."
        available = [(row, answer, "tiny")]
        jobs = scoring.extraction_jobs(row, answer, "tiny", self.catalog)
        label = pilot.canonical({"sentence": answer, "facts": [answer]}).decode().strip()
        store = scoring.JobStore(self.root, "extractions", dispatch=True)
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response(label)) as http:
            outcomes = store.evaluate(jobs)
            packet = scoring.extraction_packet(available, jobs, outcomes, {"input": "frozen"}, root=self.root)
            scoring.validate_extraction_packet(packet, available, self.catalog, store, {"input": "frozen"})
        self.assertEqual(http.call_count, 1)
        self.assertEqual(packet["responses"][0]["facts"][0]["text"], answer)
        packet["responses"][0]["facts"][0]["text"] = "Injected false human annotation."
        with self.assertRaises(ValueError):
            scoring.validate_extraction_packet(packet, available, self.catalog, store, {"input": "frozen"})

    def test_full_factscore_staged_pipeline_and_reuse(self):
        _, _, evidence, source_available = FactScoreTests().retrieval_fixture()
        row, _, short = source_available[0]
        answer = "Alice is a writer."
        available = [(row, answer, short)]
        def http(request, token):
            prompt = request["messages"][0]["content"]
            label = pilot.canonical({"sentence": answer, "facts": [answer]}).decode().strip() if "Please breakdown" in prompt else "True"
            return response(label)
        with patch.object(scoring, "load_available", return_value=(available, {"input": "frozen"})), patch.object(scoring, "SourceCatalog", return_value=self.catalog), patch.object(scoring, "EvidenceCatalog", return_value=evidence), patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", side_effect=http) as http_mock:
            extracted = scoring.factscore_stage(self.root, "staged", "extract", models=["tiny"], dispatch=True)
            ready_result = scoring.factscore_stage(self.root, "staged", "retrieval-ready", models=["tiny"], extraction_path=extracted["snapshot"])
            ready = pilot.read_json(Path(ready_result["snapshot"]).read_bytes())
            retrieval = remote_outputs(ready, evidence)
            retrieval_path = self.root / "retrieval.jsonl"
            budget.write_once(retrieval_path, b"".join(pilot.canonical(record) for record in retrieval))
            budget.write_once(self.root / "manifest.json", budget.json_bytes(remote_manifest(ready)))
            verified = scoring.factscore_stage(self.root, "staged", "verify", models=["tiny"], dispatch=True,
                extraction_path=extracted["snapshot"], retrieval_path=retrieval_path)
            retrieval[0]["controller_metadata"] = "Extra lineage does not change existing fact/passage support requests."
            other_path = self.root / "retrieval-more-lineage.jsonl"
            budget.write_once(other_path, b"".join(pilot.canonical(record) for record in retrieval))
            scoring.factscore_stage(self.root, "staged", "verify", models=["tiny"], dispatch=True,
                extraction_path=extracted["snapshot"], retrieval_path=other_path)
        self.assertEqual(http_mock.call_count, 2)  # One fresh extraction + one fresh support judgment.
        self.assertEqual(verified["metrics"]["tiny"]["macro_precision"], 1)
        self.assertEqual(verified["metrics"]["tiny"]["total_facts_verified"], 1)

    def test_legacy_retrieval_rejected_before_any_store_or_paid_evaluation(self):
        with patch.object(scoring, "load_available") as loader, patch.object(scoring, "JobStore") as store, patch.object(pilot, "http_request") as http:
            with self.assertRaises(ValueError):
                scoring.factscore_stage(self.root, "legacy", "verify", models=["tiny"], dispatch=True,
                    extraction_path=self.root / "extraction.json", retrieval_path=self.root / "legacy.json")
        loader.assert_not_called()
        store.assert_not_called()
        http.assert_not_called()


class RemoteRetrievalTests(TempCase):
    def setUp(self):
        super().setUp()
        self.ledger = self.make_ledger()
        self.catalog = scoring.SourceCatalog(pilot.DEFAULT_ROOT)
        _, _, self.evidence, self.available = FactScoreTests().retrieval_fixture()
        # Unicode and whitespace exercise the actual controller's input serialization.
        source = self.evidence.subjects["Alice"]
        source["text"] += " Alice likes caf\u00e9s."
        source["text_sha256"] = pilot.sha(source["text"].encode())
        row, _, short = self.available[0]
        self.answer = "Alice is a writer."
        self.available = [(row, self.answer, short)]
        journal_path = self.root / "scoring/v2-glm/request-journals/fixture/attempts/extraction.response.json"
        journal_path.parent.mkdir(parents=True)
        body = pilot.canonical(envelope(json.dumps({"sentence": self.answer, "facts": [self.answer]})))
        journal_raw = budget.json_bytes({"kind": "http_response", "http_status": 200, "attempt_id": "extraction",
            "request_sha256": "fixture-request", "raw_body_base64": base64.b64encode(body).decode(), "raw_body_sha256": pilot.sha(body)})
        budget.write_once(journal_path, journal_raw)
        fact = {"aya_model": short, "id": row["id"], "topic": "Alice", "fact_id": "fact-remote",
                "text": self.answer, "fact_sha256": pilot.sha(self.answer.encode()),
                "answer_sha256": pilot.sha(self.answer.encode()), "sentence_index": 0, "fact_index": 0,
                "extraction_cache_key": "extraction-key", "extraction_journal_sha256": pilot.sha(journal_raw),
                "extraction_journal": journal_path.relative_to(self.root).as_posix(),
                "extraction_response_sha256": pilot.sha(body)}
        packet = {"responses": [{"topic": "Alice", "completed": True, "facts": [fact]}]}
        self.ready = scoring.retrieval_ready(packet, self.evidence, root=self.root)
        self.records = self.outputs(self.ready)
        self.path = self.root / "remote/retrieval.jsonl"
        self.path.parent.mkdir()
        self.manifest_path = self.path.with_name("manifest.json")

    def outputs(self, ready):
        return remote_outputs(ready, self.evidence)

    def write_outputs(self, records=None, manifest_changes=None):
        self.path.write_bytes(b"".join((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
                                      for record in (self.records if records is None else records)))
        manifest = remote_manifest(self.ready, **(manifest_changes or {}))
        self.manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())

    def jobs(self, records=None):
        return scoring.remote_verification_jobs(self.ready, self.records if records is None else records,
                                                self.evidence, self.available)

    def test_export_passes_controller_stdlib_validation_and_preserves_original_facts(self):
        from v2_factscore_retrieval import validate_inputs
        validate_inputs(self.ready)  # No run(), tokenizer, torch or encoder import.
        fact = self.ready["facts"][0]
        self.assertEqual(fact["fact"], self.answer)
        self.assertEqual(fact["response_sha256"], pilot.sha(self.answer.encode()))
        self.assertEqual(self.ready["evidence"][0]["text"], self.evidence.subjects["Alice"]["text"])
        self.assertTrue(fact["extraction_journal"].startswith("scoring/v2-glm/"))
        self.assertEqual(len(fact["extraction_journal_sha256"]), 64)
        self.assertEqual(self.ready["tokenizer_revision"], scoring.TOKENIZER_REVISION)

    def test_manifest_exact_controller_serialization_and_decimal_scores(self):
        self.write_outputs()
        ready = pilot.read_json(budget.json_bytes(self.ready))
        records = scoring.read_remote_retrieval(self.path, ready)
        self.assertIsInstance(records[0]["passages"][0]["score"], Decimal)
        job = self.jobs(records)[0]
        self.assertEqual(job["lineage"]["passages_returned"], 1)
        self.assertIn("Alice likes caf\u00e9s.", job["payload"]["messages"][0]["content"])

    def test_manifest_rejects_incomplete_wrong_hash_pins_counts_and_types(self):
        for key, value in (("stage", "running"), ("input_sha256", "wrong"), ("model", "other"),
                           ("revision", "wrong"), ("tokenizer", "other"), ("tokenizer_revision", "wrong"),
                           ("facts", 0), ("facts", True), ("passage_tokens", 256.0), ("passages_per_fact", 5.0),
                           ("passage_scope", "all subjects"), ("similarity", "cosine")):
            with self.subTest(key=key, value=value):
                self.write_outputs(manifest_changes={key: value})
                with self.assertRaises(ValueError):
                    scoring.read_remote_retrieval(self.path, self.ready)
        self.write_outputs(records=[])
        with self.assertRaises(ValueError):
            scoring.read_remote_retrieval(self.path, self.ready)
        self.write_outputs()
        self.manifest_path.unlink()
        with self.assertRaises(FileNotFoundError):
            scoring.read_remote_retrieval(self.path, self.ready)

    def test_remote_rejects_original_lineage_fact_and_pin_mutation(self):
        for key, value in (("fact", "Alice is not a writer."), ("text", "changed"), ("fact_sha256", "changed"),
                           ("response_sha256", "changed"), ("id", "other-bio"), ("aya_model", "8b"),
                           ("extraction_journal_sha256", "changed"), ("sentence_index", True),
                           ("source_text_sha256", "changed"), ("status", "error"),
                           ("evidence_revision", 124), ("evidence_text_sha256", "changed"),
                           ("retrieval_model", "other"), ("retrieval_revision", "changed"),
                           ("tokenizer_model", "other"), ("tokenizer_revision", "changed"), ("query", "Alice")):
            with self.subTest(key=key):
                records = deepcopy(self.records)
                records[0][key] = value
                with self.assertRaises(ValueError):
                    self.jobs(records)
        for records in ([], self.records * 2, [None], ["not an object"]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.jobs(records)

    def test_remote_passage_integrity_sorting_and_short_page_indices(self):
        for key, value in (("title", "Bob"), ("text_sha256", "changed"), ("passage_index", True),
                           ("passage_index", -1), ("passage_index", 3), ("score", True),
                           ("score", "0.5"), ("score", float("nan")), ("score", float("inf"))):
            with self.subTest(key=key, value=value):
                records = deepcopy(self.records)
                records[0]["passages"][0][key] = value
                with self.assertRaises(ValueError):
                    self.jobs(records)
        for passages in ([], self.records[0]["passages"] * 2, self.records[0]["passages"] * 6, [None]):
            records = deepcopy(self.records)
            records[0]["passages"] = passages
            with self.subTest(passages=passages), self.assertRaises(ValueError):
                self.jobs(records)
        records = deepcopy(self.records)
        records[0]["passages"] *= 2
        records[0]["passages"][1] = {**records[0]["passages"][1], "passage_index": 1, "score": 0.9}
        with self.assertRaises(ValueError):
            self.jobs(records)

    def test_text_self_hash_cannot_hide_injection_or_removed_word_boundaries(self):
        for text in ("Alice is not a writer.", "Aliceisawriter.", "A lice is a writer."):
            records = deepcopy(self.records)
            records[0]["passages"][0].update(text=text, text_sha256=pilot.sha(text.encode()))
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.jobs(records)
        records = deepcopy(self.records)
        text = "<s> " + self.evidence.subjects["Alice"]["text"].replace(" ", "  \n").replace("caf\u00e9s", "cafe\u0301s") + " </s>"
        records[0]["passages"][0].update(text=text, text_sha256=pilot.sha(text.encode()))
        self.assertEqual(len(self.jobs(records)), 1)  # Whitespace/NFC only, never rewrite words.

    def test_top_five_full_page_retains_reverse_rank_order(self):
        records = deepcopy(self.records)
        source = self.evidence.subjects["Alice"]
        # Mock long controller chunks, not five arbitrary character/sentence spans.
        chunks = [sentence + " Biography detail." * 128 for sentence in (
            "Alice is a writer.", "Alice was born in 1990.", "Alice lives in Rome.", "Alice wrote books.", "Alice won an award.")]
        source["text"] = " ".join(chunks)
        source["text_sha256"] = pilot.sha(source["text"].encode())
        self.ready["facts"][0]["source_text_sha256"] = source["text_sha256"]
        records[0].update(source_text_sha256=source["text_sha256"], evidence_text_sha256=source["text_sha256"])
        records[0]["passages"] = [{"title": "Alice", "text": text,
            "passage_index": index, "score": 1 - index / 10,
            "text_sha256": pilot.sha(text.encode())} for index, text in enumerate(chunks)]
        prompt = self.jobs(records)[0]["payload"]["messages"][0]["content"]
        self.assertLess(prompt.index("Alice won an award"), prompt.index("Alice lives in Rome"))

    def test_score_only_lineage_change_reuses_support_journal_no_http_replay(self):
        jobs = self.jobs()
        store = scoring.JobStore(self.root, "remote-support", dispatch=True)
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response("True")) as http:
            first = store.evaluate(jobs)
            changed = deepcopy(self.records)
            changed[0]["passages"][0]["score"] = 0.7
            again_jobs = self.jobs(changed)
            again = store.evaluate(again_jobs)
        self.assertEqual(jobs[0]["cache_key"], again_jobs[0]["cache_key"])
        self.assertNotEqual(jobs[0]["lineage"]["retrieval_record_sha256"], again_jobs[0]["lineage"]["retrieval_record_sha256"])
        self.assertTrue(first[0]["ok"] and again[0]["ok"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(self.ledger.totals()["attempts_settled"], 1)

    def test_controller_jsonl_pipeline_with_receipt_and_resume(self):
        row, _, short = self.available[0]
        def http(request, token):
            prompt = request["messages"][0]["content"]
            label = json.dumps({"sentence": self.answer, "facts": [self.answer]}) if "Please breakdown" in prompt else "True"
            return response(label)
        with patch.object(scoring, "load_available", return_value=(self.available, {"input": "frozen"})), patch.object(scoring, "SourceCatalog", return_value=self.catalog), patch.object(scoring, "EvidenceCatalog", return_value=self.evidence), patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", side_effect=http) as http_mock:
            extracted = scoring.factscore_stage(self.root, "controller-staged", "extract", models=[short], dispatch=True)
            ready_result = scoring.factscore_stage(self.root, "controller-staged", "retrieval-ready", models=[short], extraction_path=extracted["snapshot"])
            self.ready = pilot.read_json(Path(ready_result["snapshot"]).read_bytes())
            self.records = self.outputs(self.ready)
            self.write_outputs()
            verified = scoring.factscore_stage(self.root, "controller-staged", "verify", models=[short], dispatch=True,
                extraction_path=extracted["snapshot"], retrieval_path=self.path)
            resumed = scoring.factscore_stage(self.root, "controller-staged", "verify", models=[short], dispatch=True,
                extraction_path=extracted["snapshot"], retrieval_path=self.path)
        self.assertEqual(http_mock.call_count, 2)
        self.assertEqual(verified["metrics"][short]["macro_precision"], 1)
        self.assertEqual(resumed["metrics"], verified["metrics"])


class ProvenanceTests(TempCase):
    def setUp(self):
        super().setUp()
        self.ledger = self.make_ledger()
        self.catalog = scoring.SourceCatalog(pilot.DEFAULT_ROOT)
        _, _, self.evidence, _ = FactScoreTests().retrieval_fixture()
        source = self.evidence.subjects["Alice"]
        text_path = self.root / source["text_file"]
        text_path.parent.mkdir(parents=True)
        text_path.write_bytes(source["text"].encode())
        record = {"topic": "Alice", "title": source["title"], "url": source["url"], "revision": source["revision"],
                  "files": [{"path": source["text_file"], "sha256": source["text_sha256"]}]}
        manifest = {"ready": True, "count": 180, "records": [record] +
                    [{"topic": f"fixture-subject-{i}", "revision": 1, "files": []} for i in range(179)]}
        raw = pilot.canonical(manifest)
        (self.root / "data/v2/factscore-evidence-manifest.json").write_bytes(raw)
        self.evidence.manifest_sha256 = pilot.sha(raw)
        import v2_compute_jobs
        pin = patch.object(v2_compute_jobs, "FROZEN_EVIDENCE_SHA256", self.evidence.manifest_sha256)
        pin.start()  # Synthetic 180-subject fixture pin; production constant/file is untouched.
        self.addCleanup(pin.stop)
        self.answer = "Alice is a writer."
        row = {"id": "bio", "dataset": "factscore_bios", "question": "Tell me about Alice.", "metadata": {"topic": "Alice"}}
        self.available = [(row, self.answer, "tiny")]
        jobs = scoring.extraction_jobs(row, self.answer, "tiny", self.catalog)
        store = scoring.JobStore(self.root, "provenance", dispatch=True)
        label = json.dumps({"sentence": self.answer, "facts": [self.answer]})
        with patch.object(pilot, "token", return_value="secret"), patch.object(pilot, "http_request", return_value=response(label)):
            outcomes = store.evaluate(jobs)
        self.packet = scoring.extraction_packet(self.available, jobs, outcomes, {"input": "fixture"}, root=self.root)
        scoring.immutable(store.directory / "snapshots/extract-fixture.json", self.packet)

    def export(self, packet=None):
        return scoring.retrieval_ready(self.packet if packet is None else packet, self.evidence, root=self.root)

    def validate(self, ready):
        from v2_compute_jobs import validate_factscore_payload
        # Supply this fixture's approved source/answer inputs; keep the controller's
        # snapshot/request/raw-journal rederivation and all provenance checks real.
        with patch.object(scoring, "load_available", return_value=(self.available, {"input": "fixture"})), patch.object(scoring, "SourceCatalog", return_value=self.catalog), patch.object(scoring, "JobStore", side_effect=AssertionError("Validator must be read-only")):
            validate_factscore_payload(self.root, ready)

    def test_export_satisfies_controller_independent_file_and_raw_journal_validator(self):
        ready = self.export()
        self.validate(ready)  # Validation only; never prepare/submit/run.
        fact = ready["facts"][0]
        self.assertFalse(Path(fact["extraction_journal"]).is_absolute())
        self.assertTrue(fact["extraction_journal"].startswith("scoring/v2-glm/"))
        journal = pilot.read_json((self.root / fact["extraction_journal"]).read_bytes())
        raw_body = base64.b64decode(journal["raw_body_base64"], validate=True)
        self.assertEqual(fact["extraction_response_sha256"], pilot.sha(raw_body))
        self.assertEqual(ready["provenance"]["evidence_manifest_sha256"], self.evidence.manifest_sha256)
        ref = ready["provenance"]["facts_file"]
        raw = (self.root / ref["path"]).read_bytes()
        self.assertEqual(ref["sha256"], pilot.sha(raw))
        self.assertEqual([json.loads(line) for line in raw.splitlines()], ready["facts"])
        self.assertEqual(fact["fact"], self.answer)

    def test_controller_requires_original_bound_extraction_snapshot(self):
        ready = self.export()
        snapshot = self.root / "scoring/v2-glm/results/provenance/snapshots/extract-fixture.json"
        packet = json.loads(snapshot.read_bytes())
        packet["responses"][0]["facts"][0]["text"] = "Changed snapshot fact."
        snapshot.write_bytes(pilot.canonical(packet))
        with self.assertRaisesRegex(ValueError, "no bound original GLM extraction snapshot"):
            self.validate(ready)

    def test_export_is_deterministic_immutable_and_never_rewrites_raw_receipt(self):
        ready = self.export()
        journal_path = self.root / ready["facts"][0]["extraction_journal"]
        journal_raw = journal_path.read_bytes()
        again = self.export()
        self.assertEqual(ready, again)
        self.assertEqual(journal_path.read_bytes(), journal_raw)
        export_path = self.root / ready["provenance"]["facts_file"]["path"]
        export_path.write_bytes(export_path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.export()

    def test_export_rejects_fact_and_receipt_hash_tampering_or_external_paths(self):
        for key, value in (("extraction_journal", "../outside.json"),
                           ("extraction_journal", "data/v2/non-glm.json"),
                           ("extraction_response_sha256", "changed"), ("extraction_journal_sha256", "changed"),
                           ("text", "Alice is not a writer."), ("fact_index", 1)):
            with self.subTest(key=key):
                packet = deepcopy(self.packet)
                packet["responses"][0]["facts"][0][key] = value
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.export(packet)

    def test_controller_detects_changed_source_file_or_extraction_body(self):
        ready = self.export()
        export_path = self.root / ready["provenance"]["facts_file"]["path"]
        original = export_path.read_bytes()
        export_path.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            self.validate(ready)
        export_path.write_bytes(original)
        journal_path = self.root / ready["facts"][0]["extraction_journal"]
        journal = json.loads(journal_path.read_bytes())
        body = json.loads(base64.b64decode(journal["raw_body_base64"]))
        body["choices"][0]["message"]["content"] = json.dumps({"facts": ["not the exported fact"]})
        body_raw = pilot.canonical(body)
        journal.update(raw_body_base64=base64.b64encode(body_raw).decode(), raw_body_sha256=pilot.sha(body_raw))
        journal_path.write_bytes(pilot.canonical(journal))
        with self.assertRaises(ValueError):
            self.validate(ready)

    def test_no_fact_extraction_exports_empty_jsonl_without_padding_or_judgments(self):
        packet = deepcopy(self.packet)
        packet["responses"][0]["facts"] = []
        ready = self.export(packet)
        self.assertEqual(ready["facts"], [])
        self.assertEqual(ready["evidence"], [])
        ref = ready["provenance"]["facts_file"]
        self.assertEqual((self.root / ref["path"]).read_bytes(), b"")
        self.assertEqual(ref["sha256"], pilot.sha(b""))


if __name__ == "__main__":
    unittest.main()
