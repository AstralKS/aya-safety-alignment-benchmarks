"""Offline contract tests using copies of the pinned, already-downloaded inputs.

Run: python -m unittest discover -s scripts -p test_prepare_v2_data.py -v
No source in the workspace is modified; fixtures live under the approved temp
directory. The real source shapes/duplicate anomalies are intentionally tested.
"""

from collections import Counter
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prepare_v2_data as preparation


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


class PreparationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        approved_temp = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\AstralK\AppData\Local")) / "Temp/opencode"
        approved_temp.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="prepare-v2-tests-", dir=approved_temp)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        paths = {"data/evaluation.jsonl", "data/manifest.json", *preparation.RECEIPTS}
        paths.add("scoring/v2-glm/preflight-input-integrity.json")
        for receipt_path in preparation.RECEIPTS:
            receipt = json.loads((preparation.DEFAULT_ROOT / receipt_path).read_bytes())
            paths.update(entry["path"] for entry in receipt["files"])
        paths.update(f"runs/{model}-baseline/checkpoint/responses.jsonl" for model in ("tiny", "8b"))
        for model in ("tiny", "8b"):
            paths.update((f"runs/{model}-baseline/launch.json", f"notebooks/{model}-baseline.ipynb",
                          f"runs/{model}-baseline/checkpoint/manifest.json",
                          f"runs/{model}-baseline/checkpoint/chat_template.txt"))
        for relative in paths:
            destination = cls.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(preparation.DEFAULT_ROOT / relative, destination)
        cls.input_hashes = {relative: sha((cls.root / relative).read_bytes()) for relative in paths}
        cls.manifest = preparation.assemble(cls.root)
        cls.expected = {name: (cls.root / "data/v2" / name).read_bytes() for name in preparation.OUTPUT_NAMES}

    @contextmanager
    def changed_file(self, relative, content):
        path = self.root / relative
        original = path.read_bytes()
        try:
            if content is None:
                path.unlink()
            else:
                path.write_bytes(content)
            yield path
        finally:
            path.write_bytes(original)

    @contextmanager
    def empty_destinations(self):
        """Remove only fixture outputs, restoring them even after a failed assertion."""
        with ExitStack() as stack:
            for name in preparation.OUTPUT_NAMES:
                stack.enter_context(self.changed_file(f"data/v2/{name}", None))
            yield

    def assert_clean_assembly_rejected(self, message):
        with self.assertRaisesRegex(preparation.PreparationError, message):
            preparation.assemble(self.root)
        self.assertTrue(all(not (self.root / "data/v2" / name).exists() for name in preparation.OUTPUT_NAMES))

    def test_matching_source_and_receipt_tamper_rejected_on_empty_destination(self):
        for receipt_path, source_path in (
            (preparation.RECEIPTS[0], preparation.SELFAWARE_PATH),
            (preparation.RECEIPTS[1], preparation.PARQUET_PATHS["mixed_entities"]),
        ):
            with self.subTest(receipt=receipt_path):
                original = (self.root / source_path).read_bytes()
                if source_path.endswith(".json"):
                    document = json.loads(original)
                    document["example"][0]["question"] += " Changed source."
                    altered = json.dumps(document).encode("utf-8")
                else:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    rows = preparation._read_parquet(original)
                    rows[0]["prompt"] += " Changed source."
                    buffer = io.BytesIO()
                    pq.write_table(pa.Table.from_pylist(rows), buffer)
                    altered = buffer.getvalue()
                receipt = json.loads((self.root / receipt_path).read_bytes())
                entry = next(entry for entry in receipt["files"] if entry["path"] == source_path)
                entry.update(sha256=sha(altered), bytes=len(altered))
                with self.empty_destinations(), self.changed_file(source_path, altered):
                    with self.changed_file(receipt_path, json.dumps(receipt).encode("utf-8")):
                        self.assert_clean_assembly_rejected("Source receipt hash mismatch")

    def test_matching_original_data_and_manifest_tamper_rejected_on_empty_destination(self):
        lines = (self.root / "data/evaluation.jsonl").read_bytes().splitlines(keepends=True)
        first = json.loads(lines[0])
        first["question"] += " Changed original question."
        altered = json.dumps(first).encode("utf-8") + b"\n" + b"".join(lines[1:])
        manifest = json.loads((self.root / "data/manifest.json").read_bytes())
        manifest["evaluation_sha256"] = sha(altered)
        with self.empty_destinations(), self.changed_file("data/evaluation.jsonl", altered):
            with self.changed_file("data/manifest.json", json.dumps(manifest).encode("utf-8")):
                self.assert_clean_assembly_rejected("Frozen input hash mismatch")

    def test_checkpoint_text_tamper_rejected_on_empty_destination(self):
        for model in ("tiny", "8b"):
            relative = f"runs/{model}-baseline/checkpoint/responses.jsonl"
            lines = (self.root / relative).read_bytes().splitlines(keepends=True)
            first = json.loads(lines[0])
            first["response"] += " Changed saved response."
            altered = json.dumps(first).encode("utf-8") + b"\n" + b"".join(lines[1:])
            with self.subTest(model=model), self.empty_destinations(), self.changed_file(relative, altered):
                self.assert_clean_assembly_rejected("Frozen input hash mismatch")

    def test_matching_preflight_record_and_checkpoint_tamper_is_rejected(self):
        relative = "runs/tiny-baseline/checkpoint/responses.jsonl"
        altered = (self.root / relative).read_bytes()[:-1] + b" \n"
        receipt_path = "scoring/v2-glm/preflight-input-integrity.json"
        receipt = json.loads((self.root / receipt_path).read_bytes())
        entry = next(entry for entry in receipt["files"] if entry["path"] == relative)
        entry.update(sha256=sha(altered), bytes=len(altered))
        with self.empty_destinations(), self.changed_file(relative, altered):
            with self.changed_file(receipt_path, json.dumps(receipt).encode("utf-8")):
                self.assert_clean_assembly_rejected("Preflight integrity receipt hash mismatch")

    def test_generation_contract_binds_saved_runtime_and_native_templates(self):
        self.assertIn("generation_contract", set(self.manifest))
        self.assertIn("input_integrity", set(self.manifest))
        integrity = self.manifest["input_integrity"]
        self.assertEqual(integrity["receipt_sha256"], "9bedb8765581c17e94d474fc2f1b355280548b181f30a0dc59af7bf699db1e51")
        self.assertEqual({entry["path"] for entry in integrity["files"]}, set(preparation.FROZEN_INPUT_SHA256))
        contract = self.manifest["generation_contract"]
        self.assertEqual(contract["decoding"]["do_sample"], False)
        self.assertEqual(contract["decoding"]["num_beams"], 1)
        self.assertEqual(contract["decoding"]["temperature_policy"], "not passed; retain model generation_config value, inactive during greedy decoding")
        self.assertIsNone(contract["decoding"]["temperature_override"])
        self.assertEqual(contract["prompting"]["new_system_prompt"], "")
        self.assertIs(contract["prompting"]["reflection"], False)
        self.assertEqual(contract["chat_template_policy"]["add_generation_prompt"], True)
        self.assertEqual(contract["chat_template_policy"]["tokenize"], False)
        self.assertEqual(contract["chat_template_policy"]["padding_side"], "left")
        self.assertEqual(contract["max_new_tokens"], {
            "hallulens_precisewikiqa": 256, "hallulens_nonexistentrefusal": 512,
            "selfaware": 256, "factscore_bios": 1536,
        })
        revisions = {"tiny": "af89d219b53ed9b13b8a4645f9c8028973510324",
                     "8b": "5062468bf9bc0c6035fd64e06274333ec127d980"}
        for model in ("tiny", "8b"):
            runtime = json.loads((self.root / f"runs/{model}-baseline/checkpoint/manifest.json").read_bytes())
            entry = contract["models"][model]
            self.assertEqual(entry["model_id"], runtime["model_id"])
            self.assertEqual(entry["revision"], revisions[model])
            self.assertEqual(entry["weight_dtype"], "float16")
            self.assertIsNone(entry["quantization"])
            self.assertEqual(entry["eos_token_id"], runtime["eos_token_id"])
            self.assertEqual(entry["pad_token_id"], runtime["pad_token_id"])
            self.assertEqual(entry["chat_template"]["sha256"], runtime["chat_template_sha256"])
            self.assertEqual(entry["chat_template"]["selection"], "default")
            checkpoint = f"runs/{model}-baseline/checkpoint/responses.jsonl"
            self.assertEqual(entry["saved_checkpoint"]["sha256"], sha((self.root / checkpoint).read_bytes()))
            for metadata in entry["runtime_metadata"]:
                self.assertEqual(metadata["sha256"], sha((self.root / metadata["path"]).read_bytes()))
            self.assertEqual(len(entry["runtime_metadata"]), 4)

    def test_receipt_anchors_are_checked_before_json_parsing(self):
        for relative, reader, message in (
            (preparation.RECEIPTS[0], preparation._verified_sources, "Source receipt hash mismatch"),
            (preparation.PREFLIGHT_PATH, preparation._verified_preflight, "Preflight integrity receipt hash mismatch"),
        ):
            with self.subTest(path=relative), self.changed_file(relative, b"not JSON"):
                with patch.object(preparation.json, "loads", side_effect=AssertionError("Parsed an untrusted receipt")):
                    with self.assertRaisesRegex(preparation.PreparationError, message):
                        reader(self.root)

    def test_each_frozen_input_is_anchored_on_empty_destination(self):
        for relative in preparation.FROZEN_INPUT_SHA256:
            with self.subTest(path=relative), self.empty_destinations():
                with self.changed_file(relative, (self.root / relative).read_bytes() + b" "):
                    self.assert_clean_assembly_rejected("Frozen input hash mismatch")

    def test_runtime_metadata_tamper_rejected_on_empty_destination(self):
        for model in ("tiny", "8b"):
            for relative in (f"runs/{model}-baseline/launch.json", f"notebooks/{model}-baseline.ipynb",
                             f"runs/{model}-baseline/checkpoint/manifest.json",
                             f"runs/{model}-baseline/checkpoint/chat_template.txt"):
                with self.subTest(path=relative), self.empty_destinations():
                    with self.changed_file(relative, (self.root / relative).read_bytes() + b" "):
                        self.assert_clean_assembly_rejected("Runtime metadata hash mismatch")

    def test_counts_strata_and_raw_preservation(self):
        rows = read_jsonl(self.root / "data/v2/evaluation.jsonl")
        self.assertEqual(len(rows), 9527)
        self.assertEqual(Counter(row["dataset"] for row in rows), {
            "simpleqa_verified": 1000, "abstention_bench": 978,
            "hallulens_precisewikiqa": 1000, "hallulens_nonexistentrefusal": 3000,
            "selfaware": 3369, "factscore_bios": 180,
        })
        precise = [row for row in rows if row["dataset"] == "hallulens_precisewikiqa"]
        self.assertEqual(Counter(row["metadata"]["h_score_cat"] for row in precise), {i: 100 for i in range(10)})
        mixed = [row for row in rows if row["dataset"] == "hallulens_nonexistentrefusal" and row["component"] == "Mixed"]
        generated = [row for row in rows if row["dataset"] == "hallulens_nonexistentrefusal" and row["component"] == "Generated"]
        self.assertEqual(Counter(row["metadata"]["type"] for row in mixed), {key: 500 for key in ("animal", "plant", "bacteria", "medicine")})
        self.assertEqual(len(generated), 1000)
        for dataset in preparation.NEW_COUNTS:
            subset = [row for row in rows if row["dataset"] == dataset]
            self.assertEqual(len(subset), len({row["question"] for row in subset}))
        selections = self.manifest["selection"]
        self.assertEqual(selections["precise_wiki"]["duplicate_exclusions"][0]["source_index"], 945)
        self.assertEqual(selections["precise_wiki"]["duplicate_exclusions"][0]["kept_source_index"], 349)
        self.assertEqual(selections["generated_entities"]["unique_question_count"], 1237)
        self.assertEqual(len(selections["generated_entities"]["duplicate_exclusions"]), 63)
        self.assertEqual(len(selections["generated_entities"]["allocations"]), 35)
        self.assertEqual(Counter(row["metadata"]["type_"] for row in generated), selections["generated_entities"]["allocations"])
        for row in precise + mixed + generated:
            self.assertEqual(row["question"], row["metadata"]["prompt"])
        for row in precise:
            self.assertEqual(row["reference_answers"], [row["metadata"]["answer"]])
            self.assertTrue(all(key in row["metadata"] for key in ("reference", "pageid", "revid", "title")))
        for relative, digest in self.input_hashes.items():
            self.assertEqual(sha((self.root / relative).read_bytes()), digest, relative)

    def test_original_rows_and_saved_response_hashes_are_exact(self):
        old_lines = (self.root / "data/evaluation.jsonl").read_bytes().splitlines(keepends=True)
        retained = [line for line in old_lines if json.loads(line)["dataset"] in ("simpleqa_verified", "abstention_bench")]
        self.assertEqual(self.expected["evaluation.jsonl"].splitlines(keepends=True)[:1978], retained)
        reuse = json.loads(self.expected["reuse.json"])
        for short, model in preparation.MODELS.items():
            path = self.root / f"runs/{short}-baseline/checkpoint/responses.jsonl"
            checkpoint = {row["id"]: row for row in read_jsonl(path)}
            actual = reuse["models"][short]
            self.assertEqual(actual["checkpoint"]["sha256"], sha(path.read_bytes()))
            self.assertEqual(len(actual["responses"]), 1978)
            self.assertEqual([row["id"] for row in actual["responses"]], [json.loads(line)["id"] for line in retained])
            for row in actual["responses"]:
                self.assertEqual(row["response_sha256"], sha(checkpoint[row["id"]]["response"].encode("utf-8")))
                self.assertEqual(row["model_id"], model)

    def test_generation_has_exact_allowlist_and_no_reused_rows(self):
        generation = read_jsonl(self.root / "data/v2/generation.jsonl")
        evaluation = {row["id"]: row for row in read_jsonl(self.root / "data/v2/evaluation.jsonl")}
        allowed = {"id", "dataset", "component", "question", "system_prompt", "max_new_tokens"}
        self.assertEqual(len(generation), 7549)
        self.assertEqual(len({row["id"] for row in generation}), 7549)
        self.assertFalse({"v2-factscore_bios-2", "v2-factscore_bios-127", "v2-factscore_bios-162"} & {row["id"] for row in generation})
        for row in generation:
            self.assertEqual(set(row), allowed)
            self.assertEqual(row, {key: evaluation[row["id"]][key] for key in allowed})
            self.assertTrue(row["id"].startswith("v2-"))
            self.assertEqual(row["system_prompt"], "")
            self.assertEqual(row["max_new_tokens"], {
                "hallulens_precisewikiqa": 256, "hallulens_nonexistentrefusal": 512,
                "selfaware": 256, "factscore_bios": 1536,
            }[row["dataset"]])

    def test_selfaware_and_biographies_preserve_source_contract(self):
        rows = read_jsonl(self.root / "data/v2/evaluation.jsonl")
        original = json.loads((self.root / preparation.SELFAWARE_PATH).read_bytes())["example"]
        selfaware = [row for row in rows if row["dataset"] == "selfaware"]
        for index, (row, raw) in enumerate(zip(selfaware, original, strict=True)):
            self.assertEqual(row["id"], f"v2-selfaware-{index}")
            self.assertEqual(row["question"], raw["question"])
            self.assertEqual(row["reference_answers"], raw["answer"])
            self.assertIs(row["should_abstain"], not raw["answerable"])
            self.assertEqual(row["metadata"], {**raw, "source_index": index})
        anomaly = next(row for row in selfaware if row["metadata"]["question_id"] == 3149)
        self.assertFalse(anomaly["should_abstain"])
        self.assertEqual(Counter(row["should_abstain"] for row in selfaware), {False: 2338, True: 1031})
        self.assertEqual(sum(row["reference_answers"] is None for row in selfaware), 1031)
        bios = [row for row in rows if row["dataset"] == "factscore_bios"]
        self.assertEqual(len({row["metadata"]["topic"] for row in bios}), 180)
        self.assertEqual([row["metadata"]["source_index"] for row in bios],
                         [i for i in range(183) if i not in {2, 127, 162}])
        self.assertEqual([row["id"] for row in bios],
                         [f"v2-factscore_bios-{i}" for i in range(183) if i not in {2, 127, 162}])
        for row in bios:
            self.assertEqual(row["question"], f"Question: Tell me a bio of {row['metadata']['topic']}.")
            self.assertIsNone(row["reference_answers"])
            self.assertIsNone(row["should_abstain"])
            self.assertEqual(set(row["metadata"]), {"topic", "source_index", "source_path", "zip_member"})

    def test_repeat_assemble_and_verify_are_read_only_and_identical(self):
        before = {name: (self.root / "data/v2" / name).stat().st_mtime_ns for name in preparation.OUTPUT_NAMES}
        actual_open = Path.open

        def read_only(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                raise AssertionError(f"Unexpected write during no-op: {path}")
            return actual_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", read_only):
            self.assertEqual(preparation.assemble(self.root), self.manifest)
            self.assertEqual(preparation.verify(self.root), self.manifest)
        for name, content in self.expected.items():
            path = self.root / "data/v2" / name
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, before[name])

    def test_source_tamper_including_unparsed_evidence_is_rejected(self):
        for relative in (preparation.PARQUET_PATHS["precise_wiki"], preparation.FACTSCORE_PATH,
                         "data/v2/raw/HalluLens/tasks/shortform/precise_wikiqa.py"):
            with self.subTest(source=relative):
                with self.changed_file(relative, (self.root / relative).read_bytes() + b"tampered"):
                    for action in (preparation.assemble, preparation.verify):
                        with self.assertRaisesRegex(preparation.PreparationError, "Source hash mismatch"):
                            action(self.root)

    def test_original_evaluation_hash_is_required(self):
        with self.changed_file("data/evaluation.jsonl", (self.root / "data/evaluation.jsonl").read_bytes() + b"\n"):
            with self.assertRaisesRegex(preparation.PreparationError, "Frozen input hash mismatch"):
                preparation.verify(self.root)

    def test_differing_last_destination_blocks_all_writes(self):
        with self.changed_file("data/v2/evaluation.jsonl", None) as missing:
            with self.changed_file("data/v2/manifest.json", b"user-owned destination\n") as different:
                with self.assertRaisesRegex(preparation.PreparationError, "refusing overwrite"):
                    preparation.assemble(self.root)
                self.assertFalse(missing.exists())
                self.assertEqual(different.read_bytes(), b"user-owned destination\n")

    def test_verify_rederives_every_output_including_manifest(self):
        for name in preparation.OUTPUT_NAMES:
            with self.subTest(output=name):
                with self.changed_file(f"data/v2/{name}", self.expected[name] + b" "):
                    with self.assertRaisesRegex(preparation.PreparationError, "Destination differs"):
                        preparation.verify(self.root)
        with self.changed_file("data/v2/generation.jsonl", None):
            with self.assertRaisesRegex(preparation.PreparationError, "Missing assembled destination"):
                preparation.verify(self.root)

    def test_checkpoint_metadata_and_response_changes_are_detected(self):
        # Exercise semantic validation after the public API's hash gate. Public
        # first-assembly tamper rejection is tested separately on empty outputs.
        frozen = {path: (self.root / path).read_bytes() for path in preparation.FROZEN_INPUT_SHA256}
        relative = "runs/tiny-baseline/checkpoint/responses.jsonl"
        original = frozen[relative]
        lines = original.splitlines(keepends=True)
        cases = (("status", "failed", "status is not ok"),
                 ("model_id", "wrong-model", "model mismatch"),
                 ("dataset", "selfaware", "dataset mismatch"),
                 ("response", None, "response is not a string"),
                 ("id", "unknown", "Unknown checkpoint ID"))
        for key, value, message in cases:
            with self.subTest(field=key, value=value):
                first = json.loads(lines[0])
                first[key] = value
                content = json.dumps(first).encode("utf-8") + b"\n" + b"".join(lines[1:])
                with self.assertRaisesRegex(preparation.PreparationError, message):
                    preparation._legacy({**frozen, relative: content})
        with self.assertRaisesRegex(preparation.PreparationError, "Duplicate checkpoint ID"):
            preparation._legacy({**frozen, relative: original + lines[0]})
        with self.assertRaisesRegex(preparation.PreparationError, "Missing reusable checkpoint IDs"):
            preparation._legacy({**frozen, relative: b"".join(lines[1:])})

    def test_receipt_pins_and_only_known_404_are_accepted(self):
        relative = preparation.RECEIPTS[1]
        accepted = self.manifest["source_receipts"][1]["accepted_download_errors"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["url"], preparation.KNOWN_404)
        for mutation in ("pin", "error"):
            receipt = json.loads((self.root / relative).read_bytes())
            if mutation == "pin":
                receipt["files"][0]["url"] = receipt["files"][0]["url"].replace(preparation.PINS["alexanderstern/hallulens"], "main")
            else:
                receipt["errors"].append({"url": "https://example.invalid/missing", "error": "404"})
            with self.subTest(mutation=mutation):
                with self.changed_file(relative, json.dumps(receipt).encode("utf-8")):
                    with self.assertRaisesRegex(preparation.PreparationError, "Source receipt hash mismatch"):
                        preparation.verify(self.root)

    def test_largest_remainder_lexical_ties_and_dedup_before_strata(self):
        self.assertEqual(preparation._largest_remainder({"z": 1, "b": 1, "a": 1}, 2), {"a": 1, "b": 1, "z": 0})
        self.assertEqual(preparation._largest_remainder({"a": 5, "b": 3, "c": 2}, 6), {"a": 3, "b": 2, "c": 1})
        rows = [{"prompt": "same", "type": "a"}, {"prompt": "other", "type": "b"},
                {"prompt": "same", "type": "b"}, {"prompt": "third", "type": "b"}]
        chosen, details = preparation._select(rows, "fixture", "type", {"a": 1, "b": 3}, target=2)
        self.assertEqual(details["unique_strata_counts"], {"a": 1, "b": 2})
        self.assertEqual(details["allocations"], {"a": 1, "b": 1})
        self.assertEqual(details["duplicate_exclusions"][0]["kept_source_index"], 0)
        self.assertEqual(len(chosen), 2)
        with self.assertRaisesRegex(preparation.PreparationError, "Insufficient unique rows"):
            preparation._select(rows, "fixture", "type", None, allocations={"b": 3})

    def test_hash_ranking_uses_seed_source_original_index_and_exact_prompt(self):
        rows = [{"prompt": f"Exact question {i}? ", "type": "a"} for i in range(12)]
        rows[5]["prompt"] = rows[0]["prompt"]
        chosen, _ = preparation._select(rows, "fixture", "type", {"a": 12}, allocations={"a": 5})
        expected_indices = sorted(
            (i for i in range(12) if i != 5),
            key=lambda i: sha(f"20261111:fixture:{i}:Exact question {i}? ".encode("utf-8")),
        )[:5]
        self.assertEqual([index for index, _ in chosen], expected_indices)

    def test_selfaware_duplicates_are_rejected_never_silently_removed(self):
        original = (self.root / preparation.SELFAWARE_PATH).read_bytes()
        for field, message in (("question", "Duplicate SelfAware question"), ("question_id", "duplicate SelfAware ID")):
            document = json.loads(original)
            document["example"][1][field] = document["example"][0][field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(preparation.PreparationError, message):
                    preparation._selfaware({preparation.SELFAWARE_PATH: json.dumps(document).encode("utf-8")})

    def test_manifest_output_hashes_and_overlap_report(self):
        # Preserve the reviewed 183-subject outputs as historic trust anchors.
        historic_payloads = {
            "evaluation.jsonl": "d0ae4eae4ed696a39461d8ebb28eac7f024cdf8d66f237819f67c65ab45e1679",
            "generation.jsonl": "8a04869e154fc9773ee90a5d53dc80199a8832cac9a81c54ec05cb3e0fc8d274",
            "reuse.json": "bc12dcf726e965533b391f708be4992f9633c226585459330971e2e5c8bc6574",
        }
        selected_payloads = {
            "evaluation.jsonl": "c69c16f295bd7f2e8ee62e7ab3f537efa87859fec04b01dac4d0bd03288354b3",
            "generation.jsonl": "96994ee2392780c17bfcf3709f33bffefd4ef9d931b5b177ede072eacd0c21bd",
            "reuse.json": "bc12dcf726e965533b391f708be4992f9633c226585459330971e2e5c8bc6574",
        }
        preserved = preparation.DEFAULT_ROOT / "data/v2/review-drafts/pre-exclusion-183-20261111/data/v2"
        self.assertEqual(sha((preserved / "manifest.json").read_bytes()),
                         "b91dfed5bbd94fef2ab37ae9c4327b26ffd28c4db4c10f945e610777e7c94bff")
        removed_ids = {"v2-factscore_bios-2", "v2-factscore_bios-127", "v2-factscore_bios-162"}
        for name, expected_hash in historic_payloads.items():
            historic = (preserved / name).read_bytes()
            self.assertEqual(sha(historic), expected_hash)
            if name.endswith(".jsonl"):
                # Byte-for-byte filtering proves nothing else changed in payloads.
                selected = b"".join(line for line in historic.splitlines(keepends=True)
                                    if json.loads(line)["id"] not in removed_ids)
                self.assertEqual(self.expected[name], selected)
            else:
                self.assertEqual(self.expected[name], historic)
        for name, entry in self.manifest["outputs"].items():
            self.assertEqual(entry["sha256"], sha(self.expected[name]))
            self.assertEqual(entry["bytes"], len(self.expected[name]))
            self.assertEqual(entry["sha256"], selected_payloads[name])
        selected = self.manifest["selection"]["factscore_bios"]
        self.assertEqual(selected["original_count"], 183)
        self.assertEqual(selected["selected_count"], 180)
        self.assertEqual([r["topic"] for r in selected["excluded_subjects"]], ["William Post", "Carlos Alfonso", "Jagan"])
        self.assertTrue(any("183" in text and "180" in text and "local exclusion" in text
                            for text in self.manifest["disclosures"]))
        inputs = [{"id": "a", "dataset": "x", "component": "one", "question": "exact"},
                  {"id": "b", "dataset": "y", "component": "two", "question": "exact"},
                  {"id": "c", "dataset": "z", "component": "three", "question": " exact"}]
        overlaps = preparation._overlaps(inputs)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual([item["id"] for item in overlaps[0]["items"]], ["a", "b"])
        self.assertEqual(len(inputs), 3)


if __name__ == "__main__":
    unittest.main()
