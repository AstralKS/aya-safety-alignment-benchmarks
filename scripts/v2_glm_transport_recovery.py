"""Authorize exactly one replacement attempt for a transport-ambiguous GLM call.

A dispatched call whose transport failed leaves no usable response. The ledger keeps its
full reserve as unknown (conservative), and this receipt is the only way the scorer is
allowed to spend one replacement attempt for that cache key. The abandoned attempt id is
recorded so the authorization can never be reused for a second failure.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import v2_glm_budget as budget
import v2_glm_pilot as client

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_once(path, raw):
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite evidence: {path}") from error


def authorize(root, run_id, cache_key, attempt_id, reason):
    root = Path(root).resolve()
    results = root / "scoring/v2-glm/results" / run_id
    require(results.is_dir(), "Unknown scoring run")
    contract_path = root / "scoring/v2-glm/request-journals" / cache_key / "job.json"
    attempts = contract_path.parent / "attempts"
    require(contract_path.is_file() and attempts.is_dir(), "Unknown cache key")
    contract = client.read_json(contract_path.read_bytes())
    require(contract.get("cache_key") == cache_key, "Request contract belongs to another cache key")
    journal_path = attempts / (attempt_id + ".response.json")
    require(journal_path.is_file(), "Ambiguous attempt has no raw journal; refuse to authorize")
    journal = client.read_json(journal_path.read_bytes())
    require(journal.get("kind") == "transport_error" and journal.get("attempt_id") == attempt_id,
            "Only a transport-error journal can be replaced")
    require(journal.get("request_sha256") == contract["request_sha256"], "Journal request hash changed")
    ledger = budget.Ledger(root / "scoring/budget.json").snapshot()
    attempt = ledger.get("v2_glm", {}).get("attempts", {}).get(attempt_id)
    require(attempt is not None and attempt.get("state") == "reserved",
            "Ambiguous attempt is not an unsettled reservation")
    require(attempt.get("job_cache_key") == cache_key, "Attempt belongs to another cache key")
    require(attempt.get("request_sha256") == contract["request_sha256"], "Attempt request hash changed")
    receipt = {
        "kind": "glm-transport-replacement-v1",
        "run_id": run_id,
        "cache_key": cache_key,
        "request_sha256": contract["request_sha256"],
        "abandoned_attempt_ids": [attempt_id],
        "journal_path": journal_path.relative_to(root).as_posix(),
        "journal_sha256": client.sha(journal_path.read_bytes()),
        "reserve_retained_usd": attempt["reserve_usd"],
        "reason": reason,
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "max_replacement_attempts": 1,
    }
    path = results / "recovery" / f"transport-replacement-{cache_key[:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    for existing in path.parent.glob("transport-replacement-*.json"):
        if existing == path:
            continue
        other = client.read_json(existing.read_bytes())
        require(other.get("cache_key") != cache_key, "A replacement is already authorized for this cache key")
    write_once(path, json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return {"receipt": path.relative_to(root).as_posix(), **receipt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--reason", default="Transport ambiguity with no usable response")
    arguments = parser.parse_args()
    print(json.dumps(authorize(arguments.root, arguments.run_id, arguments.cache_key,
                               arguments.attempt_id, arguments.reason), indent=2))
