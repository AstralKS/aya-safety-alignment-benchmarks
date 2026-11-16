"""Fail-closed shared-ledger transactions for the initial GLM pilot."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


CEILING = Decimal("18")
INPUT_BOUND = 1_048_576
OUTPUT_BOUND = 2048
INPUT_RATE = Decimal("0.15") / 1_000_000
OUTPUT_RATE = Decimal("0.50") / 1_000_000
ATTEMPT_BOUND = INPUT_BOUND * INPUT_RATE + OUTPUT_BOUND * OUTPUT_RATE
LEGACY = {
    "reserved_upper_usd": Decimal("5.877233977999998"),
    "reported_cost_usd": Decimal("2.303337629999998"),
    "requests": 8049,
    "settled_requests": 7446,
    "ceiling_usd": CEILING,
}


class BudgetError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BudgetError(message)


def money(value):
    require(not isinstance(value, bool), "Boolean money")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BudgetError("Invalid money") from error
    require(result.is_finite() and result >= 0, "Nonfinite/negative money")
    return result


def json_bytes(value):
    # Preserve unknown legacy keys, including Decimal values read from JSON.
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False, default=lambda x: str(x) if isinstance(x, Decimal) else _bad(x))
            + "\n").encode("utf-8")


def _bad(value):
    raise TypeError(type(value).__name__)


def write_once(path, content):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_replace(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        write_once(temporary, content)
        os.replace(temporary, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def file_lock(path, timeout=10.0):
    """Exclusive filesystem lock; never remove someone else's/stale lock."""
    lock = Path(str(path) + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            # Atomic directory creation avoids Windows delete-pending file-handle
            # races under threaded/process contention. An existing lock is never
            # removed by a contender, regardless of its age or owner.
            lock.mkdir()
            break
        except (FileExistsError, PermissionError):
            if time.monotonic() >= deadline:
                raise BudgetError(f"Lock busy/stuck (manual investigation required): {lock}")
            time.sleep(0.01)
    try:
        yield
    finally:
        lock.rmdir()


def integer(value, label):
    require(type(value) is int and value >= 0, f"Invalid {label}")
    return value


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate ledger/journal JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise BudgetError("Nonfinite ledger/journal JSON")
    return json.loads(raw, parse_float=Decimal, parse_constant=invalid, object_pairs_hook=pairs)


def usage_price(usage, nonempty):
    require(isinstance(usage, dict), "Missing/invalid usage")
    prompt = integer(usage.get("prompt_tokens"), "prompt_tokens")
    completion = integer(usage.get("completion_tokens"), "completion_tokens")
    require(not nonempty or completion > 0, "Nonempty response with zero completion_tokens")
    require(integer(usage.get("total_tokens"), "total_tokens") == prompt + completion,
            "Inconsistent total_tokens")
    for name, bound in (("prompt_tokens_details", prompt), ("completion_tokens_details", completion)):
        if name in usage:
            details = usage[name]
            if details is None:
                continue  # Optional provider detail object; never a required total.
            require(isinstance(details, dict), f"Invalid {name}")
            for key, value in details.items():
                if value is None:
                    continue  # Unknown optional breakdown/cache count is unpriced.
                require(integer(value, f"{name}.{key}") <= bound, f"{name}.{key} exceeds token total")
    priced = prompt * INPUT_RATE + completion * OUTPUT_RATE
    provider = money(usage["estimated_cost"]) if "estimated_cost" in usage else None
    charged = max(priced, provider or Decimal(0))
    over = prompt > INPUT_BOUND or completion > OUTPUT_BOUND or charged > ATTEMPT_BOUND
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "token_priced_usd": str(priced),
            "provider_estimated_usd": None if provider is None else str(provider),
            "accounted_usd": str(charged), "bound_failure": over}


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    def _read(self):
        raw = self.path.read_bytes()
        value = read_json(raw)
        self._validate(value)
        return value, raw

    def _validate(self, value):
        require(isinstance(value, dict), "Ledger must be an object")
        for key in ("ceiling_usd", "reserved_upper_usd", "reported_cost_usd"):
            money(value.get(key))
        require(money(value["ceiling_usd"]) == CEILING, "Shared ceiling must be exactly $18")
        require(money(value["reserved_upper_usd"]) >= money(value["reported_cost_usd"]),
                "Inclusive reserve below legacy reported cost")
        for key in ("requests", "settled_requests"):
            integer(value.get(key), key)
        require(value["settled_requests"] <= value["requests"], "Invalid legacy counters")
        v2 = value.get("v2_glm")
        if v2 is None:
            for key, expected in LEGACY.items():
                actual = money(value.get(key)) if isinstance(expected, Decimal) else value.get(key)
                require(actual == expected, f"Unexpected legacy baseline: {key}")
            return
        require(isinstance(v2, dict) and v2.get("schema_version") == 1, "Invalid v2 ledger")
        baseline = v2.get("legacy_baseline", {})
        for key, expected in LEGACY.items():
            require((money(baseline.get(key)) if isinstance(expected, Decimal) else baseline.get(key)) == expected,
                    f"Altered legacy baseline: {key}")
        attempts = v2.get("attempts")
        require(isinstance(attempts, dict), "Invalid durable attempts")
        accounted, settled = Decimal(0), 0
        for attempt_id, attempt in attempts.items():
            require(isinstance(attempt_id, str) and attempt_id and isinstance(attempt, dict), "Invalid attempt")
            require(attempt.get("reserve_usd") == str(ATTEMPT_BOUND), "Altered attempt reservation")
            require(attempt.get("state") in ("reserved", "settled"), "Invalid attempt state")
            if attempt["state"] == "settled":
                accounted += money(attempt.get("accounted_usd"))
                settled += 1
                priced = money(attempt.get("token_priced_usd"))
                provider = attempt.get("provider_estimated_usd")
                require(money(attempt["accounted_usd"]) == max(priced, money(provider) if provider is not None else Decimal(0)),
                        "Invalid settlement arithmetic")
                expected = usage_price({
                    "prompt_tokens": attempt.get("prompt_tokens"), "completion_tokens": attempt.get("completion_tokens"),
                    "total_tokens": integer(attempt.get("prompt_tokens"), "prompt_tokens")
                                    + integer(attempt.get("completion_tokens"), "completion_tokens"),
                    **({"estimated_cost": provider} if provider is not None else {}),
                }, False)
                require(all(attempt.get(key) == val for key, val in expected.items()), "Altered settlement token pricing/bounds")
            else:
                accounted += ATTEMPT_BOUND
        require(money(value["reserved_upper_usd"]) == LEGACY["reserved_upper_usd"] + accounted,
                "Inclusive reserve arithmetic mismatch")
        require(money(value["reported_cost_usd"]) == LEGACY["reported_cost_usd"], "Legacy reported cost changed")
        require(value["requests"] == LEGACY["requests"] + len(attempts), "Request counter mismatch")
        require(value["settled_requests"] == LEGACY["settled_requests"] + settled, "Settlement counter mismatch")

    def snapshot(self):
        with file_lock(self.path):
            value, _ = self._read()
            return value

    def start(self, directory, required=30 * ATTEMPT_BOUND):
        """Copy/hash the shared ledger before this run's first reservation."""
        with file_lock(self.path):
            value, raw = self._read()
            if required is not None:
                require(CEILING - money(value["reserved_upper_usd"]) >= required, "Insufficient shared pilot headroom")
            write_once(Path(directory) / "ledger-before.json", raw)
            write_once(Path(directory) / "ledger-before-sha256.json",
                       json_bytes({"sha256": hashlib.sha256(raw).hexdigest()}))

    def reserve(self, attempt_id, pilot_id, request_sha256, job_cache_key=None):
        with file_lock(self.path):
            value, raw = self._read()
            require(CEILING - money(value["reserved_upper_usd"]) >= ATTEMPT_BOUND, "Insufficient shared headroom")
            v2 = value.setdefault("v2_glm", {
                "schema_version": 1, "legacy_baseline": dict(LEGACY),
                "initial_ledger_sha256": hashlib.sha256(raw).hexdigest(), "attempts": {},
            })
            require(attempt_id not in v2["attempts"], "Duplicate attempt ID")
            v2["attempts"][attempt_id] = {
                "pilot_id": pilot_id, "request_sha256": request_sha256,
                "state": "reserved", "reserve_usd": str(ATTEMPT_BOUND),
            }
            if job_cache_key is not None:
                require(isinstance(job_cache_key, str) and len(job_cache_key) == 64, "Invalid job cache key")
                v2["attempts"][attempt_id]["job_cache_key"] = job_cache_key
            value["reserved_upper_usd"] = str(money(value["reserved_upper_usd"]) + ATTEMPT_BOUND)
            value["requests"] += 1
            self._validate(value)
            atomic_replace(self.path, json_bytes(value))

    def settle(self, attempt_id, accounting, journal_path):
        """Only release reserve after durable raw-response journaling; idempotent."""
        journal_path = Path(journal_path)
        raw = journal_path.read_bytes()
        journal = read_json(raw)
        require(journal.get("attempt_id") == attempt_id and journal.get("kind") == "http_response",
                "Settlement requires the matching raw response journal")
        require(journal.get("http_status") == 200, "Settlement requires HTTP 200")
        try:
            body = base64.b64decode(journal["raw_body_base64"], validate=True)
        except (ValueError, KeyError, TypeError) as error:
            raise BudgetError("Invalid raw response journal") from error
        require(hashlib.sha256(body).hexdigest() == journal.get("raw_body_sha256"), "Raw response journal hash mismatch")
        envelope = read_json(body)
        require(isinstance(envelope, dict) and envelope.get("model") in
                ("zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash:deepinfra"), "Untrusted journal model")
        choices = envelope.get("choices")
        require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), "Untrusted journal choices")
        choice = choices[0]
        require(type(choice.get("index")) is int and choice["index"] == 0, "Untrusted journal choice index")
        message = choice.get("message")
        require(isinstance(message, dict) and message.get("role") == "assistant", "Untrusted journal message")
        content = message.get("content")
        require(content is None or isinstance(content, str), "Untrusted journal final content")
        require(not message.get("tool_calls") and not message.get("function_call"), "Untrusted journal tool envelope")
        nonempty = bool(content) or any(bool(message.get(key)) for key in ("reasoning", "reasoning_content", "reasoning_details"))
        require(usage_price(envelope.get("usage"), nonempty) == accounting, "Settlement differs from raw journal usage")
        journal_sha = hashlib.sha256(raw).hexdigest()
        with file_lock(self.path):
            value, _ = self._read()
            attempt = value.get("v2_glm", {}).get("attempts", {}).get(attempt_id)
            require(attempt is not None, "Unreserved attempt")
            require(journal.get("request_sha256") == attempt["request_sha256"], "Settlement request hash mismatch")
            expected = {**accounting, "response_journal_sha256": journal_sha}
            if attempt["state"] == "settled":
                require(all(attempt.get(key) == val for key, val in expected.items()), "Conflicting double settlement")
                return
            amount = money(accounting.get("accounted_usd"))
            attempt.update(expected)
            attempt["state"] = "settled"
            value["reserved_upper_usd"] = str(money(value["reserved_upper_usd"]) - ATTEMPT_BOUND + amount)
            value["settled_requests"] += 1
            self._validate(value)
            atomic_replace(self.path, json_bytes(value))

    def totals(self, pilot_id=None):
        value = self.snapshot()
        attempts = [a for a in value.get("v2_glm", {}).get("attempts", {}).values()
                    if pilot_id is None or a["pilot_id"] == pilot_id]
        settled = [a for a in attempts if a["state"] == "settled"]
        return {
            "token_priced_estimates_usd": str(sum((money(a["token_priced_usd"]) for a in settled), Decimal(0))),
            "provider_estimates_usd": str(sum((money(a["provider_estimated_usd"]) for a in settled
                                               if a["provider_estimated_usd"] is not None), Decimal(0))),
            "provider_estimates_present": sum(a["provider_estimated_usd"] is not None for a in settled),
            "unknown_reserves_usd": str((len(attempts) - len(settled)) * ATTEMPT_BOUND),
            "attempts_reserved": len(attempts), "attempts_settled": len(settled),
            "shared_inclusive_reserved_upper_usd": str(money(value["reserved_upper_usd"])),
            "legacy_reported_usd": str(money(value["reported_cost_usd"])),
            "remaining_shared_headroom_usd": str(CEILING - money(value["reserved_upper_usd"])),
            "provider_estimates_are_not_invoices": True,
        }
