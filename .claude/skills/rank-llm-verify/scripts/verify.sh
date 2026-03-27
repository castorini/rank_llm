#!/usr/bin/env bash
# rank-llm-verify: validate RankLLM artifact files.
#
# Usage:
#   bash verify.sh <artifact-path> [artifact-type]
#
# artifact-type: request-input | rerank-output | invocations-history | trec-output

set -euo pipefail

ARTIFACT_PATH="${1:?Usage: verify.sh <artifact-path> [artifact-type]}"
ARTIFACT_TYPE="${2:-}"
FAILURES=0

if [[ -z "${NO_COLOR:-}" ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; NC=''
fi

pass() { printf '%b\n' "${GREEN}OK${NC} $1"; }
fail() { printf '%b\n' "${RED}FAIL${NC} $1"; FAILURES=$((FAILURES + 1)); }
warn() { printf '%b\n' "${YELLOW}WARN${NC} $1"; }

echo "=== File Integrity ==="

if [[ ! -f "$ARTIFACT_PATH" ]]; then
  fail "File not found: $ARTIFACT_PATH"
  exit 1
fi
pass "File exists"

LINE_COUNT=$(wc -l < "$ARTIFACT_PATH" | tr -d ' ')
BYTE_COUNT=$(wc -c < "$ARTIFACT_PATH" | tr -d ' ')
if [[ "$BYTE_COUNT" -eq 0 ]]; then
  fail "File is empty"
  exit 1
fi
pass "File is non-empty"

if [[ -z "$ARTIFACT_TYPE" ]]; then
  DETECTED_TYPE=$(rank-llm view "$ARTIFACT_PATH" --output json 2>/dev/null | python3 -c '
import json
import sys

try:
    envelope = json.load(sys.stdin)
    artifacts = envelope.get("artifacts", [])
    if artifacts:
        print(artifacts[0].get("value", {}).get("artifact_type", ""))
except Exception:
    pass
')
  if [[ -n "$DETECTED_TYPE" ]]; then
    ARTIFACT_TYPE="$DETECTED_TYPE"
    pass "Auto-detected artifact type: $ARTIFACT_TYPE"
  else
    warn "Could not auto-detect artifact type"
  fi
fi

echo
echo "=== Content Validation (${ARTIFACT_TYPE:-unknown}) ==="

if ! python3 - "$ARTIFACT_PATH" "$ARTIFACT_TYPE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
artifact_type = sys.argv[2]
failures = 0


def fail(message: str) -> None:
    global failures
    print(f"FAIL {message}")
    failures += 1


def ok(message: str) -> None:
    print(f"OK {message}")


def load_jsonl(file_path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON on line {line_number}: {exc.msg}")
            continue
        if not isinstance(record, dict):
            fail(f"line {line_number} is not a JSON object")
            continue
        records.append(record)
    return records


if artifact_type in {"request-input", "rerank-output"}:
    records = load_jsonl(path)
    if not records:
        fail("no JSONL records loaded")
    else:
        ok(f"loaded {len(records)} JSONL records")
    qids: list[str] = []
    for index, record in enumerate(records, start=1):
        if "query" not in record or "candidates" not in record:
            fail(f"record {index} missing query or candidates")
            continue
        candidates = record["candidates"]
        if not isinstance(candidates, list) or not candidates:
            fail(f"record {index} has no candidates")
            continue
        query = record["query"]
        if isinstance(query, dict) and query.get("qid"):
            qids.append(str(query["qid"]))
        for candidate_index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                fail(f"record {index} candidate {candidate_index} is not an object")
                continue
            if artifact_type == "request-input":
                if "doc" not in candidate and "text" not in candidate:
                    fail(f"record {index} candidate {candidate_index} missing doc/text")
            else:
                required = {"docid", "score", "doc"}
                missing = sorted(required - candidate.keys())
                if missing:
                    fail(
                        f"record {index} candidate {candidate_index} missing {', '.join(missing)}"
                    )
    duplicate_count = len(qids) - len(set(qids))
    if duplicate_count:
        fail(f"found {duplicate_count} duplicate qid values")
    elif qids:
        ok(f"no duplicate qid values across {len(qids)} records")

elif artifact_type == "invocations-history":
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON: {exc.msg}")
        payload = []
    if not isinstance(payload, list):
        fail("top-level JSON value is not a list")
        payload = []
    else:
        ok(f"loaded {len(payload)} top-level records")
    for index, record in enumerate(payload, start=1):
        if not isinstance(record, dict):
            fail(f"record {index} is not an object")
            continue
        history = record.get("invocations_history")
        if not isinstance(history, list):
            fail(f"record {index} missing invocations_history list")
            continue
        for invocation_index, invocation in enumerate(history, start=1):
            if not isinstance(invocation, dict):
                fail(
                    f"record {index} invocation {invocation_index} is not an object"
                )

elif artifact_type == "trec-output":
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        fail("no non-empty TREC lines found")
    else:
        ok(f"loaded {len(lines)} TREC lines")
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 6:
            fail(f"line {line_number} does not have 6 columns")
            continue
        try:
            int(parts[3])
        except ValueError:
            fail(f"line {line_number} has a non-integer rank field")
        try:
            float(parts[4])
        except ValueError:
            fail(f"line {line_number} has a non-numeric score field")

else:
    fail("unknown artifact type; pass one explicitly")

sys.exit(1 if failures else 0)
PY
then
  FAILURES=$((FAILURES + 1))
fi

echo
echo "=== Summary ==="
if [[ "$FAILURES" -eq 0 ]]; then
  pass "All checks passed"
  exit 0
fi

fail "$FAILURES check groups failed"
exit 1
