# Retrieval and reranking through HTTP and MCP

The CLI, FastAPI HTTP server, and MCP tools share reranking options, input
normalization, and validation. HTTP and MCP execution is synchronous. Request
files, prompt templates, few-shot files, qrels, and output paths belong to the
server's filesystem; mount files into the server when clients run elsewhere.

## Installation and startup

```bash
pip install -e '.[api]'
rank-llm serve http --model-path rank_identity --port 8082
```

`rank_identity` preserves candidate order and is useful for checking connectivity.
Install the corresponding inference extra for another model, such as
`.[api,openai]` or `.[api,vllm]`. Local dataset retrieval also requires
`.[pyserini]` and JDK 21. Connecting to an existing Pyserini HTTP service does not
require Pyserini or Java in the RankLLM HTTP process.

```bash
pip install -e '.[mcp]'
rank-llm serve mcp --transport stdio
# Or: rank-llm serve mcp --transport http --port 8000
```

The MCP server continues to register Pyserini's own tools, so its existing `mcp`
extra and startup still require Pyserini. RankLLM's service retrieval operation
itself uses HTTP.

## Capabilities

| Workflow | CLI | HTTP | MCP |
| --- | --- | --- | --- |
| Supplied candidates | `rerank --input-json` / `--stdin` | `POST /v1/rerank` | `rerank` |
| Dataset topics or a supplied query | `rerank --dataset` | `POST /v1/retrieve-and-rerank` | `retrieve_and_rerank` |
| JSON/JSONL request file | `rerank --requests-file` | `POST /v1/retrieve-and-rerank` | `retrieve_and_rerank` |
| Pyserini HTTP service | `rerank --dataset --query --retriever-host` | `POST /v1/retrieve-and-rerank` | `retrieve_and_rerank` |
| Validation / dry run | Existing `--validate-only` / `--dry-run` | Not exposed | Not exposed |

HTTP returns the `castorini.cli.v1` envelope, also used by `rank-llm --output json`.
Ranked records are in `artifacts[0].value`. Existing MCP execution tools return
lists of result records.
`GET /healthz` reports server health. Interactive API documentation and the
generated schemas are available at `/docs` and `/openapi.json`.

## Direct reranking

```bash
rank-llm --output json rerank --model-path rank_identity \
  --input-json '{"query":{"text":"cats","qid":"q1"},"candidates":["first passage",{"text":"second passage"}]}' \
  --top-k-rerank 1

curl -s http://localhost:8082/v1/rerank -H 'Content-Type: application/json' -d '{
  "query": {"text":"cats","qid":"q1"},
  "candidates": ["first passage", {"text":"second passage"}],
  "overrides": {"top_k_rerank":1}
}'
```

Equivalent MCP call: `rerank` with arguments:

```json
{"model_path":"rank_identity","query_text":"cats","query_id":"q1","candidates":["first passage",{"text":"second passage"}],"top_k_rerank":1}
```

Queries may be strings or `{ "text": "...", "qid": "..." }` objects in HTTP/CLI
direct inputs. Candidates may be strings, objects with `text`, or objects with
`doc` containing text or a document object. Omitted candidate IDs and scores
default to the one-based position and `0.0`. Document objects retain their fields.

## Dataset retrieval

```bash
rank-llm rerank --model-path rank_identity --dataset dl19 \
  --retrieval-method bm25 --top-k-candidates 100 --max-queries 2

curl -s http://localhost:8082/v1/retrieve-and-rerank \
  -H 'Content-Type: application/json' -d '{
    "dataset":"dl19","retrieval_method":"bm25","top_k_candidates":100,"max_queries":2
  }'
```

Equivalent MCP call: `retrieve_and_rerank` with arguments:

```json
{"model_path":"rank_identity","dataset":"dl19","retrieval_method":"bm25","top_k_candidates":100,"max_queries":2}
```

Omit `query` to process dataset topics, or supply it to search for one query.
`query_id` defaults to `1` for supplied retrieval queries. Dataset retrieval
requires an explicit `retrieval_method`.

## Request files and output artifacts

Request files contain an array of requests (`.json`) or one request per line
(`.jsonl`), using the same query/candidate forms as direct input. Choose exactly
one input source. Do not combine request files with a dataset, retrieval method,
query, or retriever host.

```bash
rank-llm rerank --model-path castorini/rank_zephyr_7b_v1_full \
  --requests-file /data/requests.jsonl --output-jsonl-file /data/ranked.jsonl \
  --output-trec-file /data/ranked.trec

curl -s http://localhost:8082/v1/retrieve-and-rerank \
  -H 'Content-Type: application/json' -d '{
    "requests_file":"/data/requests.jsonl",
    "output_jsonl_file":"/data/ranked.jsonl","output_trec_file":"/data/ranked.trec",
    "overrides":{"model_path":"castorini/rank_zephyr_7b_v1_full"}
  }'
```

Equivalent MCP call: `retrieve_and_rerank` with arguments:

```json
{"model_path":"castorini/rank_zephyr_7b_v1_full","requests_file":"/data/requests.jsonl","output_jsonl_file":"/data/ranked.jsonl","output_trec_file":"/data/ranked.trec"}
```

Existing pipeline artifact behavior is preserved: model-backed dataset runs
use computed output paths unless overridden; request-file runs write requested
outputs. Identity/random baselines return results without writing pipeline
artifacts. To record request-file invocations, supply `invocations_history_file`
and enable `populate_invocations_history`. Optional `qrels_file` retains the
pipeline's evaluation behavior; single supplied queries skip dataset-topic
evaluation.

## Pyserini HTTP service

```bash
rank-llm rerank --model-path rank_identity --dataset msmarco-v2.1-doc \
  --query 'cats' --query-id q1 --retrieval-method bm25 \
  --retriever-host http://localhost:8081 --top-k-candidates 20 --top-k-rerank 10

curl -s http://localhost:8082/v1/retrieve-and-rerank \
  -H 'Content-Type: application/json' -d '{
    "dataset":"msmarco-v2.1-doc","query":"cats","query_id":"q1",
    "retrieval_method":"bm25","retriever_host":"http://localhost:8081",
    "top_k_candidates":20,"overrides":{"top_k_rerank":10}
  }'
```

Equivalent MCP call: `retrieve_and_rerank` with arguments:

```json
{"model_path":"rank_identity","dataset":"msmarco-v2.1-doc","query":"cats","query_id":"q1","retrieval_method":"bm25","retriever_host":"http://localhost:8081","top_k_candidates":20,"top_k_rerank":10}
```

An explicit HTTP(S) `retriever_host` selects service retrieval. Omission selects
local retrieval; it does not silently fall back to an HTTP service. Service mode
requires a dataset/index, nonempty query, and BM25. The existing `ServiceRetriever`
calls Pyserini's `/v1/{index}/search` endpoint and preserves the supplied query ID.

## Options, validation, and errors

HTTP inference settings go in `overrides`; they override the defaults supplied
at server startup. Retrieval source/workflow fields stay at the top level.
MCP execution tools accept these same settings as flat arguments. All interfaces
support LiteLLM, pointwise vLLM, remote listwise vLLM, reasoning effort, and passage
word limits. Backend selectors cannot be combined. Pointwise vLLM and remote
listwise vLLM require `base_url`.

HTTP caches models by their construction settings; changing pass counts, output
limits, or invocation-history flags reuses the model. CLI JSON mode sends
pipeline diagnostics to stderr so stdout remains a single JSON response.

All execution requests are validated before retrieval or model initialization.
HTTP invalid inputs return 400, upstream failures return 502, and unexpected
execution failures return 500, with details in the common error envelope.
MCP reports invalid execution calls as tool errors.

REST and MCP expose retrieval/reranking execution only. They do not offer a
validation tool or accept validation/dry-run request flags. The existing CLI
validation, dry-run, evaluation, analysis, and inspection commands remain
CLI-only; their exposure through other interfaces is outside this change.

## Package layout

The three interfaces live under `rank_llm.api`:

```text
api/
  cli/               # Command parsing and terminal presentation
  rest/              # HTTP request parsing, FastAPI routes, and model cache
  mcp/               # MCP tools and server
  operations.py      # Shared execution accepting option objects
  options.py         # Shared option definitions, argument conversion, and validation
  introspection.py   # Published schemas and inspection helpers
```

Serialization, responses, and error classification also live in `api/` and are
shared by the interfaces. Request-file reading and candidate normalization live
in `rank_llm.data`, so core retrieval does not depend on an interface package.
Importing `rank_llm.api` or using the CLI does not load FastAPI or FastMCP.

The `rank-llm` command is unchanged. Python imports move from `rank_llm.cli` to
`rank_llm.api.cli`, from `rank_llm.api.app`/`runtime` to
`rank_llm.api.rest.app`/`runtime`, and from `rank_llm.server.mcp` to
`rank_llm.api.mcp`. The MCP module entrypoint is now
`python -m rank_llm.api.mcp`; `rank-llm serve mcp` remains available.

## Breaking migration from Flask

The Flask server, `rank_llm.server.flask.api` entrypoint, and
`GET /api/model/{model}/index/{index}/{retriever_port}` route have been removed.
Start `rank-llm serve http` and migrate clients to the POST examples above.

| Old Flask field | New retrieval POST field |
| --- | --- |
| Model URL segment | `overrides.model_path` or server `--model-path` |
| Index URL segment | `dataset` |
| Retriever port URL segment | `retriever_host`, e.g. `http://localhost:8081` |
| `query` | `query` |
| `qid` | `query_id` |
| `hits_retriever` | `top_k_candidates` |
| `hits_reranker` | `overrides.top_k_rerank` |
| `num_passes` | `overrides.num_passes` |
| `retrieval_method` | `retrieval_method` (explicit `bm25` for service mode) |

Use canonical model identifiers such as `castorini/rank_zephyr_7b_v1_full`,
`castorini/rank_vicuna_7b_v1`, `castorini/first_mistral`, or `rank_identity`.
Supply the desired model settings explicitly instead of relying on Flask presets.
To preserve Flask's former hit limits, explicitly request 20 candidates and 10
results; the shared defaults are 100 candidates and `top_k_rerank=-1` (return all
retrieved candidates). HTTP clients must read result records from
`artifacts[0].value` instead of treating the response itself as one record.
