# RankLLM API: CLI, REST, and MCP

## Shared reranking operations

All three interfaces expose the same two operations, with shared options, input
normalization, and validation:

- **Rerank** takes a query and supplied candidates, then returns ranked results.
- **Retrieve and rerank** retrieves candidates from a dataset or Pyserini HTTP
  service before reranking them. It also accepts a request file containing
  queries and candidates, in which case retrieval is skipped.

| Operation | CLI | REST | MCP |
| --- | --- | --- | --- |
| Rerank | `rank-llm rerank --input-json ...` or `--stdin` | `POST /v1/rerank` | `rerank` |
| Retrieve and rerank | `rank-llm rerank --dataset ...` or `--requests-file ...` | `POST /v1/retrieve-and-rerank` | `retrieve_and_rerank` |

For direct reranking, CLI and REST inputs contain `query` and `candidates`.
A query can be a string or an object such as `{"text":"cats","qid":"q1"}`.
MCP takes `query_text` and `query_id` separately. Candidates can be strings,
objects with `text`, or objects with `doc` containing text or a document object.
Missing candidate IDs and scores default to the one-based position and `0.0`.

Retrieve-and-rerank inputs select one of these sources:

- **Dataset:** supply `dataset` and an explicit `retrieval_method`. Omit `query`
  to process dataset topics, or supply it to search for one query.
- **Pyserini HTTP service:** also supply `retriever_host`, a nonempty `query`,
  and `retrieval_method="bm25"`. The dataset identifies the index. Omitting
  `retriever_host` selects local retrieval.
- **Request file:** supply `requests_file`, without a dataset, retrieval method,
  query, or retriever host. Files contain a JSON array (`.json`) or one request
  per line (`.jsonl`), using the direct query/candidate format above.

Common options include `model_path`, `top_k_rerank`, `num_passes`, and
`prompt_template_path`. Retrieval options include `top_k_candidates` (default
100), `max_queries`, and `query_id`. A `top_k_rerank` of `-1` returns all
candidates. The CLI uses hyphenated flags, REST puts model/reranking options in
`overrides` and retrieval fields at the top level, and MCP takes flat arguments.
Backend selectors cannot be combined; pointwise vLLM and remote listwise vLLM
require `base_url`.

Model-backed dataset runs write pipeline artifacts to computed paths unless
`output_jsonl_file`, `output_trec_file`, or `invocations_history_file` overrides
them. Request-file runs write explicitly requested outputs; recording their
invocation history requires both `populate_invocations_history` and
`invocations_history_file`. Optional `qrels_file` controls pipeline evaluation;
single supplied queries skip dataset-topic evaluation.


### Installation

The base package (`pip install -e .`) includes the CLI and Python library.
Neither requires server extras. The `api` extra is for the REST server, `mcp`
is for the MCP server (including HTTP transport), and `server` installs both.

Install `.[vllm]` for the RankZephyr examples and run them on a machine with a
compatible GPU. Other models may require a different inference extra, such as
`.[openai]` for hosted OpenAI models.
Local dataset retrieval requires `.[pyserini]` and JDK 21. Server-side file paths
refer to the server's filesystem.

## CLI

Use `rank-llm COMMAND --help` for the full argument list. Add `--output json` for a
`castorini.cli.v1` response envelope; reranking results are in
`artifacts[0].value`, and diagnostics go to stderr.

In addition to the shared reranking operations, the CLI provides validation,
evaluation, analysis, cache generation, and inspection. These additional
RankLLM operations are currently CLI-only; REST/MCP support is planned for a
follow-up PR.

### Rerank

Rerank supplied candidates with `--input-json`, or read the same JSON from
standard input with `--stdin`:

```bash
rank-llm --output json rerank --model-path castorini/rank_zephyr_7b_v1_full \
  --input-json '{"query":{"text":"cats","qid":"q1"},"candidates":["first passage",{"text":"second passage"}]}' \
  --top-k-rerank 1
```

Retrieve and rerank dataset topics:

```bash
rank-llm rerank --model-path castorini/rank_zephyr_7b_v1_full --dataset dl19 \
  --retrieval-method bm25 --top-k-candidates 100 --max-queries 2
```

Retrieve from a Pyserini HTTP service and rerank one query:

```bash
rank-llm rerank --model-path castorini/rank_zephyr_7b_v1_full --dataset msmarco-v2.1-doc \
  --query cats --query-id q1 --retrieval-method bm25 \
  --retriever-host http://localhost:8081 --top-k-candidates 20 --top-k-rerank 10
```

Rerank a request file and save results:

```bash
rank-llm rerank --model-path castorini/rank_zephyr_7b_v1_full \
  --requests-file /data/requests.jsonl --output-jsonl-file /data/ranked.jsonl \
  --output-trec-file /data/ranked.trec
```

### Validate and dry-run

`validate rerank` checks a direct payload or request file without inference.
`rerank --validate-only` and `rerank --dry-run` also check the workflow options.
They do not load models or contact retrieval services.

```bash
rank-llm validate rerank --input-json '{"query":"cats","candidates":["first passage"]}'
rank-llm validate rerank --requests-file /data/requests.jsonl
rank-llm rerank --model-path castorini/rank_zephyr_7b_v1_full --dataset dl19 --retrieval-method bm25 --dry-run
```

### Evaluate

Aggregate trec_eval metrics across stored runs. The command matches filenames by
model name, context size, candidate count, and dataset under retrieval-method
subdirectories, then writes an aggregate JSONL report. Use the model name
embedded in those filenames.

```bash
rank-llm evaluate --model-name rank_zephyr_7b_v1_full \
  --context-size 4096 --rerank-results-dirname /data/rerank_results
```

### Analyze

Count response errors in stored invocation histories:

```bash
rank-llm analyze --files /data/invocations_history.json --verbose
```

### Generate a retrieval cache

`retrieve-cache` joins an existing TREC run with corpus and query sources to
produce a retrieval JSON cache. It can also write a truncated TREC run with
`--output-trec-file`.

```bash
rank-llm retrieve-cache --trec-file /data/run.trec \
  --collection-file /data/collection.tsv --query-file /data/queries.tsv \
  --output-file /data/retrieval_cache.json --topk 20
```

### Inspect prompts

List bundled templates, show a template and its placeholders, or render it with
sample input without running a model. `show` and `render` accept a bundled name
or a path to a custom YAML template.

```bash
rank-llm prompt list
rank-llm prompt show rank_zephyr_template
rank-llm prompt render rank_zephyr_template \
  --input-json '{"query":"cats","candidates":["first passage"]}'
```

### View artifacts

Summarize and sample request/ranking JSONL files, TREC runs, or invocation
histories:

```bash
rank-llm view /data/ranked.jsonl --records 2
```

### Inspect commands, schemas, and environment

`describe` returns command metadata, `schema` returns a named input/output
schema, and `doctor` reports environment readiness, optional dependencies, and
loaded configuration.

```bash
rank-llm describe rerank
rank-llm schema rerank-direct-input
rank-llm doctor
```

### Start servers

`serve` starts either interface. See the following sections for installation
and request examples.

```bash
rank-llm serve http --model-path castorini/rank_zephyr_7b_v1_full --port 8082
rank-llm serve mcp
```

## REST API

Install `api` for FastAPI and Uvicorn, together with `vllm` to run RankZephyr:

```bash
pip install -e '.[api,vllm]'
rank-llm serve http --model-path castorini/rank_zephyr_7b_v1_full --port 8082
```

The REST API exposes the two shared operations. Model/reranking options in
`overrides` replace the startup defaults for that request. Execution is
synchronous, and the server reuses models with matching construction settings.

### Rerank

```bash
curl -s http://localhost:8082/v1/rerank -H 'Content-Type: application/json' -d '{
  "query": {"text":"cats","qid":"q1"},
  "candidates": ["first passage", {"text":"second passage"}],
  "overrides": {"top_k_rerank":1}
}'
```

### Retrieve and rerank

Process dataset topics:

```bash
curl -s http://localhost:8082/v1/retrieve-and-rerank \
  -H 'Content-Type: application/json' -d '{
    "dataset":"dl19","retrieval_method":"bm25","top_k_candidates":100,"max_queries":2
  }'
```

For retrieval through a separate Pyserini HTTP service, supply `retriever_host`.
See the [Pyserini REST guide](https://github.com/castorini/pyserini/blob/master/docs/usage-rest.md)
for starting and configuring that service. This mode does not require Pyserini
or Java in the RankLLM REST process. RankLLM does not expose Pyserini's standalone
search and document routes.

```bash
curl -s http://localhost:8082/v1/retrieve-and-rerank \
  -H 'Content-Type: application/json' -d '{
    "dataset":"msmarco-v2.1-doc","query":"cats","query_id":"q1",
    "retrieval_method":"bm25","retriever_host":"http://localhost:8081",
    "top_k_candidates":20,"overrides":{"top_k_rerank":10}
  }'
```

To rerank a request file on the server and save outputs:

```bash
curl -s http://localhost:8082/v1/retrieve-and-rerank \
  -H 'Content-Type: application/json' -d '{
    "requests_file":"/data/requests.jsonl",
    "output_jsonl_file":"/data/ranked.jsonl","output_trec_file":"/data/ranked.trec",
    "overrides":{"model_path":"castorini/rank_zephyr_7b_v1_full"}
  }'
```

### Responses and service information

Responses use the same `castorini.cli.v1` envelope as CLI JSON output, with ranked
records in `artifacts[0].value`. Invalid requests return HTTP 400, upstream
failures return 502, and unexpected execution failures return 500. Errors are
included in the response envelope. There is no standalone validation or dry-run
endpoint.

`GET /healthz` reports process health. Interactive documentation is available at
`/docs` and `/redoc`, and the OpenAPI schema is at `/openapi.json`.

The former Flask server and its `GET /api/model/...` route have been removed.
Use `rank-llm serve http` and the POST requests above instead.

## MCP

Install the `mcp` extra; the separate REST `api` extra is not required, even for
MCP over HTTP. The current MCP extra includes FastMCP, Pyserini, and the OpenAI
and vLLM backends. JDK 21 is also required for Pyserini's tools.

```bash
pip install -e '.[mcp]'
rank-llm serve mcp
# For a client that launches the server over stdio:
rank-llm serve mcp --transport stdio
```

HTTP is the default transport, on port 8000. Connect your MCP client to
`http://localhost:8000/mcp`; use `--port` to change the port. For stdio, configure
the client to launch `rank-llm` with arguments
`["serve", "mcp", "--transport", "stdio"]`. MCP over HTTP is separate from the
REST API above.

The two RankLLM tools take flat arguments, including a required `model_path`.
They execute synchronously and return lists of ranked result records. Invalid
calls are reported as tool errors. There is no RankLLM validation or dry-run tool.

### Rerank

Call `rerank` with:

```json
{
  "model_path": "castorini/rank_zephyr_7b_v1_full",
  "query_text": "cats",
  "query_id": "q1",
  "candidates": ["first passage", {"text": "second passage"}],
  "top_k_rerank": 1
}
```

### Retrieve and rerank

Call `retrieve_and_rerank` for dataset topics:

```json
{
  "model_path": "castorini/rank_zephyr_7b_v1_full",
  "dataset": "dl19",
  "retrieval_method": "bm25",
  "top_k_candidates": 100,
  "max_queries": 2
}
```

For a query against a Pyserini HTTP service, use:

```json
{
  "model_path": "castorini/rank_zephyr_7b_v1_full",
  "dataset": "msmarco-v2.1-doc",
  "query": "cats",
  "query_id": "q1",
  "retrieval_method": "bm25",
  "retriever_host": "http://localhost:8081",
  "top_k_candidates": 20,
  "top_k_rerank": 10
}
```

For a request file on the server, use:

```json
{
  "model_path": "castorini/rank_zephyr_7b_v1_full",
  "requests_file": "/data/requests.jsonl",
  "output_jsonl_file": "/data/ranked.jsonl",
  "output_trec_file": "/data/ranked.trec"
}
```

The server also exposes Pyserini's search, document, index, fusion, and evaluation
tools. See the [Pyserini MCP guide](https://github.com/castorini/pyserini/blob/master/docs/usage-mcp.md)
for those operations and client setup examples. Pyserini's `eval_hits` evaluates
supplied hits for one query; it is separate from RankLLM's CLI `evaluate` command.
