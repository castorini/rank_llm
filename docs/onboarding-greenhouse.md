# Greenhouse Reranking with Gaggle
## Setup

Clone the repo and configure a [Modal](https://modal.com/docs/guide/getting-started) profile for your own workspace:

```bash
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
uvx --from modal modal setup
uvx --from modal modal profile list
export MODAL_PROFILE=your-modal-profile
```

Replace `your-modal-profile` with the name shown by `profile list`. Run the commands below from the repo root. Modal builds the image in `scripts/modal_gaggle.py` with Python 3.12, Java 21, PyTorch 2.9.1, Transformers 5.16.1, and `rank_llm`.

## Run the rerankers

Start with a query and two passages:

```bash
cat > /tmp/gaggle-input.json <<'JSON'
{"query":"what are the symptoms of iron deficiency anemia?","candidates":["Iron deficiency anemia causes fatigue and pale skin.","Iron is a metal used in construction."]}
JSON

uvx --from modal modal run scripts/modal_gaggle.py --input-path /tmp/gaggle-input.json
```

This uses the soup checkpoint. To try the four individual runs, pass each model ID:

```bash
for model in \
  castorini/gaggle-reranker-run1-20260926 \
  castorini/gaggle-reranker-run2-20261004 \
  castorini/gaggle-reranker-run3-20261002 \
  castorini/gaggle-reranker-run4-20261005; do
  uvx --from modal modal run scripts/modal_gaggle.py \
    --input-path /tmp/gaggle-input.json --model-path "$model"
done
```

These were the scores on an A100:

| Model | Anemia passage | Metal passage |
|---|---:|---:|
| Soup | 22.976 | 14.265 |
| Run 1 | 21.429 | 12.668 |
| Run 2 | 21.634 | 12.868 |
| Run 3 | 23.933 | 15.350 |
| Run 4 | 24.724 | 16.184 |

These results agree with the results on the Hugging Face model card.

## Run the pretrained backbones

The backbone script just scores continuations.

```bash
cat > /tmp/gaggle-backbone-input.json <<'JSON'
{"prompt":"The capital of France is","continuations":[" Paris"," Berlin"]}
JSON

uvx --from modal modal run scripts/modal_gaggle_backbone.py \
  --input-path /tmp/gaggle-backbone-input.json
uvx --from modal modal run scripts/modal_gaggle_backbone.py \
  --input-path /tmp/gaggle-backbone-input.json \
  --model castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929
```

The first command uses the [causal backbone](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924); the second uses the [bidirectional backbone](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929). Both assign a higher log probability to ` Paris`:

| Backbone | ` Paris` | ` Berlin` |
|---|---:|---:|
| Causal | -0.99846 | -7.73540 |
| Bidirectional | -0.52638 | -12.86721 |

## Reproduce the reranking results

For an end-to-end run, retrieve 100 BM25 passages per query, rerank them with the soup checkpoint, and evaluate nDCG@10. Open a shell in the Modal image:

```bash
uvx --from modal modal shell scripts/modal_gaggle.py::rerank
```

Inside that shell, run:

```bash
python - <<'PY'
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.reranker import Reranker
from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever
from rank_llm.retrieve.retrieval_method import RetrievalMethod
from rank_llm.retrieve.topics_dict import TOPICS

model = Reranker(Reranker.create_model_coordinator(
    "castorini/gaggle-reranker-20261005", None, False,
    device="cuda", batch_size=16,
))
for dataset in ("dl19", "covid"):
    requests = PyseriniRetriever(dataset, RetrievalMethod.BM25).retrieve(k=100)
    results = model.rerank_batch(requests, rank_end=100)
    print(dataset, EvalFunction.from_results(results, TOPICS[dataset]))
PY
```

The reproduced scores agree with the [reported Greenhouse results](https://github.com/castorini/rank_llm/pull/463#issuecomment-6274605647):

| Collection | Queries | Reproduced nDCG@10 | Reported nDCG@10 |
|---|---:|---:|---:|
| DL19 | 43 | 0.7552 | 0.757 |
| TREC-COVID | 50 | 0.8563 | 0.859 |

+ Results reproduced by [@MankyDanky](https://github.com/MankyDanky) on 2026-10-19 using a Modal A100.
+ Results reproduced by [@tanvirsarao](https://github.com/tanvirsarao) on 2026-11-20 using a Modal A100.
