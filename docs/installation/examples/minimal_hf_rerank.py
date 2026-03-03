from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.monot5 import MonoT5


model = MonoT5(
    model="castorini/monot5-base-msmarco",
    device="cpu",  # switch to "cuda" if GPU is available
    batch_size=4,
)
reranker = Reranker(model)

request = Request(
    query=Query(text="what is bm25", qid="q1"),
    candidates=[
        Candidate(
            docid="d1",
            score=0.0,
            doc={"text": "BM25 is a bag-of-words ranking function used in search."},
        ),
        Candidate(
            docid="d2",
            score=0.0,
            doc={
                "text": "Neural reranking uses cross-encoders to score query-document relevance."
            },
        ),
        Candidate(
            docid="d3",
            score=0.0,
            doc={"text": "How to bake sourdough bread at home."},
        ),
    ],
)

result = reranker.rerank(request, rank_start=0, rank_end=3)
for i, candidate in enumerate(result.candidates, start=1):
    print(f"{i}. {candidate.docid} score={candidate.score:.6f}")
