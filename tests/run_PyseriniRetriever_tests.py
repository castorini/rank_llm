import sys

# sys.path.append('../rank_llm')

# from rank_llm.src.rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod

from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod
from rank_llm.retrieve.indices_dict import INDICES

valid_inputs = [
    ("dl19", RetrievalMethod.BM25),
    ("dl19", RetrievalMethod.BM25_RM3),
    ("dl20", RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL),
    ("dl20", RetrievalMethod.D_BERT_KD_TASB),
    ("dl20", RetrievalMethod.OPEN_AI_ADA2),
    # ("dl22", RetrievalMethod.REP_LLAMA)
]

failure_inputs = [
    ("dl23", RetrievalMethod.BM25),  # dataset error
    ("dl23", RetrievalMethod.BM25_RM3),  # dataset error
    ("dl18", RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL),  # dataset error
    ("dl19", RetrievalMethod.UNSPECIFIED),  # retrieval method error
    ("dl16", RetrievalMethod.UNSPECIFIED),  # dataset and retrieval method error
    ("dl21", RetrievalMethod.D_BERT_KD_TASB),
    ("covid", RetrievalMethod.OPEN_AI_ADA2),
]

# class RetrievalMethod(Enum):
#     UNSPECIFIED = "unspecified"
#     BM25 = "bm25"
#     BM25_RM3 = "bm25_rm3"
#     SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++_EnsembleDistil_ONNX"
#     D_BERT_KD_TASB = "distilbert_tas_b"
#     OPEN_AI_ADA2 = "openai-ada2"
#     REP_LLAMA = "rep-llama"


def run_valid_input_tests(input_pairs):
    for dataset, retrieval_method in input_pairs:
        retriever = PyseriniRetriever(dataset, retrieval_method)
        assert retriever._dataset == dataset
        assert retriever._retrieval_method == retrieval_method
        assert retriever._searcher is not None
        key = retrieval_method.value
        if key == "bm25_rm3":
            key = "bm25"
        assert retriever._get_index() == INDICES[key][dataset]

    print("Valid inputs tests passed")


def run_failure_input_tests(input_pairs):
    count = 0
    for dataset, retrieval_method in input_pairs:
        try:
            retriever = PyseriniRetriever(dataset, retrieval_method)
        except:
            print("Exception raised correctly")
            count += 1

    print(f"{count}/{len(input_pairs)} exceptions raised correctly")


def run_retrieve_tests(inputs):
    pass


if __name__ == "__main__":
    run_valid_input_tests(valid_inputs)
    run_failure_input_tests(failure_inputs)

    # inputs = [("dl19", RetrievalMethod.BM25)]

    # rub
