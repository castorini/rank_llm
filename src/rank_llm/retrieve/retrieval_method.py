from enum import Enum


class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    BM25_RM3 = "bm25_rm3"
    SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++_EnsembleDistil_ONNX"
    D_BERT_KD_TASB = "distilbert_tas_b"
    OPEN_AI_ADA2 = "openai-ada2"
    REP_LLAMA = "rep-llama"
    CUSTOM_INDEX = "custom_index"

    def __str__(self) -> str:
        return self.value
