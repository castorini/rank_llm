# How to Use External Integrations of RankLLM

## [LangChain](https://github.com/langchain-ai/langchain)
**Install all the packages:**
```bash
pip install langchain-community faiss-gpu torch transformers sentence-transformers huggingface-hub rank_llm
```

**Install the document example:**
https://github.com/hwchase17/chat-your-data/blob/master/state_of_the_union.txt

**Set up the base vector store retriever:**
```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import os

device = "cuda"

documents = TextLoader("state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en", # or any model of your choice
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})
```

**Retrieval without reranking:**
```python
query = "What was done to Russia?"
docs = retriever.invoke(query)
pretty_print_docs(docs)
```

**All the field arguments to RankLLMRerank:**
```
model_path: str = Field(default="rank_zephyr")
top_n: int = Field(default=3)
window_size: int = Field(default=20)
context_size: int = Field(default=4096)
prompt_mode: str = Field(default="rank_GPT")
num_gpus: int = Field(default=1)
num_few_shot_examples: int = Field(default=0)
few_shot_file: Optional[str] = Field(default=None)
use_logits: bool = Field(default=False)
use_alpha: bool = Field(default=False)
variable_passages: bool = Field(default=False)
stride: int = Field(default=10)
use_azure_openai: bool = Field(default=False)
model_coordinator: Any = Field(default=None, exclude=True)
```

**Retrieval with reranking (default RankLLM model is rank_zephyr):**
```python
torch.cuda.empty_cache()

compressor = RankLLMRerank(top_n=3, model_path="rank_zephyr")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

del compressor

compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)
```

## [Rerankers](https://github.com/AnswerDotAI/rerankers)
**Install the packages:**
```bash
pip install "rerankers[rankllm]"
```

**All the field arguments and defaults to Rreranker (with model_type="rankllm"):**
```
model: str = "rank_zephyr",
window_size: int = 20,
context_size: int = 4096,
prompt_mode: PromptMode = PromptMode.RANK_GPT,
num_few_shot_examples: int = 0,
few_shot_file: Optional[str] = None,
num_gpus: int = 1,
variable_passages: bool = False,
use_logits: bool = False,
use_alpha: bool = False,
stride: int = 10,
use_azure_openai: bool = False,
```

**Usage:**
```python
from rerankers import Reranker

ranker = Reranker('rank_zephyr', model_type="rankllm")

results = ranker.rank(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1])
print(results)
```

## [Llama Index](https://github.com/run-llama/llama_index)
**Install the packages:**
```bash
pip install llama-index-core llama-index-embeddings-huggingface llama-index-postprocessor-rank-llm rank_llm transformers requests
```

**All the field arguments and defaults for RankLLMRerank:**
```
model: str = Field(
  description="Model name.",
  default="rank_zephyr"
)
top_n: Optional[int] = Field(
  description="Number of nodes to return sorted by reranking score."
)
window_size: int = Field(
  description="Reranking window size. Applicable only for listwise and pairwise models.",
  default=20
)
batch_size: Optional[int] = Field(
  description="Reranking batch size. Applicable only for pointwise models."
)
context_size: int = Field(
  description="Maximum number of tokens for the context window.",
  default=4096
)
prompt_mode: PromptMode = Field(
  description="Prompt format and strategy used when invoking the reranking model.",
  default=PromptMode.RANK_GPT
)
num_gpus: int = Field(
  description="Number of GPUs to use for inference if applicable.",
  default=1
)
num_few_shot_examples: int = Field(
  description="Number of few-shot examples to include in the prompt.",
  default=0
)
few_shot_file: Optional[str] = Field(
  description="Path to a file containing few-shot examples, used if few-shot prompting is enabled.",
  default=None
)
use_logits: bool = Field(
  description="Whether to use raw logits for reranking scores instead of probabilities.",
  default=False
)
use_alpha: bool = Field(
  description="Whether to apply an alpha scaling factor in the reranking score calculation.",
  default=False
)
variable_passages: bool = Field(
  description="Whether to allow passages of variable lengths instead of fixed-size chunks.",
  default=False
)
stride: int = Field(
  description="Stride to use when sliding over long documents for reranking.",
  default=10
)
use_azure_openai: bool = Field(
  description="Whether to use Azure OpenAI instead of the standard OpenAI API.",
  default=False
)
```

**Load data and build index:**
```python
import os
import requests
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.rankllm_rerank import RankLLMRerank

# Load Wikipedia content
wiki_titles = ["Vincent van Gogh"]
data_path = Path("data_wiki")
data_path.mkdir(exist_ok=True)

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]
    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Set HuggingFace embedder
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

# Load and index documents
documents = SimpleDirectoryReader("data_wiki").load_data()
index = VectorStoreIndex.from_documents(documents)
```

**Retrieval + RankLLM Reranking:**
```python
def get_retrieved_nodes(
    query_str,
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=False,
    model="rank_zephyr",
    window_size=None,
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)
    retrieved_nodes.reverse()

    if with_reranker:
        # configure reranker
        reranker = RankLLMRerank(
            model=model, top_n=reranker_top_n, window_size=window_size
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

        # clear cache, rank_zephyr uses 16GB of GPU VRAM
        del reranker
        torch.cuda.empty_cache()

    return retrieved_nodes


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))
```

Running the test:
```python
# Without RankLLM
new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    with_reranker=False,
)
visualize_retrieved_nodes(new_nodes[:3])

# With RankLLM
new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles?",
    vector_top_k=50,
    reranker_top_n=3,
    with_reranker=True,
    model="rank_zephyr",
    window_size=15,
)
visualize_retrieved_nodes(new_nodes)
```
