from typing import List, Dict, Any, Tuple
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.data import Request, Result, Candidate, Query
from rank_llm.rerank import PromptMode

class InsertRankReranker(RankListwiseOSLLM):
    def __init__(self, model_path: str, **kwargs):
        # remove deprecated params - we don't need template since we override create_prompt anyway
        kwargs.pop('prompt_mode', None)
        kwargs.pop('prompt_template_path', None)
        
        # just use the same init as the parent class
        super().__init__(
            model=model_path,
            name=f"insertrank_{model_path.split('/')[-1]}",
            **kwargs
        )
    
    def create_insertrank_listwise_prompt(self, query: str, candidates: List[Candidate], max_length: int = 300):
        """create insertrank prompt following the paper format"""
        
        # follow the paper's format - start with telling the model about bm25 scores
        prompt = f"You are also given the BM25 scores from a lexical retrieval system.\n\n"
        prompt += f"Query: {query}\n\n"
        
        # add each doc with its bm25 score (candidates should already be sorted by bm25)
        # paper puts document content first, then the score below it
        for i, candidate in enumerate(candidates):
            content = self.convert_doc_to_prompt_content(candidate.doc, max_length)
            prompt += f"[{i+1}]. {content}\nBM25 score: {candidate.score:.3f}\n\n"
        
        prompt += f"Rerank the above {len(candidates)} passages based on their relevance to the query. "
        prompt += "Consider both the document content and the BM25 scores as signals for relevance. "
        prompt += f"Return the ranking as a list of numbers from 1 to {len(candidates)}, most relevant first."
        
        return prompt
    
    def convert_doc_to_prompt_content(self, doc: Dict[str, Any], max_length: int) -> str:
        """extract content from document for the prompt"""
        # try different field names that might contain the document text
        if "text" in doc:
            content = doc["text"]
        elif "segment" in doc:
            content = doc["segment"]
        elif "contents" in doc:
            content = doc["contents"]
        else:
            content = str(doc)  # fallback
        
        # cut it short if its too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
            
        # prepend title if we have one
        if "title" in doc and doc["title"]:
            content = f"Title: {doc['title']} Content: {content}"
            
        return content.strip()
    
    
    def create_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[str, int]:
        """override create_prompt to use our custom insertrank format"""
        
        candidates = result.candidates[rank_start:rank_end]
        query_text = result.query.text
        
        # use our custom prompt that includes bm25 scores
        prompt = self.create_insertrank_listwise_prompt(query_text, candidates)
        
        return prompt, len(candidates)

    
    def parse_reranking_response(self, response: str, original_candidates: List[Candidate]) -> List[Candidate]:
        """parse the llm response to get the reranked candidates"""
        
        try:
            # sometimes the response comes as a tuple, handle that
            if isinstance(response, tuple):
                response_str = str(response[0])
            else:
                response_str = str(response).strip()
            
            # extract all numbers from response - should be something like "1, 3, 2, 4, 5" or "[1, 3, 2, 4, 5]"
            import re
            numbers = re.findall(r'\d+', response_str)
            rankings = [int(num) for num in numbers if 1 <= int(num) <= len(original_candidates)]
            
            # reorder the candidates based on what the llm said
            if len(rankings) >= len(original_candidates):
                reranked = []
                used_indices = set()
                
                # go through the rankings and add candidates in that order
                for rank in rankings[:len(original_candidates)]:
                    idx = rank - 1  # convert to 0-based indexing
                    if 0 <= idx < len(original_candidates) and idx not in used_indices:
                        reranked.append(original_candidates[idx])
                        used_indices.add(idx)
                
                # if we missed any candidates somehow, add them at the end
                for i, candidate in enumerate(original_candidates):
                    if i not in used_indices:
                        reranked.append(candidate)
                
                return reranked
            else:
                return original_candidates  # not enough rankings, just return original
                
        except Exception as e:
            return original_candidates  # something went wrong, return original order