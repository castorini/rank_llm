from flask import Flask, jsonify, request

from rank_llm import retrieve_and_rerank
from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode


app = Flask(__name__)

print("Loading default model...")
# Load default rank-zephyr model upon server initialization
default_agent = RankListwiseOSLLM(
    model="castorini/rank_zephyr_7b_v1_full",
    context_size=4096,
    prompt_mode=PromptMode.RANK_GPT,
    num_few_shot_examples=0,
    device="cuda",
    num_gpus=1,
    variable_passages=False,
    window_size=20,
    system_message=None,
)

@app.route('/api/model/<string:model_path>/collection/<string:dataset>/retriever/<string:host>/query=<string:query>&hits_retriever=<int:top_k_retrieve>&hits_reranker=<int:top_k_rerank>&qid=<int:qid>', methods=['GET'])
def search(model_path, dataset,host, query,top_k_retrieve, top_k_rerank,qid):
    try:
        # Assuming the function is called with these parameters and returns a response
        response = retrieve_and_rerank.retrieve_and_rerank(
            dataset=dataset,
            query=query,
            model_path=model_path,
            host="http://localhost:" + host,
            interactive=True, 
            top_k_rerank=top_k_rerank,
            top_k_retrieve=top_k_retrieve,
            qid=qid,
            exec_summary=False,
            default_agent=default_agent
        )

        return jsonify(response[0]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# http://localhost:8082/ base url 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=False)
