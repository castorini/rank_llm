import argparse
from flask import Flask, jsonify, request

from rank_llm import retrieve_and_rerank
from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.api_keys import get_openai_api_key, get_azure_openai_args
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rankllm import PromptMode

""" API URL FORMAT

http://localhost:8082/api/model/{model_name}/index/{index_name}/retriever/{retriever_base_host}?query={query}&hits_retriever={top_k_retriever}&hits_reranker={top_k_reranker}&qid={qid}&num_passes={num_passes}

hits_retriever, hits_reranker, qid, and num_passes are OPTIONAL
Default to 20, 5, None, and 1 respectively

"""
 

def create_app(model, port, use_azure_openai=False):

    app = Flask(__name__)
    if model == 'rank_zephyr':
        print(f"Loading {model} model...")
        # Load specified model upon server initialization
        default_agent = RankListwiseOSLLM(
            model=f"castorini/{model}_7b_v1_full",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device="cuda",
            num_gpus=1,
            variable_passages=True,
            window_size=20,
            system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        )
    elif model == 'rank_vicuna':
        print(f"Loading {model} model...")
        # Load specified model upon server initialization
        default_agent = RankListwiseOSLLM(
            model=f"castorini/{model}_7b_v1",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device="cuda",
            num_gpus=1,
            variable_passages=False,
            window_size=20,
        )
    elif 'gpt' in model:
        print(f"Loading {model} model...")
        # Load specified model upon server initialization
        openai_keys = get_openai_api_key()
        print(openai_keys)
        default_agent = SafeOpenai(
            model=model,
            context_size=8192,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            keys=openai_keys,
            **(get_azure_openai_args() if use_azure_openai else {}),
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    @app.route('/api/model/<string:model_path>/index/<string:dataset>/<string:host>', methods=['GET'])
    def search(model_path, dataset, host):

        query = request.args.get('query',type=str)
        top_k_retrieve = request.args.get('hits_retriever',default=20,type=int)
        top_k_rerank = request.args.get('hits_reranker',default=5,type=int)
        qid = request.args.get('qid',default=None,type=str)
        num_passes = request.args.get('num_passes',default=1,type=int)

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
                default_agent=default_agent,
                num_passes=num_passes,
            )

            return jsonify(response[0]), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app, port

def main():
    parser = argparse.ArgumentParser(description="Start the RankLLM Flask server.")
    parser.add_argument('--model', type=str, default='rank_zephyr', help='The model to load (e.g., rank_zephyr).')
    parser.add_argument('--port', type=int, default=8082, help='The port to run the Flask server on.')
    parser.add_argument('--use_azure_openai', action='store_true', help='Use Azure OpenAI API.')
    args = parser.parse_args()

    app, port = create_app(args.model, args.port, args.use_azure_openai)
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()
