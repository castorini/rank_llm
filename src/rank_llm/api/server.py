from flask import Flask, jsonify, request

from rank_llm import retrieve_and_rerank

app = Flask(__name__)

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
        )
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# http://localhost:8082/ base url 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)
