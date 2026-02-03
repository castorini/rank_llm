import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from rank_llm.server.mcp.mcp_rankllm import main

if __name__ == "__main__":
    main()
