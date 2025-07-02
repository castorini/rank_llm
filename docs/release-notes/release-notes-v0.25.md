# RankLLM Release Notes (v0.25)

+ **Release date:** PLACEHOLDER, 2025

## Summary of Changes

### Major Changes
+ Added PromptTemplate with InferenceHandlers to be responsible for prompt generation
    - Created inference handler classes for all reranking methods: ListwiseInferenceHandler (with SingleTurnListwiseInferenceHandler and MultiTurnListwiseInferenceHandler as subclasses), RankFIDInferenceHandler, PairwiseInferenceHandler, and PointwiseInferenceHandler
    - Added fewshot prompt injection feature to all inference handlers
    - Created default templates for all current PromptModes
    - Removed old prompt generation code for all rerankers and replaced it with inference handler logic
+ Soft deprecation of PromptMode
    - Modified response analysis class to use regex patterns from the PromptTemplate instead of PromptMode
    - Modified reranker classes to have PromptMode as optional argument and added default PromptTemplate paths instead
    - Removed PromptMode from demos and response analysis
+ Updated and optimized first stage corpus caching logic
    - Migrated commonly used corpus to HuggingFace Datasets instead of github
    - Modified retrieval logic to remove unnecessary code and added new function to optimize caching
    - Added new indices and topics that's supported by Pyserini
+ Added regression tests to prevent silent errors

### Minor Changes
+ Updated 2CR pages
    - Added 2CR page (yaml file, html template, row template) and updated page generation logic for MSMARCO-v2 corpus
    - Updated all 2CR commands and scores with respect to the new changes
+ Improved clean response function to effectively remove thinking traces as well as better handling of "fake" digits and moved it to listwise inference handler class
+ Added new arguments (listwise rerankers): is_thinking, reasoning_token_budget, populate_invocations_history, max_queries
    - Modified RankListwiseOSLLM to use these arguments to effectively remove thinking traces in responses
+ Moved training dependencies to optional dependencies in pyproject
+ Added doc for how to use rankllm with external integrations
+ Updated/added new unittests for full coverage of all new features

## Contributors

### This Release

Sorted by number of commits:

+ Daniel Guo ([clides](https://github.com/clides))
+ Sahel Sharifymoghaddam ([sahel-sh](https://github.com/sahel-sh))
+ Lily Ge ([lilyjge](https://github.com/lilyjge))

### All Time

All contributors with 500 or more additions, sorted by number of additions, [according to GitHub](https://github.com/castorini/rank_llm/graphs/contributors?selectedMetric=additions):

+ Sahel Sharifymoghaddam ([sahel-sh](https://github.com/sahel-sh))
+ Ronak ([ronakice](https://github.com/ronakice))
+ Daniel Guo ([clides](https://github.com/clides))
+ Ryan Nguyen ([xpbowler](https://github.com/xpbowler))
+ Yidi Chen ([XKTZ](https://github.com/XKTZ))
+ Akintunde Oladipo ([theyorubayesian](https://github.com/theyorubayesian))
+ Steven Chen ([wu-ming233](https://github.com/wu-ming233))
+ charlie-liuu ([charlie-liuu](https://github.com/charlie-liuu))
+ Jason Zhang ([yilinjz](https://github.com/yilinjz))
+ Andre Slavescu ([AndreSlavescu](https://github.com/AndreSlavescu))
+ Richard Fan ([Richard5678](https://github.com/Richard5678))
