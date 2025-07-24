# RankLLM Release Notes (v0.25.0)

+ **Release date:** July 23, 2025
+ **Pyserini dependency:** v1.2.0

## Summary of Changes

+ Added support for prompt templates using yaml files. The current default templates can be found [here](/src/rank_llm/rerank/prompt_templates)
    - After this change, prompts for model coordinators are entirely generated based on the prompt templates so prompt_mode is no longer needed and will be deprecated in v0.30.0 (prompt_mode has been changed to an optional argument for creating model coordinators and analyzing responses)
+ Added support for thinking/reasoning models
    - The thinking traces for reasoning models can now be toggled off by setting the `is_thinking` argument to True
    - A budget for the reasoning token usage can now be specified via the `reasoning_token_budget` argument
+ Extended the support for adding few-shot examples to more model coordinators
    - For supported model coordinators, the number of few-shot examples to use can be specified via the `num_few_shot_examples` argument and a json file containing the few-shot examples can be used via the `few_shot_file` argument
+ Improved/added test coverage for all new features and added regression tests to check for silent errors
+ Updated documentations:
    - Added all of the results from the [rankllm paper](https://arxiv.org/pdf/2505.19284) to 2CR [pages](/src/rank_llm/2cr)
    - Updated training python dependencies to 3.10 and moved them to optional dependencies in pyproject, created conda config file to install environment and dependencies
    - Added [documentation](/docs/external-integrations.md) with instructions on using rankllm with external integrations
+ Other QoL changes/bug fixes:
    - Updated and optimized first stage corpus caching logic and migrated commonly used corpus to HuggingFace Datasets [repo](https://huggingface.co/datasets/castorini/rank_llm_data) and added support for new indices and topics that's supported by [Pyserini](https://github.com/castorini/pyserini/)
    - Added ability to save the model's inference history via `populate_invocations_history` argument by running CLI command
    - Added ability to choose the number of queries to run via `max_queries` argument so tests/experiments can be ran faster with fewer queries
    - Bug fix: improved clean response function to effectively remove thinking traces as well as better handling of "fake" digits
    - Bug fix: added chat template to rank_vicuna's tokenizer
    - Bug fix: fixed rank_fid's LLM generation error with newer versions of Huggingface transformers (v4.50+)

## Contributors

### This Release

Sorted by number of commits:

+ Daniel Guo ([clides](https://github.com/clides))
+ Sahel Sharifymoghaddam ([sahel-sh](https://github.com/sahel-sh))
+ Lily Ge ([lilyjge](https://github.com/lilyjge))
+ Steven Chen ([wu-ming233](https://github.com/wu-ming233))
+ Anthony Miyaguchi ([acmiyaguchi](https://github.com/acmiyaguchi))

### All Time

All contributors with 3 or more commits, sorted by number of commits, [according to GitHub](https://github.com/castorini/rank_llm/graphs/contributors):

+ Sahel Sharifymoghaddam ([sahel-sh](https://github.com/sahel-sh))
+ Daniel Guo ([clides](https://github.com/clides))
+ Ronak ([ronakice](https://github.com/ronakice))
+ Ryan Nguyen ([xpbowler](https://github.com/xpbowler))
+ Andre Slavescu ([AndreSlavescu](https://github.com/AndreSlavescu))
+ Steven Chen ([wu-ming233](https://github.com/wu-ming233))
+ andrewxucs ([andrewxucs](https://github.com/andrewxucs))
+ Jason Zhang ([yilinjz](https://github.com/yilinjz))
+ Yidi Chen ([XKTZ](https://github.com/XKTZ))
+ Patrick Yi ([pjyi2147](https://github.com/pjyi2147))
+ Lily Ge ([lilyjge](https://github.com/lilyjge))
