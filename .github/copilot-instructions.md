# RankLLM - GitHub Copilot Instructions

RankLLM is a Python package for reranking with Large Language Models (pointwise, pairwise, and listwise). It supports multiple backends including vLLM, SGLang, TensorRT-LLM, and provides various reranking models including RankZephyr, RankVicuna, MonoT5, DuoT5, and LiT5.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Environment Setup
- **Python Requirements**: Python 3.11+ required (Python 3.12.3 verified working)
- **Java Requirements**: JDK 21 required for Anserini dependency. Note: JDK 17 may be present by default but JDK 21 is specifically required
- **Operating System**: Linux or Windows only. **NOT compatible with macOS** (Intel or Apple Silicon)

### Installation and Build Process
- **CRITICAL**: Network timeouts are common during pip installations. If pip install fails with timeout errors, retry multiple times or use alternative installation methods
- **Install Dependencies** (Expected time: 2-5 minutes, NEVER CANCEL - Set timeout to 10+ minutes):
  ```bash
  # Basic installation
  pip install -e .
  
  # With all optional dependencies
  pip install -e .[all]
  
  # For retrieval functionality
  pip install -e .[pyserini]
  
  # For training
  pip install -e .[training]
  ```
- **Python Path Setup**: When running without full installation, set PYTHONPATH:
  ```bash
  export PYTHONPATH=/path/to/rank_llm/src
  ```

### Development Tools and Linting
- **Pre-commit Setup** (Expected time: 1-2 minutes):
  ```bash
  pip install pre-commit black isort flake8
  pre-commit install
  ```
- **Format and Lint Code** (Expected time: 10-30 seconds):
  ```bash
  pre-commit run --all-files
  ```
- **Manual Formatting**:
  ```bash
  black .
  isort --profile=black .
  flake8 --ignore=E501 --select=F401 .
  ```

### Testing
- **Unit Tests** (Expected time: 1-5 seconds, NEVER CANCEL - Set timeout to 2+ minutes):
  ```bash
  python -m unittest discover test/
  ```
- **Specific Test Suites**:
  ```bash
  python -m unittest discover -s test/analysis
  python -m unittest discover -s test/evaluation  
  python -m unittest discover -s test/rerank
  python -m unittest discover -s test/retrieve
  ```
- **Regression Tests** (Expected time: 15-45 minutes per test, NEVER CANCEL - Set timeout to 60+ minutes):
  ```bash
  bash regression_test.sh
  ```

### Running the Application

#### Basic Usage Example
```bash
# Set environment
export PYTHONPATH=/path/to/rank_llm/src

# Run main script with RankZephyr
python src/rank_llm/scripts/run_rank_llm.py \
  --model_path=castorini/rank_zephyr_7b_v1_full \
  --top_k_candidates=100 \
  --dataset=dl20 \
  --retrieval_method=SPLADE++_EnsembleDistil_ONNX \
  --prompt_template_path=src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml \
  --context_size=4096 \
  --variable_passages
```

#### Other Model Examples
```bash
# MonoT5 (pointwise reranker)
python src/rank_llm/scripts/run_rank_llm.py \
  --model_path=castorini/monot5-3b-msmarco-10k \
  --top_k_candidates=1000 \
  --dataset=dl19 \
  --retrieval_method=bm25 \
  --prompt_template_path=src/rank_llm/rerank/prompt_templates/monot5_template.yaml \
  --context_size=512

# DuoT5 (pairwise reranker)  
python src/rank_llm/scripts/run_rank_llm.py \
  --model_path=castorini/duot5-3b-msmarco-10k \
  --top_k_candidates=50 \
  --dataset=dl19 \
  --retrieval_method=bm25 \
  --prompt_template_path=src/rank_llm/rerank/prompt_templates/duot5_template.yaml

# FirstMistral (with logits optimization)
python src/rank_llm/scripts/run_rank_llm.py \
  --model_path=castorini/first_mistral \
  --top_k_candidates=100 \
  --dataset=dl20 \
  --retrieval_method=SPLADE++_EnsembleDistil_ONNX \
  --prompt_template_path=src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml \
  --context_size=4096 \
  --variable_passages \
  --use_logits \
  --use_alpha
```

#### Common Demo Scripts
All demo scripts are located in `src/rank_llm/demo/`:
```bash
cd src/rank_llm/demo
python readme_snippet.py
python rerank_rank_gpt.py
python rerank_zephyr.py
```

## Validation Requirements

### **CRITICAL**: Validation Steps for Changes
- **ALWAYS** run unit tests after making changes: `python -m unittest discover test/`
- **ALWAYS** run linting before commits: `pre-commit run --all-files`
- **ALWAYS** test imports work: `python -c "import rank_llm; print('Success')"`
- **For core functionality changes**: Run specific regression tests relevant to your changes
- **For model changes**: Test with a simple demo script to ensure end-to-end functionality

### Network and Installation Issues
- **pip install timeouts**: Common issue, retry multiple times or use `--timeout 300` flag
- **Missing dependencies**: Core dependencies include dacite, tqdm, pandas, transformers, vllm
- **Java dependency**: Verify JDK 21 is installed with `java -version`

## Navigation and Code Structure

### Key Directories
- `src/rank_llm/`: Main source code
  - `rerank/`: Reranking models and logic (listwise, pointwise, pairwise)
  - `retrieve/`: Retrieval functionality and Pyserini integration
  - `evaluation/`: TREC evaluation tools
  - `analysis/`: Response analysis tools
  - `scripts/`: Command-line tools and utilities
  - `demo/`: Example usage scripts
- `test/`: Unit tests organized by module
- `training/`: Model training scripts and documentation
- `docs/`: Additional documentation and release notes

### Important Files
- `src/rank_llm/scripts/run_rank_llm.py`: Main CLI entry point (247 lines)
- `src/rank_llm/retrieve_and_rerank.py`: Core end-to-end functionality
- `src/rank_llm/data.py`: Data structures and models
- `regression_test.sh`: Comprehensive regression testing
- `pyproject.toml`: Package configuration and dependencies

### Common Workflow Patterns
- **Adding new rerankers**: Extend classes in `src/rank_llm/rerank/`
- **Adding retrieval methods**: Modify `src/rank_llm/retrieve/`
- **Testing changes**: Add tests in corresponding `test/` subdirectory
- **CLI modifications**: Update `src/rank_llm/scripts/run_rank_llm.py`

## Build and CI Information

### GitHub Actions Workflow
- **Lint job**: Runs pre-commit hooks (black, isort, flake8)
- **Unit tests**: Matrix strategy testing analysis, evaluation, rerank modules
- **Requirements**: Java 21, Python 3.11
- **Duration**: Lint ~1 minute, Tests ~2-5 minutes each

### Expected Timing and Timeouts
- **pip install**: 2-10 minutes (network dependent) - **NEVER CANCEL, set 15+ minute timeout**
- **Unit tests**: 1-5 seconds total - **Set 2+ minute timeout for safety**
- **Linting**: 10-30 seconds - **Set 2+ minute timeout**
- **Regression tests**: 5 test cases, 15-45 minutes per test - **NEVER CANCEL, set 60+ minute timeout**
- **Demo scripts**: 1-10 minutes depending on model downloads - **Set 15+ minute timeout**
- **Model inference**: Varies by model size and hardware (seconds to minutes per query batch)

### Known Issues and Workarounds
- **Network timeouts**: Retry pip installations multiple times
- **Java version**: Ensure JDK 21, not just JDK 17
- **Import errors**: Verify PYTHONPATH includes `/src` directory
- **Missing dacite**: Core dependency often missing, install with `pip install dacite`
- **macOS incompatibility**: Use Linux or Windows development environments only

## Troubleshooting

### Common Error Messages
- `ModuleNotFoundError: No module named 'dacite'`: Install dependencies with `pip install -e .`
- `ReadTimeoutError`: Network timeout, retry pip installation
- `ImportError` for rank_llm modules: Set `PYTHONPATH=/path/to/rank_llm/src`
- Test import failures: Install missing dependencies

### Quick Diagnostic Commands
```bash
# Check environment
python --version  # Should be 3.11+
java -version    # Should be JDK 21
echo $PYTHONPATH # Should include .../src

# Test basic functionality
python -c "import rank_llm; print('✓ Basic import works')"
python -c "from rank_llm.data import DataWriter; print('✓ Dependencies installed')"

# Run minimal test
python -m unittest test.analysis.test_response_analyzer -v
```

### Performance Expectations
- **First-time setup**: 10-20 minutes including dependency installation
- **Regular development cycle**: 30 seconds (lint + test)
- **Full regression testing**: 2-4 hours (run overnight or in CI)

## External Dependencies and Integration
- **Anserini**: Java-based information retrieval toolkit (requires JDK 21)
- **vLLM**: High-throughput LLM inference (optional)
- **SGLang**: Structured generation language (optional)  
- **TensorRT-LLM**: NVIDIA optimized inference (optional)
- **HuggingFace**: Model hosting and transformers library
- **TREC evaluation**: Standard IR evaluation metrics

**Remember**: Always validate your changes work by running tests and ensuring imports succeed. The codebase has excellent test coverage and regression testing - use them to verify your changes don't break existing functionality.