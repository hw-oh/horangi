# 🐯 Horangi - Korean LLM Benchmark Evaluation Framework

**Horangi** is an open-source benchmark framework for comprehensively evaluating Korean LLM performance.

By integrating [WandB/Weave](https://wandb.ai/site/weave) and [Inspect AI](https://inspect.ai-safety-institute.org.uk/), it evaluates Korean LLMs along two axes: General Language Performance (GLP) and Alignment Performance (ALT), providing standardized benchmark datasets and evaluation pipelines.
- 📦 Over 20 Korean benchmarks are registered in [Weave](https://wandb.ai/horangi/horangi4/weave/objects), allowing you to start evaluation immediately without separate data preparation.
  - You can add new benchmarks. See [Horangi benchmark documentation](./docs/README_benchmark.md) for details.
- 🔓 You can evaluate API models (OpenAI, Anthropic, Google, etc.) as well as open-source models served via vLLM using the same standards.
- 📊 Evaluation results are automatically logged to Weave, enabling sample-level analysis, model comparison, and leaderboard generation.
- 🏆 Check out the official leaderboard operated by W&B at **[Horangi Leaderboard](https://horangi.ai)**.
  - Manages evaluation runs with W&B Models and tracks results with Weave to provide a **fully automated leaderboard**.
  - The leaderboard automatically updates when new models are evaluated, always reflecting the latest results.

### 📬 Contact

| | |
|---|---|
| Leaderboard Registration | [Application Form](https://docs.google.com/forms/d/e/1FAIpQLSdQERNX8jCEuqzUiodjnUdAI7JRCemy5sgmVylio-u0DRb9Xw/viewform) |
| Enterprise Inquiries | contact-kr@wandb.com |

---

## 📋 Table of Contents

- [Features](#-features)
- [Viewing Results](#-viewing-results)
- [Supported Benchmarks](#-supported-benchmarks)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration Guide](#️-configuration-guide)
- [Evaluating Open-source Models with vLLM](#️-evaluating-open-source-models-with-vllm)
- [SWE-bench Evaluation (Code Generation)](#-swe-bench-evaluation-code-generation)
- [Troubleshooting](#-troubleshooting)

---
## ✨ Features

- 🇰🇷 **20+ Korean benchmarks** supported
- 📊 **Automatic WandB/Weave logging** - Experiment tracking and result comparison
- 🚀 **Various model support** - OpenAI, Claude, Gemini, Solar, EXAONE, etc.
- 🛠️ **CLI support** - Easy execution with `horangi` command
- 📈 **Automatic leaderboard generation** - Model comparison in Weave UI
### 📈 Viewing Results

After evaluation completes, you can view detailed results at the Weave URL in the output:
See [Horangi Weave documentation](./docs/README_weave.md) for more details.
- **Per-sample scores and responses**
- **Model comparison**
- **Aggregated metrics**
- **Automatic leaderboard generation**
![Weave Leaderboard](./docs/assets/leaderboard.png)

---

## 📊 Supported Benchmarks

### General Language Performance (GLP)

Evaluates general language model capabilities including language understanding, knowledge, reasoning, coding, and function calling.

| Evaluation Area | Benchmark | Description | Samples | Source |
|----------------|----------|------|--------:|------|
| **Syntax Analysis** | `ko_balt_700_syntax` | Sentence structure analysis, grammatical validity evaluation | 100 | [snunlp/KoBALT-700](https://huggingface.co/datasets/snunlp/KoBALT-700) |
| **Semantic Analysis** | `ko_balt_700_semantic` | Context-based inference, semantic consistency evaluation | 100 | [snunlp/KoBALT-700](https://huggingface.co/datasets/snunlp/KoBALT-700) |
| | `haerae_bench_v1_rc` | Reading comprehension-based semantic interpretation | 100 | [HAERAE-HUB/HAE_RAE_BENCH_1.0](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0) |
| **Expression** | `ko_mtbench` | Writing, roleplay, humanities expression (LLM Judge) | 80 | [LGAI-EXAONE/KoMT-Bench](https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench) |
| **Information Retrieval** | `squad_kor_v1` | QA-based information retrieval | 100 | [KorQuAD/squad_kor_v1](https://huggingface.co/datasets/KorQuAD/squad_kor_v1) |
| **General Knowledge** | `kmmlu` | Common sense, STEM fundamentals | 100 | [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) |
| | `haerae_bench_v1_wo_rc` | Multi-turn QA-based knowledge evaluation | 100 | [HAERAE-HUB/HAE_RAE_BENCH_1.0](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0) |
| **Expert Knowledge** | `kmmlu_pro` | Advanced expertise in medicine, law, engineering, etc. | 100 | [LGAI-EXAONE/KMMLU-Pro](https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro) |
| | `ko_hle` | Korean expert-level difficult problems | 100 | [cais/hle](https://huggingface.co/datasets/cais/hle) + Custom translation |
| **Common Sense Reasoning** | `ko_hellaswag` | Sentence completion, next sentence prediction | 100 | [davidkim205/ko_hellaswag](https://huggingface.co/datasets/davidkim205/ko_hellaswag) |
| **Mathematical Reasoning** | `hrm8k` | Korean math reasoning (GSM8K, KSM, MATH, MMMLU, OMNI_MATH combined) | 100 | [HAERAE-HUB/HRM8K](https://huggingface.co/datasets/HAERAE-HUB/HRM8K) |
| | `ko_aime2025` | AIME 2025 advanced math | 30 | [allganize/AIME2025-ko](https://huggingface.co/datasets/allganize/AIME2025-ko) |
| **Abstract Reasoning** | `ko_arc_agi` | Visual/structural reasoning, abstract problem solving | 100 | [ARC-AGI](https://arcprize.org/) |
| **Coding** | `swebench_verified_official_80` | GitHub issue resolution | 80 | [SWE-bench](https://www.swebench.com/) |
| **Function Calling** | `bfcl` | Function calling accuracy (single, multi-turn, irrelevance detection) | 258 | [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) |

### Alignment Performance (ALT)

Evaluates model safety and alignment including controllability, ethics, harm/bias prevention, and hallucination prevention.

| Evaluation Area | Benchmark | Description | Samples | Source |
|----------------|----------|------|--------:|------|
| **Controllability** | `ifeval_ko` | Instruction following, command compliance | 100 | [allganize/IFEval-Ko](https://huggingface.co/datasets/allganize/IFEval-Ko) |
| **Ethics/Morality** | `ko_moral` | Social norm compliance, safe language generation | 100 | [AI Hub Ethics Data](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=558) |
| **Harm Prevention** | `korean_hate_speech` | Hate speech, offensive speech detection and suppression | 100 | [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) |
| **Bias Prevention** | `kobbq` | Bias evaluation against specific groups/attributes | 100 | [naver-ai/kobbq](https://huggingface.co/datasets/naver-ai/kobbq) |
| **Hallucination Prevention** | `ko_truthful_qa` | Factuality verification, evidence-based response | 100 | Custom translation |
| | `ko_hallulens_wikiqa` | Wikipedia QA-based hallucination evaluation | 100 | [facebookresearch/HalluLens](https://github.com/facebookresearch/HalluLens) + Custom translation |
| | `ko_hallulens_longwiki` | Long context Wikipedia hallucination evaluation | 100 | [facebookresearch/HalluLens](https://github.com/facebookresearch/HalluLens) + Custom translation |
| | `ko_hallulens_nonexistent` | Fictional entity refusal ability evaluation | 100 | [facebookresearch/HalluLens](https://github.com/facebookresearch/HalluLens) + Custom translation |


<details>
<summary>📦 Dataset References (Weave)</summary>

Datasets are uploaded to the `horangi/horangi4` project:

| Dataset | Weave Ref |
|----------|-----------|
| KoHellaSwag_mini | `weave:///horangi/horangi4/object/KoHellaSwag_mini:latest` |
| KoAIME2025_mini | `weave:///horangi/horangi4/object/KoAIME2025_mini:latest` |
| IFEval_Ko_mini | `weave:///horangi/horangi4/object/IFEval_Ko_mini:latest` |
| HAERAE_Bench_v1_mini | `weave:///horangi/horangi4/object/HAERAE_Bench_v1_mini:latest` |
| KoBALT_700_mini | `weave:///horangi/horangi4/object/KoBALT_700_mini:latest` |
| KMMLU_mini | `weave:///horangi/horangi4/object/KMMLU_mini:latest` |
| KMMLU_Pro_mini | `weave:///horangi/horangi4/object/KMMLU_Pro_mini:latest` |
| SQuAD_Kor_v1_mini | `weave:///horangi/horangi4/object/SQuAD_Kor_v1_mini:latest` |
| KoTruthfulQA_mini | `weave:///horangi/horangi4/object/KoTruthfulQA_mini:latest` |
| KoMoral_mini | `weave:///horangi/horangi4/object/KoMoral_mini:latest` |
| KoARC_AGI_mini | `weave:///horangi/horangi4/object/KoARC_AGI_mini:latest` |
| HRM8K_mini | `weave:///horangi/horangi4/object/HRM8K_mini:latest` |
| KoreanHateSpeech_mini | `weave:///horangi/horangi4/object/KoreanHateSpeech_mini:latest` |
| KoBBQ_mini | `weave:///horangi/horangi4/object/KoBBQ_mini:latest` |
| KoHLE_mini | `weave:///horangi/horangi4/object/KoHLE_mini:latest` |
| KoHalluLens_WikiQA_mini | `weave:///horangi/horangi4/object/KoHalluLens_WikiQA_mini:latest` |
| KoHalluLens_LongWiki_mini | `weave:///horangi/horangi4/object/KoHalluLens_LongWiki_mini:latest` |
| KoHalluLens_NonExistent_mini | `weave:///horangi/horangi4/object/KoHalluLens_NonExistent_mini:latest` |
| BFCL_mini | `weave:///horangi/horangi4/object/BFCL_mini:latest` |
| KoMTBench_mini | `weave:///horangi/horangi4/object/KoMTBench_mini:latest` |
| SWEBench_Verified_80_mini | `weave:///horangi/horangi4/object/SWEBench_Verified_80_mini:latest` |

</details>

---


## 📁 Project Structure

```
horangi/
├── horangi.py              # @task function definitions (entry point)
├── run_eval.py             # Full benchmark execution script
├── configs/
│   ├── base_config.yaml    # Global default settings
│   └── models/             # Model configuration files
├── src/
│   ├── benchmarks/         # Benchmark configurations
│   ├── core/               # Core logic
│   ├── scorers/            # Custom Scorers
│   ├── solvers/            # Custom Solvers
│   └── cli/                # CLI entry point
├── create_benchmark/       # Dataset creation scripts
└── logs/                   # Evaluation logs
```

> 📖 **How to add new benchmarks**: See [docs/README_benchmark.md](docs/README_benchmark.md).

---


## 📦 Installation

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation Steps

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/wandb-korea/horangi.git
cd horangi

# Install dependencies
uv sync
```

### Environment Variables

Copy `.env.sample` to create a `.env` file or set environment variables directly:

```bash
# Provide the API key for the model(s) you intend to use
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
UPSTAGE_API_KEY=your_upstage_api_key

# W&B settings
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_wandb_project
# inspect_ai settings
INSPECT_WANDB_WEAVE_ENABLED=true_or_false
INSPECT_WANDB_MODELS_ENABLED=true_or_false
# swebench server settings
SWE_API_KEY=your_swebench_server_api_key
```

---

## 🚀 Quick Start

### 1. List Available Benchmarks

```bash
uv run horangi --list
```

### 2. Run Benchmarks

There are **two ways** to specify a model:

#### Method A: Direct specification with `--model` option (for simple tests)

```bash
# Basic execution
uv run horangi kmmlu --model openai/gpt-4o

# Limit samples (for testing)
uv run horangi kmmlu --model openai/gpt-4o -T limit=10
```

#### Method B: Use configuration file with `--config` option (recommended)

Configuration files (`configs/models/*.yaml`) allow you to pre-define API endpoints, generation parameters, metadata, etc.

```bash
# Use configuration file (configs/models/gpt-4o.yaml)
uv run horangi kmmlu --config gpt-4o

# Limit samples
uv run horangi kmmlu --config gpt-4o -T limit=10

# Batch run multiple benchmarks (using run_eval.py)
uv run python run_eval.py --config gpt-4o --only kmmlu,kobbq
```

> **💡 Tip**: Using `--config` makes it convenient to reuse settings for custom API endpoints (vLLM, Ollama, etc.).

---

## ⚙️ Configuration Guide

### Configuration File Structure

```
configs/
├── base_config.yaml      # Global default settings
└── models/               # Per-model settings
    ├── _template.yaml    # Template
    ├── gpt-4o.yaml
    └── solar-pro2-251215.yaml
```

### Model Configuration Format

```yaml
# configs/models/my-model.yaml

wandb:
  run_name: "my-model: high-effort"    # W&B run display name

metadata:
  release_date: "2025-01-01"           # Model release date
  size_category: null                  # "Small (<10B)", "Medium (10-30B)", "Large (30B<)"
  model_size: null                     # e.g., "4B", "32B", "70B"
  context_window: 128000
  max_output_tokens: 4096

model:
  name: my-model-name                  # Model name
  client: litellm                      # litellm | openai
  provider: openai                     # anthropic, openai, xai, google, etc.
  # base_url: https://...              # For OpenAI-compatible APIs
  api_key_env: OPENAI_API_KEY

  params:
    max_tokens: 4096
    temperature: 0.0
    # reasoning_effort: high           # For reasoning models
    # timeout: 3600
    # max_retries: 10

benchmarks:
  bfcl:
    use_native_tools: true
  swebench_verified_official_80:
    max_tokens: 16384
```

### Adding a New Model

```bash
# 1. Copy template
cp configs/models/_template.yaml configs/models/my-model.yaml

# 2. Edit configuration
vi configs/models/my-model.yaml

# 3. Run
uv run horangi kmmlu --config my-model -T limit=5
```

### `--model` vs `--config`

| Method | When to Use | Example |
|------|----------|------|
| `--model` | Simple execution, one-time tests | `--model openai/gpt-4o` |
| `--config` | Repeated use, OpenAI-compatible API, per-benchmark settings | `--config solar-pro2-251215` |

---

## 🖥️ Evaluating Open-source Models with vLLM

Here's how to serve open-source models with vLLM on a GPU server and run benchmarks locally.

### 1. Run vLLM Server on GPU Server

```bash
# Install vLLM
pip install vllm

# Serve model (auto-downloads from HuggingFace)
vllm serve LGAI-EXAONE/EXAONE-4.0.1-32B\
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name EXAONE-4.0.1-32B
  --api_key my-secret-key
```

> **💡 `--served-model-name`**: By default, vLLM uses the full HuggingFace path (`LGAI-EXAONE/EXAONE-4.0.1-32B`) as the model name. This option lets you specify a shorter alias, making config file writing easier.

### 2. Create Model Configuration File

```yaml
# configs/models/EXAONE-4.0.1-32B.yaml

wandb:
  run_name: "EXAONE-4.0.1-32B"

metadata:
  release_date: "2025-07-29"
  size_category: "Large (30B<)"
  model_size: 32000000000
  context_window: 32768
  max_output_tokens: 4096

model:
  name: LGAI-EXAONE/EXAONE-4.0.1-32B   # HuggingFace model path
  client: openai                        # vLLM provides OpenAI-compatible API
  provider: lgai                        # Model provider (for metadata)
  base_url: http://YOUR_SERVER_IP:8000/v1
  api_key_env: VLLM_API_KEY

  params:
    max_tokens: 4096
    temperature: 0.0

benchmarks:
  bfcl:
    use_native_tools: false             # Text-based recommended for open-source models
  ko_mtbench:
    temperature: 0.7
```

### 3. Run Benchmarks

```bash
# Set environment variable
export VLLM_API_KEY=my-secret-key

# Test run
uv run horangi kmmlu --config EXAONE-4.0.1-32B -T limit=5

# Full benchmarks
uv run python run_eval.py --config EXAONE-4.0.1-32B
```

---

## 🔧 SWE-bench Evaluation (Code Generation)

SWE-bench is a benchmark that evaluates the ability to fix bugs in real open-source projects.

📖 **Detailed setup guide**: [docs/README_swebench.md](docs/README_swebench.md)

### Quick Start

```bash
# 1. Run server (Linux environment with Docker)
uv run python src/server/swebench_server.py --host 0.0.0.0 --port 8000

# 2. Client setup (macOS, etc.)
export SWE_SERVER_URL=http://YOUR_SERVER:8000

# 3. Run evaluation
uv run horangi swebench_verified_official_80 --config gpt-4o -T limit=5
```

---

## 📚 References
- [WandB Weave](https://wandb.ai/site/weave)
- [Inspect AI Documentation](https://inspect.ai-safety-institute.org.uk/)
- [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)
- [inspect-wandb (fork)](https://github.com/hw-oh/inspect_wandb)
- [inspect_evals (fork)](https://github.com/hw-oh/inspect_evals)

