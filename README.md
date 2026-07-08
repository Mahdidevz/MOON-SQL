<div align="center">

# MOON-SQL

### Schema-Enhanced & Adaptive Text-to-SQL Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Community-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langchain)
[![Groq](https://img.shields.io/badge/Groq-API-F55036?style=flat-square&logo=groq&logoColor=white)](https://groq.com/)
[![Spider](https://img.shields.io/badge/Benchmark-Spider-4B8BBE?style=flat-square)](https://yale-lily.github.io/spider)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

*A high-performance, parallelised Text-to-SQL pipeline powered by large language models up to 96-100% accuracy, BM25-driven schema linking, and iterative self-correction.*

</div>

---

## 📖 Overview

MOON-SQL converts natural-language questions into executable SQLite queries against the [Spider](https://yale-lily.github.io/spider) benchmark. It achieves this through a multi-stage pipeline that first enriches raw database schemas with semantically relevant cell values (via BM25 retrieval), then drives an LLM through two targeted generation rounds — an initial draft, followed by a self-reflection and correction pass — to produce high-accuracy SQL with minimal hallucination.

The entire pipeline is designed for production throughput: all LLM calls are distributed across a configurable thread pool, each stage checkpoints its partial output, and evaluation runs in parallel across CPU cores.

---

## 🏗️ Architecture

```
Natural Language Question
         │
         ▼
┌─────────────────────────┐
│   Stage 0 · Preprocess  │  BM25 content retrieval · schema normalisation
│   src/preprocess/       │  fuzzy cell-value matching · skeleton extraction
└────────────┬────────────┘
             │  preprocessed_data.json
             ▼
┌─────────────────────────┐
│  Stage 1 · First Round  │  Prompt → LLM (Llama-3 via Groq)
│  src/pipeline/          │  Token-aware model routing (short / long context)
│  first_module.py        │  Parallel chunk generation + auto-resume
└────────────┬────────────┘
             │  first_round.sql
             ▼
┌─────────────────────────┐
│  Stage 2 · Third Round  │  Execute draft SQL against SQLite
│  src/pipeline/          │  Reflect on errors → Correct → Retry (N times)
│  third_module.py        │  ReflectTool + CorrectTool + SQLGenerateTool
└────────────┬────────────┘
             │  third_round.sql
             ▼
┌─────────────────────────┐
│  Stage 3 · Post-process │  Append db_id · produce final predict JSON
│  src/utils/             │
└────────────┬────────────┘
             │  predict_dev_50.json
             ▼
┌─────────────────────────┐
│  Evaluation             │  Exact Match Accuracy + VES scoring
│  src/evaluate/          │  Multi-core parallel SQL execution
└─────────────────────────┘
```

### Key Design Decisions

| Component | Technology | Purpose |
|---|---|---|
| Schema linking | BM25 (`rank_bm25`) + fuzzy match (`rapidfuzz`) | Surface relevant cell values from live DB |
| LLM backend | `langchain-community` → Groq API | Stateless, retryable chat completions |
| Model routing | `tiktoken` token counting | Short prompts → fast model; long prompts → extended-context model |
| Parallelism | `concurrent.futures.ThreadPoolExecutor` | Saturate API rate limits with minimal overhead |
| Fault tolerance | Per-chunk file checkpointing | Resume interrupted runs without re-processing completed samples |
| Evaluation | `func_timeout` + `multiprocessing` | Bounded-time SQL execution across all test cases |

---

## ⚙️ Prerequisites

### Environment Variables

MOON-SQL routes all LLM traffic through an OpenAI-compatible endpoint (e.g., Groq). Set the following before running any script:

```bash
export OPENAI_API_KEY="gsk_..."          # Your Groq API key
export OPENAI_API_BASE="https://api.groq.com/openai/v1"
```

> **Groq API keys** can be obtained from [console.groq.com](https://console.groq.com). The free tier is sufficient for evaluation on the Spider dev set.

### System Requirements

- Python **3.10+**
- SQLite3 (ships with Python's standard library)
- 4+ CPU cores recommended for parallel evaluation

---

## 📦 Data Layout

```
data/
├── spider/
│   └── dev_50.json              # Spider dev questions (50-sample subset)
├── spider_data/
│   ├── tables.json              # Database schemas
│   └── database/               # SQLite .db files, one per schema
├── generate_datasets/
│   └── preprocessed_data.json  # Output of Stage 0 (auto-created)
└── intermediate_datasets/
    ├── first_round_test.sql     # Output of Stage 1 (auto-created)
    ├── third_round.sql          # Output of Stage 2 (auto-created)
    └── predict_dev_50.json      # Final predictions (auto-created)
```

The `intermediate_datasets/` and `generate_datasets/` directories are created automatically by `script/run.sh` if they do not exist.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mahdidevz/MOON-SQL.git
cd MOON-SQL

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API credentials
export OPENAI_API_KEY="gsk_..."
export OPENAI_API_BASE="https://api.groq.com/openai/v1"
```

---

## 🧑‍💻 Usage

### Generation

Run the full pipeline — preprocessing, first-round generation, third-round self-correction, and post-processing — with a single command:

```bash
bash script/run.sh
```

Key parameters inside `script/run.sh` that you can adjust:

| Variable | Default | Description |
|---|---|---|
| `short_model_name` | `llama-3.3-70b-versatile` | Model for prompts under 3 800 tokens |
| `long_model_name` | `llama-3.3-70b-versatile` | Model for longer prompts |
| `PROCESS_NUM` | `1` | Preprocessing worker threads |
| `API_CALL_NUM` | `2` | Parallel LLM call threads |
| `RETRY_NUM` | `10` | Self-correction attempts per failed query |

### Evaluation

After generation, compute **Exact Match Accuracy** and **Valid Efficiency Score (VES)**:

```bash
bash script/eval.sh
```

The evaluation script runs two independent scorers in sequence:

- `src/evaluate/evaluation.py` — set-based exact match accuracy
- `src/evaluate/evaluation_ves.py` — execution efficiency score

Key parameters inside `script/eval.sh`:

| Variable | Default | Description |
|---|---|---|
| `num_cpus` | `4` | CPU cores for parallel SQL execution |
| `meta_time_out` | `30.0` | Per-query execution timeout (seconds) |
| `data_mode` | `dev_50` | Must match the filename stem of the prediction JSON |

---

## 📁 Project Structure

```
MOON-SQL/
├── src/
│   ├── pipeline/
│   │   ├── first_module.py      # Stage 1: initial SQL generation
│   │   └── third_module.py      # Stage 2: reflection & correction
│   ├── preprocess/
│   │   ├── preprocessing.py     # Main preprocessing orchestrator
│   │   ├── add_content.py       # BM25 DB content retrieval
│   │   ├── bridge_content_encoder.py  # Fuzzy cell-value matching
│   │   └── table_generator.py   # Schema string builders
│   ├── evaluate/
│   │   ├── evaluation.py        # Exact match accuracy
│   │   └── evaluation_ves.py    # Valid Efficiency Score
│   ├── utils/
│   │   ├── tools.py             # Schema/FK helpers, SQL runner
│   │   └── append_db_id.py      # Post-process: attach db_id to predictions
│   ├── prompts.py               # All LLM prompt templates
│   └── process_sql.py           # SQL parsing utilities
├── script/
│   ├── run.sh                   # End-to-end generation pipeline
│   ├── eval.sh                  # Evaluation runner
│   └── generate_train_data.sh   # Training data generation (optional)
├── data/                        # Spider dataset (see Data Layout above)
├── requirements.txt
└── README.md
```

---

## 🔬 How Self-Correction Works

The third-round module (`third_module.py`) implements a **Reflect → Correct** loop for every generated query:

1. The draft SQL from Stage 1 is executed against the live SQLite database.
2. If execution raises an error, a `ReflectTool` prompts the LLM to diagnose the failure in natural language.
3. The reflection is fed into a `CorrectTool` together with the original question, schema, and foreign keys, producing a revised query.
4. Steps 1–3 repeat up to `RETRY_NUM` times or until the query executes without error.

This loop is run in parallel across all dataset samples, with each worker independently checkpointing its progress.

---

## 📊 Prompt Templates

MOON-SQL uses a library of structured prompt templates (`src/prompts.py`) covering the full generation lifecycle:

| Template | Stage | Purpose |
|---|---|---|
| `sql_simple_prompt` | First round | Direct SQL generation from schema + question |
| `sql_simple_prompt_kg` | First round | Same, with external knowledge hint |
| `sql_middle_prompt` | Third round | Refined generation after schema re-linking |
| `reflect_prompt` | Third round | Error diagnosis |
| `correct_prompt` | Third round | Query correction given reflection |
| `schema_link_prompt` | Optional | Table/column selection |

---

## 🤝 Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change. Ensure that `script/eval.sh` still passes on the Spider dev set before submitting.

---

<div align="center">
<sub>Built with LangChain · Groq · rank_bm25 · rapidfuzz · Spider</sub>
</div>
