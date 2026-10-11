# 🍳 DataChef

> **Let your data tell its story.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/engine-Polars-orange.svg)](https://pola.rs/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)

**DataChef** is an open-source AI agent for autonomous dataset investigation. It pairs open-weight language models with a local, deterministic analytical engine (powered by Polars and SciPy) to explore, audit, and investigate tabular datasets through conversational queries and autonomous discovery.

Unlike standard chatbots that hallucinate numerical facts, DataChef adheres strictly to the principle that **computation is ground truth**: the model reasons, plans, and selects analytical tools, while all calculations and statistics are computed locally on your machine.

---

## 🌟 Key Highlights

- **🔒 Local-First Privacy Guarantee:** Raw dataset rows are **never** transmitted to the cloud. Only column schemas, summary profile statistics, and concise tool execution results cross the network boundary.
- **🛡️ Computation as Ground Truth:** Numerical claims are traced to local tool output; unsupported numerical claims are flagged in the finding.
- **⚡ 11 Deterministic Analytical Tools:**
  - `get_column_stats`: Descriptive statistics (mean, median, std, min, max, quantiles).
  - `compare_groups`: Group-level comparisons with Cohen's $d$ effect sizes and Kruskal–Wallis tests.
  - `get_correlation`: Pairwise Pearson & Spearman correlation matrices.
  - `detect_outliers`: Tukey IQR fences & Iglewicz–Hoaglin modified Z-scores.
  - `analyze_trend`: Temporal monotonicity and period-over-period delta checks.
  - `get_value_counts` & `run_cross_tabulation`: Categorical distributions and cross-tabs.
  - `get_missing_summary`: Missing value patterns and completeness rates.
  - `get_group_distribution` & `get_top_correlations`: Focused grouped and ranked summaries.
  - `get_duplicate_summary`: Duplicate counts and bounded row hashes without exposing raw rows.
- **🚀 Autonomous Discovery Mode:** Scans datasets across 7 deterministic statistical checks without invoking AI models, generates heuristic candidate findings, and deep-dives into high-priority patterns.
- **📊 Interactive Visualizations:** Auto-generates dark-themed Plotly charts (histograms, box plots, group bars, correlation heatmaps, trend lines).
- **📜 Auditable & Reproducible:** Every investigation step, tool call, argument, and result is recorded into immutable NDJSON logs for verification.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                    │
│          Streamlit UI · Upload · Chat · Reports         │
└────────────────────────┬────────────────────────────────┘
                         │ user queries / rendered findings
┌────────────────────────▼────────────────────────────────┐
│                 DATASET CONTEXT LAYER                   │
│    Schema · Profile · Investigation State · History     │
└──────────┬──────────────────────────────┬───────────────┘
           │ context snapshot             │ tool results
┌──────────▼──────────┐      ┌────────────▼───────────────┐
│   AI REASONING      │      │   ANALYTICAL EXECUTION     │
│       LAYER         │◄────►│         LAYER              │
│  Orchestration ·    │      │  Polars · SciPy ·          │
│  Hypothesis ·       │      │  Comparisons · Anomalies · │
│  Tool Selection ·   │      │  Correlations · Trends     │
│  Interpretation     │      └────────────────────────────┘
└──────────┬──────────┘
           │ API requests / responses
┌──────────▼──────────────────────────────────────────────┐
│               PROVIDER INTEGRATION LAYER                │
│    Open-Source Models via huggingface_hub (Apache 2.0)  │
│         (Llama 3.1 70B, Qwen 2.5, DeepSeek, etc.)       │
└─────────────────────────────────────────────────────────┘
```

For detailed specifications, see:
- [Architecture Guide](architecture.md)
- [Agent Design & Bounded Loop](agents.md)

---

## 🚀 Quick Start

### 1. Installation

On Windows PowerShell, create an isolated environment and install the project with its development tools:

```powershell
git clone https://github.com/ajatshatru01/DataChef.git
Set-Location DataChef
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.10–3.13 are supported. On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

If PowerShell blocks activation, either run `Set-ExecutionPolicy -Scope Process Bypass` for that terminal or invoke `.\.venv\Scripts\python.exe` directly.

### 2. Configuration

Create your local configuration:

```powershell
Copy-Item .env.example .env
```

Choose one provider.

#### Option A: local Ollama

No API key is needed. Install [Ollama](https://ollama.com/), then in a separate terminal:

```powershell
ollama pull gemma4
ollama serve
```

Set these values in `.env`:

```env
DATACHEF_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4
```

Use a model tagged for tool support. Smaller models run on less hardware but may choose tools less reliably.

#### Option B: Hugging Face Inference Providers

Create a token with Inference Providers permission, then set:

```env
DATACHEF_PROVIDER=huggingface
HF_API_KEY=hf_your_token_here
HF_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
HF_INFERENCE_PROVIDER=auto
```

Tool support and availability depend on the selected model/provider combination.

### 3. Run the app

```powershell
python -m streamlit run datachef/ui/app.py
```

Open your browser at `http://localhost:8501`.

After installation, the equivalent console command is `datachef`. If the script directory is not on `PATH`, use the `python -m streamlit` command above.

---

## 🧪 Running Tests

Run the complete suite:

```powershell
python -m pytest
```

Run static checks:

```powershell
python -m ruff check datachef tests
python -m compileall -q datachef
```

The suite covers:
- Dataset profiler and column role inference
- Group comparator with Cohen's $d$ effect sizes
- Tool validator schema boundaries
- Minimal context builder and token budgets
- Finding composer and fact/interpretation separation
- Autonomous discovery engine candidate generators
- End-to-end multi-turn investigation loop with mock providers
- True turn/tool-call limits and batched tool calls
- Reproducible NDJSON records and path safety
- Headless Streamlit startup and sample-dataset ingestion

## Troubleshooting

- **Ollama connection warning:** make sure `ollama serve` is running and `OLLAMA_MODEL` exactly matches `ollama list`.
- **Hugging Face error:** verify `HF_API_KEY`, model availability, and that the chosen provider supports chat completion with tools.
- **No useful tool calls:** use a tool-capable model; very small local models may return plain text instead of structured calls.
- **Port 8501 already in use:** run `python -m streamlit run datachef/ui/app.py --server.port 8502`.
- **Start clean:** delete `.env`, recreate it from `.env.example`, and restart Streamlit after configuration changes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
