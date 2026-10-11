# Contributing to DataChef

Thank you for your interest in contributing to **DataChef**! We welcome bug reports, feature requests, statistical tool contributions, and documentation improvements.

---

## 🧭 Guiding Architectural Principles

Before contributing code, please keep in mind our three core tenets:

1. **Computation is ground truth:** The AI model is an orchestrator and interpreter, not a source of arithmetic or statistical numbers. Every fact, metric, and comparison must originate from deterministic execution in Polars or SciPy.
2. **Data stays local by default:** Raw dataset rows are never transmitted over the network boundary to any model API. Only schema details, summary statistics, and concise tool results may be passed to the reasoning layer.
3. **The investigation loop is bounded and auditable:** All tool calls must be validated before execution, bounded by step/turn/time limits, and recorded in human-readable NDJSON format.

---

## 🛠️ Development Setup

1. **Clone and Install:**
   ```bash
   git clone https://github.com/ajatshatru01/DataChef.git
   cd DataChef
   pip install -e ".[dev]"
   ```

2. **Environment Configuration:**
   Copy `.env.example` to `.env`, then configure either local Ollama or `HF_API_KEY`.

3. **Running the Application:**
   ```bash
   python -m streamlit run datachef/ui/app.py
   ```

4. **Running Tests:**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Static Checks:**
   ```bash
   python -m ruff check datachef tests
   python -m compileall -q datachef
   ```

---

## 🔌 Adding a New Analytical Tool

Adding a deterministic analytical tool requires:
1. Implement the deterministic logic in `datachef/analytics/`.
2. Define the JSON schema and handler binding in `datachef/tools/definitions.py`.
3. Register the schema in `ToolRegistry` and add any necessary constraints to `ToolValidator` in `datachef/tools/registry.py`.
4. Add unit tests in `tests/unit/`.

The AI reasoning layer will automatically discover the tool via schema injection without requiring prompt modifications!
