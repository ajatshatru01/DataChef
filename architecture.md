# DataChef — Architecture

> *Let your data tell its story.*

This document describes the system architecture of DataChef: an open-source AI agent for autonomous dataset investigation. It covers the layered component model, data flows, module responsibilities, safety boundaries, and the design decisions that shape the initial release.

---

## Table of Contents

1. [Architectural Philosophy](#1-architectural-philosophy)
2. [High-Level Layer Model](#2-high-level-layer-model)
3. [Component Map](#3-component-map)
4. [Layer Specifications](#4-layer-specifications)
   - 4.1 [Presentation Layer](#41-presentation-layer)
   - 4.2 [Dataset Context Layer](#42-dataset-context-layer)
   - 4.3 [AI Reasoning Layer](#43-ai-reasoning-layer)
   - 4.4 [Provider Integration Layer](#44-provider-integration-layer)
   - 4.5 [Analytical Execution Layer](#45-analytical-execution-layer)
   - 4.6 [Persistence Layer](#46-persistence-layer)
5. [Core Data Flows](#5-core-data-flows)
   - 5.1 [Dataset Upload and Profiling](#51-dataset-upload-and-profiling)
   - 5.2 [Conversational Query](#52-conversational-query)
   - 5.3 [Autonomous Discovery](#53-autonomous-discovery)
   - 5.4 [Iterative Investigation Loop](#54-iterative-investigation-loop)
6. [Tool Registry](#6-tool-registry)
7. [Context Management and Token Budget](#7-context-management-and-token-budget)
8. [Safety and Validation Boundaries](#8-safety-and-validation-boundaries)
9. [Proposed Repository Structure](#9-proposed-repository-structure)
10. [Technology Stack Reference](#10-technology-stack-reference)
11. [Extension Points](#11-extension-points)
12. [Design Decisions and Trade-offs](#12-design-decisions-and-trade-offs)

---

## 1. Architectural Philosophy

DataChef is built around three foundational constraints that shape every design decision:

**Computation is ground truth.** The AI model is a reasoning and orchestration layer, not the source of numerical facts. Every metric, comparison, and statistical result must originate from the local analytical engine. The model interprets evidence; it does not fabricate it.

**Data stays local by default.** Raw dataset rows are never transmitted to a hosted inference API. Only schema information, profiling summaries, and concise tool results cross that boundary. With the Ollama provider, reasoning also runs locally. Users are informed when derived information may be transmitted.

**The investigation loop is bounded and auditable.** Every tool call is validated before execution. The number of calls per investigation, the execution time, and the size of returned data are all subject to explicit limits. Each step of the investigation is recorded in a format that makes it possible to understand and reproduce the reported conclusions.

These constraints deliberately exclude certain capabilities from the initial release — unrestricted code execution, vector databases, RAG pipelines, and multi-agent delegation — so that the core investigation workflow can be evaluated and stabilized first.

---

## 2. High-Level Layer Model

```
┌─────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                    │
│          Streamlit UI · Upload · Chat · Reports         │
└────────────────────────┬────────────────────────────────┘
                         │ user events / rendered findings
┌────────────────────────▼────────────────────────────────┐
│                 DATASET CONTEXT LAYER                   │
│    Schema · Profile · Investigation State · History     │
└──────────┬──────────────────────────────┬───────────────┘
           │ context snapshot             │ tool results
┌──────────▼──────────┐      ┌────────────▼───────────────┐
│   AI REASONING      │      │   ANALYTICAL EXECUTION     │
│       LAYER         │◄────►│         LAYER              │
│  Orchestration ·    │      │  Profiling · Stats ·       │
│  Hypothesis ·       │      │  Comparisons · Anomalies · │
│  Tool Selection ·   │      │  Correlations · Trends     │
│  Interpretation     │      └────────────────────────────┘
└──────────┬──────────┘
           │ API requests / responses
┌──────────▼──────────────────────────────────────────────┐
│               PROVIDER INTEGRATION LAYER                │
│       Auth · Retry · Timeout · Response Normalization    │
│          Local Ollama or Hugging Face Providers          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   PERSISTENCE LAYER                     │
│      Investigation Records · Session State · Cache      │
└─────────────────────────────────────────────────────────┘
```

The Persistence Layer is accessed by the Context, Reasoning, and Presentation layers as needed. It has no direct connection to the Provider Integration Layer.

---

## 3. Component Map

| Layer | Module / Package | Primary Responsibility |
|---|---|---|
| Presentation | `datachef/ui/` | Streamlit pages, upload widget, chat interface, finding cards, history view |
| Dataset Context | `datachef/context/` | DatasetProfile, InvestigationState, ContextBuilder, DatasetStore |
| AI Reasoning | `datachef/agent/` | InvestigationAgent, PromptBuilder, ToolCallParser, FindingComposer |
| Provider Integration | `datachef/providers/` | BaseProvider, OllamaProvider, HuggingFaceProvider, response normalization, retry logic |
| Analytical Execution | `datachef/analytics/` | Profiler, StatEngine, GroupComparator, AnomalyDetector, CorrelationEngine, TrendAnalyzer, DiscoveryEngine |
| Tool Registry | `datachef/tools/` | ToolRegistry, ToolSchema, ToolValidator, tool definitions |
| Persistence | `datachef/persistence/` | SessionStore, InvestigationRecord, ProfileCache |

---

## 4. Layer Specifications

### 4.1 Presentation Layer

**Package:** `datachef/ui/`

The Presentation Layer is a Streamlit application composed of discrete page modules. Its sole responsibility is rendering state and forwarding user actions to the layers beneath it. It holds no analytical logic and performs no direct API calls.

**Pages and components:**

| Component | Description |
|---|---|
| `upload_page.py` | File upload widget for CSV and Parquet. Triggers validation and profiling on submission. |
| `chat_page.py` | Conversational interface. Renders the investigation thread and forwards user messages to the agent. |
| `discovery_page.py` | Initiates and displays autonomous discovery runs. Shows progress and candidate findings. |
| `history_page.py` | Lists past investigations with expandable step-by-step records. |
| `finding_card.py` | Reusable component that renders a single finding with its metrics, chart, and tool-call trace. |
| `progress_indicator.py` | Live indicator showing the current agent step during an active investigation. |

**Interaction contract:** The UI layer reads from `InvestigationState` and `SessionStore`. It dispatches actions (upload, query, discover, replay) to the agent and context layers. It never reads raw dataset rows directly.

**Visualization:** Plotly figures are constructed in the Analytical Execution Layer and passed upward as serialized figure objects. The UI layer renders them; it does not construct them.

---

### 4.2 Dataset Context Layer

**Package:** `datachef/context/`

The Dataset Context Layer is the memory of a running DataChef session. It maintains the loaded dataset, the computed profile, the current investigation state, and the structured history of past investigations.

**Core data structures:**

**`DatasetProfile`** — A Pydantic model representing the full initial profile of a dataset.

```
DatasetProfile
├── source_path: str
├── dataset_fingerprint: str
├── row_count: int
├── column_count: int
├── columns: List[ColumnProfile]
│   ├── name: str
│   ├── dtype: str              # polars dtype name
│   ├── inferred_role: Literal["numeric", "categorical", "datetime", "text", "id"]
│   ├── null_count: int
│   ├── null_pct: float
│   ├── n_unique: int
│   ├── stats: NumericStats | CategoricalStats | None
│   └── sample_values: List[Any]
├── duplicate_row_count: int
├── correlation_matrix: Dict[str, Dict[str, float]] | None
└── profiled_at: datetime
```

**`InvestigationState`** — A mutable object that tracks the active investigation turn by turn.

```
InvestigationState
├── session_id: str
├── dataset_profile: DatasetProfile
├── conversation_history: List[Message]    # OpenAI message format
├── tool_call_log: List[ToolCallRecord]
├── findings: List[Finding]
└── turn_count: int
```

**`ContextBuilder`** — Produces the minimal context snapshot that is safe to send to the inference API for a given investigation step. It selects from the profile only those fields relevant to the current question and appends concise summaries of recent tool results. Raw rows are never included.

**`DatasetStore`** — Manages the in-memory Polars DataFrame and provides read-only access to the analytical engine. No layer above the analytical engine is permitted to mutate the stored DataFrame.

---

### 4.3 AI Reasoning Layer

**Package:** `datachef/agent/`

The AI Reasoning Layer is the orchestration core of DataChef. It manages the iterative investigation loop: interpreting user intent, building prompts, sending requests to the provider, parsing tool calls, dispatching them to the analytical engine, evaluating results, and deciding whether to continue or conclude.

**`InvestigationAgent`** — The central controller. It exposes two primary entry points:

- `investigate(user_message: str) → Finding` — handles a conversational query.
- `discover() → List[Finding]` — initiates autonomous discovery.

Both methods run the bounded investigation loop described in §5.4.

**`PromptBuilder`** — Constructs the system prompt and the per-turn user message. The system prompt is stable across a session and contains:
- The agent's role and operating constraints.
- The dataset schema and a condensed profile summary.
- The tool registry (tool names, descriptions, and JSON schemas).
- Safety and reproducibility instructions.

The per-turn message is constructed by `ContextBuilder` and contains the current question or discovery objective plus recent tool results.

**`ToolCallParser`** — Parses the model's response to extract structured tool calls. Handles both strict JSON tool-call format and, as a fallback, JSON embedded in markdown code blocks. Malformed outputs are rejected before reaching the validator.

**`FindingComposer`** — Synthesizes accumulated tool results, the model's interpretation, and investigation metadata into a structured `Finding` object. It separates computed facts from the model's interpretation and flags any limitations identified during the investigation.

**Bounded loop invariants (enforced by `InvestigationAgent`):**

| Limit | Default | Configurable |
|---|---|---|
| Max tool calls per investigation | 10 | Yes |
| Max investigation wall-clock time | 120 s | Yes |
| Max tokens per tool result | 2 000 | Yes |
| Max turns before forced conclusion | 8 | Yes |

---

### 4.4 Provider Integration Layer

**Package:** `datachef/providers/`

The Provider Integration Layer isolates model communication behind a stable interface. The reasoning layer interacts only with `BaseProvider`; it has no knowledge of Ollama or Hugging Face client details and never handles authentication tokens.

**`BaseProvider` (Abstract)**

```python
class BaseProvider(ABC):
    @abstractmethod
    def chat_completion(
        self,
        messages: List[dict],
        tools: List[dict],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        ...
```

**`ProviderResponse`** (Pydantic model)

```
ProviderResponse
├── content: str | None
├── tool_calls: List[ParsedToolCall]
├── finish_reason: str
├── usage: TokenUsage
└── raw_response: dict
```

**`HuggingFaceProvider`** — Uses `huggingface_hub.InferenceClient` with the chat-completion and function-calling APIs. It handles:

- API key injection from environment variables (never from code or committed configuration files).
- Configurable model identifier and inference provider.
- Request timeout enforcement.
- Exponential backoff on 429 (rate limit) and 5xx responses.
- Graceful degradation when usage credits are exhausted.
- Response normalization into `ProviderResponse`.

**`OllamaProvider`** — Uses Ollama's local `/api/chat` endpoint and supports native tool calls, request timeouts, response normalization, and retries for connection failures, rate limits, and server errors.

**Provider configuration** is loaded from environment variables and an optional local `.env` file excluded from version control:

```
DATACHEF_PROVIDER=ollama    # ollama or huggingface
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4

HF_API_KEY=...              # required only for huggingface
HF_MODEL_ID=...
HF_INFERENCE_PROVIDER=auto
```

Additional backends, such as direct `llama.cpp` integration, can be added by implementing `BaseProvider` without touching the reasoning layer.

---

### 4.5 Analytical Execution Layer

**Package:** `datachef/analytics/`

The Analytical Execution Layer performs all dataset computations on the user's machine. It exposes a collection of deterministic, validated Python functions that the tool registry maps to named tools. The reasoning layer never calls these functions directly; it goes through the tool registry and validator.

**Sub-modules:**

| Module | Class | Responsibility |
|---|---|---|
| `profiler.py` | `DatasetProfiler` | Computes the initial `DatasetProfile` using Polars, DuckDB, and SciPy |
| `stat_engine.py` | `StatEngine` | Descriptive statistics, value counts, percentiles, grouped aggregations, cross-tabulations |
| `group_comparator.py` | `GroupComparator` | Group means, medians, std devs, Cohen's d, optional Kruskal–Wallis test |
| `anomaly_detector.py` | `AnomalyDetector` | IQR-fence and modified Z-score outlier detection |
| `correlation_engine.py` | `CorrelationEngine` | Pearson and Spearman pairwise correlations; ranked association lists |
| `trend_analyzer.py` | `TrendAnalyzer` | Monotonic trend detection; period-over-period changes; rolling statistics |
| `discovery_engine.py` | `DiscoveryEngine` | Orchestrates predefined check suite; produces ranked `CandidateFinding` list |
| `chart_builder.py` | `ChartBuilder` | Constructs serialized Plotly figures for all supported chart types |

---

### 4.6 Persistence Layer

**Package:** `datachef/persistence/`

The Persistence Layer provides lightweight local storage using JSON files and, optionally, SQLite. It does not introduce a separate database server or a cloud-hosted storage service.

**`SessionStore`** — Manages the active `InvestigationState`, dataset, and agent components in Streamlit session state. Completed findings are persisted by `InvestigationRecord`; uploaded raw datasets are deliberately not persisted by DataChef.

**`InvestigationRecord`** — An immutable, serialized record of a completed investigation. It contains:
- The original question or discovery objective.
- The sequence of tool calls with validated arguments and returned results.
- The model configuration used (model ID, provider, temperature).
- The final finding.

Records are stored as newline-delimited JSON under `~/.datachef/investigations/`.

**`ProfileCache`** — Caches the `DatasetProfile` for a given file path and modification time so that re-uploading the same dataset does not re-run the full profiling computation.

---

## 5. Core Data Flows

### 5.1 Dataset Upload and Profiling

```
User uploads file (CSV / Parquet)
        │
        ▼
[UI] Validates file type and size
        │ valid path
        ▼
[DatasetStore] Loads DataFrame with Polars
        │
        ▼
[ProfileCache] Check cache by (path, mtime)
        │
        ├── cache hit ──────────────────────► [Context] Store profile
        │
        └── cache miss
                │
                ▼
        [DatasetProfiler] Compute full profile
                │
                ▼
        [ProfileCache] Write cache entry
                │
                ▼
        [Context] Store DatasetProfile in InvestigationState
                │
                ▼
        [UI] Display profile summary to user
```

Profiling is always performed locally. No profile data is sent to the inference API during this step.

---

### 5.2 Conversational Query

```
User submits natural-language question
        │
        ▼
[UI] Appends user message to conversation history
        │
        ▼
[InvestigationAgent] Entry: investigate(user_message)
        │
        ▼
[ContextBuilder] Produce context snapshot
  (schema + condensed profile + relevant recent results)
        │
        ▼
[PromptBuilder] Build messages array for API call
        │
        ▼
[Configured Provider] Send chat completion request
        │
        ▼
Model response
        │
        ├── text-only response ──► [FindingComposer] Compose finding → UI
        │
        └── tool call in response
                │
                ▼
        [ToolCallParser] Extract and validate tool call structure
                │
                ▼
        [ToolValidator] Validate tool name, arguments, column names
                │
                ├── invalid ──────► Return error message to model
                │
                └── valid
                        │
                        ▼
                [ToolRegistry] Dispatch to analytical function
                        │
                        ▼
                [Analytical Engine] Execute and return ToolResult
                        │
                        ▼
                [InvestigationState] Append ToolCallRecord
                        │
                        ▼
                [InvestigationAgent] Check loop limits
                        │
                        ├── limit reached ──► [FindingComposer] Force conclusion
                        │
                        └── continue ──────► Next model turn (loop back)
```

---

### 5.3 Autonomous Discovery

```
User initiates discovery mode
        │
        ▼
[DiscoveryEngine] Run predefined check suite on DatasetProfile
        │   (outlier scan, group difference scan, correlation scan,
        │    trend scan, missingness scan, duplicate scan)
        │
        ▼
List[CandidateFinding]  (ranked by heuristic score)
        │
        ▼
[InvestigationAgent] Receive candidate list
        │
        ▼
[PromptBuilder] Build discovery prioritization prompt
        │
        ▼
[Configured Provider] Request model to select and order candidates
        │
        ▼
Model returns ordered list of candidates to investigate
        │
        ▼
For each selected candidate (up to configured maximum):
        │
        └──► [InvestigationAgent] investigate(candidate) → Finding
                (follows the Iterative Investigation Loop in §5.4)
        │
        ▼
[FindingComposer] Aggregate findings into discovery report
        │
        ▼
[SessionStore] Persist investigation records
        │
        ▼
[UI] Display discovery findings
```

---

### 5.4 Iterative Investigation Loop

This is the bounded loop at the heart of every investigation. It is the same regardless of whether the trigger is a conversational query or a discovery candidate.

```
┌──────────────────────────────────────────┐
│           INVESTIGATION LOOP             │
│                                          │
│  ┌─────────────────────────────────┐     │
│  │ 1. Build context snapshot       │     │
│  │    (schema, profile, results)   │     │
│  └────────────────┬────────────────┘     │
│                   │                      │
│  ┌────────────────▼────────────────┐     │
│  │ 2. Send to hosted model         │     │
│  └────────────────┬────────────────┘     │
│                   │                      │
│         ┌─────────▼──────────┐           │
│         │  text conclusion?  │──YES──────┼──► Compose Finding → EXIT
│         └─────────┬──────────┘           │
│                   │ NO (tool call)        │
│  ┌────────────────▼────────────────┐     │
│  │ 3. Parse & validate tool call   │     │
│  └────────────────┬────────────────┘     │
│                   │                      │
│         ┌─────────▼──────────┐           │
│         │    valid call?     │──NO───────┼──► Return error to model
│         └─────────┬──────────┘           │
│                   │ YES                  │
│  ┌────────────────▼────────────────┐     │
│  │ 4. Execute analytical tool      │     │
│  └────────────────┬────────────────┘     │
│                   │                      │
│  ┌────────────────▼────────────────┐     │
│  │ 5. Append ToolCallRecord        │     │
│  └────────────────┬────────────────┘     │
│                   │                      │
│         ┌─────────▼──────────┐           │
│         │  limits reached?   │──YES──────┼──► Force conclusion → EXIT
│         └─────────┬──────────┘           │
│                   │ NO                   │
│                   └─── (back to step 1)  │
└──────────────────────────────────────────┘
```

**Limits checked at step 5:**
- `turn_count >= MAX_TURNS`
- `tool_call_count >= MAX_TOOL_CALLS`
- `elapsed_time >= MAX_WALL_SECONDS`

---

## 6. Tool Registry

**Package:** `datachef/tools/`

The Tool Registry is the interface contract between the AI reasoning layer and the analytical execution layer. It defines every operation the model is permitted to invoke.

**`ToolSchema`** — A Pydantic model that mirrors the OpenAI function-calling schema:
```
ToolSchema
├── name: str
├── description: str
├── parameters: JsonSchemaObject     # validated against model output
└── handler: Callable                # reference to analytical function
```

**`ToolValidator`** — Validates incoming tool calls against the schema:
- Tool name must exist in the registry.
- All required parameters must be present.
- Column names must exist in the loaded dataset.
- Enum parameters must match allowed values.
- Numeric parameters must be within allowed ranges.

**Initial tool set:**

| Tool Name | Analytical Function | Description |
|---|---|---|
| `get_column_stats` | `StatEngine.describe` | Descriptive statistics for one or more columns |
| `compare_groups` | `GroupComparator.compare` | Compare a numeric column across a categorical grouping column |
| `get_correlation` | `CorrelationEngine.pairwise` | Pairwise correlations between numeric columns |
| `detect_outliers` | `AnomalyDetector.detect` | Find candidate outliers in a numeric column |
| `get_value_counts` | `StatEngine.value_counts` | Frequency table for a categorical column |
| `analyze_trend` | `TrendAnalyzer.monotonic_trend` | Trend direction and strength for a time-indexed column |
| `get_missing_summary` | `StatEngine.missing_summary` | Missing value counts and patterns across columns |
| `get_group_distribution` | `StatEngine.group_distribution` | Distribution statistics for a numeric column per group |
| `get_top_correlations` | `CorrelationEngine.top_n` | Strongest N correlations above a threshold |
| `run_cross_tabulation` | `StatEngine.cross_tab` | Frequency cross-tabulation of two categorical columns |

New tools are added by implementing a handler function in the Analytical Execution Layer, defining a `ToolSchema`, and registering both in `ToolRegistry`. The reasoning layer requires no changes.

---

## 7. Context Management and Token Budget

Sending unnecessary data to the inference API increases cost and latency. `ContextBuilder` applies the following rules to construct each request:

**Always included:**
- Dataset schema (column names, dtypes, inferred roles).
- Aggregated profile statistics (null percentages, row/column count, top categories).
- Tool registry descriptions and JSON schemas.

**Conditionally included:**
- Condensed summaries of the last N tool results (default: 3).
- Correlation matrix only if correlation questions are relevant.
- Sample values only when needed for column disambiguation.

**Never included:**
- Raw dataset rows.
- Full profiling data when only a subset is relevant.
- Previous investigation records from unrelated sessions.

**Token budget enforcement:** `ContextBuilder` estimates the token count of the assembled context and truncates or summarizes tool results if the budget is exceeded before sending to the provider.

---

## 8. Safety and Validation Boundaries

DataChef applies validation at three explicit boundaries:

```
                   ┌─────────────────────────────┐
  Model output     │   BOUNDARY 1: Tool Call     │
  ─────────────►   │   Parser + Validator        │
                   │   (schema, columns, types)  │
                   └──────────────┬──────────────┘
                                  │ validated call
                   ┌──────────────▼──────────────┐
  Validated call   │   BOUNDARY 2: DatasetStore  │
  ─────────────►   │   Read-only access enforced │
                   │   No mutation permitted      │
                   └──────────────┬──────────────┘
                                  │ results
                   ┌──────────────▼──────────────┐
  Tool results     │   BOUNDARY 3: Result Caps   │
  ─────────────►   │   Max rows, max tokens      │
                   │   Truncated before context  │
                   └─────────────────────────────┘
```

**Additional safeguards:**
- API credentials are read exclusively from environment variables. They are never hardcoded, logged, or included in any serialized investigation record.
- Model-generated Python code is not executed. The model selects from a closed set of registered tools.
- Source datasets are opened read-only. The analytical engine does not write to them.
- All investigation records written to disk are sanitized to exclude raw dataset rows.
- The application displays a disclosure notice explaining which derived information may be transmitted when a hosted provider is selected.

---

## 9. Proposed Repository Structure

```
datachef/
├── datachef/
│   ├── __init__.py
│   ├── config.py                  # Application configuration from env
│   ├── ui/
│   │   ├── app.py                 # Streamlit entry point
│   │   ├── upload_page.py
│   │   ├── chat_page.py
│   │   ├── discovery_page.py
│   │   ├── history_page.py
│   │   ├── finding_card.py
│   │   └── progress_indicator.py
│   ├── context/
│   │   ├── models.py              # DatasetProfile, InvestigationState, Finding
│   │   ├── dataset_store.py       # DatasetStore (Polars DataFrame wrapper)
│   │   ├── context_builder.py     # ContextBuilder
│   │   └── profile_cache.py
│   ├── agent/
│   │   ├── investigation_agent.py # InvestigationAgent (main loop)
│   │   ├── prompt_builder.py
│   │   ├── tool_call_parser.py
│   │   └── finding_composer.py
│   ├── providers/
│   │   ├── base.py                # BaseProvider ABC
│   │   ├── huggingface.py         # HuggingFaceProvider
│   │   ├── ollama.py              # OllamaProvider
│   │   └── registry.py            # Provider selection by config
│   ├── tools/
│   │   ├── registry.py            # ToolRegistry, ToolValidator
│   │   ├── schemas.py             # ToolSchema definitions
│   │   └── definitions.py         # All tool JSON schemas and handlers
│   ├── analytics/
│   │   ├── profiler.py
│   │   ├── stat_engine.py
│   │   ├── group_comparator.py
│   │   ├── anomaly_detector.py
│   │   ├── correlation_engine.py
│   │   ├── trend_analyzer.py
│   │   ├── discovery_engine.py
│   │   └── chart_builder.py
│   └── persistence/
│       ├── session_store.py
│       ├── investigation_record.py
│       └── profile_cache.py
├── tests/
│   ├── unit/
│   │   ├── test_profiler.py
│   │   ├── test_group_comparator.py
│   │   ├── test_tool_validator.py
│   │   ├── test_context_builder.py
│   │   └── test_finding_composer.py
│   ├── integration/
│   │   ├── test_investigation_loop.py
│   │   └── test_discovery_engine.py
│   └── fixtures/
│       ├── sample_numeric.csv
│       ├── sample_mixed.csv
│       └── sample_timeseries.parquet
├── docs/
│   ├── architecture.md            # This document
│   ├── agents.md
│   └── contributing.md
├── .env.example                   # Template for environment variables
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 10. Technology Stack Reference

| Domain | Library | Version Policy | Notes |
|---|---|---|---|
| Data loading | Polars | >= 0.20 | Primary DataFrame engine |
| Data loading | PyArrow | >= 14 | Parquet backend for Polars |
| Analytical SQL | DuckDB | >= 0.10 | In-process SQL for complex aggregations |
| Statistics | NumPy | >= 1.26 | Numerical arrays |
| Statistics | SciPy | >= 1.12 | Statistical tests, distributions |
| Hosted AI client | huggingface-hub | >= 0.23 | Inference Providers chat completion and tools |
| Local AI client | requests | >= 2.31 | Ollama `/api/chat` transport |
| Validation | Pydantic | >= 2.0 | Data models and tool schema validation |
| UI | Streamlit | >= 1.33 | Application interface |
| Visualization | Plotly | >= 5.20 | Interactive charts |
| Testing | pytest | >= 8.0 | Unit and integration tests |
| CI | GitHub Actions | — | Linting, tests, packaging on push |

---

## 11. Extension Points

The modular design explicitly supports the following future extensions without requiring changes to the core investigation loop:

| Extension | Entry Point |
|---|---|
| New inference backend (for example llama.cpp) | Implement `BaseProvider` |
| New analytical tool | Add handler + `ToolSchema` + register in `ToolRegistry` |
| New discovery check | Add check function in `DiscoveryEngine` |
| New dataset format (JSON, Excel) | Extend `DatasetStore` |
| New chart type | Add method to `ChartBuilder` |
| Alternative UI (FastAPI + React) | Replace `datachef/ui/` while reusing all other layers |
| Experiment tracking | Replace `SessionStore` with a compatible implementation |

---

## 12. Design Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Polars over Pandas | Faster for large files; stricter typing; modern API | Smaller ecosystem; some contributors may be unfamiliar |
| DuckDB for SQL aggregations | Expressive for grouped analytics; no server required | Additional dependency |
| Streamlit for initial UI | Low-friction for Python developers; built-in state management | Less flexible than a custom frontend |
| Local and hosted providers | Users can choose privacy/local control or hosted scale | Local models require hardware; hosted inference may cost money |
| Closed tool set over code execution | Predictable, auditable, safe | Less flexible than a REPL or code-gen approach |
| JSON file persistence | Simple, zero-dependency, inspectable | Not suitable for concurrent or large-scale use |
| Pydantic v2 for all models | Fast validation; JSON schema generation for tools | Breaking changes from v1; newer dependency |
| Provider-neutral response model | Keeps the agent loop independent of Ollama/Hugging Face details | Adapters must normalize provider-specific formats |
