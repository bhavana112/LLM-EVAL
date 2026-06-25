# Production LLM Evaluation & Experimentation Platform

A production-quality, modular, and developer-centric platform designed for evaluating, tracking, comparing, and auditing Large Language Model (LLM) performance, prompts, latencies, and generation configurations.

---

## 🚀 Key Capabilities

- **Unified Provider Abstraction**: A decoupled interface enabling seamless switching and registration of LLM providers (Google Gemini, OpenAI, Anthropic).
- **Benchmark Evaluation Engine**: Executes test datasets using DeepEval quality metrics (Exact Match, Jaccard, and custom metrics).
- **Persistent Experiment Manager**: Structured serialization saving completed evaluation runs to localized directories.
- **Reporting & Regression Detector**: Programmatic generation of reports and comparison deltas across sequential runs to detect performance drifts or regressions.
- **AI-Assisted Failure Analysis**: Automated grouping of failed cases into thematic categories (e.g. Hallucinations) and generation of root-cause summaries and recommendations using LLMs.
- **Streamlit Interactive Dashboard**: A visualization layer to monitor, compare, and audit runs through clean charts, tables, and KPI metrics cards.

---

## 📂 Folder Structure

```text
llm-eval-platform/
├── backend/
│   ├── api/            # FastAPI Web Server (Endpoints for runs, providers, datasets)
│   ├── core/           # Configuration management, Base storage interfaces, and Base exceptions
│   ├── providers/      # LLM Provider wrappers (Gemini, OpenAI, Anthropic adapters)
│   ├── evaluation/     # Pluggable DeepEval benchmarks runner and dataset loader
│   ├── experiments/    # Runs serialization and disk management (ExperimentManager, JSONStorage)
│   ├── reporting/      # Runs summarization and sequential regression detection
│   └── analysis/       # AI failure filtering, category grouping, and prompt builders
├── dashboard/          # Streamlit UI Visualization (Charts, tables, filters, pages)
│   ├── app.py          # Dashboard entry point and routing manager
│   ├── utils.py        # Adaptable REST API client and local analytics wrappers
│   ├── components/     # UI cards, tables, filters, and sidebar layout
│   └── pages/          # Home, experiments logs, reports, comparisons, regression, analysis pages
├── tests/              # Verification suites (3 pytest test files)
├── .env.example        # Settings template
├── requirements.txt    # Platform dependencies
└── README.md           # This comprehensive guide
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python**: Version 3.10 or higher
- **Virtualenv**: Virtual environment manager

### 2. Environment Setup
Install dependencies inside a local virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to create your local `.env` settings file:

```bash
cp .env.example .env
```

Add your credentials and storage settings in `.env`:
```ini
# Storage configuration (choose "memory" or "json" file storage)
DATABASE_BACKEND=json

# API keys for providers
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

## 💻 Running the Platform

### Start the Backend FastAPI Server
Run the API web server using Uvicorn:

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Start the Streamlit Dashboard
Launch the visualization dashboard in a separate terminal:

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🔄 Core Workflows

### A. Provider Abstraction
All LLM communications are abstractly routed through `backend/providers/base.py`. Adding a new LLM provider requires implementing a subclass of `LLMProvider` and registering it with `ProviderFactory.register("name", ProviderClass)`.

### B. Evaluation Workflow & Engine
The `EvaluationEngine` takes a dataset (containing prompts and target ground truths) and runs queries through the Provider Layer. Completed completions are evaluated by pluggable test case evaluators, yielding scores and execution latencies.

### C. Experiment Lifecycle
Every evaluation run is structured as an `Experiment` and saved via `JSONStorage` under `experiments/exp_[timestamp]_[id]/` containing `experiment.json`.

### D. Reporting & Regression Detection
The `ReportGenerator` calculates statistical summaries (min/max/avg latency, pass rate, metric averages) from experiment outputs to save `report.json`. The `RegressionDetector` calculates pairwise comparisons against standard baselines to identify improvements or regressions (verdict `Better` / `Worse`).

### E. AI-Assisted Failure Analysis
The `FailureAnalyzer` identifies failed runs, wraps them into formatted prompt templates constructed by the `PromptBuilder`, and query the provider for thematic categorization (e.g. Hallucinations), pattern recognition, and recommendations.

---

## 🧪 Running Tests

Verify the integrity of all computational packages using pytest:

```bash
pytest
```
