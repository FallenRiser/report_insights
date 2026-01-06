# Smart Report Insights Engine

A high-performance backend analysis engine that competes with Microsoft Power BI's Quick Insights and Report Insights, powered by local Ollama LLM for conversational analysis.

## Features

### 🚀 Quick Insights
Automatic, fast analysis similar to Power BI Quick Insights:
- **Trend Detection** - Linear regression, Mann-Kendall test, change point detection
- **Outlier Detection** - IQR, Z-score, Isolation Forest, LOF, consensus-based
- **Correlation Analysis** - Pearson, Spearman, Mutual Information
- **Pattern Recognition** - Majority detection, distribution patterns, frequency analysis
- **Seasonality Detection** - FFT-based period detection, STL decomposition

### 📊 Report Insights
Comprehensive, in-depth analysis:
- **Data Profiling** - Complete column statistics and type detection
- **Key Influencers** - Random Forest + SHAP value explanations
- **Decomposition Trees** - Hierarchical measure breakdown with AI-suggested splits
- **Statistical Analysis** - Confidence intervals, group comparisons, normality tests

### 💬 Conversational Analysis
Natural language queries via local Ollama:
- Context-aware responses
- Conversation history
- Suggested follow-up questions
- No cloud dependency - 100% local

## Performance

| Metric | Target |
|--------|--------|
| CSV Parse (100MB) | < 2 seconds |
| Quick Insights (100MB) | < 5 seconds |
| Report Insights (100MB) | < 15 seconds |
| Chat Response | < 3 seconds |

### Optimizations
- **Polars** for 10-100x faster data processing vs Pandas
- **Numba JIT** for critical numerical operations
- **Async processing** for parallel analysis
- **Smart caching** with TTL

## Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running

### Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Or use pip directly
pip install -r requirements.txt
```

### Start Ollama
```bash
# Pull a model (e.g., llama3.2)
ollama pull llama3.2

# Ollama should be running on http://localhost:11434
```

## Running the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

## API Endpoints

### Upload & Sessions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload` | Upload CSV file |
| GET | `/api/v1/sessions` | List all sessions |
| GET | `/api/v1/sessions/{id}` | Get session info |
| DELETE | `/api/v1/sessions/{id}` | Delete session |

### Quick Insights
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/quick-insights/{session_id}` | All quick insights |
| GET | `/api/v1/quick-insights/{session_id}/trends` | Trends only |
| GET | `/api/v1/quick-insights/{session_id}/outliers` | Outliers only |
| GET | `/api/v1/quick-insights/{session_id}/correlations` | Correlations |

### Report Insights
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/report-insights/{session_id}` | Full report |
| POST | `/api/v1/report-insights/{session_id}/key-influencers` | Key influencers |
| POST | `/api/v1/report-insights/{session_id}/decomposition` | Decomposition tree |
| GET | `/api/v1/report-insights/{session_id}/statistics` | Detailed stats |
| GET | `/api/v1/report-insights/{session_id}/seasonality` | Seasonality |
| GET | `/api/v1/report-insights/{session_id}/patterns` | Patterns |

### Chat (Conversational Analysis)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat/{session_id}` | Send message |
| GET | `/api/v1/chat/{session_id}/history` | Chat history |
| DELETE | `/api/v1/chat/{session_id}/history` | Clear history |
| GET | `/api/v1/chat/{session_id}/suggest` | Suggested questions |
| GET | `/api/v1/chat/status` | LLM status |

## Usage Example

```python
import httpx

# Upload a CSV file
with open("data.csv", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/upload",
        files={"file": ("data.csv", f, "text/csv")}
    )
    session_id = response.json()["session_id"]

# Get quick insights
insights = httpx.get(
    f"http://localhost:8000/api/v1/quick-insights/{session_id}"
).json()

print(f"Found {len(insights['insights'])} insights")
for insight in insights["insights"][:5]:
    print(f"- {insight['title']}")

# Chat about the data
response = httpx.post(
    f"http://localhost:8000/api/v1/chat/{session_id}",
    json={"message": "What are the main trends in my data?"}
)
print(response.json()["response"])
```

## Configuration

Environment variables (or `.env` file):

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=120

# Analysis
ANALYSIS_QUICK_INSIGHTS_TOP_N=10
ANALYSIS_OUTLIER_IQR_MULTIPLIER=1.5
ANALYSIS_CORRELATION_SIGNIFICANCE_LEVEL=0.05

# Server
MAX_FILE_SIZE_MB=500
SESSION_TTL_HOURS=24
```

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── config.py               # Configuration management
├── api/
│   ├── routes/             # API endpoints
│   └── schemas/            # Pydantic models
├── core/
│   ├── csv_parser.py       # High-performance parsing
│   ├── data_profiler.py    # Data profiling
│   └── cache.py            # Caching utilities
├── analysis/
│   ├── statistical.py      # Descriptive statistics
│   ├── trends.py           # Trend detection
│   ├── outliers.py         # Anomaly detection
│   ├── correlations.py     # Correlation analysis
│   ├── key_influencers.py  # Feature importance
│   ├── decomposition.py    # Decomposition trees
│   ├── patterns.py         # Pattern recognition
│   └── seasonality.py      # Seasonal analysis
├── insights/
│   ├── quick_generator.py  # Quick insights
│   ├── report_generator.py # Report insights
│   └── ranker.py           # Insight ranking
└── llm/
    ├── ollama_client.py    # Ollama API client
    ├── prompts.py          # Prompt templates
    ├── context_builder.py  # Context management
    └── conversation.py     # Chat handling
```

## License

MIT License
