# Deficiency Data Chatbot

A production-ready LLM-powered chatbot for querying FDA deficiency data using OpenAI function calling.

## Features

- **Natural Language Queries**: Ask questions in plain English
- **15 Query Functions**: From simple filters to complex analytics
- **Smart Data Visualization**: Tables, charts, and formatted results
- **Export Capabilities**: Download results as CSV or Excel
- **Conversation Memory**: Multi-turn conversations with context
- **Error Handling**: Retry logic and graceful error recovery
- **Logging**: Comprehensive logging for debugging

## Architecture

```
User Question → OpenAI GPT-4 → Function Calling → Pandas DataFrame → Formatted Response
```

### Why Function Calling > Text-to-SQL

| Aspect | Text-to-SQL | Function Calling |
|--------|-------------|------------------|
| Error Rate | High (SQL hallucination) | Low (typed functions) |
| LLM Calls | 3-5 per query | 1-2 per query |
| Complexity | 13+ components | 4 components |
| Debugging | Hard | Easy |

## Quick Start

### 1. Setup Environment with UV

```bash
cd text2sql

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install fastapi "uvicorn[standard]" python-dotenv pandas openpyxl pyarrow openai streamlit httpx
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

### 3. Run the Application

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501

## Example Queries

### Basic Queries
- "How many deficiencies are there for USA?"
- "Show API deficiencies from 2023"
- "What are the top categories?"
- "Compare USA vs EU markets"

### Advanced Analytics
- "List all products having nitrosamine impurity related deficiency"
- "List all products approved in the last financial year, marketwise"
- "List deficiency points with keyword 'dissolution', by product"
- "Which products are under review (no approval yet), by market"
- "List outlier products with high response duration by letter type"
- "Which products got approval in US within 10 months without CR letter"
- "List products responded later than average, show highest deviation"

## Project Structure

```
text2sql/
├── app.py                    # Streamlit frontend (production-ready)
├── backend/
│   ├── __init__.py
│   ├── api.py                # FastAPI endpoints (optional)
│   ├── data_service.py       # Data loading & 15 query functions
│   ├── llm_service.py        # OpenAI with retry logic
│   └── tools.py              # Function definitions for LLM
├── data/
│   └── deficiency_data.xlsx  # Your data file
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Available Data Fields

| Column | Description |
|--------|-------------|
| GEOGRAPHY | Geographic region (NAG, EM, EUG, etc.) |
| Markets | Market/Country (USA, EU, CHINA, etc.) |
| VERTICAL | RA Function (API, Injectables, OSD, etc.) |
| PRODUCTNAME | Pharmaceutical product name |
| RECEIVEDDATE | Date deficiency was received |
| RESPONSEDATE | Date response was submitted (null if pending) |
| ApprovalDate | Date approved (null if under review) |
| CATEGORY | Main category (Finished Product, DMF, API, etc.) |
| DEFICIENCYLETTERTYPE | Letter type (CR-Major, CR-Minor, DRL-Quality, etc.) |
| DEFICIENCYDESCRIPTION | Full description text |

## Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | gpt-4o-mini |
| `DATA_FILE_PATH` | Path to data file | data/deficiency_data.xlsx |

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment Variables for Production

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_MODEL=gpt-4o-mini
DATA_FILE_PATH=/app/data/deficiency_data.xlsx
```

### Security Considerations

1. **API Keys**: Use secrets management (AWS Secrets Manager, Azure Key Vault)
2. **Data**: Ensure data file is not exposed publicly
3. **Network**: Deploy behind a reverse proxy (nginx)
4. **Authentication**: Add authentication layer if needed

## Troubleshooting

**"No OpenAI API Key" warning**
- Ensure `.env` file exists with valid API key
- Or enter key directly in sidebar

**Data not loading**
- Check `data/deficiency_data.xlsx` exists
- Verify file permissions

**Slow responses**
- First query loads data into memory (slower)
- Subsequent queries are faster
- Consider using GPT-4o-mini for cost/speed balance

**Complex queries timing out**
- Increase `max_tool_calls` in llm_service.py
- Some queries may need multiple tool calls

## License

Internal use only.
