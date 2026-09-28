# AI Market Narrator

The **AI Narrative** Streamlit tab runs a two-node LangGraph workflow for the selected ticker:

1. **Retriever agent** refreshes up to 30 recent Yahoo Finance news items and loads local earnings-call transcript files, splits them into chunks, and upserts them into a ticker-filtered persistent Chroma collection. It retrieves the most relevant excerpts for the current question.
2. **Synthesis agent** reads the latest per-ticker prediction saved by the app (including the model version and conformal interval) and produces a plain-English explanation grounded in the retrieved evidence. Citations use `[S1]`, `[S2]`, and so on; source text, dates, and links are displayed below the explanation.

Source coverage and causality are deliberately distinguished: financial news and transcripts are context for interpreting the prediction, not proof of why the ML model produced it. If source evidence is sparse or the interval crosses zero, the narrative should say so. The explanation is not investment advice.

## Setup

1. Install the project dependencies with `pip install -r requirements-dev.txt` (or `pip install -r requirements.txt` for the app runtime).
2. Set `OPENAI_API_KEY` in `.env` for local runs or Streamlit secrets/environment for deployment. The default chat model is `gpt-4o-mini`; embeddings default to `text-embedding-3-small`. `OPENAI_API_BASE` can point to an OpenAI-compatible endpoint.
3. Optionally configure `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` (default Langfuse cloud URL). Without Langfuse keys the workflow still runs, and each agent's inputs, outputs, status, and timing are recorded locally in `logs/agent_traces.jsonl`.
4. Start the dashboard with `streamlit run app/app.py`, select a ticker, and click **Generate AI Narrative**. Uploaded `.txt` and `.md` transcripts are saved under `data/earnings_transcripts/<TICKER>/` and indexed during generation.

The Chroma database is stored under `data/market_narrator/chroma/`. Both it and uploaded transcripts are ignored by git; provide your own transcript text. News availability and freshness depend on Yahoo Finance. No transcript vendor/API is assumed.

The model registry's existing job was to retain model versions, not predictions. The app now also writes the latest prediction separately for each ticker to `outputs/latest_predictions.json`, so a ticker's explanation is paired with its own point estimate and calibrated interval.

## Faithfulness evaluation

Generate at least one narrative with source context, then run:

```bash
python scripts/evaluate_narrative_faithfulness.py
# or: make eval-narrative
```

This reads successful synthesis-agent records from `logs/agent_traces.jsonl`, evaluates each response against the exact retrieved contexts using RAGAS **Faithfulness**, prints scores and their mean, and writes `outputs/narrative_faithfulness.json`. It uses `OPENAI_API_KEY` and `RAGAS_EVAL_MODEL` (default `gpt-4o-mini`). The evaluation is an LLM-based estimate, not a guarantee of truth.
