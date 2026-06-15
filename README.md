# Public Agent Chatbot

This repository contains the backend for the Mesa public-agent chatbot. The bot is built as a LangGraph workflow that accepts a user query, checks safety and clarity, looks for cached answers, retrieves document-grounded context from RAG, and returns a final humanized response.

The public agent is intentionally scoped to the knowledge available in the project documents. It should not act like a general-purpose chatbot. If the answer is not available from the indexed RAG documents, it responds gracefully with the configured support fallback message.

## What The Bot Does

- Checks MongoDB cold memory first for exact or semantic FAQ-style cache hits using cosine similarity (0.80 threshold).
- Uses an LLM classifier (gpt-4o-mini with structured output) to route queries as: toxic, greeting, vague, or valid.
- Filters abusive, unsafe, or prompt-attack style input using gpt-4o with zero temperature.
- Responds to simple greetings with warm, personalized messages.
- Handles unclear or vague questions with context-aware clarification requests.
- Runs RAG over Word, PDF, text, and Markdown documents stored in `rag/rag_docs` for valid queries.
- Loads recent chat history from MongoDB as hot memory (2-turn context, 1-hour TTL) during RAG retrieval.
- Uses Qdrant as the vector database for document chunks (1000-char chunks with 200-char overlap).
- Uses OpenAI embeddings with `text-embedding-3-small` for both cache lookup and RAG retrieval.
- Uses OpenAI chat completion calls (gpt-4o) to generate and humanize final responses.
- Performs LLM-based quality checks on RAG-generated responses with retry logic.
- Stores successful answers back into FAQ cache when appropriate (10-day TTL, max 20 items).
- Persists the current user and assistant turn into hot memory after finalization.
- Supports local CLI testing and FastAPI endpoints with streaming support.
- Supports graph-level streaming through Server-Sent Events.

## High-Level Flow

```text
START
 -> check_cache (FAQ lookup with semantic similarity)
 -> router (LLM classification: toxic | greeting | vague | valid)
    ├─ toxic
    │   -> abuse_filter (security validation)
    │   -> finalize_response
    │   -> persist_memory
    │   -> END
    ├─ greeting
    │   -> greeting_handler (warm greeting response)
    │   -> finalize_response
    │   -> persist_memory
    │   -> END
    ├─ vague
    │   -> vague_handler (clarification request)
    │   -> finalize_response
    │   -> persist_memory
    │   -> END
    └─ valid
        -> rag (document retrieval + response generation)
        -> quality_check (LLM validation with retry)
        -> store_cache (FAQ caching if successful)
        -> finalize_response
        -> persist_memory
        -> END
```

Important behavior:

- Cache hits bypass all other processing and go directly to finalization.
- Router uses gpt-4o-mini with structured output validation for fast classification.
- Abuse, greeting, and vague routes return early without RAG processing.
- Only valid queries reach the RAG pipeline.
- Quality check can retry RAG generation once or escalate to human support fallback.
- Hot memory (recent chat history) is only loaded during RAG retrieval, not stored separately in graph state.
- RAG uses document chunks from `rag/rag_docs` with chat history context.

## Graph Nodes

**check_cache**: Semantic FAQ lookup via MongoDB cold memory. Returns cached response if similarity > threshold.

**router**: LLM classifier (gpt-4o-mini) with structured output. Routes to abuse, greeting, vague, or rag node.

**abuse_filter**: Security validation using gpt-4o. Evaluates query for toxicity, jailbreak attempts, prompt injection.

**greeting_handler**: LLM-based (gpt-4o-mini, temp=0.7) warm greeting response. Acknowledges user and offers help.

**vague_handler**: Requests clarification from user. Returns a template response asking for more specific details.

**rag**: Subgraph that loads chat history, retrieves relevant documents, and generates response using gpt-4o.

**quality_check**: Validates RAG-generated response. Retries once on failure; escalates to human support after retry.

**store_cache**: Persists successful RAG responses to FAQ cache if quality check passed.

**finalize_response**: Polishes final answer (syntax, tone). Applies to all response types (cache, abuse, greeting, vague, RAG).

**persist_memory**: Saves current turn (user query + assistant response) to hot memory for session context.

## Main Folders

```text
agents/
  Standalone pure business logic for abuse handling, vague detection,
  and greeting responses. Can be imported and tested independently.

caching/
  MongoDB backend for hot memory (chat history, 1-hour TTL, 2-turn limit)
  and cold memory (semantic FAQ cache, 10-day TTL, max 20 items).

graph/
  Main LangGraph orchestration layer with node implementations that wrap
  the standalone agent logic with graph state management.
  - agents/: Node implementations (abuse.py, vague.py, greeting_handler.py, etc.)
  - conditions/: Edge routing logic (router classification, edge functions)
  - prompts/: LLM prompts and persona configuration
  - states/: LangGraph state schema definition
  - tools/: Tool definitions and descriptions for LLM calls

rag/
  Document ingestion, chunking, embeddings, Qdrant retrieval, RAG generation,
  and the RAG subgraph. Handles document-grounded answer synthesis.

rag/rag_docs/
  Source documents that should be chunked and indexed for RAG.
  Supports: .pdf, .docx, .txt, .md
```

## Required Environment Variables

Create a `.env` file in the repo root with these values:

```env
PUBLIC_AGENT_OPENAI_APIKEY=
QDRANT_URL=
QDRANT_API_KEY=
CHATBOT_RD_PRODUCT_OVERVIEW_URL=
CHATBOT_RW_CHAT_HISTORY_URL=
MONGODB_DATABASE_NAME=
```

Optional:

```env
RAG_QDRANT_COLLECTION=documents
RAG_SCORE_THRESHOLD=0.0
```

The default Qdrant collection is `documents`. Both indexing and retrieval use the same shared RAG config, so the bot does not accidentally index one collection and query another.

## Configuration

**Cache Settings** (in `caching/`):

- `CACHE_SIMILARITY_THRESHOLD`: Cosine similarity threshold for semantic cache matching (default: 0.80).
- `HOT_MEMORY_TTL_HOURS`: Time-to-live for chat history in hours (default: 1).
- `HOT_MEMORY_TURN_LIMIT`: Maximum conversation turns to keep per session (default: 2).
- `COLD_MEMORY_TTL_DAYS`: Time-to-live for FAQ cache in days (default: 10).
- `COLD_MEMORY_MAX_ITEMS`: Maximum FAQ cache entries before purging oldest (default: 20).

**RAG Settings** (in `rag/config.py`):

- `RAG_QDRANT_COLLECTION`: Qdrant collection name (default: "documents").
- `RAG_SCORE_THRESHOLD`: Minimum relevance score for document retrieval (default: 0.0).
- Document chunking: 1000 characters per chunk, 200-character overlap.

**LLM Models**:

- Router and greeting responses: gpt-4o-mini (for speed).
- Abuse filtering and finalization: gpt-4o (for quality).
- RAG generation and quality check: gpt-4o (for accuracy).
- All embeddings: text-embedding-3-small.

## Install

From the repo root:

```powershell
pip install -r requirements.txt
```

## Index RAG Documents

Place supported files inside:

```text
rag/rag_docs/
```

Supported formats:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

Then index them into Qdrant:

```powershell
python index_rag_docs.py
```

The script loads the files, chunks them, embeds them, creates the Qdrant collection if needed, and upserts the document chunks.

Expected collection:

```text
documents
```

Do not use older upload scripts that write to a different collection such as `portal_documentation`.

## Run Locally In CLI

Use this for end-to-end chatbot testing without a frontend:

```powershell
python local_app.py
```

The CLI will print the answer, route, cache status, and session id after each query.

## Run Local API

Start the FastAPI server:

```powershell
python local_app.py --server
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Chat endpoint:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What does Mesa help with?","session_id":"test-session"}'
```

Streaming endpoint:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/chat/stream" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What does Mesa help with?","session_id":"test-session"}'
```

Available endpoints:

- `GET /health`
- `POST /chat`
- `POST /chat/stream`

## Memory And Cache

**Cache Priority**: Cold cache is checked first at graph entry point. Cache hits bypass all other processing.

**Hot Memory** (Recent Chat History):

- Stores recent chat history by `session_id` (limit: 2 turns, auto-rotated).
- Backed by MongoDB with 1-hour TTL (automatic expiration).
- Only loaded during RAG retrieval to provide conversational context.
- Not used for abuse filtering, routing, or greeting responses.

**Cold Memory** (FAQ Cache):

- Stores normalized query and response pairs from successful previous interactions.
- Uses OpenAI `text-embedding-3-small` embeddings with NumPy cosine similarity.
- Exact string match checked first; semantic match as fallback.
- Similarity threshold: 0.80 (configurable via `CACHE_SIMILARITY_THRESHOLD`).
- TTL: 10 days; max items: 20 (oldest items auto-purged when limit exceeded).
- Backed by MongoDB.
- Only populated after quality check passes on RAG-generated responses.

**RAG Memory**:

- Stores only document chunks in Qdrant (chunked at 1000 characters with 200-character overlap).
- Does not store chat history.
- Does not store FAQ cache entries.
- Uses same embeddings model as cache (`text-embedding-3-small`) for consistency.

**Router Classification**:

- Uses gpt-4o-mini with structured output validation (Pydantic models).
- Classifies queries into: `toxic`, `greeting`, `vague`, or `valid`.
- Allows early exits for non-standard inputs before expensive RAG operations.

## Utility Scripts

**clear_cache.py**

Clear the MongoDB FAQ cache:

```powershell
python clear_cache.py
```

**list_cache.py**

List all cached Q&A pairs with hit counts and timestamps:

```powershell
python list_cache.py
```

## Out-Of-Scope Behavior

If the answer is not supported by the indexed documents, the bot uses this fallback:

```text
I sincerely apologize, but I don't have the information needed to answer your request at this time. We are currently working hard to expand my capabilities and knowledge base to serve you better. For now, please reach out to **support@company.com**, and our team will be happy to assist you manually with this feature or data point.
```

## Developer Notes

- `app.py` is a placeholder for potential async deployment frameworks.
- `local_app.py` is the primary local CLI and FastAPI server entrypoint.
- `langgraph.json` exposes the compiled graph schema for LangGraph Studio and API introspection.
- The graph is the main orchestrator. Router classification and conditional routing are implemented via edge functions and conditional branching.
- The RAG pipeline is implemented as a subgraph (`rag/subgraph.py`) and called as a single node from the main graph.
- Agent logic is split into two locations for separation of concerns:
  - `agents/`: Pure business logic functions (testable in isolation).
  - `graph/agents/`: Node wrappers that manage graph state and invoke the pure logic.
- Quality check node has built-in retry: if response fails quality check, RAG is retried once before falling back to human support message.
- Session IDs are auto-generated with UUID4 if not provided.
- All timestamps in MongoDB use UTC.
