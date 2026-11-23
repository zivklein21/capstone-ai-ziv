# capstone-ai-ziv

## Project
- Python 3.11, managed with uv. Add dependencies with `uv add`, never `uv pip install`.
- Package lives in `src/capstone_ai_ziv/` (src layout). Run everything with `uv run`.
- Secrets come from `.env` through `core/config.py` (`Settings`). Never commit `.env`.
- Tests: `uv run pytest tests/unit tests/integration` must stay green.

## RAG
### Stack decisions
- Generation: Anthropic SDK (Claude), via `core/ai_client.py`.
- Vector stores: Chroma for local development, Pinecone for production.
- Text splitting: `langchain-text-splitters` (import `langchain_text_splitters`).
- All retrieval code lives in `src/capstone_ai_ziv/rag/`.

### Quality rules
- Always validate retrieval results before passing to generation.
- Every pipeline stage must be independently testable.
- Log chunk counts and token usage at each stage.

### Constraints
- Do not hardcode API keys.
- Do not skip the embedding step by passing raw text to the LLM.