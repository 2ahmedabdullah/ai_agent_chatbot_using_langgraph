"""Descriptions of tools/nodes used by the public-agent graph."""

TOOL_DESCRIPTIONS = {
    "abuse_filter": "Sanitizes input and blocks abusive or prompt-injection-style requests.",
    "vague_handler": "Detects empty, very short, greeting-only, or under-specified queries.",
    "load_hot_memory": "Loads recent chat history for the active session from MongoDB.",
    "check_cache": "Checks MongoDB FAQ cache with exact match first, then semantic embedding similarity.",
    "rag_node": "Runs the RAG subgraph, retrieving context and generating a document-grounded raw answer.",
    "finalize_response": "Supervisor final pass that edits raw node output into the final user response.",
    "quality_check": "Checks whether the final response should be accepted, retried, or escalated.",
    "persist_memory": "Stores successful answers in FAQ cache and stores the turn in hot memory.",
}
