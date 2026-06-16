import os
from rag.subgraph import invoke_rag
from dotenv import load_dotenv

load_dotenv()

# 1. Define your "Golden Dataset" based on the Leadership Blog
eval_questions = [
    "What are the main problems with traditional assessments?",
    "How does AI shift the value from interesting insight to real impact?",
    "What is the future of leadership assessment according to AssessCenter?"
]

def run_local_eval():
    print("Starting RAG Evaluation...")
    for query in eval_questions:
        print(f"\nTesting Query: {query}")
        try:
            # This calls your actual graph
            response = invoke_rag(query=query)
            
            print(f"Response: {response['raw_response'][:150]}...")
            print(f"Contexts Retrieved: {len(response.get('retrieved_context', []))}")
            
        except Exception as e:
            print(f"Eval failed for this query: {e}")
            print("Note: This is expected if Qdrant/OpenAI credits are down.")

if __name__ == "__main__":
    run_local_eval()