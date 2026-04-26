import os
from pymongo import MongoClient
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

def list_cache_entries():
    uri = os.getenv("CHATBOT_RW_CHAT_HISTORY_URL")
    db_name = os.getenv("MONGODB_DATABASE_NAME")
    
    mongo_client = MongoClient(uri)
    db = mongo_client[db_name] if db_name else mongo_client["chatbot_db"]
    
    cache_col = db["public_agent_cache_messages"]
    
    # Fetch all entries, sorted by last_accessed descending
    all_entries = list(cache_col.find({}).sort("last_accessed", -1))
    
    if not all_entries:
        print("Cache is currently empty!")
        return

    print(f"Existing Cache Entries ({len(all_entries)}):\n")
    
    seen_ids = set()
    for i, entry in enumerate(all_entries, 1):
        if entry["_id"] in seen_ids:
            continue  # skip duplicate prints
        seen_ids.add(entry["_id"])
        
        print(f"--- Entry {i} ---")
        pprint({
            "_id": str(entry.get("_id")),
            "user_query": entry.get("user_query"),
            "response": entry.get("response")[:150] + ("..." if len(entry.get("response")) > 150 else ""),
            "hit_count": entry.get("hit_count"),
            "last_accessed": entry.get("last_accessed"),
            "created_at": entry.get("created_at"),
        })
        print("\n")

if __name__ == "__main__":
    list_cache_entries()