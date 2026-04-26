import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def manual_clear_cache():
    uri = os.getenv("CHATBOT_RW_CHAT_HISTORY_URL")
    db_name = os.getenv("MONGODB_DATABASE_NAME")
    
    mongo_client = MongoClient(uri)
    db = mongo_client[db_name] if db_name else mongo_client["chatbot_db"]
    
    # Target the global cache collection
    cache_col = db["public_agent_cache_messages"]
    
    count = cache_col.count_documents({})
    result = cache_col.delete_many({})
    
    print(f"SUCCESS: Deleted {result.deleted_count} stabilized slots from Atlas.")
    print("The cache is now empty and ready for new 'Top 20' queries.")

if __name__ == "__main__":
    manual_clear_cache()