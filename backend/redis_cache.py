from datetime import datetime
import redis
import json
import os

from models import ChatSessionResponse

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# Caching Chat Sessions
def cache_chat_session(chat_id, user_id, chat_data: ChatSessionResponse, ttl=3600):
    chat_dict = chat_data.model_dump()  # ✅ Convert Pydantic model to dictionary
    redis_client.setex(f"chat:{chat_id}:{user_id}", ttl, json.dumps(chat_dict))

def get_cached_chat_session(chat_id, user_id):
    data = redis_client.get(f"chat:{chat_id}:{user_id}")
    if data:
        chat_dict = json.loads(data)  # ✅ Convert JSON string back to dictionary
        return ChatSessionResponse(**chat_dict)  # ✅ Convert dictionary back to Pydantic model
    return json.loads(data) if data else None

def delete_cached_chat_session(chat_id, user_id):
    redis_client.delete(f"chat:{chat_id}:{user_id}")

# ✅ Update cached chat session when a new message arrives
def update_cached_chat_session(chat_id, user_id, new_message: dict):
    cached_chat = get_cached_chat_session(chat_id, user_id)
    if cached_chat:
        cached_chat.updated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cached_chat.messages.append(new_message)  # ✅ Append new message
        cache_chat_session(chat_id, user_id, cached_chat)  # ✅ Update Redis cache