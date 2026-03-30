
# ============================================================
# Test MongoDB connection 
# ============================================================
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import os
# load_dotenv()
# client = MongoClient(os.getenv('MONGO_URI_ADMIN'))
# db = client[os.getenv('MONGO_DB_NAME', 'portfolio_rag')]
# for name in db.list_collection_names():
#     count = db[name].count_documents({})
#     print(f'{name}: {count} documents')

# Expected (similar structure):
# weekly_vectors: 68 documents
# pnl_vectors: 31 documents
# newsletter_vectors: 119 documents
# context_vectors: 15358 documents

# ============================================================
# Test Auth 
# ============================================================
# from auth_helper import verify_login
# result = verify_login('admin', 'eTnoH$2001')
# print(result)

# Expected:
# {'username': 'admin1', 'role': 'admin'}

# ============================================================
# Test Vector Search
# ============================================================
from dotenv import load_dotenv
load_dotenv()
from multiagent import build_agent_system
import asyncio

orchestrator = build_agent_system()
result = asyncio.run(orchestrator.run_parallel('what are the key macro risks'))
print(result['market']['analysis'][:300])