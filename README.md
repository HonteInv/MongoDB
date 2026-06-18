# MongoDB

## Cluster Creation and Set Up
1. Pre-set up info
    - Access to the existing Atlas cluster (portfolio-cluster)
    - Database: portfolio_rag
    - Atlas → cluster → Collections → Create Collection

2. Search Index Config
    - Regular Index: Exact lookups and does not allow for vector similarity search. Created through the Indexes Tab.
    - Altas Vector Search: Sorts on fields and similarity search over embeddings. 

**NOTE**
    - numDimensions field: 3072 for OpenAI text-embedding-3-large, 768 if switch to gemini 
        - must match the embedding model or else will cause an index failure 
    - standard search is cosine for embeddings 

3. Current Fields Used
    - report_day, report_month, region, context_type, section, source
    - content_type -> separates narrative from numbers (deliberate desig)
        - Daily reports mix commentary with embedded tables (exhibits), so they are now tagged separately so we don't embed numerical data as prose. (similar set up to pnl) 

## Set Up 
1. set up virtual environment (Python version 3.11 and up)
2. install requirements 
    * `brew install cmake` if failure building wheel 
3. 

## PNL 
This is stored in a separate collection with different logic for accuracy. 