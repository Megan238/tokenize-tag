from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import asyncio
from fastapi.concurrency import run_in_threadpool
from threading import Lock

from tag import tokenize_and_tag_batch, tokenize_and_tag

app = FastAPI(title="Tokenize & Tag API", version="0.1")

REQ_SEM = asyncio.Semaphore(10)
DICT_LOCK = Lock()  

class OneReq(BaseModel):
    keyword: str = Field(..., min_length=1)

class BatchReq(BaseModel):
    keywords: List[str] = Field(..., min_length=1)

def _safe_tokenize_and_tag(keyword: str) -> Dict:
    with DICT_LOCK:
        return tokenize_and_tag(keyword)

def _safe_tokenize_and_tag_batch(keywords: List[str]) -> List[Dict]:
    with DICT_LOCK:
        return tokenize_and_tag_batch(keywords)

@app.post("/tokenize_and_tag")
async def api_one(body: OneReq) -> Dict:
    try:
        async with REQ_SEM:
            return await run_in_threadpool(_safe_tokenize_and_tag, body.keyword)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize_and_tag_batch")
async def api_batch(body: BatchReq) -> List[Dict]:
    try:
        async with REQ_SEM:
            return await run_in_threadpool(_safe_tokenize_and_tag_batch, body.keywords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
