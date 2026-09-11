"""
用户接口 Agent (gateway)
职责：接收用户请求 → 投递任务队列 → 返回经过校验的推理结果
通信：Redis 消息队列 + SSE 流式返回
"""
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Redis 通道名
PENDING_QUEUE = "tasks:pending"          # 待调度队列
RESULT_PREFIX = "tasks:{task_id}:result" # 结果订阅通道
RESULT_KEY = "tasks:{task_id}:result"    # 结果存储键

r: redis.Redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global r
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    await r.ping()
    print("[gateway] Redis 连接成功", flush=True)
    yield
    await r.close()


app = FastAPI(title="多智能体推理调度 - 用户接口 Agent", lifespan=lifespan)


class InferRequest(BaseModel):
    prompt: str
    strategy: str = "auto"        # auto | fp16 | int4
    max_tokens: int = 256         # 预期输出长度（供推理策略 Agent 判断任务复杂度）


@app.post("/infer")
async def infer(req: InferRequest):
    """接收推理请求，投递到任务队列"""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "prompt": req.prompt.strip(),
        "strategy": req.strategy,        # 用户可指定，默认 auto
        "max_tokens": req.max_tokens,    # 预期输出长度
        "status": "pending",
    }

    await r.lpush(PENDING_QUEUE, json.dumps(task, ensure_ascii=False))
    print(f"[gateway] 接收任务 {task_id[:8]}: {task['prompt'][:30]}...", flush=True)

    return {"task_id": task_id, "status": "dispatched"}


@app.get("/infer/{task_id}")
async def get_result(task_id: str):
    """查询任务最终结果（非阻塞）"""
    data = await r.get(RESULT_KEY.format(task_id=task_id))
    if data is None:
        return {"task_id": task_id, "status": "processing"}
    return json.loads(data)


@app.get("/infer/{task_id}/stream")
async def stream_result(task_id: str):
    """SSE 流式返回经过校验的推理结果"""
    async def event_stream():
        # 先检查是否已有结果
        cached = await r.get(RESULT_KEY.format(task_id=task_id))
        if cached is not None:
            yield f"data: {cached}\n\n"
            return

        # 订阅结果通道
        pubsub = r.pubsub()
        await pubsub.subscribe(RESULT_PREFIX.format(task_id=task_id))

        try:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield f"data: {msg['data']}\n\n"
                    data = json.loads(msg["data"])
                    if data.get("status") == "done":
                        break
        finally:
            await pubsub.unsubscribe(RESULT_PREFIX.format(task_id=task_id))
            await pubsub.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
