"""
一致性校验 Agent (validator)
职责：对推理结果做交叉校验（用另一精度模型复核）→ 标记一致性 → 发布最终结果
通信：Redis 消息队列 + httpx 调用 vLLM
"""
import asyncio
import json
import os
from difflib import SequenceMatcher

import httpx
import redis.asyncio as redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
VLLM_FP16_URL = os.getenv("VLLM_FP16_URL", "http://vllm-fp16:8000/v1")
VLLM_INT4_URL = os.getenv("VLLM_INT4_URL", "http://vllm-int4:8000/v1")
VLLM_FP16_MODEL = os.getenv("VLLM_FP16_MODEL", "qwen-fp16")
VLLM_INT4_MODEL = os.getenv("VLLM_INT4_MODEL", "qwen-int4")

VALIDATING_QUEUE = "tasks:validating"    # 输入：待校验结果
RESULT_CHANNEL = "tasks:{task_id}:result"  # 输出：结果发布通道
RESULT_KEY = "tasks:{task_id}:result"    # 输出：结果存储键

# 一致性阈值（FP16 与 INT4 结果会有差异，阈值不宜过高）
SIMILARITY_THRESHOLD = 0.25


def similarity(a: str, b: str) -> float:
    """两个文本的相似度（基于序列匹配）"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


async def cross_validate(
    client: httpx.AsyncClient,
    task: dict,
) -> dict:
    """
    交叉校验：用「另一精度模型」复核原结果，比对一致性。
    - 若 strategy 用 FP16，则用 INT4 复核（反之亦然）
    - 两结果都非空且相似度达标 → validated=True
    """
    prompt = task["prompt"]
    original = task.get("result", "")
    used = task.get("strategy_used", "fp16")

    # 选择复核模型（与原模型相反精度）
    if used == "fp16":
        url, model, cross_tag = VLLM_INT4_URL, VLLM_INT4_MODEL, "int4"
    else:
        url, model, cross_tag = VLLM_FP16_URL, VLLM_FP16_MODEL, "fp16"

    try:
        resp = await client.post(
            f"{url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.3,   # 复核用低温度，更稳定
            },
            timeout=120,
        )
        resp.raise_for_status()
        cross_result = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        cross_result = ""
        print(f"[validator] 交叉复核失败: {e}", flush=True)

    # 计算一致性
    sim = similarity(original, cross_result)
    validated = bool(original) and bool(cross_result) and sim >= SIMILARITY_THRESHOLD

    task["cross_result"] = cross_result
    task["similarity"] = round(sim, 3)
    task["validated"] = validated
    task["confidence"] = "high" if sim >= 0.5 else ("medium" if validated else "low")
    task["final_result"] = original if validated else "校验未通过，结果可信度低"
    task["status"] = "done"

    print(
        f"[validator] 任务 {task['task_id'][:8]} 校验完成: "
        f"validated={validated}, 相似度={sim:.3f}, 置信度={task['confidence']}",
        flush=True,
    )
    return task


async def main():
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    await r.ping()
    print("[validator] Redis 连接成功，等待校验任务...", flush=True)

    async with httpx.AsyncClient() as client:
        while True:
            _, task_raw = await r.brpop(VALIDATING_QUEUE, timeout=0)
            task = json.loads(task_raw)

            task = await cross_validate(client, task)

            # 发布结果 + 存储结果
            result_json = json.dumps(task, ensure_ascii=False)
            await r.publish(RESULT_CHANNEL.format(task_id=task["task_id"]), result_json)
            await r.set(RESULT_KEY.format(task_id=task["task_id"]), result_json, ex=3600)


if __name__ == "__main__":
    asyncio.run(main())
