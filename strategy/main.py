"""
推理策略 Agent (strategy)
职责：从任务队列取任务 → 规则控制器选择 FP16/INT4 → 调用对应推理引擎

调度算法参考 ModeSwitch-LLM 的规则控制器设计（https://github.com/ModeSwitch-LLM/ModeSwitch-LLM）：
  - 量化模型(INT4)在多数场景下质量够用、延迟/能耗更低 → 作为默认
  - FP16 仅用于「高复杂度 + 长输出 + 资源充足」的场景，或作为保守回退
  - 内存压力 / 批处理压力 → 强制降级 INT4 保吞吐
  - 手写规则优于学习型路由器（学习型路由器 CPU 开销高且无明显收益）
"""
import asyncio
import json
import os
from dataclasses import dataclass

import httpx
import redis.asyncio as redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
VLLM_FP16_URL = os.getenv("VLLM_FP16_URL", "http://vllm-fp16:8000/v1")
VLLM_INT4_URL = os.getenv("VLLM_INT4_URL", "http://vllm-int4:8000/v1")
VLLM_FP16_MODEL = os.getenv("VLLM_FP16_MODEL", "qwen-fp16")
VLLM_INT4_MODEL = os.getenv("VLLM_INT4_MODEL", "qwen-int4")

PENDING_QUEUE = "tasks:pending"          # 输入：待调度任务
VALIDATING_QUEUE = "tasks:validating"    # 输出：待校验结果
GPU_USAGE_KEY = "gpu:usage"              # 显存监控数据

# ==================== 规则控制器阈值（可调）====================
PROMPT_MEDIUM = 20         # 中等复杂度 prompt 长度阈值（>20字符走 FP16）
PROMPT_HARD = 100          # 高复杂度 prompt 长度阈值（>100字符标记高复杂度）
LONG_OUTPUT = 200          # 预期长输出阈值（max_tokens）
MEM_PRESSURE = 0.98        # GPU 显存占用压力阈值（单卡模式下两个引擎共用GPU，静态占用已达90%，调高避免误判降级）
BATCH_PRESSURE = 50        # 队列积压任务数阈值（演示16并发不触发，保留作极端保护）


@dataclass
class Decision:
    """调度决策结果"""
    strategy: str
    reason: str


def choose_strategy(
    prompt: str,
    gpu_usage: dict,
    queue_depth: int,
    user_strategy: str = "auto",
    max_tokens: int = 256,
) -> Decision:
    """
    规则控制器：根据任务特征 + 系统状态选择 FP16 / INT4。

    特征（参考 ModeSwitch-LLM）：
      - prompt 长度（复杂度）
      - 预期输出长度（max_tokens）
      - GPU 内存压力（显存占用）
      - 批处理压力（队列积压）
    """
    # 1. 用户显式指定 → 优先遵从
    if user_strategy in ("fp16", "int4"):
        return Decision(user_strategy, "用户显式指定精度")

    prompt_len = len(prompt)
    long_output = max_tokens > LONG_OUTPUT

    gpu0 = gpu_usage.get("gpu0", 0.0)   # FP16 卡占用
    gpu1 = gpu_usage.get("gpu1", 0.0)   # INT4 卡占用
    mem_pressure = gpu0 > MEM_PRESSURE or gpu1 > MEM_PRESSURE
    batch_pressure = queue_depth > BATCH_PRESSURE

    # 2. 资源紧张（内存/批处理压力）→ 降级 INT4 保吞吐
    if mem_pressure:
        return Decision("int4", f"显存压力高(gpu0={gpu0:.2f}/gpu1={gpu1:.2f})，降级 INT4")
    if batch_pressure:
        return Decision("int4", f"队列积压 {queue_depth} 个任务，降级 INT4 提升吞吐")

    # 3. 高复杂度 + 预期长输出 → FP16 保质量（资源充足时）
    if prompt_len > PROMPT_HARD and long_output:
        return Decision("fp16", f"高复杂度(prompt={prompt_len})+长输出({max_tokens})，用 FP16 保质量")

    # 4. 中等复杂度 → FP16（质量优先）
    if prompt_len > PROMPT_MEDIUM:
        return Decision("fp16", f"中等复杂度(prompt={prompt_len})，用 FP16")

    # 5. 短任务 → INT4（默认量化，低延迟高吞吐）
    return Decision("int4", f"短任务(prompt={prompt_len})，用 INT4 低延迟")


async def call_vllm(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int) -> str:
    """调用 vLLM 生成回答"""
    resp = await client.post(
        f"{url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def main():
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    await r.ping()
    print("[strategy] Redis 连接成功，等待任务...", flush=True)

    async with httpx.AsyncClient() as client:
        while True:
            # 阻塞等待任务
            _, task_raw = await r.brpop(PENDING_QUEUE, timeout=0)
            task = json.loads(task_raw)

            # 采集系统状态：显存占用 + 队列积压
            gpu_raw = await r.get(GPU_USAGE_KEY) or "{}"
            gpu_usage = json.loads(gpu_raw)
            queue_depth = await r.llen(PENDING_QUEUE)

            # 规则控制器决策
            decision = choose_strategy(
                prompt=task["prompt"],
                gpu_usage=gpu_usage,
                queue_depth=queue_depth,
                user_strategy=task.get("strategy", "auto"),
                max_tokens=task.get("max_tokens", 256),
            )

            # 分发到对应推理引擎
            if decision.strategy == "fp16":
                url, model, gpu = VLLM_FP16_URL, VLLM_FP16_MODEL, "gpu0"
            else:
                url, model, gpu = VLLM_INT4_URL, VLLM_INT4_MODEL, "gpu1"

            print(
                f"[strategy] 任务 {task['task_id'][:8]} → {decision.strategy} | "
                f"{decision.reason} | 队列={queue_depth}",
                flush=True,
            )

            try:
                result = await call_vllm(
                    client, url, model, task["prompt"], task.get("max_tokens", 256)
                )
            except Exception as e:
                result = f"[推理失败] {e}"
                print(f"[strategy] 任务 {task['task_id'][:8]} 推理失败: {e}", flush=True)

            # 附加结果，投递到校验队列
            task["result"] = result
            task["gpu_used"] = gpu
            task["strategy_used"] = decision.strategy
            task["strategy_reason"] = decision.reason   # 决策理由（演示用）
            task["status"] = "validating"

            await r.lpush(VALIDATING_QUEUE, json.dumps(task, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
