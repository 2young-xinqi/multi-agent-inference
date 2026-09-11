"""
显存监控 Agent (monitor)
职责：周期性采集 nvidia-smi 显存占用 → 写入 Redis，供 strategy Agent 做负载决策
"""
import asyncio
import json
import os
import subprocess

import redis.asyncio as redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GPU_USAGE_KEY = "gpu:usage"
GPU_USAGE_EXPIRE = 5   # 数据 5 秒过期，避免读到陈旧数据

INTERVAL = 2           # 采集间隔（秒）


def collect_gpu_usage() -> dict:
    """调用 nvidia-smi 采集各 GPU 显存占用率"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore")

        usage = {}
        for line in out.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            idx, used, total = parts
            if int(total) > 0:
                usage[f"gpu{idx}"] = round(int(used) / int(total), 3)
        return usage
    except Exception as e:
        print(f"[monitor] nvidia-smi 采集失败: {e}", flush=True)
        return {}


async def main():
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    await r.ping()
    print("[monitor] Redis 连接成功，开始监控显存...", flush=True)

    while True:
        usage = collect_gpu_usage()
        if usage:
            await r.set(GPU_USAGE_KEY, json.dumps(usage), ex=GPU_USAGE_EXPIRE)
            print(f"[monitor] GPU 显存占用: {usage}", flush=True)
        await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
