"""
高并发负载测试脚本
提交混合 max_tokens 的任务，验证推理策略 Agent 的规则控制器是否正确分流 FP16/INT4。

任务分类（对应 strategy 的规则表）：
  - short  ：短 prompt + 短输出 → 预期 INT4（默认量化，低延迟）
  - medium ：中等复杂度 → 预期 FP16（质量优先）
  - hard   ：高复杂度 + 长输出 → 预期 FP16（保质量）
  - 批量压力：并发 > 10 个任务时，队列积压触发降级 INT4
"""
import asyncio
import time

import httpx

GATEWAY_URL = "http://localhost:8080"

# 三类任务：明确触发不同调度规则
TASKS = [
    # (标签, prompt, max_tokens)
    # ---- short：短任务 → INT4 ----
    ("short", "什么是 Transformer 模型？", 64),
    ("short", "1+1 等于几？", 64),
    ("short", "你好，请介绍一下你自己", 64),
    ("short", "今天天气怎么样？", 64),
    # ---- medium：中等复杂度 → FP16 ----
    ("medium", "解释 Transformer 中的多头注意力机制是什么？请说明它的主要作用。", 128),
    ("medium", "对比分析 RNN 和 Transformer 两种序列建模架构的优缺点。", 128),
    # ---- hard：高复杂度 + 长输出 → FP16 ----
    ("hard", "详细解释 Transformer 中自注意力机制（Self-Attention）的计算流程，"
             "包括 Q、K、V 矩阵的作用、Scaled Dot-Product Attention 的公式推导，"
             "以及为什么需要除以根号 d_k 进行缩放。请给出完整的数学说明。", 512),
    ("hard", "从并行计算能力、长距离依赖建模、训练效率、可扩展性等多个角度，"
             "全面分析 Transformer 相比 RNN 和 LSTM 的优势与不足，"
             "并讨论其在大语言模型中的应用价值。", 512),
]

# 循环次数：生成超过 BATCH_PRESSURE(10) 的任务量，触发批处理降级规则
REPEAT = 2  # 8 任务 × 2 = 16 个任务，> 10 触发队列积压


async def submit_one(client: httpx.AsyncClient, label: str, prompt: str, max_tokens: int, idx: int) -> dict:
    """提交单个任务并等待最终结果"""
    t0 = time.time()

    # 1. 提交任务（带 max_tokens）
    resp = await client.post(
        f"{GATEWAY_URL}/infer",
        json={"prompt": prompt, "max_tokens": max_tokens},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    # 2. 轮询结果（最多等待 120 秒）
    deadline = time.time() + 120
    data = None
    while time.time() < deadline:
        resp = await client.get(f"{GATEWAY_URL}/infer/{task_id}")
        data = resp.json()
        if data.get("status") == "done":
            break
        await asyncio.sleep(0.3)

    elapsed = time.time() - t0
    return {
        "idx": idx,
        "label": label,
        "task_id": task_id[:8],
        "prompt_len": len(prompt),
        "max_tokens": max_tokens,
        "strategy": data.get("strategy_used", "?"),
        "reason": data.get("strategy_reason", ""),
        "gpu": data.get("gpu_used", "?"),
        "validated": data.get("validated"),
        "confidence": data.get("confidence"),
        "elapsed": round(elapsed, 2),
        "result": (data.get("final_result") or "")[:30],
    }


async def main():
    # 生成任务列表（混合 + 循环以制造并发压力）
    tasks_list = []
    for _ in range(REPEAT):
        for label, prompt, mt in TASKS:
            tasks_list.append((label, prompt, mt))

    print(f"=== 高并发负载测试：共 {len(tasks_list)} 个请求 ===\n")
    print(f"任务构成：short×4 + medium×2 + hard×2，循环 {REPEAT} 次\n")

    t_start = time.time()
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *[
                submit_one(client, label, prompt, mt, i)
                for i, (label, prompt, mt) in enumerate(tasks_list)
            ]
        )
    total_elapsed = time.time() - t_start

    # 统计
    fp16 = [r for r in results if r["strategy"] == "fp16"]
    int4 = [r for r in results if r["strategy"] == "int4"]
    validated = sum(1 for r in results if r["validated"])
    avg_elapsed = sum(r["elapsed"] for r in results) / len(results)

    print("\n=== 测试结果统计 ===")
    print(f"总请求数: {len(results)}")
    print(f"总耗时: {total_elapsed:.2f}s")
    print(f"吞吐量: {len(results) / total_elapsed:.2f} req/s")
    print(f"FP16 任务: {len(fp16)} 个")
    print(f"INT4 任务: {len(int4)} 个")
    print(f"校验通过: {validated}/{len(results)}")
    print(f"平均延迟: {avg_elapsed:.2f}s\n")

    # 按标签统计策略分配（验证规则是否生效）
    print("=== 各任务类型的策略分配 ===")
    for label in ["short", "medium", "hard"]:
        subset = [r for r in results if r["label"] == label]
        if not subset:
            continue
        fp16_n = sum(1 for r in subset if r["strategy"] == "fp16")
        int4_n = sum(1 for r in subset if r["strategy"] == "int4")
        print(f"  {label:8s} ({len(subset)}个) → FP16:{fp16_n}  INT4:{int4_n}")

    # 打印明细（含决策理由）
    print("\n=== 任务明细（含调度理由）===")
    for r in results:
        print(
            f"[{r['idx']:2d}] {r['label']:6s} | {r['strategy']} | {r['gpu']} "
            f"| 校验={r['validated']} | {r['elapsed']}s"
        )
        print(f"       理由: {r['reason']}")
        print(f"       结果: {r['result']}")


if __name__ == "__main__":
    asyncio.run(main())
