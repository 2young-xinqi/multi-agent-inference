# 多智能体协作的推理调度系统

在 Docker 环境下构建基于多智能体（Multi-Agent）协作的推理调度系统：三个 Agent 协同完成对 FP16 / INT4 双精度推理引擎的自动调度、显存监控与一致性校验，提升高并发场景下的推理吞吐量。

## 系统架构

```
 高并发请求 ──▶ gateway(用户接口Agent) ──▶ Redis(任务队列) ──▶ strategy(推理策略Agent)
                                                                  │
                                    短任务→INT4 / 复杂任务→FP16    │
                                          ┌───────────────────────┴──────────┐
                                          ▼                                  ▼
                                   vllm-fp16 (GPU0)                  vllm-int4 (GPU0)
                                          └───────────────────────┬──────────┘
                                                                  ▼
                                            validator(一致性校验Agent) ──▶ 返回结果

 monitor(显存监控Agent) ──▶ 每2s采集 nvidia-smi ──▶ Redis ──▶ strategy 做负载决策
```

## 三个核心 Agent

| Agent | 容器名 | 职责 |
|-------|--------|------|
| 用户接口 Agent | `agent-gateway` | 接收请求、投递任务、SSE 返回结果 |
| 推理策略 Agent | `agent-strategy` | 按任务复杂度 + GPU 负载选择 FP16/INT4 |
| 一致性校验 Agent | `agent-validator` | 用另一精度模型交叉复核，标记一致性 |

旁路：`agent-monitor` 显存监控 Agent，每 2s 采集 `nvidia-smi` 写入 Redis。

**调度决策**由 `strategy/main.py` 的 `choose_strategy` 规则控制器完成，优先级为：用户显式指定 → 显存/队列压力降级 INT4 → 高复杂度+长输出 FP16 → 中等复杂度 FP16 → 短任务 INT4。各阈值的取值与调整原因见 [DEPLOYMENT.md](DEPLOYMENT.md) 第 6.3 节。

**一致性校验**由 `validator` 用「另一精度模型」交叉复核：原结果走 FP16 就用 INT4 复核（反之亦然），以 `SequenceMatcher` 计算相似度，≥ 0.25 判定通过，并标记 confidence。

## 目录结构

```
multi-agent-inference/
├── docker-compose.yml      # 7 个服务编排
├── .env.example            # 环境变量说明（变量已配在 compose 中）
├── gateway/                # 用户接口 Agent（FastAPI + SSE）
├── strategy/               # 推理策略 Agent（规则控制器）
├── validator/              # 一致性校验 Agent
├── monitor/                # 显存监控 Agent
├── client/                 # 高并发负载测试脚本
└── models/                 # 模型权重（下载生成，不入库）
```

## 快速开始

```bash
# 1. 下载模型权重（约 1.7GB，详见 DEPLOYMENT.md 步骤 1）
#    FP16 引擎：Qwen2-0.5B-Instruct
#    INT4 引擎：Qwen2-0.5B-Instruct-GPTQ-Int4

# 2. 构建并启动全部服务
docker compose up -d --build

# 3. 提交任务验证
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "什么是 Transformer 模型？"}'
# 返回 {"task_id":"...", "status":"dispatched"}

# 4. 查询结果（替换为返回的 task_id）
curl http://localhost:8080/infer/<task_id>
```

环境要求：Docker 24.0+ / Compose v2.0+、NVIDIA GPU + Container Toolkit、显存 ≥ 6GB（单卡模式）。本项目在 **WSL2 + Docker Desktop** 下验证通过。

高并发演示（`client/load_test.py`，16 并发）、日志观察方法与预期输出见 [DEPLOYMENT.md](DEPLOYMENT.md) 第 5 节。

## 文档

| 文档 | 内容 |
|------|------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | 环境要求、完整部署流程（含 ModelScope 下载模型）、演示流程、单卡显存适配、踩坑记录、调度阈值说明、验收标准 |
