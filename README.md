# 多智能体协作的推理调度系统

在 Docker 环境下构建基于多智能体（Multi-Agent）协作的推理调度系统，通过三个 Agent 的协同，实现对 FP16/INT4 双精度推理引擎的自动调度、显存监控与一致性校验，提升高并发场景下的推理吞吐量。

## 系统架构

```
 高并发请求 ──▶ gateway(用户接口Agent) ──▶ Redis(任务队列) ──▶ strategy(推理策略Agent)
                                                                  │
                                    简单任务→INT4 / 复杂任务→FP16  │
                                          ┌───────────────────────┴──────────┐
                                          ▼                                  ▼
                                   vllm-fp16 (GPU0)                  vllm-int4 (GPU0/1)
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

## 目录结构

```
multi-agent-inference/
├── docker-compose.yml
├── .env.example    # 环境变量模板（复制为 .env 使用）
├── gateway/        # 用户接口 Agent
├── strategy/       # 推理策略 Agent
├── validator/      # 一致性校验 Agent
├── monitor/        # 显存监控 Agent
└── client/         # 高并发负载测试脚本
```

## 部署流程

### 1. 前置条件

- Docker + Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit（`nvidia-smi` 可用）
- 至少 6GB 显存（本配置用两个 0.5B 小模型，单卡可运行）

### 2. 预下载模型（可选，推荐）

```bash
docker run --rm \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v $(pwd)/volumes/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.6.3 \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B-Instruct'); snapshot_download('Qwen/Qwen2-0.5B-Instruct-AWQ')"
```

### 3. 启动全部服务

> 环境变量（Redis 地址、两个 vLLM 引擎地址与模型名）已在 `docker-compose.yml` 的
> `environment` 段中配好，可直接启动。如需调整，改 `docker-compose.yml` 即可；
> `.env.example` 列出了各服务读取的全部变量名，供参考。

```bash
cd multi-agent-inference
docker compose up -d --build
```

### 4. 查看状态

```bash
docker compose ps
nvidia-smi   # 确认两个 vLLM 实例各自占用显存
```

### 5. 查看日志

```bash
docker compose logs -f strategy    # 观察调度决策
docker compose logs -f validator   # 观察一致性校验
docker compose logs -f monitor     # 观察显存监控
```

## 演示流程

### 1. 单个请求

```bash
# 提交任务
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "什么是 Transformer 模型？"}'
# 返回 {"task_id":"...", "status":"dispatched"}

# 查询结果
curl http://localhost:8080/infer/<task_id>
```

### 2. 高并发负载测试

```bash
cd client
pip install httpx
python load_test.py
```

脚本会并发提交混合任务（简单 + 复杂），并输出：
- 总吞吐量（req/s）
- FP16 / INT4 任务分配数量
- 校验通过率
- 平均延迟

### 3. 观察三 Agent 协同

在三个终端分别运行：
```bash
docker compose logs -f strategy    # 终端1：调度决策
docker compose logs -f validator   # 终端2：校验结果
docker compose logs -f monitor     # 终端3：显存监控
```

再运行负载测试，可看到任务自动分流到 FP16/INT4、显存占用波动、结果经过校验。

## 调度策略说明

`strategy` Agent 的决策规则（`strategy/main.py` 的 `choose_strategy`）：

1. 用户显式指定精度 → 用指定精度
2. 复杂任务（prompt 长度 > 200）→ FP16 高精度
3. GPU0（FP16 卡）显存占用 > 85% → 降级到 INT4
4. 简单任务 → INT4 低精度（高吞吐）

## 一致性校验说明

`validator` Agent 用「另一精度模型」交叉复核：
- 原结果用 FP16 → 用 INT4 复核（反之亦然）
- 计算两结果相似度（SequenceMatcher）
- 相似度 ≥ 0.25 判定一致，标记 confidence（high/medium/low）

## 双卡模式切换

`docker-compose.yml` 中默认单卡（两引擎共用 GPU 0）。若有多卡，修改：

```yaml
vllm-int4:
  environment:
    - NVIDIA_VISIBLE_DEVICES=1    # 改为 GPU 1
```

并适当调高两个引擎的 `gpu-memory-utilization`（如 0.85）。

## 常见问题

**Q: 模型下载失败（SSL 错误）？**
A: 已配置 `HF_ENDPOINT=https://hf-mirror.com` 国内镜像，若仍失败可先手动预下载模型。

**Q: INT4 AWQ 模型下载失败？**
A: 将 `docker-compose.yml` 中 `vllm-int4` 的 `--model Qwen/Qwen2-0.5B-Instruct-AWQ` 改为 `Qwen/Qwen2-0.5B-Instruct`，并移除 `--quantization awq`（用同一模型模拟双引擎）。

**Q: 单卡显存不足？**
A: 降低两个引擎的 `--gpu-memory-utilization`（如 0.3 / 0.25）。
