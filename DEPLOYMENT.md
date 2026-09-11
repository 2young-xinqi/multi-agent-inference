# 多智能体协作的推理调度系统 — Docker 部署流程

> 项目二：在 Docker 环境下构建基于多智能体（Multi-Agent）协作的推理调度系统，实现对 FP16/INT4 双精度推理引擎的自动调度、显存监控与一致性校验。

---

## 1. 系统架构

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

**三个核心 Agent**：

| Agent | 容器名 | 职责 |
|-------|--------|------|
| 用户接口 Agent | `agent-gateway` | 接收请求、投递任务、返回结果（FastAPI + SSE） |
| 推理策略 Agent | `agent-strategy` | 规则控制器选择 FP16/INT4（参考 ModeSwitch-LLM） |
| 一致性校验 Agent | `agent-validator` | 用另一精度模型交叉复核，标记一致性 |

**旁路**：`agent-monitor` 显存监控 Agent，每 2s 采集 `nvidia-smi` 写入 Redis。

---

## 2. 环境要求

| 依赖 | 要求 | 检查命令 |
|------|------|---------|
| Docker | 24.0+ | `docker --version` |
| Docker Compose | v2.0+ | `docker compose version` |
| NVIDIA 驱动 + Container Toolkit | 能跑 nvidia-smi | `nvidia-smi` |
| GPU 显存 | ≥ 6GB（单卡模式） | `nvidia-smi` |
| 磁盘 | ≥ 4GB（模型 + 镜像） | — |

> 注意：本项目在 **WSL2 + Docker Desktop** 环境下运行，WSL2 默认内存 7.44GB 需要特殊适配（见第 6 节踩坑记录）。

---

## 3. 目录结构

```
multi-agent-inference/
├── docker-compose.yml      # 7 个服务编排
├── .env.example            # 环境变量说明（变量已配在 compose 中）
├── gateway/                # 用户接口 Agent（FastAPI）
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── strategy/               # 推理策略 Agent（规则控制器）
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── validator/              # 一致性校验 Agent
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── monitor/                # 显存监控 Agent
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
└── client/
    ├── load_test.py        # 高并发负载测试脚本
    └── requirements.txt
```

---

## 4. 完整部署流程（从零开始）

### 步骤 1：下载模型（ModelScope）

由于 HuggingFace（含 hf-mirror）连接不稳定，使用 **ModelScope（阿里魔搭）** 下载 Qwen 模型。

以下命令在**项目根目录**执行（`cache_dir` 为相对路径，下载后会生成 `models/models/...` 的目录层级）：

```powershell
# 安装 modelscope
pip install modelscope

# 下载 FP16 模型（约 1GB）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B-Instruct', cache_dir='models')"

# 下载 INT4 量化模型（约 0.7GB，GPTQ）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4', cache_dir='models')"
```

下载后目录结构（ModelScope 缓存格式）：

```
models/models/
├── Qwen--Qwen2-0.5B-Instruct/snapshots/master/           # FP16
└── Qwen--Qwen2-0.5B-Instruct-GPTQ-Int4/snapshots/master/  # INT4
```

### 步骤 2：构建并启动服务

```powershell
cd multi-agent-inference

# 构建镜像 + 后台启动全部服务
docker compose up -d --build
```

### 步骤 3：查看服务状态

```powershell
docker compose ps
```

预期 7 个服务：

| 容器 | 作用 | 端口 |
|------|------|------|
| agent-redis | 消息队列 | 6379 |
| agent-gateway | 用户接口 Agent | 8080 |
| agent-strategy | 推理策略 Agent | — |
| agent-validator | 一致性校验 Agent | — |
| agent-monitor | 显存监控 Agent | — |
| vllm-fp16 | FP16 推理引擎 | 8001 |
| vllm-int4 | INT4 推理引擎 | 8002 |

### 步骤 4：等待模型加载

```powershell
docker compose logs -f vllm-fp16
# 看到 "Uvicorn running on socket ('0.0.0.0', 8000)" 即就绪

docker compose logs -f vllm-int4
```

验证引擎健康（vLLM 的 `/health` 返回 200 空 body）：

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://localhost:8001/health   # 200
curl.exe -s -o NUL -w "%{http_code}" http://localhost:8002/health   # 200
```

---

## 5. 演示流程

### 5.1 单请求验证

```powershell
# 提交任务
Invoke-RestMethod -Uri http://localhost:8080/infer -Method Post -ContentType "application/json" -Body '{"prompt": "What is Transformer model?"}'

# 查询结果（复制返回的 task_id）
Invoke-RestMethod http://localhost:8080/infer/<task_id>
```

返回字段：`strategy_used`、`gpu_used`、`validated`、`confidence`、`final_result`。

### 5.2 高并发演示（核心）

开 3 个终端观察 Agent 协同：

```powershell
docker compose logs -f strategy    # 调度决策
docker compose logs -f validator   # 一致性校验
docker compose logs -f monitor     # 显存监控
```

第 4 个终端跑负载测试：

```powershell
cd client
pip install httpx
python load_test.py
```

### 5.3 预期输出

**strategy 日志（FP16/INT4 分流）**：

```
[strategy] 任务 xxx → int4 | 短任务(prompt=8)，用 INT4 低延迟
[strategy] 任务 xxx → fp16 | 中等复杂度(prompt=38)，用 FP16
[strategy] 任务 xxx → fp16 | 高复杂度(prompt=128)+长输出(512)，用 FP16 保质量
```

**validator 日志（一致性校验）**：

```
[validator] 任务 xxx 校验完成: validated=True, 相似度=0.870, 置信度=high
```

**monitor 日志（显存监控）**：

```
[monitor] GPU 显存占用: {'gpu0': 0.90}   # 高并发时升到 90%
```

---

## 6. 关键配置与踩坑记录

### 6.1 单卡 6GB 显存适配

两个 vLLM 引擎共用 GPU 0，必须限制显存：

```yaml
vllm-fp16:
  command: >
    --max-model-len 1024          # 上下文长度减半
    --gpu-memory-utilization 0.50 # 预留 3GB
    --max-num-seqs 8              # 关键：减少激活值预留
    --swap-space 1                # CPU swap 降到 1GB
    --enforce-eager               # 禁用 CUDA graph
    --dtype float16

vllm-int4:
  command: >
    --max-model-len 1024
    --gpu-memory-utilization 0.30 # 预留 1.8GB
    --max-num-seqs 8
    --swap-space 1
    --quantization gptq
    --enforce-eager
```

### 6.2 踩坑记录（按解决顺序）

| # | 错误现象 | 根因 | 解决方案 |
|---|---------|------|---------|
| 1 | 模型下载 SSL 失败 | hf-mirror 连不上 | 改用 ModelScope |
| 2 | `GPU blocks: 0` | `--max-num-seqs` 默认 256，激活值预留过大 | 降到 8 |
| 3 | `No heartbeat from MQLLMEngine` | WSL2 内存 7.44GB，swap 4GB×2 超限 | `--swap-space 1` + `shm_size 2gb` |
| 4 | 所有任务走 INT4 | 显存压力阈值 0.85 误判（单卡静态占用 90%） | 调高到 0.98 |
| 5 | medium 任务被判短任务 | `PROMPT_MEDIUM=200` 太高 | 调到 20 |
| 6 | 队列积压全降级 | `BATCH_PRESSURE=10` 太低 | 调到 50 |

### 6.3 规则控制器阈值（`strategy/main.py`）

```python
PROMPT_MEDIUM = 20    # >20 字符 → FP16
PROMPT_HARD = 100     # >100 字符 → 高复杂度
LONG_OUTPUT = 200     # max_tokens > 200 → 长输出
MEM_PRESSURE = 0.98   # 显存压力阈值（单卡静态占用 90%，调高避免误判）
BATCH_PRESSURE = 50   # 队列积压阈值（16 并发不触发）
```

调度规则（参考 ModeSwitch-LLM）：

1. 用户显式指定精度 → 遵从
2. 显存压力高 → 降级 INT4
3. 队列积压多 → 降级 INT4
4. 高复杂度 + 长输出 → FP16 保质量
5. 中等复杂度 → FP16
6. 短任务 → INT4 低延迟

### 6.4 双卡模式切换

`docker-compose.yml` 中默认单卡（两个引擎共用 GPU 0，故显存阈值调高到 0.98）。若有多卡，把 INT4 引擎绑到另一张卡：

```yaml
vllm-int4:
  environment:
    - NVIDIA_VISIBLE_DEVICES=1    # 改为 GPU 1
```

改用双卡后每张卡的静态占用下降，可适当调高两个引擎的 `--gpu-memory-utilization`（如 0.85），并相应下调 `MEM_PRESSURE` 使显存压力判定重新生效。

---

## 7. 停止与清理

```powershell
# 停止所有服务（保留数据）
docker compose down

# 停止并删除数据卷
docker compose down -v

# 彻底清理（含镜像）
docker compose down -v --rmi all
```

---

## 8. 验收标准

| 核心任务 | 验收点 |
|---------|--------|
| 角色定义 | 3 个 Agent 独立容器运行 |
| 通信框架 | Redis 消息队列传递任务 |
| 推理优化 | strategy 日志显示 FP16/INT4 正确分流 |
| 高并发演示 | load_test.py 16 并发，队列积压可见 |
| 显存监控 | monitor 日志显示 GPU 占用波动 |
| 一致性校验 | validator 日志显示相似度 + validated 判定 |
