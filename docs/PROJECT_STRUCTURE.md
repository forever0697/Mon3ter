# Mon3ter — 项目结构说明

> 本地 AI 桌面伴侣工程，方案 B（Python 后端 + Electron 前端 + WebSocket 通信）
> 当前阶段: Phase 1 — 核心骨架搭建

---

## 顶层目录

```
Mon3ter/                              # 项目根目录
│
├── main.py                           # 🚀 应用入口，生命周期管理
├── server.py                         # 📡 WebSocket 服务端，前后端通信管道
├── test_cli.py                       # ⌨️  CLI 测试客户端（Phase 1-3 交互界面）
├── config.yaml                       # ⚙️ 全局配置中心（所有可调参数）
├── requirements.txt                  # 📦 Python 依赖清单
│
├── mon3ter/                          # 🐍 所有 Python 源码（一个包）
│   ├── __init__.py
│   ├── core/          (推理引擎)
│   ├── personality/   (人格系统)
│   ├── memory/        (记忆系统)
│   ├── conversation/  (对话管理器)
│   ├── web/           (联网抓取)
│   ├── learning/      (能力学习 — Phase 3)
│   ├── voice/         (语音交互 — Phase 5)
│   └── avatar/        (2D 形象 — Phase 4)
│
├── docs/                              # 📖 项目文档
│   ├── PROJECT_STRUCTURE.md           #   你正在看的文件
│   └── Technical_Chain.md             #   完整技术方案
│
├── A_Test/                            # 🧪 模块验证脚本（不参与生产代码）
├── frontend/                          # 🖼️ Electron 前端 (Phase 4)
├── models/                            # 📥 离线模型文件
├── data/                              # 💾 运行时数据（自动生成）
└── .venv/                             # 🐍 Python 虚拟环境
```

> **设计思路**: 根目录只留入口文件和资源目录。所有 Python 代码都在 `mon3ter/` 包里，`from mon3ter.core.xxx import ...` 统一前缀，一目了然。

---

## 核心模块详解

所有模块都在 `mon3ter/` 包下，import 路径以 `mon3ter.` 开头。

### `mon3ter/core/` — 推理引擎

| 文件 | 作用 |
|------|------|
| `engine.py` | 封装 llama-cpp-python，提供模型加载、`chat()` 阻塞推理、`chat_stream()` 流式推理、卸载。全局单例。 |
| `types.py` | 全局数据类型定义。`Message`、`ModelConfig`、`InferenceRequest`、`MemoryItem`、`Emotion` 等所有模块共享的数据结构。 |

**核心职责**: 唯一持有模型实例的模块。其他模块不直接碰 llama.cpp。

---

### `mon3ter/personality/` — 人格系统

| 文件 | 作用 |
|------|------|
| `character.yaml` | 角色卡。定义名字、性格、说话风格、情绪触发规则。用户可直接编辑换皮。 |
| `character.py` | 加载 YAML，根据当前上下文动态拼装 System Prompt。 |
| `emotion_state.py` | 情绪状态机。基于规则的快速情绪切换（不依赖 LLM 判断）。 |

**核心职责**: 回答"她是谁"和"她现在心情怎样"。输入对话上下文，输出拼好的 System Prompt。

---

### `mon3ter/memory/` — 记忆系统

| 文件 | 作用 |
|------|------|
| `vector_store.py` | ChromaDB 向量库封装。存储/检索对话摘要的 embedding。 |
| `models.py` | SQLAlchemy ORM 表定义。`conversation_summaries`、`user_facts`、`chat_logs` 三张表。 |
| `retrieval.py` | 混合检索。向量相似度 + SQLite FTS 关键词 + 结构化事实，加权排序。 |
| `summarizer.py` | 调用 LLM 对对话做摘要，对旧记忆做压缩合并。 |

**核心职责**: 回答"她记得什么"。对话后写入，对话前检索。

---

### `mon3ter/conversation/` — 对话管理器

| 文件 | 作用 |
|------|------|
| `orchestrator.py` | **总指挥**。持有所有模块引用，编排一次对话的完整流程：收消息→查记忆→拼 prompt→推理→存记忆→回复。 |
| `session.py` | 当前对话的上下文窗口管理。滑动窗口最多 20 轮。 |
| `greeting.py` | 开机问候生成。根据时间、天气、用户称呼拼问候语。 |
| `idle_detector.py` | 用户空闲检测。通过 Win32 API 检测键鼠空闲时间，触发主动搭话。 |
| `topic_pool.py` | 主动话题池。固定话题 + 联网话题 + 用户相关话题，24h 内不重复。 |

**核心职责**: 系统的总调度中心。它自己不干具体活（不推理、不查库、不拼 prompt），只编排顺序和传递数据。

---

### `mon3ter/web/` — 联网抓取

| 文件 | 作用 |
|------|------|
| `weather.py` | 从 wttr.in 获取天气数据。缓存 1 小时。 |
| `news.py` | 新闻/热搜抓取。支持 RSS 和页面解析。缓存 2 小时。 |
| `scheduler.py` | APScheduler 定时任务管理。天气每小时、新闻每 2 小时、每日摘要早 8 点。 |
| `cache.py` | 本地 JSON 缓存读写。网络不可用时返回过期缓存 + 标注时间。 |
| `browser.py` | Playwright 动态页面渲染。用户主动请求时按需启动 headless Chromium。 |

**核心职责**: 所有外部数据的入口。网络不可用时优雅降级，不影响离线核心对话。

---

### 未来模块（Phase 3-5，占位）

| 目录 | 阶段 | 说明 |
|------|------|------|
| `mon3ter/learning/` | Phase 3 | RAG 知识库注入 + LoRA 偏好微调 + 数据集构建 |
| `mon3ter/avatar/` | Phase 4 | Live2D 渲染桥接 + 情绪→动作映射 |
| `frontend/` | Phase 4 | Electron + Vue3 桌面壳 + Live2D 画布 |
| `mon3ter/voice/` | Phase 5 | faster-whisper 语音识别 + GPT-SoVITS 语音合成 + 唤醒词 |

---

## 非代码目录

| 目录 | 说明 |
|------|------|
| `A_Test/` | **模块验证脚本**。每个新依赖安装后在这里写快速验证。如现有 `Test.py` 验证大模型和嵌入模型加载。不参与生产运行，不修改。 |
| `models/llm/` | GGUF 量化模型。`qwen2.5-7b-instruct-q4_k_m.gguf` (4.4GB) ✅ 已就位。 |
| `models/embedding/` | BGE-M3 嵌入模型 (2.2GB) ✅ 已就位。 |
| `models/asr/` | 语音识别模型（Phase 5 预留）。 |
| `models/tts/` | 语音合成模型（Phase 5 预留）。 |
| `data/chroma/` | ChromaDB 向量库持久化文件。程序自动生成。 |
| `data/summaries.db` | SQLite 数据库。程序自动生成。 |
| `data/cache/` | 联网数据缓存 JSON。程序自动生成。 |
| `data/logs/` | 对话日志。 |
| `.venv/` | Python 3.13 虚拟环境，所有依赖已安装。 |

---

## 关键设计原则

1. **源码统一在 `mon3ter/` 包**: 根目录干净，import 路径统一 `from mon3ter.xxx import ...`
2. **模块单向依赖**: `core` ← `memory` ← `personality` ← `conversation`。越往右越"高层"，右调左不反向。
3. **推理引擎是黑盒**: 其他模块不直接碰模型，只通过 `engine.chat()` / `engine.chat_stream()` 接口。
4. **对话管理器是总指挥**: 它持有所有模块引用，但自己不干具体活——编排顺序、传递数据。
5. **Test 不碰生产**: `A_Test/` 是独立验证区，生产代码不引用它。
6. **先跑起来再变好看**: Phase 1 是 CLI 对话，Phase 4 才加 GUI，Phase 5 才加语音——每步都是独立可用的增量。
