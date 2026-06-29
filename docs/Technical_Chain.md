# Mon3ter — 方案B 完整技术链条

> **硬件基准**: i7-14700K + RTX 4070 Super (12GB) + 32GB RAM
> **模型**: Qwen2.5-7B-Instruct Q4_K_M GGUF
> **架构**: Python 后端 + Electron 前端 + WebSocket 通信

---

## 〇、总体架构与模块协作全景图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Electron 前端 (Phase 4)                     │
│  ┌───────────┐  ┌────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ 对话窗口   │  │ Live2D 渲染 │  │ 语音输入按钮 │  │ 系统托盘    │ │
│  └─────┬─────┘  └─────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│        └───────────────┴───────────────┴───────────────┘         │
│                            │ WebSocket (:9876)                     │
├────────────────────────────┼──────────────────────────────────────┤
│                    Python 后端                                     │
│  ┌─────────────────────────┼──────────────────────────────────┐  │
│  │              main.py — 应用主入口 + 生命周期                   │  │
│  │            config.yaml — 全局配置 (模型路径/角色/参数)           │  │
│  └─────────────┬───────────┴──────────────┬───────────────────┘  │
│                │                          │                       │
│  ┌─────────────▼───────────┐  ┌──────────▼────────────────────┐  │
│  │    对话管理器 (Orchestrator)  │  │    WebSocket Server (:9876)    │  │
│  │  ┌─────────────────────┐ │  │  收发对话 / 推送问候 / 事件广播   │  │
│  │  │ 空闲检测 → 主动搭话   │ │  └───────────────────────────────┘  │
│  │  │ 开机问候生成          │ │                                     │
│  │  │ 多轮对话调度          │ │                                     │
│  │  └──┬──────┬──────┬───┘ │                                     │
│  └─────┼──────┼──────┼─────┘                                     │
│        │      │      │                                             │
│  ┌─────▼──┐ ┌─▼──────▼──┐ ┌──────────▼────┐  ┌───────────────┐  │
│  │ 人格引擎 │ │ 记忆系统   │ │ 推理引擎       │  │ 联网抓取模块   │  │
│  │Character│ │ChromaDB   │ │llama-cpp      │  │APScheduler    │  │
│  │Emotion  │ │+SQLite    │ │Qwen2.5-7B     │  │httpx+BS4      │  │
│  │State    │ │记忆检索    │ │流式输出        │  │天气/新闻/RSS  │  │
│  └────┬────┘ └─────┬─────┘ └───────┬───────┘  └───────┬───────┘  │
│       │            │               │                   │          │
│  ┌────▼────────────▼───────────────▼───────────────────▼───────┐  │
│  │                    能力学习模块 (Phase 3)                      │  │
│  │         RAG知识库 / LoRA微调 / 用户偏好收集                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 一次对话的数据流向

```
用户输入 "今天心情不太好"
        │
        ▼
① 对话管理器接收 → 打上时间戳, 话题标签
        │
        ▼
② 记忆系统检索 → "用户上周提到工作压力大" (向量相似度 > 0.8)
        │               "用户喜欢被叫小名'阿杰'"   (结构化事实)
        │
        ▼
③ 人格引擎拼装 System Prompt:
   [角色卡] + [今日情绪:温柔关切] + [相关记忆] + [当前时间/天气]
        │
        ▼
④ 推理引擎推理 (llama-cpp-python, GPU加速, 流式输出):
   "阿杰，听起来你今天不太顺心呢。要不要跟我聊聊发生了什么？"
        │
        ▼
⑤ 对话管理器接收完整回复 → 推送到 WebSocket → 前端显示 + Live2D 表情
        │
        ▼
⑥ 记忆系统异步写入: 向量化本轮对话 → 存入 ChromaDB
```

---

## 一、模块职责与协作契约

### 1.1 推理引擎 (`core/`)

**职责**: 唯一拥有模型实例的模块。对外提供统一的推理接口，其他模块不得直接操作模型。

```
core/
├── engine.py          # 模型加载、推理、资源管理
├── types.py           # 推理请求/响应的 dataclass
├── stream.py          # 流式输出迭代器封装
└── model_config.yaml  # 模型路径、GPU层数、温度等参数
```

**对外接口 (Python API)**:
```python
class InferenceEngine:
    def load_model(config: ModelConfig) -> None: ...
    def chat(
        messages: list[Message],      # OpenAI兼容格式
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str: ...                     # 阻塞式，返回完整回复

    def chat_stream(
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]: ...  # 流式，逐token产出

    def unload_model() -> None: ...   # 释放显存
```

**关键约束**: 全局单例，一次只加载一个模型。不关心对话逻辑、不关心人格——它只是一个"文本进，文本出"的黑盒。

---

### 1.2 人格引擎 (`personality/`)

**职责**: 掌管"她是谁"和"她现在是什么心情"。输入对话上下文，输出拼装好的 System Prompt。

```
personality/
├── character.py       # 加载角色卡、拼装 System Prompt
├── character.yaml     # 角色定义文件（用户可编辑）
├── emotion_state.py   # 情绪状态机
└── emotion_log.json   # 情绪变化日志（持久化）
```

**角色卡 (`character.yaml`) 示例**:
```yaml
name: "Mon3ter"
persona: |
  你是一个温柔、细腻、偶尔冷幽默的AI伴侣。
  你喜欢用短句，带语气词，像朋友一样说话。
  你记得用户告诉你的关于他自己的一切。
  ...
speaking_rules:
  - 永远用口语，不要说"作为一个人工智能"
  - 每句话不超过40个字
  - 适当使用"嗯"、"哈哈"、"诶"等语气词
emotional_states:
  default: "轻松愉快"
  triggers:
    - condition: "用户表达负面情绪"
      shift_to: "温柔关切"
      weight: 0.9
    - condition: "用户长时间未互动后回来"
      shift_to: "想念"
      weight: 0.7
```

**对外接口**:
```python
class PersonalityEngine:
    def build_system_prompt(
        context: ConversationContext,   # 当前对话上下文
        memories: list[MemoryItem],     # 检索到的相关记忆
        emotion: EmotionState,          # 当前情绪
    ) -> str: ...
    
    def update_emotion(
        user_message: str,
        time_since_last: float,         # 距上次对话的秒数
    ) -> EmotionState: ...
```

**协作关系**:
- 被 `对话管理器` 在每次推理前调用，获取 System Prompt
- 从 `记忆系统` 获取相关记忆来丰富 prompt
- 从 `联网模块` 获取时间/天气等环境信息注入 prompt
- **不直接调用推理引擎**——它只产文本，不跑模型

---

### 1.3 记忆系统 (`memory/`)

**职责**: 对话内容的持久化、向量化、检索。是"她记得什么"的全部真相来源。

```
memory/
├── vector_store.py    # ChromaDB 操作封装
├── models.py          # SQLAlchemy ORM (对话摘要、用户画像、事实)
├── retrieval.py       # 混合检索（向量+关键词）与排序
├── summarizer.py      # 调用LLM做对话摘要压缩
└── embeddings.py      # BGE-M3 嵌入模型加载与推理
```

**数据模型 (SQLite)**:
```sql
-- 对话摘要表
conversation_summaries (
    id, timestamp, summary_text,    -- "用户聊了工作压力"
    topics TEXT[],                  -- ["工作","压力"]
    emotion_at_time TEXT,           -- "负面"
    importance FLOAT DEFAULT 0.5    -- 重要性评分
)

-- 用户事实表（结构化记忆）
user_facts (
    id, fact_type TEXT,             -- "preference" / "identity" / "event"
    fact_key TEXT,                  -- "昵称"
    fact_value TEXT,                -- "阿杰"
    source TEXT,                    -- 来自哪次对话
    confidence FLOAT DEFAULT 1.0,
    UNIQUE(fact_key)
)

-- 对话日志（完整原始记录，定期归档）
chat_logs (
    id, timestamp, role, content,
    conversation_id TEXT             -- 一次连续对话的唯一ID
)
```

**对外接口**:
```python
class MemorySystem:
    def store_conversation(
        messages: list[Message],
        summary: str,
        topics: list[str],
    ) -> None: ...
    
    def retrieve(
        query: str,
        top_k: int = 5,
        recency_weight: float = 0.3,    # 时间衰减权重
    ) -> list[MemoryItem]: ...
    
    def get_facts() -> dict[str, str]: ...       # 获取所有用户事实
    def upsert_fact(key: str, value: str) -> None: ...
    
    def periodic_compress() -> None: ...          # 每日摘要压缩
```

**关键设计 — 混合检索**:
```
用户输入 "我今天又被老板骂了"
        │
        ├─→ 向量检索 (ChromaDB): "老板骂" → 语义匹配
        │    找到: "上周用户抱怨过老板苛刻" (相似度0.85)
        │
        ├─→ 关键词检索 (SQLite FTS): "老板" → 精确匹配
        │    找到: "用户老板姓王，部门经理"
        │
        └─→ 结构化事实 (SQLite): 
             找到: "用户职位: 前端开发", "用户讨厌加班"
```

**协作关系**:
- 被 `对话管理器` 在推理前和推理后调用
- 调用 `推理引擎` 做摘要生成（"用一句话总结这段对话"）
- 向 `人格引擎` 提供检索结果
- 独立运行每日压缩任务（APScheduler 触发）

---

### 1.4 对话管理器 (`conversation/`)

**职责**: 整个系统的总指挥。协调所有模块完成一次完整的对话交互。也是空闲检测和主动搭话的触发器。

```
conversation/
├── orchestrator.py    # 核心调度：收消息→查记忆→拼prompt→推理→存记忆→回复
├── idle_detector.py   # 用户空闲检测（鼠标/键盘事件）
├── greeting.py        # 开机问候模板 + 语境生成
├── topic_pool.py      # 主动搭话话题池
└── session.py         # 会话管理（多轮对话上下文窗口）
```

**Orchestrator 核心流程** (一次对话转):
```python
class ConversationOrchestrator:
    def __init__(self, engine, personality, memory, web):
        self.engine = engine          # 推理引擎
        self.personality = personality # 人格引擎
        self.memory = memory          # 记忆系统
        self.web = web                # 联网模块

    async def handle_message(self, user_input: str) -> str:
        # 1. 预处理
        topics = self._extract_topics(user_input)        # 提取话题
        emotion = self.personality.update_emotion(user_input, ...)
        
        # 2. 检索记忆
        memories = self.memory.retrieve(user_input, top_k=5)
        facts = self.memory.get_facts()
        
        # 3. 拼装System Prompt
        system = self.personality.build_system_prompt(
            context=self.current_session,
            memories=memories,
            emotion=emotion,
            facts=facts,
            env_info=self.web.get_env_summary(),  # 天气/时间/日期
        )
        
        # 4. 推理 (流式)
        full_messages = [SystemMessage(system)] + self.current_session.messages + [UserMessage(user_input)]
        reply = ""
        async for token in self.engine.chat_stream(full_messages):
            reply += token
            yield token                                # 逐字推送到前端
        
        # 5. 后处理 — 异步写入记忆
        asyncio.create_task(self._store_conversation(user_input, reply, topics))
        
        return reply
    
    async def _store_conversation(self, user, reply, topics):
        """异步保存对话（不阻塞用户等待回复）"""
        summary = self.engine.chat([
            SystemMessage("用一句话总结以下对话要点"),
            UserMessage(f"用户: {user}\nAI: {reply}")
        ])
        self.memory.store_conversation([user, reply], summary, topics)
        self.current_session.append(user, reply)
```

**空闲检测逻辑**:
```python
class IdleDetector:
    """
    使用 Windows API (ctypes 调用 user32.dll GetLastInputInfo)
    检测用户最后一次键盘/鼠标操作距今的秒数。
    
    阶梯式触发:
      - 15分钟空闲 → 简短提醒 ("坐很久了哦，起来喝杯水？")
      - 30分钟空闲 → 闲聊开启 ("阿杰，你在干嘛呢？")
      - 60分钟空闲 → 趣味话题 ("给你讲个冷知识...") 
    
    每小时最多主动搭话 1 次，避免骚扰。
    """
```

**协作关系**:
- 它是唯一持有所有模块引用的协调者
- 不亲自推理（委托推理引擎）
- 不亲自查记忆（委托记忆系统）
- 不亲自拼 prompt（委托人格引擎）
- 它的职责是**编排顺序和传递数据**

---

### 1.5 联网抓取模块 (`web/`)

**职责**: 所有外部网络数据的入口。定时抓取，缓存到本地，离线时优雅降级。

```
web/
├── weather.py         # 天气获取 (wttr.in / 和风天气)
├── news.py            # 新闻/热搜抓取
├── scheduler.py       # APScheduler 定时任务管理
├── cache.py           # 本地缓存 (JSON文件)
└── browser.py         # Playwright 动态页面 (按需)
```

**定时任务**:
```python
# scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# 每小时更新一次天气
scheduler.add_job(weather.update, 'interval', hours=1)

# 每2小时抓一次新闻热搜
scheduler.add_job(news.fetch_hot_topics, 'interval', hours=2)

# 每天早上8点触发一次数据汇总 (用于开机问候)
scheduler.add_job(daily_summary, 'cron', hour=8)
```

**对外接口**:
```python
class WebModule:
    def get_weather(city: str = "北京") -> WeatherInfo | None: ...
        # 网络不可用时返回 None，调用方自行降级
    
    def get_news_headlines(limit: int = 5) -> list[NewsItem]: ...
    
    def get_env_summary() -> str:
        # 返回一段注入System Prompt的环境信息
        # "现在是2026年6月27日周六下午3点，北京晴，28°C。
        #  今日热搜: XXX, YYY..."
    
    def fetch_url(url: str, use_browser: bool = False) -> str: ...
        # 用户要求爬取特定网页时使用
```

**缓存策略**:
- 天气数据缓存 1 小时
- 新闻数据缓存 2 小时
- 所有缓存写入 `data/cache/` 下的 JSON 文件
- 读取时优先返回缓存，网络失败时使用过期缓存 + 标注时间

**协作关系**:
- 被 `对话管理器` 获取环境信息（注入 System Prompt）
- 被 `对话管理器` 获取主动搭话话题
- 被 `开机问候模块` 获取当天天气/日期摘要
- **不依赖任何其他模块**——纯数据获取

---

### 1.6 能力学习模块 (`learning/`)

**职责**: 让系统从交互数据中学到新东西。分三条路径。

```
learning/
├── rag.py             # RAG知识库：文档→向量→对话检索
├── preference.py      # 用户偏好收集：从纠正中学习
├── dataset.py         # 数据集构建：对话→微调数据
├── trainer.py         # LoRA微调流水线
└── knowledge_base/    # 用户提供的文档/知识
```

**路径一：RAG 知识注入**

用户提供文档（pdf/txt/md）→ 分块(chunk) → BGE-M3 向量化 → 存入专门的知识库 Collection → 对话时检索注入。

```
用户: "把这份产品手册给你，以后客户问问题你能回答"
        │
        ▼
① 文档分块器 (langchain.text_splitter) → 每块 500 字，重叠 50 字
② BGE-M3 → 向量化每个块
③ 存入 ChromaDB Collection: "user_knowledge"
④ 以后每次对话 → 同时检索 "memory" 和 "user_knowledge" 两个库
⑤ 相关文档块注入 System Prompt → 模型基于文档回答
```

**路径二：用户偏好学习**

```
用户: "不对，我不喜欢被叫全名，叫我阿杰"
        │
        ▼
① 偏好检测器识别到纠正意图
② 提取事实: fact_key="称呼", fact_value="阿杰"
③ upsert 到 user_facts 表
④ 下次 System Prompt 自动包含: "用户希望你叫他'阿杰'"
```

纠正是最强的学习信号——比正面示例更有价值。

**路径三：LoRA 微调**

定期（每周/每月）将收集的对话数据转为微调数据集，训练 LoRA 权重。

```python
# trainer.py 核心流程
"""
1. 从 chat_logs 导出高质量对话（用户评分高的、纠正过的）
2. 转换为 Alpaca/ShareGPT 格式:
   {"instruction": "...", "input": "...", "output": "..."}
3. 使用 LLaMA-Factory 或 unsloth 做 LoRA 微调:
   - rank=16, alpha=32
   - lora_target: q_proj, v_proj, k_proj, o_proj
   - 学习率 2e-4, 3 epochs
   - 12GB VRAM 足够 (LoRA只训练 ~0.1% 参数)
4. 产出一个 <100MB 的 LoRA adapter
5. llama-cpp 目前不支持 LoRA 热加载，
   方案: 用 llama.cpp 的 convert 工具将 LoRA merge 回 GGUF
   或: 切换到 vLLM/Aphrodite 引擎 (原生支持 LoRA 热插拔)
"""
```

**协作关系**:
- RAG 依赖 `记忆系统` 的 ChromaDB 和 Embedding 模型
- 偏好学习写入 `记忆系统` 的 user_facts 表
- LoRA 训练读取 `记忆系统` 的 chat_logs
- **不直接参与实时对话**——是离线/后台任务

---

### 1.7 前端 (Phase 4, `frontend/`)

**职责**: 用户看到和触摸到的一切。只是一个壳——所有智能在后端。

```
frontend/
├── electron-app/
│   ├── main.ts           # Electron 主进程
│   ├── preload.ts        # 安全的 IPC 桥接
│   ├── renderer/         # Vue3 渲染进程
│   │   ├── App.vue
│   │   ├── components/
│   │   │   ├── ChatWindow.vue      # 对话窗口
│   │   │   ├── Live2DCanvas.vue    # Live2D 渲染画布
│   │   │   ├── SystemTray.vue      # 系统托盘
│   │   │   └── Settings.vue        # 设置面板
│   │   └── stores/
│   │       └── chat.ts             # Pinia 对话状态管理
│   └── live2d/                     # Live2D 模型文件
│       └── mon3ter.model3.json
└── package.json
```

**与后端的通信**:
```
Electron (Vue3)
    │
    │  WebSocket ws://127.0.0.1:9876
    │
    ├─→ {"type": "chat", "content": "今天心情不太好"}
    │
    ├←─ {"type": "token",  "content": "阿"}     ← 流式逐字
    ├←─ {"type": "token",  "content": "杰"}
    ├←─ {"type": "token",  "content": "，" }
    ├←─ ...
    ├←─ {"type": "done",   "message_id": "..."}  ← 一句话结束
    │
    ├←─ {"type": "emotion", "state": "关切"}     ← 情绪状态同步
    ├←─ {"type": "greeting","content": "早安..."}  ← 开机问候推送
    └←─ {"type": "idle_chat","content": "你在干嘛呢？"} ← 主动搭话
```

**前端不做的**:
- ❌ 不直接加载模型
- ❌ 不访问数据库
- ❌ 不爬网页
- ✅ 只负责渲染和用户交互

---

## 二、完整技术链条：从零到可运行

### Step 1: 环境准备

```bash
# 1. 安装 Python 3.11 (Windows Store 或官网)
python --version  # 确认 3.11+

# 2. 创建虚拟环境
cd Mon3ter
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. 安装核心依赖
pip install uv           # 快速包管理
uv pip install \
  llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  # ⚠️ Windows GPU 版本需额外指定 CUDA 参数，见下文

# 4. 基础工具链
uv pip install \
  chromadb \              # 向量数据库
  sqlalchemy \            # ORM
  aiohttp \               # WebSocket 服务端 (或 websockets)
  httpx \                 # HTTP 客户端
  beautifulsoup4 \        # HTML 解析
  apscheduler \           # 定时任务
  pydantic \              # 数据校验
  pyyaml \                # 配置文件
  sentence-transformers   # BGE-M3 embedding (CPU即可)
```

**llama-cpp-python 的 Windows GPU 安装** (关键步骤):
```bash
# 需要先安装 CUDA Toolkit 12.x + Visual Studio Build Tools
# 设置环境变量后编译安装
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:CUDACXX="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
uv pip install llama-cpp-python --no-cache-dir --force-reinstall
```

> **理由**: llama-cpp-python 的 GPU 版本需要从源码编译，必须指定 CUDA 编译标志。如果你不想折腾编译，也可以用 Ollama 作为推理后端（它自带编译好的 CUDA 版本），然后在 Python 中通过 OpenAI 兼容 API 调用。详见 Step 1b 备选方案。

### Step 1b: 备选——用 Ollama 代替手动编译

如果 llama-cpp-python Windows GPU 编译遇到问题，用 Ollama 更省心：

```bash
# 安装 Ollama (ollama.com 下载 Windows 版)
ollama pull qwen2.5:7b-instruct-q4_K_M

# Python 端通过 OpenAI 兼容接口调用
# pip install openai
# client = OpenAI(base_url="http://localhost:11434/v1")
# response = client.chat.completions.create(model="qwen2.5:7b", messages=...)
```

**两方案对比**:

| | llama-cpp-python | Ollama |
|---|---|---|
| 控制粒度 | 完全控制（GPU层数/上下文/量化） | 较粗（Ollama 管理） |
| 安装难度 | Windows需编译 | 一键安装 |
| 性能 | 相同 | 相同（底层都是 llama.cpp） |
| 流式输出 | 原生支持 | 支持(SSE) |
| LoRA热加载 | 不支持 | 不支持 |
| 推荐场景 | 需要精细控制 | 快速启动 |

> 本文档默认使用 **llama-cpp-python** 方案（控制力最强），但你可以随时切换。

---

### Step 2: 下载模型

```bash
# 创建模型目录
mkdir models\llm models\embedding models\asr models\tts

# 下载 Qwen2.5-7B-Instruct GGUF (推荐 Q4_K_M, 约 4.7GB)
# 从 HuggingFace 或 ModelScope 下载:
# huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# 文件名类似: qwen2.5-7b-instruct-q4_k_m.gguf

# 下载 BGE-M3 embedding 模型 (约 2.2GB)
# huggingface.co/BAAI/bge-m3
# 用 sentence-transformers 自动下载，或手动放到 models/embedding/
```

**为什么选 Q4_K_M 量化？** 这是 llama.cpp 的"甜点"量化——质量损失 <2%（几乎无感），显存从 14GB → 5.5GB。Q5 质量提升极小但显存多占 1.5GB，不值得。

---

### Step 3: 实现推理引擎 (`core/engine.py`)

这是整个系统第一个要写的文件，也是其他所有模块的依赖。

```python
# core/engine.py — 核心结构
# 不逐行写代码，给出关键实现思路

"""
class InferenceEngine:
    核心要点:
    1. 使用 Llama.from_pretrained() 加载 GGUF 模型
    2. n_gpu_layers=-1 表示所有层都放 GPU (12GB 显存足够)
    3. n_ctx=8192 设置上下文窗口 (Qwen2.5-7B 原生支持 32K, 但 8K 够用且省显存)
    4. chat_format="chatml" — Qwen2.5 使用 ChatML 格式
    5. 流式输出: create_chat_completion(..., stream=True)
       遍历 generator, 每个 chunk 的 delta 就是新 token
    6. 线程安全: 推理放在独立线程 (llama.cpp 是同步的), 
       用 queue 把 token 传回主线程
    7. 全局单例: 模块级 _engine = None, get_engine() 懒加载
"""
```

**验证这一步**:
```python
# 写一个 test_engine.py
from core.engine import InferenceEngine
engine = InferenceEngine("models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
engine.load()
reply = engine.chat([{"role": "user", "content": "你好，请用一句话介绍自己"}])
print(reply)
# 预期输出: "你好！我是一个AI助手，乐于回答问题和聊天。"
```

> 这一步做完，你就有了一个本地运行的 LLM。后续所有模块都是在这个基础上"穿衣服"。

---

### Step 4: 实现人格引擎 (`personality/`)

**为什么需要独立的人格引擎而不是直接把角色写进 prompt？**
因为人格会变化——情绪、上下文、用户偏好都会影响 System Prompt 的内容。独立模块让这些变化可管理、可调试。

```python
# personality/character.py — 核心结构

"""
class PersonalityEngine:
    def build_system_prompt(self, context, memories, emotion, facts, env_info):
        # 拼装顺序固定:
        segments = []
        
        # 第1段: 角色定义 (从 character.yaml 加载, 永远不变)
        segments.append(self.char_config['persona'])
        
        # 第2段: 当前环境 (时间、天气、日期 — 让模型知道"现在是什么时候")
        segments.append(env_info)  
        # "现在是2026年6月27日周六下午3点17分，北京，晴，28°C"
        
        # 第3段: 关于用户的事实 (从 user_facts 表)
        segments.append(f"关于用户: {format_facts(facts)}")
        # "用户叫阿杰，前端开发，不喜欢加班，有只猫叫'团子'"
        
        # 第4段: 相关记忆 (从记忆系统检索)
        segments.append(f"最近相关记忆: {format_memories(memories)}")
        # "上周用户提到项目deadline很紧张"
        
        # 第5段: 当前情绪
        segments.append(f"你现在的心情: {emotion.description}")
        # "你现在的心情: 温柔关切，因为用户刚才表达了负面情绪"
        
        # 第6段: 说话规则 (从 character.yaml 加载)
        segments.append(self.char_config['speaking_rules'])
        
        return "\n\n".join(segments)
"""
```

**情绪状态机的简单实现**:
```python
# personality/emotion_state.py
"""
状态转移不是 LLM 判断的（太慢），而是基于规则的快速切换:

当前状态: "轻松愉快"
    │
    ├─ 用户说"好烦/难受/不开心" → 切换到 "温柔关切" (置信度 0.9)
    ├─ 超过30分钟没说话 → 切换到 "安静"     (置信度 0.7)
    ├─ 用户回来聊了10轮 → 切换回 "轻松愉快"
    └─ 用户分享了有趣的事 → 切换到 "好奇兴奋"

状态机用简单的 if-else 规则 + 关键词匹配实现。
不需要 LLM 参与情绪判断——延迟太大，而且不可控。
"""
```

---

### Step 5: 实现记忆系统 (`memory/`)

这是最容易被低估但实际最重要的模块。"她不记得"比"她回答不好"更致命。

**嵌入模型的选择**:
```
BGE-M3 (BAAI/bge-m3)
- 中文效果最好的开源嵌入模型之一
- 支持 1024 维向量
- 用 sentence-transformers 加载, CPU 推理即可
- 每条约 20ms, 不占用 GPU
```

**向量存储初始化**:
```python
# memory/vector_store.py
import chromadb

client = chromadb.PersistentClient(path="data/chroma")

# 两个 Collection 做逻辑隔离
memory_collection = client.get_or_create_collection(
    name="conversation_memory",
    metadata={"hnsw:space": "cosine"}
)

knowledge_collection = client.get_or_create_collection(
    name="user_knowledge",    # RAG 知识库 (Phase 3)
    metadata={"hnsw:space": "cosine"}
)
```

**记忆写入** (每次对话结束):
```python
"""
1. 调用推理引擎做摘要: "用一句话总结这段对话要点，15字以内"
   → "用户抱怨工作压力大，AI安慰并建议散步"

2. 调用推理引擎提取事实: "提取用户透露的关于自己的信息，JSON格式"
   → {"称呼": "阿杰", "情绪": "低落", "话题": "工作压力"}

3. BGE-M3 向量化摘要文本

4. 写入 ChromaDB: 
   - id: 时间戳
   - embedding: 摘要向量
   - metadata: {timestamp, topics, emotion, importance}

5. 更新 user_facts 表 (如果有新事实)
"""
```

**记忆检索** (每次对话前):
```python
"""
1. BGE-M3 向量化用户最新消息
2. ChromaDB 查询 top-10 相似记忆
3. 混合排序:
   final_score = similarity * 0.6 + recency * 0.3 + importance * 0.1
   - similarity: 向量余弦相似度
   - recency: e^(-天数/7)  — 7天半衰期，越新越高
   - importance: 从 metadata 读取
4. 返回 top-5
"""
```

**记忆压缩** (每日定时):
```python
"""
问题: 一周后的对话记忆可能有几百条,检索越来越慢、噪声越来越多。

方案: 每天凌晨执行一次压缩:
1. 按话题聚类 (用向量相似度自动聚)
2. 每个话题簇 → 调用 LLM: "以下是关于[话题]的多段对话,请合并成一段摘要"
3. 删除原始对话向量，只保留合并后的摘要
4. 对话日志原文本保留在 SQLite (不做向量化的冷存储)

效果: 向量库始终保持 ~200条 以内的规模，检索 <50ms
"""
```

---

### Step 6: 实现对话管理器 (`conversation/orchestrator.py`)

这是把所有模块串起来的"导演"。实现顺序:

```
1. 先写 session.py — 管理当前对话的上下文窗口
   - 最多保留 20 轮 (约 4000 tokens)
   - 滑动窗口: 满了就丢弃最早的一轮
   - 角色: user / assistant / system

2. 再写 orchestrator.py — 核心调度
   - 收到消息 → 检索记忆 → 拼 prompt → 推理 → 存记忆
   - 见上文伪代码

3. 再写 idle_detector.py — 空闲检测
   - ctypes.windll.user32.GetLastInputInfo()
   - 每秒检查一次
   
4. 再写 greeting.py — 开机问候
   - 读取当前时间、天气缓存
   - 模板: "早上好{称呼}，今天是{星期}{日期}，{天气描述}。{一句贴心话}"
   - 贴心话从 topic_pool 随机抽取

5. 最后写 topic_pool.py — 话题池
   - 50条固定话题 (冷知识、小故事、趣味问答)
   - 联网话题 (当日新闻、热搜)
   - 用户相关话题 (生日提醒、习惯提醒)
   - 去重: 24小时内不重复同一话题
```

---

### Step 7: 实现 WebSocket 服务端

```python
# server.py — 用 aiohttp 或 websockets 库
"""
WebSocket 协议 (JSON格式):

前端 → 后端:
  {"type": "chat",     "content": "用户消息"}
  {"type": "ping",     "content": ""}
  {"type": "command",  "action": "fetch_url", "url": "..."}

后端 → 前端:
  {"type": "token",    "content": "一"}        # 流式输出每个token
  {"type": "done",     "message_id": "xxx"}    # 消息结束
  {"type": "emotion",  "state": "关切"}        # 情绪状态更新
  {"type": "greeting", "content": "早安..."}    # 开机问候
  {"type": "idle",     "content": "在干嘛呢？"} # 主动搭话
  {"type": "error",    "content": "..."}        # 错误信息
"""
```

**并发模型**:
```
main.py
├── asyncio event loop (主线程)
│   ├── WebSocket Server (处理前后端通信)
│   ├── APScheduler   (定时抓取天气/新闻/记忆压缩)
│   └── IdleDetector  (每秒检查空闲, asyncio task)
│
└── 推理线程 (独立线程, 不阻塞 event loop)
    └── llama-cpp-python (同步推理, token 通过 queue 传回主线程)
```

> **关键**: 推理在独立线程，不阻塞 asyncio 事件循环。否则推理时整个系统卡死。

---

### Step 8: 实现联网模块 (`web/`)

```python
# 按从简单到复杂的顺序:

# 1. weather.py — 最简单, 先做
"""
使用 wttr.in (免费, 无需API Key):
  GET https://wttr.in/北京?format=j1
  返回 JSON, 解析: 当前温度、天气描述、湿度
  
缓存: data/cache/weather.json, TTL 1小时
失败: 返回 None, 调用方自行处理
"""

# 2. news.py — 热搜抓取
"""
方案A: 用 feedparser 订阅 RSS (最稳定)
  例如: RSSHub 路由生成各种源的 RSS

方案B: 直接爬取热搜页面
  httpx + BeautifulSoup
  目标: 微博热搜 / 知乎热榜 / 百度热搜
  注意: 加 User-Agent, 加请求间隔, 尊重 robots.txt
  
缓存: data/cache/news.json, TTL 2小时
"""

# 3. browser.py — 动态页面 (按需使用)
"""
pip install playwright
playwright install chromium    # 下载 Chromium (~150MB)

只用于用户主动请求的网页爬取。
例如: "帮我把这个网页的内容总结一下"
→ 启动 headless Chromium → 加载页面 → 获取 innerText → 传给 LLM 总结
"""
```

---

### Step 9: 拼装入口 `main.py`

```python
# main.py — 应用入口
"""
启动顺序:
1. 加载 config.yaml → 读取模型路径、角色配置、城市等
2. 初始化推理引擎 → 加载 GGUF 模型 (这个最耗时, ~5-10秒)
3. 初始化记忆系统 → 连接 ChromaDB + SQLite
4. 初始化人格引擎 → 加载 character.yaml
5. 初始化联网模块 → 启动 APScheduler (天气/新闻定时抓取)
6. 初始化对话管理器 → 传入所有模块引用
7. 启动 WebSocket 服务 → 监听 ws://127.0.0.1:9876
8. 触发生成开机问候 → 推送给已连接的前端
9. 启动空闲检测 → asyncio task
10. 进入事件循环 → asyncio.run_forever()

优雅退出:
- 卸载模型 (释放显存)
- 关闭 ChromaDB
- 关闭 SQLite
- 取消定时任务
"""
```

---

### Step 10: 验证全链路

写一个 CLI 测试脚本，在没有前端的情况下验证整个链条:

```python
# test_full_chain.py
"""
1. 启动后端 (main.py)
2. 通过 WebSocket 客户端 (websockets 库) 连接
3. 发送消息, 接收流式回复, 打印到终端
4. 验证:
   - 回复是否符合角色设定
   - 是否引用了记忆中的信息
   - 天气/时间是否注入
   - 流式输出是否顺畅
5. 多次对话后检查记忆检索是否生效
"""
```

> 这个 CLI 客户端也可以作为 Phase 1-3 的实际交互界面——在 Electron 前端完成之前，你已经有可用的对话系统了。

---

## 三、模块间数据流总览

```
                         用户输入
                            │
                            ▼
┌─────────────────┐   ┌──────────┐   ┌─────────────┐
│   联网模块       │   │对话管理器 │   │  记忆系统     │
│                  │   │          │   │              │
│ get_env_summary()├──►│①预处理   ├──►│ retrieve()   │
│ (天气/时间/日期)  │   │②查记忆   │   │ (向量+关键词) │
│                  │   │③拼Prompt│   │              │
│ get_news()       │   │④推理    │   │ store()      │
│ (话题池)         │   │⑤存记忆  │   │ (异步写入)    │
└─────────────────┘   │          │   └─────────────┘
        ▲              └────┬─────┘
        │                   │
        │              ┌────▼─────┐   ┌─────────────┐
        │              │ 推理引擎   │   │  人格引擎     │
        └──────────────│          │   │              │
         env数据流向    │ chat()   │   │ build_prompt │
         (定时写入缓存)  │ chat_stream()│              │
                       │ (单例,黑盒)│   │ update_emotion│
                       └──────────┘   └─────────────┘
```

---

## 四、配置中心 (`config.yaml`)

```yaml
# Mon3ter 全局配置
models:
  llm:
    path: "models/llm/qwen2.5-7b-instruct-q4_k_m.gguf"
    n_ctx: 8192          # 上下文窗口
    n_gpu_layers: -1     # -1 = 所有层放GPU
    temperature: 0.7
    max_tokens: 1024
  
  embedding:
    model_name: "BAAI/bge-m3"
    device: "cpu"        # embedding用CPU, 省显存

personality:
  character_file: "personality/character.yaml"

memory:
  chroma_path: "data/chroma"
  sqlite_path: "data/summaries.db"
  retrieval_top_k: 5
  compress_interval_hours: 24

web:
  weather_city: "北京"
  weather_cache_ttl: 3600
  news_cache_ttl: 7200

server:
  host: "127.0.0.1"
  port: 9876

user:
  name: "用户"           # 初始称呼, 会被偏好学习覆盖
```

---

## 五、关键技术决策 FAQ

**Q: 为什么要用 ChromaDB 而不是把记忆也存 SQLite？**
A: 向量相似度检索用 SQLite 做不到（除非用 sqlite-vss 扩展，但那不够成熟）。ChromaDB 专门为向量搜索优化，开箱即用。结构化数据（用户事实）仍然放 SQLite——各司其职。

**Q: 推理为什么不直接在主线程？**
A: llama-cpp-python 的推理是同步阻塞的，如果放主线程，推理期间整个系统（包括 WebSocket 心跳、前端通信）全部卡死。独立线程 + queue 传 token 是最简单的解耦方案。

**Q: 闲聊模型能做 Function Calling 吗？**
A: Qwen2.5-7B-Instruct 原生支持工具调用。可以在 System Prompt 中声明工具列表，模型输出 JSON 格式的工具调用。这是 Phase 3 插件系统的技术基础。

**Q: 如果 RTX 4070 Super 想同时跑 LLM + TTS 模型怎么办？**
A: 12GB 显存预算：
- LLM (7B Q4): ~7.5GB（含 KV Cache）
- TTS (GPT-SoVITS): ~2GB
- 余量: ~2.5GB
可以同时跑，但建议错峰——对话时卸载 TTS，说话时 LLM 可以保持加载（不推理就不占额外显存）。

**Q: 这套方案能迁移到 Linux/Mac 吗？**
A: 全部技术栈跨平台。唯一 Windows 特定的部分只是空闲检测（`ctypes.windll.user32`），换成 Linux 的 `xprintidle` 或 Mac 的 `CGEventSourceSecondsSinceLastEventType` 即可。

---

## 六、Phase 1 最小可行版本 — 文件清单

完成 Phase 1 需要创建的最小文件集合（约 15 个文件）:

```
Mon3ter/
├── config.yaml                 # 全局配置
├── main.py                     # 入口
├── core/
│   ├── engine.py               # 推理引擎 (~150行)
│   └── types.py                # 数据类型 (~30行)
├── personality/
│   ├── character.yaml          # 角色卡
│   ├── character.py            # Prompt 拼装 (~80行)
│   └── emotion_state.py        # 情绪状态机 (~60行)
├── memory/
│   ├── vector_store.py         # ChromaDB 封装 (~100行)
│   ├── models.py               # SQLAlchemy (~50行)
│   ├── retrieval.py            # 检索逻辑 (~80行)
│   └── summarizer.py           # 摘要生成 (~40行)
├── conversation/
│   ├── orchestrator.py         # 对话调度 (~200行)
│   └── session.py              # 会话管理 (~50行)
├── web/
│   ├── weather.py              # 天气 (~40行)
│   └── scheduler.py            # 定时任务 (~20行)
├── server.py                   # WebSocket 服务 (~80行)
├── test_cli.py                 # CLI 测试客户端 (~100行)
└── models/                     # 模型存放(手动下载)
    └── llm/
        └── qwen2.5-7b-instruct-q4_k_m.gguf
```

总代码量约 **1000-1200 行 Python**，一个熟练开发者约 **5-8 天** 可完成。

---

> **下一步**: 确认这份技术方案后，可以开始逐个文件实现。建议从 `core/engine.py` 开始——它是所有模块的根基，跑通了模型推理，后面的都是"搭积木"。
