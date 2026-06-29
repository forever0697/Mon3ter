# Mon3ter `core/` 模块代码解析

> `core/` 是整个 Mon3ter 的根基。只做两件事：**定义所有模块共享的数据结构**（types.py）和**封装本地 LLM 推理**（engine.py）。两个文件、约 690 行代码，但每一个设计决策都影响全局。

---

## 文件定位

```
mon3ter/core/
│
├── __init__.py                  # 空文件, 标识 Python 包
├── types.py            ~426行   # 全局"词汇表"——所有数据类型定义
├── engine.py           ~264行   # 推理引擎——唯一的模型入口
└── CORE_MODULE_ANALYSIS.md      # 你正在看的文件
```

**依赖方向**: `types.py` 零内部依赖 → `engine.py` 依赖 `types.py`。其他所有模块（`server.py`/`main.py`/未来的 `personality`/`memory` 等）都依赖这两个文件。`core/` 不依赖任何其他模块。

---

## `types.py` — 全局数据类型（~426 行）

### 设计分层

types.py 并没有杂乱的堆砌类型。它按**业务域**分了 7 个区块，外加一段配置层，共 22 个类型：

```
推理引擎层
├── Message              对话消息（OpenAI 兼容格式）
├── ModelLoadConfig      模型加载参数（pydantic, 带校验）
├── SamplingParams       采样参数（pydantic, 带范围约束）
└── InferenceRequest     一次推理的完整请求

人格系统层
├── Emotion              情绪枚举（6 种状态, 带描述和 emoji）
└── EmotionState         当前情绪快照

记忆系统层
├── MemoryItem           向量检索结果
├── UserFact             用户结构化事实（key-value）
└── ConversationSummary  对话摘要

对话管理层
└── ConversationContext  多轮对话上下文窗口

联网模块层
├── WeatherInfo          天气数据
├── NewsItem             新闻条目
└── EnvSummary           注入 System Prompt 的环境摘要

通信层
└── WSMessage            WebSocket 消息协议

配置层（pydantic）
├── AppConfig            全局配置根
├── ModelsSection        模型配置（load + sampling + embedding）
├── PersonalitySection   人格配置
├── MemorySection        记忆配置
├── WebSection           联网配置
├── ServerSection        服务端配置
└── UserSection          用户配置
```

### 关键设计决策

**1. 核心类型用 `@dataclass(slots=True)`，配置用 pydantic `BaseModel`**

```python
# 核心数据：轻量、快速、零开销
@dataclass(slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str

# 配置：需要校验、反序列化、默认值、约束
class ServerSection(BaseModel):
    port: int = Field(default=9876, gt=1024, lt=65536)
```

为什么不用 pydantic 做所有类型？因为 `Message` 每秒会被创建几十次（对话历史里每条消息一个实例），用 `slots=True` 的 dataclass 内存开销远小于 pydantic BaseModel。而配置对象在启动时创建一次、全局只读，用 pydantic 换校验能力是值得的。

**2. `ModelLoadConfig` vs `SamplingParams` — 加载和采样分离**

这是 P1 改进（1.3）的核心。之前 `ModelConfig` 是一个混合体：

```
旧: ModelConfig (混杂)
  ├── model_path      加载参数 ← 改了就需重载模型
  ├── n_ctx           加载参数
  ├── n_gpu_layers    加载参数
  ├── temperature     采样参数 ← 每次对话都可以不同
  └── max_tokens      采样参数

新:
  ModelLoadConfig     纯加载参数 → 传给 engine.load()
  SamplingParams      纯采样参数 → InferenceRequest 携带, 每请求可变
```

这解决了一个实际的偷渡问题：之前 `server.py` 回退模式读 `engine.config.temperature` 获取默认采样值——引擎不应该关心采样默认值是什么，那是配置层的事。

**3. `ConversationContext` — 两段式上下文窗口**

```
prefix (固定不变)          history (可滚动)
┌──────────────┐  ┌─────┬─────┬─────┬─────┐
│ System Prompt │  │ U1  │ A1  │ U2  │ A2  │ ...  ← 最多 20 轮
│ 角色卡        │  └─────┴─────┴─────┴─────┘
│ 环境信息      │        ↑ 超出后自动丢弃最早一轮
│ 用户事实      │
└──────────────┘
```

`all_messages()` 拼接两段后一次性传给推理引擎。`_trim()` 只裁剪 `history` 段，`prefix` 永远保留。

和旧版 `_trim` 的区别：旧版假设 messages 全是 user/assistant 交替，一旦插入 system 消息就会静默失败（`StopIteration` → break → 无限增长）。新版显式管理 prefix 和 history，裁剪按 role 配对删除，即使 messages 中有 system/tool 消息也能正确工作。

**4. `WSMessage` — 嵌套 `extra`**

```python
# 旧：extra 平铺到 JSON 顶层——extra={"type":"hacked"} 会覆盖主字段
{"type": "chat", "content": "你好", **extra}

# 新：嵌套——无法覆盖
{"type": "chat", "content": "你好", "extra": {...}}
```

P1 改进（1.6），消除协议层的字段覆盖风险。

**5. `AppConfig.from_yaml()` — 配置加载、校验、路径解析一行完成**

```python
config = AppConfig.from_yaml("config.yaml", root_dir=ROOT_DIR)
```

它做了三件事：
1. 读 YAML
2. 解析相对路径（`models/llm/...` → `C:/AI_Project/Mon3ter/models/llm/...`）
3. pydantic 校验（`n_ctx > 0`、`1024 < port < 65536`、`0 ≤ temperature ≤ 2`……）

如果 config.yaml 写了 `port: 80` 或 `temperature: 5.0`，启动时立刻报错，而非跑到一半崩溃。

同时还兼容旧版平铺格式（`models.llm_path`），读到自动转换为 `models.load.llm_path`。

---

## `engine.py` — 推理引擎（~264 行）

### 架构

```
InferenceEngine (全局单例)
│
├── 生命周期
│   ├── load()             加载 GGUF → GPU 显存
│   ├── unload()           释放显存
│   └── is_loaded          查询状态
│
├── 推理接口
│   ├── chat()             阻塞式，返回完整回复（后台摘要用）
│   ├── chat_stream()      同步流式生成器（底层）
│   └── chat_stream_async() 异步流式（server/orchestrator 统一入口）
│
├── 内部
│   ├── _lock              threading.Lock — 同时只允许一个推理
│   └── _ensure_loaded()   守卫：未加载时报清晰错误
│
└── 异常体系
    ├── EngineError               基类
    ├── ModelNotLoadedError       未加载就推理
    ├── ModelAlreadyLoadedError   重复加载
    ├── ModelFileNotFoundError    文件路径错误
    └── InferenceFailedError      推理过程崩溃
```

### 关键设计：`chat_stream_async()` — sync→async 桥接的唯一位置

这是 P1（1.1+1.2）合并修复的成果。整个项目的同步/异步矛盾在这里一次性解决：

```
调用方 (async)
    │
    └── async for token in engine.chat_stream_async(request):
            │
            │  asyncio.Queue  ←───  call_soon_threadsafe  ←───  生产者线程
            │  (async 消费)                                       (sync 生产)
            │                                                     │
            │                                               engine.chat_stream()
            │                                               (同步阻塞 in thread)
```

桥接逻辑用三个组件：
- **`asyncio.Queue`**: 线程安全的异步队列，承载 token
- **`loop.call_soon_threadsafe`**: 把同步线程产出的 token 安全推入 event loop
- **`None` sentinel**: 生产者结束时推入 `None`，消费者收到后退出循环

异常透传：如果同步推理线程内崩溃，异常对象被推入 queue，消费者在 async 端重新 raise。

**此前这个桥接逻辑散落在 `server.py` 里，而且有 `list()` 物化的 bug。现在收归 engine 内部，server.py/orchestrator 只需一行 `async for`。**

### 全局单例管理

```python
# 模块级变量
_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()

get_engine(config=None)    # 获取或创建, 首次必须传 config
reset_engine()              # 卸载模型 + 销毁实例
```

为什么不用依赖注入而用模块级单例？
- 模型加载一次要 5-10 秒，显存占 7.5GB，全局只能有一个实例
- 如果每个需要推理的模块都自己 new 一个引擎，显存立刻爆掉
- 模块级单例 + `get_engine()` 是最简单的方式：导入即用，无需四处传参

`reset_engine()` 在应用退出时调用（`main.py` 的 `shutdown()`），确保显存被正确释放。

### 线程锁的作用域

```python
with self._lock:
    result = self._model.create_chat_completion(...)
```

锁保护的是 llama.cpp 的 `create_chat_completion()` 调用。llama.cpp 不是线程安全的——两个线程并发调用会导致未定义行为。这把锁确保任意时刻只有一个推理在跑。

当前局限（改进计划 1.9）：锁不区分前台和后台任务。如果未来实现后台摘要生成，它会和用户对话抢同一把锁。短期方案是后台任务只在用户空闲时调度。

---

## 文件间协作全景

```
config.yaml
    │
    ├─→ types.py  AppConfig.from_yaml()  ──→  pydantic校验 → AppConfig实例
    │
    ├─→ main.py
    │     ├── AppConfig.from_yaml(config.yaml)
    │     ├── ModelLoadConfig(...)  ──→  engine.load()
    │     └── SamplingParams(...)   ──→  server 回退默认值
    │
    ├─→ server.py
    │     ├── WSMessage.from_json()    ← 客户端发来的 JSON
    │     ├── InferenceRequest(...)    ← 构造推理请求
    │     └── engine.chat_stream_async() ← 异步流式推理
    │
    └─→ 未来模块
          ├── personality → 读 types.Emotion / 构造 Message.system()
          ├── memory      → 读写 types.MemoryItem / types.UserFact
          ├── conversation → 持有所有引用, 调度整个对话流程
          └── web         → 产 types.WeatherInfo / types.NewsItem
```

所有数据流回到 `types.py` 的类型定义上——这就是为什么它被称为"词汇表"。任何模块之间的数据传递，用的都是 `types.py` 中定义的结构。不会出现"A 传 dict、B 收 dict、key 名对不上"的隐式契约问题。

---

## 代码质量特征

| 维度 | 评价 |
|------|------|
| **类型标注** | 完整使用 `slots=True` dataclass、`Literal`、`| None` 联合类型 |
| **错误处理** | 自定义异常层级（`EngineError` → 四个子类），调用方可按粒度 catch |
| **文档** | 每个 public class/method 有 docstring，关键设计决策有行内注释 |
| **线程安全** | Lock 保护推理、Lock 保护单例创建、`call_soon_threadsafe` 桥接线程 |
| **资源管理** | `load()`/`unload()` 显式生命周期、`reset_engine()` 确保释放 |
| **可测性** | 配置用 `from_yaml()` 一次反序列化、采样参数独立于加载参数、单例可重置 |
