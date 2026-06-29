"""
Mon3ter 应用入口

启动顺序:
  1. 加载 config.yaml (pydantic 校验)
  2. 创建数据目录
  3. 初始化推理引擎 → 加载 GGUF 模型
  4. 初始化各功能模块 (按可用性渐进加载)
  5. 启动 WebSocket 服务
  6. 触发生成开机问候
  7. 进入事件循环, 等待用户交互

用法:
  python main.py                  # 使用默认 config.yaml
  python main.py --config my.yaml # 指定配置文件
  python main.py --verbose        # 开启详细日志
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False) -> None:
    """配置全局日志格式与级别。"""
    import io

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
    )

    handler = logging.StreamHandler(utf8_stdout)
    handler.setLevel(level)
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    for noisy in ("chromadb", "urllib3", "httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("mon3ter")


# ═══════════════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════════════

def load_config(config_path: str | None = None) -> "AppConfig":
    """
    从 YAML 文件加载全局配置（pydantic 校验）。

    相对路径自动以 ROOT_DIR 为基准解析。
    """
    from mon3ter.core.types import AppConfig

    path = Path(config_path) if config_path else ROOT_DIR / "config.yaml"
    config = AppConfig.from_yaml(path, root_dir=ROOT_DIR)

    logger.info("配置加载完成: %s", path)
    logger.debug(
        "  模型路径: %s | GPU层: %d | 上下文: %d",
        config.models.load.llm_path,
        config.models.load.n_gpu_layers,
        config.models.load.n_ctx,
    )
    return config


# ═══════════════════════════════════════════════════════════════════════
# 数据目录初始化
# ═══════════════════════════════════════════════════════════════════════

def ensure_data_dirs(config: "AppConfig") -> None:
    """创建所有运行时需要的目录。"""
    dirs = [
        ROOT_DIR / "data" / "chroma",
        ROOT_DIR / "data" / "cache",
        ROOT_DIR / "data" / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("目录就绪: %s", d.relative_to(ROOT_DIR))


# ═══════════════════════════════════════════════════════════════════════
# 模块初始化 (渐进式 — 未完成的模块跳过, 不影响核心运行)
# ═══════════════════════════════════════════════════════════════════════

def init_engine(config: "AppConfig"):
    """
    初始化推理引擎 — 加载 GGUF 模型。

    这是整个系统唯一的硬依赖。
    """
    from mon3ter.core.engine import get_engine
    from mon3ter.core.types import ModelLoadConfig

    mc = ModelLoadConfig(
        model_path=config.models.load.llm_path,
        n_ctx=config.models.load.n_ctx,
        n_gpu_layers=config.models.load.n_gpu_layers,
    )

    engine = get_engine(mc)
    engine.load()
    logger.info("✅ 推理引擎就绪")
    return engine


def init_memory(config: "AppConfig"):
    """初始化记忆系统 — ChromaDB + SQLite。"""
    try:
        from mon3ter.memory.vector_store import VectorStore
        from mon3ter.memory.models import init_db

        init_db(config.memory.sqlite_path)
        vs = VectorStore(config.memory.chroma_path)
        logger.info("✅ 记忆系统就绪")
        return vs
    except ImportError:
        logger.warning("⚠ 记忆系统模块未就绪，跳过")
        return None
    except Exception as exc:
        logger.warning("⚠ 记忆系统初始化失败: %s", exc)
        return None


def init_personality(config: "AppConfig"):
    """初始化人格引擎 — 加载角色卡。"""
    try:
        from mon3ter.personality.character import PersonalityEngine

        pe = PersonalityEngine(config.personality.character_file)
        logger.info("✅ 人格引擎就绪")
        return pe
    except ImportError:
        logger.warning("⚠ 人格引擎未就绪，跳过")
        return None
    except Exception as exc:
        logger.warning("⚠ 人格引擎初始化失败: %s", exc)
        return None


def init_web_module(config: "AppConfig"):
    """初始化联网模块 — 启动定时抓取任务。"""
    try:
        from mon3ter.web.scheduler import start_scheduler

        scheduler = start_scheduler(config.web)
        logger.info("✅ 联网模块就绪")
        return scheduler
    except ImportError:
        logger.warning("⚠ 联网模块未就绪，跳过")
        return None
    except Exception as exc:
        logger.warning("⚠ 联网模块初始化失败: %s", exc)
        return None


def init_conversation(config: "AppConfig", engine, personality, memory, web_mod):
    """初始化对话管理器 — 系统总指挥。"""
    try:
        from mon3ter.conversation.orchestrator import ConversationOrchestrator

        orch = ConversationOrchestrator(
            engine=engine,
            personality=personality,
            memory=memory,
            web=web_mod,
            user_name=config.user.name,
        )
        logger.info("✅ 对话管理器就绪")
        return orch
    except ImportError:
        logger.warning("⚠ 对话管理器未就绪，将使用简易回退模式")
        return None
    except Exception as exc:
        logger.warning("⚠ 对话管理器初始化失败: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════
# WebSocket 服务
# ═══════════════════════════════════════════════════════════════════════

async def start_server(config: "AppConfig", orchestrator, engine):
    """启动 WebSocket 服务端。"""
    from mon3ter.core.types import SamplingParams

    default_sampling = SamplingParams(
        temperature=config.models.sampling.temperature,
        max_tokens=config.models.sampling.max_tokens,
    )

    try:
        import server
        await server.start(config.server, default_sampling, orchestrator, engine)
    except ImportError:
        logger.error("❌ server.py 未实现，无法启动服务")
        raise
    except Exception as exc:
        logger.error("❌ 服务端启动失败: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════════════
# 生命周期
# ═══════════════════════════════════════════════════════════════════════

async def shutdown(engine, web_mod, exit_code: int = 0):
    """优雅退出。"""
    logger.info("正在关闭 Mon3ter...")

    if web_mod is not None:
        try:
            web_mod.shutdown()
        except Exception:
            pass

    if engine is not None:
        from mon3ter.core.engine import reset_engine
        reset_engine()

    logger.info("Mon3ter 已关闭。")
    sys.exit(exit_code)


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(description="Mon3ter - 本地 AI 桌面伴侣")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="开启 DEBUG 级别日志")
    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("══════════ Mon3ter 启动 ══════════")
    logger.info("项目根目录: %s", ROOT_DIR)

    # 1. 加载配置
    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error("配置加载失败: %s", exc)
        sys.exit(1)

    # 2. 创建数据目录
    ensure_data_dirs(config)

    # 3. 初始化推理引擎 (硬依赖)
    try:
        engine = init_engine(config)
    except Exception as exc:
        logger.error("推理引擎初始化失败: %s", exc)
        sys.exit(1)

    # 4. 初始化功能模块 (软依赖)
    memory_mod = init_memory(config)
    personality_mod = init_personality(config)
    web_mod = init_web_module(config)

    # 5. 初始化对话管理器
    orchestrator = init_conversation(
        config, engine, personality_mod, memory_mod, web_mod
    )

    # 6. 启动 WebSocket 服务
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(start_server(config, orchestrator, engine))
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as exc:
        logger.exception("运行时异常: %s", exc)
    finally:
        loop.run_until_complete(shutdown(engine, web_mod))
        loop.close()


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
