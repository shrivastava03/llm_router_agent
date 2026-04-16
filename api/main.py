"""
api/main.py
─────────────────────────────────────────────────────────────────
FastAPI Entrypoint — LLM Router Agent

Production Run Command (Docker/Render):
  uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import time
import uuid
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


import config
from core.classifier   import ComplexityClassifier
from core.router       import Router
from core.budget_guard import BudgetGuard, BudgetExceededError, LoopKillError
from core.memory       import AgentMemory
from hf_connector      import HFConnector
from agent.agent       import Agent          

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("router_agent")


# ─────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt:        str           = Field(..., min_length=1, max_length=8000)
    session_id:    Optional[str] = None
    system_prompt: Optional[str] = None

    model_config = {"json_schema_extra": {"example": {
        "prompt":     "Search the web for the latest Python version and summarize changes.",
        "session_id": "user-abc-123",
    }}}


class AgentStepOut(BaseModel):
    iteration:   int
    thought:     str
    tool:        str
    tool_input:  str
    observation: str


class CompletionResponse(BaseModel):
    session_id:         str
    answer:             str
    tier_used:          str
    total_tokens:       int
    iterations:         int
    killed:             bool
    kill_reason:        Optional[str]
    complexity_score:   float
    steps:              list[AgentStepOut]
    classifier_signals: dict


class HealthResponse(BaseModel):
    status:         str
    uptime_s:       float
    memory_entries: int


# ─────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────

class AppState:
    classifier: ComplexityClassifier
    router:     Router
    guard:      BudgetGuard
    connector:  HFConnector
    memory:     AgentMemory
    agent:      Agent             
    started_at: float


state = AppState()


# ─────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("  LLM Router Agent -- starting up")
    logger.info("=" * 50)
    state.started_at = time.time()

    state.classifier = ComplexityClassifier()
    logger.info("ComplexityClassifier ready")

    state.router = Router()
    logger.info("Router ready")

    state.guard = BudgetGuard()
    await state.guard.setup()
    logger.info("BudgetGuard ready")

    state.connector = HFConnector()
    logger.info("HFConnector ready")

    state.memory = AgentMemory(
        persist_dir = config.MEMORY.persist_dir,
        collection  = config.MEMORY.collection,
    )
    logger.info(f"AgentMemory ready  ({state.memory.count()} entries)")

    state.agent = Agent(
        classifier = state.classifier,
        router     = state.router,
        guard      = state.guard,
        connector  = state.connector,
        memory     = state.memory,
    )
    logger.info("Agent ready")

    logger.info("All systems go.\n")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "LLM Router Agent",
    description = (
        "Intelligent proxy: classifies prompt complexity, routes to cheapest "
        "capable model, runs a ReAct tool-use loop, and enforces token budgets."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({ms:.0f}ms)")
    return response


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.get("/")
@app.head("/")
async def serve_frontend():
    """Serves the UI directly from the backend via FileResponse"""
    ui_path = Path(__file__).resolve().parent.parent / "index.html"
    
    if not ui_path.exists():
        return HTMLResponse(
            content="<h1>Frontend not found!</h1><p>Ensure index.html is in the container root.</p>", 
            status_code=404
        )
        
    return FileResponse(ui_path)

@app.post("/complete", response_model=CompletionResponse)
async def complete(req: CompletionRequest):
    """
    Full agent pipeline:
      1. Classify prompt complexity
      2. Router picks model tier
      3. Agent runs ReAct loop (Think -> Tool -> Observe -> repeat)
      4. BudgetGuard watches every iteration
      5. Memory stores the completed interaction
      6. Return answer + full step trace
    """
    session_id = req.session_id or str(uuid.uuid4())

    clf = state.classifier.classify(req.prompt)
    logger.info(
        f"[{session_id[:12]}] score={clf.final_score:.3f} "
        f"tier={clf.tier.value}"
    )

    try:
        result = await state.agent.run(
            task          = req.prompt,
            session_id    = session_id,
            system_prompt = req.system_prompt,
        )

    except BudgetExceededError as e:
        raise HTTPException(status_code=402, detail={
            "error":       "budget_exceeded",
            "message":     str(e),
            "tokens_used": e.tokens_used,
            "limit":       e.limit,
        })
    except LoopKillError as e:
        raise HTTPException(status_code=429, detail={
            "error":   "loop_killed",
            "message": str(e),
            "reason":  e.reason,
        })
    except Exception as e:
        logger.error(f"[{session_id[:12]}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return CompletionResponse(
        session_id         = session_id,
        answer             = result.answer,
        tier_used          = result.tier_used,
        total_tokens       = result.total_tokens,
        iterations         = result.iterations,
        killed             = result.killed,
        kill_reason        = result.kill_reason,
        complexity_score   = clf.final_score,
        steps              = [AgentStepOut(**s.__dict__) for s in result.steps],
        classifier_signals = {
            "token_length": clf.signal_token_len,
            "keyword":      clf.signal_keyword,
            "embedding":    clf.signal_embedding,
            "structural":   clf.signal_structural,
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status         = "ok",
        uptime_s       = round(time.time() - state.started_at, 1),
        memory_entries = state.memory.count(),
    )


@app.get("/spend")
async def spend():
    report = await state.guard.spend_report()
    total  = sum(r["tokens_used"] or 0 for r in report)
    killed = [r for r in report if r["status"] == "killed"]
    return {
        "sessions": report,
        "summary": {
            "total_sessions":  len(report),
            "total_tokens":    total,
            "killed_sessions": len(killed),
            "kill_reasons":    list({r["kill_reason"] for r in killed if r["kill_reason"]}),
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {"error": "internal_server_error", "detail": str(exc)},
    )