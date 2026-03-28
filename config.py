"""
config.py
─────────────────────────────────────────────────────────────────
Single source of truth for the entire LLM Router Agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
from enum import Enum
from dotenv import load_dotenv

# Load from .env.example as requested
load_dotenv(dotenv_path=".env.example")

# ─────────────────────────────────────────────────────────────────
# Shared Types (Prevents Circular Imports)
# ─────────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    SIMPLE  = "simple"
    COMPLEX = "complex"

# ─────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
@dataclass
class ModelConfig:
    model_id:       str
    max_new_tokens: int   = 1024
    temperature:    float = 0.7
    timeout_secs:   float = 30.0

SIMPLE_MODEL = ModelConfig(
    model_id       = "llama-3.1-8b-instant", 
    max_new_tokens = 1024,
    temperature    = 0.5,
)

COMPLEX_MODEL = ModelConfig(
    model_id       = "llama-3.3-70b-versatile", 
    max_new_tokens = 4096,
    temperature    = 0.7,
)

# ─────────────────────────────────────────────────────────────────
# Classifier Settings
# ─────────────────────────────────────────────────────────────────

@dataclass
class ClassifierSettings:
    weight_token_length:   float = 0.20
    weight_keyword:        float = 0.30
    weight_embedding:      float = 0.35
    weight_structural:     float = 0.15
    embedding_model_name:  str   = "all-MiniLM-L6-v2"
    complexity_threshold:  float = 0.40
    token_len_simple_max:  int   = 60
    token_len_complex_min: int   = 300
    
CLASSIFIER = ClassifierSettings()

# ─────────────────────────────────────────────────────────────────
# Budget + Loop Settings
# ─────────────────────────────────────────────────────────────────

@dataclass
class BudgetSettings:
    max_tokens_per_session: int   = 8000
    warn_tokens_at:         int   = 6000
    max_iterations:         int   = 12
    repetition_window:      int   = 4
    repetition_threshold:   float = 0.85
    db_path:                str   = "budget_audit.db"

BUDGET = BudgetSettings()

# ─────────────────────────────────────────────────────────────────
# Connector Settings
# ─────────────────────────────────────────────────────────────────

@dataclass
class ConnectorSettings:
    max_retries:      int   = 3
    retry_base_delay: float = 2.0
    enable_fallback:  bool  = True

CONNECTOR = ConnectorSettings()

# ─────────────────────────────────────────────────────────────────
# Memory (ChromaDB)
# ─────────────────────────────────────────────────────────────────

@dataclass
class MemorySettings:
    persist_dir:     str   = ".chromadb"
    collection:      str   = "agent_memory"
    max_results:     int   = 5
    score_threshold: float = 0.75

MEMORY = MemorySettings()

# ─────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────

@dataclass
class ToolSettings:
    web_search_max_results: int   = 5
    code_executor_timeout:  float = 10.0
    file_reader_max_mb:     float = 10.0

TOOLS = ToolSettings()