"""Reporting agents: researcher, writer, and synthesist.

Each agent runs in dry_run mode by default (no API key required).
Set dry_run=False and provide ANTHROPIC_API_KEY for real LLM calls (Stage 3+).
"""

from coffee_deforestation.reporting.agents.researcher import run_researcher
from coffee_deforestation.reporting.agents.synthesist import run_synthesist
from coffee_deforestation.reporting.agents.writer import run_writer

__all__ = ["run_researcher", "run_writer", "run_synthesist"]
