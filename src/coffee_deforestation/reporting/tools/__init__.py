"""Reporting tools for the researcher agent.

Each tool is a pure function with a clear JSON-serialisable interface.
The agent calls these tools by name during its research phase.
"""

from coffee_deforestation.reporting.tools.compare_periods import compare_periods
from coffee_deforestation.reporting.tools.historical_context import get_historical_context
from coffee_deforestation.reporting.tools.hotspot_details import get_hotspot_details
from coffee_deforestation.reporting.tools.query_stats import query_stats
from coffee_deforestation.reporting.tools.render_hotspot_map import render_hotspot_map
from coffee_deforestation.reporting.tools.scratchpad import (
    scratchpad_clear,
    scratchpad_read,
    scratchpad_read_all,
    scratchpad_write,
)

__all__ = [
    "compare_periods",
    "get_historical_context",
    "get_hotspot_details",
    "query_stats",
    "render_hotspot_map",
    "scratchpad_write",
    "scratchpad_read",
    "scratchpad_read_all",
    "scratchpad_clear",
]
