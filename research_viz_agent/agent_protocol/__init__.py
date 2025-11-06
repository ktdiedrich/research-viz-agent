"""
Agent-to-Agent Communication Protocol

This module provides a standardized protocol for inter-agent communication,
enabling the research-viz-agent to interact with other AI agents.
"""

from research_viz_agent.agent_protocol.schemas import (
    AgentRequest,
    AgentResponse,
    AgentCapability,
    AgentStatus,
    ResearchQuery,
    ResearchResult
)
from research_viz_agent.agent_protocol.server import AgentServer
from research_viz_agent.agent_protocol.client import AgentClient

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "AgentCapability",
    "AgentStatus",
    "ResearchQuery",
    "ResearchResult",
    "AgentServer",
    "AgentClient"
]
