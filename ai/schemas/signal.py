from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentSignal(BaseModel):
    signal_type: Literal[
        "no_action_needed",
        "risk_detected",
        "replan_suggested",
        "clarification_needed",
        "data_sparse",
        "alternative_available",
    ] = Field(description="The type of signal this agent is emitting after completing its work.")
    message: str = Field(description="The agent's plain-English judgment about what it found and what it recommends.")
    severity: Literal["low", "medium", "high"] = Field(
        default="low",
        description="How strongly the supervisor should weight this signal when deciding next steps.",
    )
