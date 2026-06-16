from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ai.prompts import PREFERENCE_AGENT_SYSTEM_PROMPT
from ai.schemas.preferences import PreferenceContext
from ai.tools.preference_tools import make_preference_tools

logger = logging.getLogger(__name__)

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


def build_preference_executor(user_id: str, llm: BaseChatModel):
    """
    Build a LangChain 1.x agent that autonomously calls all 3 preference tools
    and returns a structured PreferenceContext via response_format.
    """
    return create_agent(
        model=llm,
        tools=make_preference_tools(user_id),
        system_prompt=PREFERENCE_AGENT_SYSTEM_PROMPT,
        response_format=PreferenceContext,
    )
