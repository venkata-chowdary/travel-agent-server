import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger("travel_agent.thinking")


def log_thinking(messages: list[BaseMessage]) -> None:
    """Walk the agent message trace and emit one log line per step.

    Each line is a JSON object with a ``type`` field so it can be parsed later
    and streamed to the frontend (e.g. via SSE or WebSocket).

    Types emitted:
    - thinking   – LLM decided to call one or more tools
    - tool_call  – individual tool invocation (name + args)
    - tool_result – what the tool returned
    - final_answer – the last text the agent produced
    """
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            is_last = i == len(messages) - 1
            if msg.tool_calls:
                logger.info(json.dumps({
                    "type": "thinking",
                    "content": f"Decided to call {len(msg.tool_calls)} tool(s)",
                }))
                for tc in msg.tool_calls:
                    logger.info(json.dumps({
                        "type": "tool_call",
                        "tool": tc["name"],
                        "args": tc["args"],
                    }))
            elif is_last:
                text = msg.content
                if isinstance(text, list):
                    text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
                logger.info(json.dumps({"type": "final_answer", "content": str(text)}))

        elif isinstance(msg, ToolMessage):
            logger.info(json.dumps({
                "type": "tool_result",
                "tool": msg.name,
                "content": str(msg.content),
            }))
