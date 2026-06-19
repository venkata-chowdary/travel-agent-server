import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


class GeminiClient(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with AFC permanently disabled."""

    def _build_request_config(self, *args, **kwargs):
        kwargs.setdefault("automatic_function_calling", {"disable": True})
        return super()._build_request_config(*args, **kwargs)


class GroqClient(ChatGroq):
    """ChatGroq that downgrades json_schema to tool_calling.

    Groq only supports json_schema on a small set of models; tool_calling
    achieves the same Pydantic-parsed result and works on every Groq model.
    """

    def with_structured_output(self, schema, *, method="function_calling", **kwargs):
        if method in ("json_schema", "tool_calling"):
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


def get_llm(model: str, temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "groq":
        groq_model = os.getenv("GROQ_MODEL", model)
        return GroqClient(model=groq_model, temperature=temperature)
    return GeminiClient(model=model, temperature=temperature)
