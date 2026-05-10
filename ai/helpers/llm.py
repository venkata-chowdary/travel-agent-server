from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiClient(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with AFC permanently disabled."""

    def _build_request_config(self, *args, **kwargs):
        kwargs.setdefault("automatic_function_calling", {"disable": True})
        return super()._build_request_config(*args, **kwargs)
