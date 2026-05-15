"""Deep Agents harness — using ``create_deep_agent`` from the upstream package.

The upstream Quickstart (https://github.com/langchain-ai/deepagents#quickstart):

    from deepagents import create_deep_agent
    agent = create_deep_agent()
    result = agent.invoke({"messages": [...]})

The customization example from the same README:

    from langchain.chat_models import init_chat_model
    agent = create_deep_agent(
        model=init_chat_model("openai:gpt-4o"),
        tools=[my_custom_tool],
        system_prompt="You are a research assistant.",
    )

The ONLY change vs that upstream sample for Foundry hosting is the LLM
binding: the README uses ``init_chat_model("openai:gpt-4o")`` (which expects
``OPENAI_API_KEY``); inside a Foundry hosted-agent container we use the
per-instance managed identity and the project's deployment instead.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI


# ---------- the only Foundry-specific change ----------

def _derive_openai_endpoint() -> str:
    explicit = os.getenv("AZURE_OPENAI_ENDPOINT")
    if explicit:
        return explicit.rstrip("/")
    project = os.getenv("PROJECT_ENDPOINT") or os.getenv("AZURE_AI_PROJECT_ENDPOINT") or ""
    m = re.match(r"^(https?://)([^.]+)\.services\.ai\.azure\.com", project)
    if not m:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not set and PROJECT_ENDPOINT not Foundry-shaped.")
    return f"{m.group(1)}{m.group(2)}.openai.azure.com"


def _get_model() -> AzureChatOpenAI:
    """Replacement for the README's ``init_chat_model("openai:gpt-4o")`` line.

    Uses the Foundry container's managed identity instead of an API key.
    """
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureChatOpenAI(
        azure_endpoint=_derive_openai_endpoint(),
        azure_deployment=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-5.4"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_ad_token_provider=token_provider,
    )


# ---------- example custom tool (deepagents already provides planning,
#            filesystem, sub-agents; this just gives the model something
#            domain-specific to call) ----------

def internet_search(query: str) -> str:
    """Stub web-search tool. Replace with Tavily / Bing / DuckDuckGo for real use."""
    return (
        f"(stub search for: {query})\n"
        "Top result: Deep Agents are an opinionated agent harness from LangChain "
        "that bundles planning, filesystem, sub-agents, and shell tools on top of "
        "LangGraph."
    )


@lru_cache(maxsize=1)
def build_app():
    """Build the deepagents harness once per container."""
    return create_deep_agent(
        model=_get_model(),
        tools=[internet_search],
        system_prompt=(
            "You are a careful research assistant. Use write_todos to plan, "
            "the filesystem tools to take notes, and internet_search to gather "
            "facts. Cite sources inline."
        ),
    )
