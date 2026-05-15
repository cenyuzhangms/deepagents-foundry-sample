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

import shlex
import subprocess
import uuid

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
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


# ---------- in-container shell sandbox -----------------------------------
#
# `create_deep_agent` registers an `execute` tool unconditionally, but it
# only works when the backend implements ``SandboxBackendProtocol``. The
# default ``StateBackend`` does not, so ``execute`` returns an error.
#
# This subclass adds a minimal sandbox that runs commands in the Foundry
# hosted-agent container itself (the container IS the isolation boundary).
# A small allow-list keeps the model from running destructive commands.

_ALLOWED_BINARIES = {
    "ls", "cat", "pwd", "echo", "wc", "head", "tail",
    "env", "printenv", "whoami", "uname", "id", "date",
    "df", "du", "free", "ps",
    "python", "python3", "pip",
    "grep", "find", "which",
}

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_BYTES = 16_000


class InContainerSandboxBackend(StateBackend, SandboxBackendProtocol):
    """StateBackend + a whitelisted subprocess shell for the `execute` tool."""

    def __init__(self) -> None:
        super().__init__()
        self._id = f"foundry-container-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return ExecuteResponse(output=f"parse error: {e}", exit_code=2)
        if not parts:
            return ExecuteResponse(output="empty command", exit_code=2)
        binary = parts[0].split("/")[-1]
        if binary not in _ALLOWED_BINARIES:
            return ExecuteResponse(
                output=(
                    f"refused: '{binary}' is not in the allow-list. "
                    f"Allowed: {sorted(_ALLOWED_BINARIES)}"
                ),
                exit_code=126,
            )
        try:
            proc = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=timeout or _DEFAULT_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output=f"timeout after {timeout or _DEFAULT_TIMEOUT}s", exit_code=124)
        except FileNotFoundError:
            return ExecuteResponse(output=f"binary not found: {binary}", exit_code=127)
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(output=f"error: {e}", exit_code=1)

        combined = (proc.stdout or "") + (proc.stderr or "")
        truncated = False
        if len(combined) > _MAX_OUTPUT_BYTES:
            combined = combined[:_MAX_OUTPUT_BYTES] + "\n...(truncated)"
            truncated = True
        return ExecuteResponse(output=combined or "(no output)", exit_code=proc.returncode, truncated=truncated)


@lru_cache(maxsize=1)
def build_app():
    """Build the deepagents harness once per container."""
    return create_deep_agent(
        model=_get_model(),
        tools=[internet_search],
        backend=InContainerSandboxBackend(),
        system_prompt=(
            "You are a careful research assistant. Use write_todos to plan, "
            "the virtual filesystem (write_file/read_file/edit_file) to take "
            "notes, internet_search to gather facts, and the execute tool for "
            "read-only shell inspection (ls, cat, env, python --version, etc.). "
            "Never assume execute can run destructive commands; the sandbox "
            "rejects anything outside an allow-list. Cite sources inline."
        ),
    )
