# Deep Agents on Microsoft Foundry Hosted Agents

A working sample proving that an **unchanged** LangChain
[`deepagents`][deepagents] harness can be deployed to **Microsoft Foundry
Hosted Agents** with a thin adapter — the same adapter that hosts the
companion [`langgraph-foundry-sample`][sister].

[`graph.py`](graph.py) is the [upstream `deepagents` README quickstart][da-readme]
plus the customization snippet, with one substantive change: the LLM is
constructed via Foundry's container managed identity instead of an
`OPENAI_API_KEY`. The Foundry glue is entirely in [`main.py`](main.py) +
[`agent.yaml`](agent.yaml) + [`Dockerfile`](Dockerfile) +
[`azure.yaml`](azure.yaml) + [`infra/`](infra/).

[deepagents]: https://github.com/langchain-ai/deepagents
[da-readme]: https://github.com/langchain-ai/deepagents#quickstart
[sister]: https://github.com/cenyuzhangms/langgraph-foundry-sample

## What you get for free from `create_deep_agent`

The harness ships with these built-in tools — no extra code to enable:

- **Planning** — `write_todos` for task breakdown and progress tracking
- **Filesystem** — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`
  (backed by LangGraph state — virtual files, not the real container FS)
- **Shell** — `execute` for running commands. The tool is *registered* by
  default but only works when the backend implements
  `SandboxBackendProtocol`. The default `StateBackend` does **not**, so this
  sample ships [`InContainerSandboxBackend`](graph.py) — a `StateBackend`
  subclass that runs commands in the Foundry container via `subprocess.run`,
  gated by a small allow-list (`ls`, `cat`, `pwd`, `env`, `python`, `pip`,
  `grep`, `find`, …). Anything outside the list is refused with exit 126.
- **Sub-agents** — `task` for delegating with isolated context windows
- **Smart prompts** — system prompts that teach the model how to use the above
- **Context management** — auto-summarization, large outputs spilled to files

You can pass your own `tools=[...]`, `backend=...`, and `system_prompt=...`
on top.

## The diff vs upstream

```diff
- from langchain.chat_models import init_chat_model
- agent = create_deep_agent(
-     model=init_chat_model("openai:gpt-4o"),
-     tools=[my_custom_tool],
-     system_prompt="You are a research assistant.",
- )
+ from langchain_openai import AzureChatOpenAI
+ from azure.identity import DefaultAzureCredential, get_bearer_token_provider
+ token_provider = get_bearer_token_provider(
+     DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
+ agent = create_deep_agent(
+     model=AzureChatOpenAI(
+         azure_endpoint=..., azure_deployment="gpt-5.4",
+         api_version="2025-04-01-preview",
+         azure_ad_token_provider=token_provider),
+     tools=[internet_search],
+     system_prompt="You are a careful research assistant. ...",
+ )
```

Everything else (planning loop, filesystem, sub-agents, shell, prompts) is
unchanged — it's all inside the installed `deepagents` package.

## Layout

```
main.py                 # Foundry adapter (~85 lines): BaseAgent subclass + msg translator
graph.py                # deepagents harness (one create_deep_agent call) + AzureChatOpenAI
agent.yaml              # Hosted-agent manifest
azure.yaml              # azd service definition
Dockerfile              # python:3.12-slim + pip install -r requirements.txt
requirements.txt        # deepagents, langchain-openai, agent-framework, ...
infra/                  # Bicep
```

`main.py` is **byte-identical** to the one in the sister
`langgraph-foundry-sample` repo (modulo the agent class name) — that's the
point: the adapter is the same for any compiled-LangGraph object, and
`create_deep_agent` returns one.

## Foundry-hosting essentials

### 1. LLM via container managed identity (no API keys)

The hosted-agent container has a per-instance MI. Use it via
`get_bearer_token_provider` instead of `OPENAI_API_KEY`. See `_get_model()`
in [`graph.py`](graph.py).

### 2. Wrap the compiled graph in `BaseAgent` and serve it

[`main.py`](main.py) — three things to remember:

- **`run` must be `def`, not `async def`.** Stream branch returns a
  `ResponseStream`; non-stream branch returns a coroutine.
- **Wrap the streaming generator in `ResponseStream(..., finalizer=...)`.**
  A bare async generator works in the CLI but the Playground crashes with
  `'async_generator' object has no attribute 'get_final_response'`.
- **Bridge the sync `app.invoke` with `asyncio.to_thread`.** LangGraph
  compiled apps are sync; the Foundry server is async.

### 3. Pinned SDK versions

```
agent-framework==1.0.0rc3
azure-ai-agentserver-agentframework==1.0.0b16
deepagents>=0.6.1
```

The rc3 surface uses `AgentResponse` / `Message` / `Content` (NOT the older
`AgentRunResponse` / `ChatMessage` / `TextContent` names). Mismatched
versions surface as `agent_version_failed` with no clear error.

## Deploy to a Foundry project

```powershell
$projectId = "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>/projects/<project>"

azd ai agent init -p $projectId -d gpt-5.4 --src .
azd deploy
```

After the first deploy, the per-agent managed identity needs:

- `AcrPull` on the project's ACR
- `Cognitive Services OpenAI User` + `Cognitive Services User` on the Foundry account
- **`Foundry User`** on the **project** scope (role definition GUID
  `53ca6127-db72-4b80-b1b0-d745d6d5456d`; the Azure CLI rejects
  `--role "Foundry User"` by name, pass the GUID)
- `Storage Blob Data Contributor` on the project's storage account

Without `Foundry User` on the project scope, invocations return
`401 PermissionDenied "Principal does not have access to API/Operation."`

## Smoke test

```powershell
azd ai agent invoke "Plan and write up a one-page brief on what Deep Agents are and how they differ from a single ReAct agent."
```

Expected: the model emits a `write_todos` plan, calls `internet_search`,
writes intermediate notes to virtual files, and returns the final brief.

## Caveats specific to Deep Agents in a Foundry container

| Built-in | Behavior in a Foundry hosted-agent container |
|---|---|
| **Filesystem tools** | Read/write the container's local filesystem. Foundry containers are **ephemeral** — files survive only within the lifetime of one container instance, not across redeploys/scale events. Use as a per-turn scratchpad, not durable storage. |
| **Shell `execute`** | Runs commands inside the agent container. The container itself is the sandbox; for a hardened public demo, omit it from `tools=[...]`. |
| **Sub-agents (`task`)** | Spawns child LangGraph runs using the same model client. Just multiplies token cost. |
| **Streaming** | Our adapter's `_stream` waits for `_invoke` to complete and emits one final chunk. Deep Agents' richer event stream (planning, tool calls) collapses to a single assistant message in the Playground. To surface intermediate updates, swap `_stream` to call `app.astream_events(...)` and forward `AgentResponseUpdate` chunks per event. |
| **Model choice** | This sample defaults to `gpt-5.4` — strong enough for the planning + sub-agents + tool loop. Smaller models (`gpt-4.1-mini`, `gpt-5.4-nano`) often skip the plan and produce shallower output. |
