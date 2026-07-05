# MCP (Model Context Protocol)

misterdev speaks MCP in both directions: it can run **as** an MCP server that an AI client drives, and it can **use** MCP servers to give a build access to external tools. Both need the `mcp` extra:

```bash
pip install 'misterdev[mcp]'
```

---

## Part A — misterdev as an MCP server

misterdev ships the `misterdev-mcp` entry point, a thin adapter over the same `ProjectOrchestrator` the CLI uses. Drive it in plain English from Claude Desktop, Claude Code, Cursor, or any MCP client.

The key property: the client only sends a short instruction and receives a short summary. **The entire orchestration — reading the codebase, symbol-graph context, multi-step reasoning, model selection, budget — runs in misterdev's own process with its own LLM key.** Your codebase never enters the client's context window, so the context-scaling misterdev exists to provide is fully preserved.

### The five tools

| Tool | Kind | What it does |
| --- | --- | --- |
| `list_projects` | read-only | List every project misterdev knows about. Start here. |
| `status` | read-only | Show a project's tasks and their state. |
| `scan` | registers | Discover projects under a directory and register them (idempotent). |
| `build` | destructive | Plan **and** execute a goal from scratch, verifying each change through the gates and reverting regressions. The main tool. |
| `run` | destructive | Execute an already-planned devplan's pending tasks (does not decompose a goal). |

`build` parameters: `path`, `goal` (plain English, or a mode word `debug` / `complete` / `review`), `budget` (default $10 — conservative for a client-triggered run; raise it explicitly), `dry_run`, `parallel`, `max_tasks`. `run` takes `path`, optional `task_id`, and `dry_run`.

The mutating tools carry honest MCP annotations. `build` and `run` are destructive (they edit files and make git commits) and refuse a dirty working tree — commit or stash first, or use `dry_run=True` to preview the plan without touching anything.

### Claude Desktop config

Add to `claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "misterdev": {
      "command": "misterdev-mcp",
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```

Then just ask: *"Have misterdev add rate limiting to the API, keep it under $5."* The client calls `build`; misterdev does the work in-process and returns a compact report of what was done, the gate results, and the cost.

---

## Part B — misterdev using MCP servers

A build can reach out to MCP servers and let the model call their discovered tools mid-task — bounded, opt-in, and constrained by an allowlist. Transports: stdio (subprocess), streamable-http, and sse. The last two connect to a remote `url` with optional auth, which is how misterdev reaches a hosted gateway like Glama that fronts many servers, without running any of them locally.

Everything here is additive and safe by construction: a missing SDK, a server that won't start, or one that hangs is simply absent — never an error, never a hang.

### Configure servers

Declare servers under `mcp.servers`. Each stdio server needs a `name` (the routing key) and a `command`; each remote server needs a `name`, `transport`, and `url`.

```yaml
mcp:
  servers:
    - name: "docs"
      command: "my-mcp-server"
      args: ["--root", "."]
      transport: "stdio"
    - name: "glama"
      transport: "http"                 # http | streamable-http | sse
      url: "https://glama.ai/mcp/instance/..."
      api_key_env: "GLAMA_API_KEY"      # Bearer token read from the environment
      # headers: { X-Custom: "value" }  # optional extra headers
  allow_tools: ["docs.search", "glama.web_fetch"]
```

For a remote gateway, the auth token is read from the environment variable named by `api_key_env` and sent as `Authorization: Bearer <token>` — the token stays out of config on disk. Extra static `headers` can be added alongside.

### The allowlist

`mcp.allow_tools` is an allowlist of tool names. An entry matches either the qualified `server.tool` form or a bare `tool` name. Omit it to allow every discovered tool. A tool not on the list is refused at both discovery and call time — a hosted gateway may expose a large catalog, so the allowlist is how you scope what a build can actually reach.

### Two levels: awareness vs. tool use

```yaml
orchestrator:
  mcp_enabled: true      # awareness only: discovered tools are described to the model
  mcp_tool_use: true     # bounded gathering loop: the model may CALL tools pre-edit
  mcp_max_tool_rounds: 3 # hard cap on gathering rounds
```

- **`mcp_enabled`** gates awareness injection only: discovered tools are described in the task context. The single-shot edit loop is unchanged.
- **`mcp_tool_use`** layers a bounded agentic pre-edit loop on top (and implies `mcp_enabled`). Each round the model sees the available tools and may reply with one line `CALL <server>.<tool> {json-args}` (or `NO_TOOL` to stop). The call runs through the timeout-guarded `call_tool` and its result is prepended to the task context, so the model gathers information *before* editing. `mcp_max_tool_rounds` (default 3) hard-caps the loop.

With `mcp_tool_use` off, the edit path is byte-identical to a build with no MCP configured; any failure degrades to gathering nothing. Plugin tools marked `gather_safe = True` participate in the same loop — see [plugins.md](plugins.md).
