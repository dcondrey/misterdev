# Container image for the misterdev MCP server (used by Glama and any MCP host).
# The server speaks MCP over stdio and lists its tools without needing an API key
# or a project, so it introspects cleanly. Provide OPENROUTER_API_KEY (or the
# provider key your project.yaml names) at runtime to actually build.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# debian-slim ships a PEP 668 "externally managed" Python that rejects
# system-wide installs, so build into a venv. Install misterdev with the mcp
# extra (the FastMCP server dependency).
RUN pip install --no-cache-dir uv \
 && uv venv /app/.venv \
 && uv pip install --python /app/.venv/bin/python '.[mcp]'

ENTRYPOINT ["/app/.venv/bin/misterdev-mcp"]
