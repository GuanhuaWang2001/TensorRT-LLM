from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from agent_flow.backends.base import Backend, BackendClient, ResultEvent
from agent_flow.types import ToolCallEvent, UsageInfo


class FakeClient(BackendClient):
    def __init__(
        self,
        *,
        text: str = "ok",
        tool_calls: list[ToolCallEvent] | None = None,
        error: Exception | None = None,
        usage: UsageInfo | None = None,
        skills: list[str] | None = None,
        turns: list[dict] | None = None,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.error = error
        self.usage = usage
        self.skills = skills
        self.turns = turns
        self.messages: list[str] = []
        self.send_count = 0
        self.closed = False

    async def list_available_skills(self) -> list[str] | None:
        return self.skills

    async def send_message(self, message: str):
        self.messages.append(message)
        self.send_count += 1
        plan = self.turns[min(self.send_count - 1, len(self.turns) - 1)] if self.turns else {}
        error = plan.get("error", self.error)
        if error is not None:
            raise error
        for tool_call in plan.get("tool_calls", self.tool_calls):
            yield tool_call
        yield ResultEvent(
            text=plan.get("text", self.text),
            usage=plan.get("usage", self.usage),
        )


class FakeBackend(Backend):
    def __init__(self, plans: list[dict] | None = None) -> None:
        self.plans = plans or [{"text": "ok"}]
        self.create_client_calls = 0
        self.enter_count = 0
        self.exit_count = 0
        self.client_exit_count = 0
        self.clients: list[FakeClient] = []

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exit_count += 1

    @asynccontextmanager
    async def create_client(
        self,
        system_prompt: str,
        model: str,
        tools: list | None = None,
        hooks: dict | None = None,
        disallowed_tools: list[str] | None = None,
        extra_mcp_servers: dict | None = None,
        cwd: Path | None = None,
    ):
        plan = self.plans[min(self.create_client_calls, len(self.plans) - 1)]
        self.create_client_calls += 1
        client = FakeClient(
            text=plan.get("text", "ok"),
            tool_calls=plan.get("tool_calls"),
            error=plan.get("error"),
            usage=plan.get("usage"),
            skills=plan.get("skills"),
            turns=plan.get("turns"),
        )
        client.system_prompt = system_prompt
        client.model = model
        client.tools = tools
        client.hooks = hooks
        client.disallowed_tools = disallowed_tools
        client.extra_mcp_servers = extra_mcp_servers
        client.cwd = cwd
        self.clients.append(client)
        try:
            yield client
        finally:
            client.closed = True
            self.client_exit_count += 1
