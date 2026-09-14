# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-level backend, model, and MCP routing for workflow agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL, BackendKind

AGENTS_FIELD = "agents"
_CONFIG_FIELDS = frozenset({"backend", "model", "reasoning_effort", "extra_mcp_servers"})
_BACKENDS = frozenset({"claude-code", "codex"})
_MCP_FIELDS = frozenset({"type", "command", "args", "env", "url", "headers"})


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Resolved runtime selection for one workflow role."""

    backend: BackendKind
    model: str
    reasoning_effort: str | None = None
    extra_mcp_servers: dict[str, Any] | None = None


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"'{path}' must be a mapping, got {type(value).__name__}")
        return {}
    return value


def _validate_runtime_fields(
    value: Any,
    path: str,
    errors: list[str],
) -> Mapping[str, Any]:
    block = _mapping(value, path, errors)
    unknown = sorted(str(key) for key in block if key not in _CONFIG_FIELDS)
    if unknown:
        errors.append(f"'{path}' has unknown field(s) {', '.join(repr(key) for key in unknown)}")

    backend = block.get("backend")
    if backend is not None and backend not in _BACKENDS:
        errors.append(f"'{path}.backend' must be one of {sorted(_BACKENDS)}, got {backend!r}")
    for name in ("model", "reasoning_effort"):
        raw = block.get(name)
        if raw is not None and (not isinstance(raw, str) or not raw.strip()):
            errors.append(f"'{path}.{name}' must be a non-empty string")

    servers = block.get("extra_mcp_servers")
    if servers is not None:
        _validate_mcp_servers(servers, f"{path}.extra_mcp_servers", errors)
    return block


def _validate_mcp_servers(value: Any, path: str, errors: list[str]) -> None:
    servers = _mapping(value, path, errors)
    for raw_name, raw_server in servers.items():
        name = str(raw_name)
        server_path = f"{path}.{name}"
        if not isinstance(raw_name, str) or not name.strip():
            errors.append(f"'{path}' server names must be non-empty strings")
            continue
        if name == "agent-tools":
            errors.append(f"'{server_path}' uses the reserved server name 'agent-tools'")
        server = _mapping(raw_server, server_path, errors)
        unknown = sorted(str(key) for key in server if key not in _MCP_FIELDS)
        if unknown:
            errors.append(
                f"'{server_path}' has unsupported field(s) "
                f"{', '.join(repr(key) for key in unknown)}"
            )
        transport = server.get("type")
        has_command = "command" in server
        has_url = "url" in server
        if transport not in (None, "stdio", "http"):
            errors.append(f"'{server_path}.type' must be 'stdio' or 'http'")
            continue
        if transport == "http" or has_url:
            if transport != "http":
                errors.append(f"'{server_path}.type' must be 'http' when 'url' is set")
            _nonempty_string(server.get("url"), f"{server_path}.url", errors)
            if has_command:
                errors.append(f"'{server_path}' cannot set both 'url' and 'command'")
            headers = server.get("headers")
            if headers is not None and not _string_mapping(headers):
                errors.append(f"'{server_path}.headers' must map strings to strings")
            for field in ("args", "env"):
                if field in server:
                    errors.append(f"'{server_path}.{field}' is only valid for stdio")
        else:
            _nonempty_string(server.get("command"), f"{server_path}.command", errors)
            args = server.get("args")
            if args is not None and (
                not isinstance(args, list) or not all(isinstance(item, str) for item in args)
            ):
                errors.append(f"'{server_path}.args' must be a list of strings")
            env = server.get("env")
            if env is not None and not _string_mapping(env):
                errors.append(f"'{server_path}.env' must map strings to strings")
            if "headers" in server:
                errors.append(f"'{server_path}.headers' is only valid for HTTP")


def _nonempty_string(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"'{path}' must be a non-empty string")


def _string_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


def validate_agents(
    data: Mapping[str, Any],
    *,
    known_roles: tuple[str, ...] | None,
) -> list[str]:
    """Return validation errors for the optional top-level ``agents`` block.

    ``known_roles=None`` performs structural validation only. This is used by
    perf-optimize's shared perf-analyze validation pass; the outer validator
    applies the optimize-specific role set afterwards.
    """
    if AGENTS_FIELD not in data:
        return []
    errors: list[str] = []
    agents = _mapping(data.get(AGENTS_FIELD), AGENTS_FIELD, errors)
    unknown = sorted(str(key) for key in agents if key not in {"defaults", "roles"})
    if unknown:
        errors.append(
            f"'{AGENTS_FIELD}' has unknown field(s) {', '.join(repr(key) for key in unknown)}"
        )
    _validate_runtime_fields(agents.get("defaults"), "agents.defaults", errors)
    roles = _mapping(agents.get("roles"), "agents.roles", errors)
    if known_roles is not None:
        bad_roles = sorted(str(role) for role in roles if role not in known_roles)
        if bad_roles:
            errors.append(
                "'agents.roles' has unknown role(s) "
                f"{', '.join(repr(role) for role in bad_roles)}; "
                f"valid roles are {list(known_roles)}"
            )
    for role, config in roles.items():
        _validate_runtime_fields(config, f"agents.roles.{role}", errors)
    return errors


def _default_model(backend: BackendKind) -> str:
    return CODEX_DEFAULT_MODEL if backend == "codex" else CLAUDE_CODE_DEFAULT_MODEL


def resolve_agent_runtime(
    data: Mapping[str, Any],
    role: str,
    *,
    legacy_backend: BackendKind,
    legacy_model: str,
) -> AgentRuntimeConfig:
    """Resolve one role after :func:`validate_agents` has succeeded."""
    agents = data.get(AGENTS_FIELD)
    if not isinstance(agents, Mapping):
        return AgentRuntimeConfig(backend=legacy_backend, model=legacy_model)
    defaults = agents.get("defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    roles = agents.get("roles")
    roles = roles if isinstance(roles, Mapping) else {}
    override = roles.get(role)
    override = override if isinstance(override, Mapping) else {}

    default_backend = defaults.get("backend")
    role_backend = override.get("backend")
    backend = role_backend or default_backend or legacy_backend

    model = override.get("model")
    if model is None:
        role_switched_from_default = role_backend is not None and role_backend != default_backend
        if defaults.get("model") is not None and not role_switched_from_default:
            model = defaults["model"]
        elif default_backend is None and role_backend is None:
            model = legacy_model
        else:
            model = _default_model(backend)

    effort = override.get("reasoning_effort", defaults.get("reasoning_effort"))
    servers = override.get("extra_mcp_servers", defaults.get("extra_mcp_servers"))
    return AgentRuntimeConfig(
        backend=backend,
        model=model,
        reasoning_effort=effort,
        extra_mcp_servers=dict(servers) if isinstance(servers, Mapping) else None,
    )
