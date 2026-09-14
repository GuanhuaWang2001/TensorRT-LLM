# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from agent_flow.agent_runtime import resolve_agent_runtime, validate_agents
from agent_flow.config import CODEX_DEFAULT_MODEL

ROLES = ("projector", "analyzer", "reporter")


def test_backend_only_switch_uses_the_selected_backend_default_model():
    runtime = resolve_agent_runtime(
        {"agents": {"defaults": {"backend": "codex"}}},
        "reporter",
        legacy_backend="claude-code",
        legacy_model="claude-legacy",
    )
    assert runtime.backend == "codex"
    assert runtime.model == CODEX_DEFAULT_MODEL


def test_role_backend_switch_does_not_inherit_another_backends_default_model():
    runtime = resolve_agent_runtime(
        {
            "agents": {
                "defaults": {"backend": "codex", "model": "gpt-default"},
                "roles": {"reporter": {"backend": "claude-code"}},
            }
        },
        "reporter",
        legacy_backend="claude-code",
        legacy_model="claude-legacy",
    )
    assert runtime.backend == "claude-code"
    assert runtime.model != "gpt-default"


def test_requested_perf_role_matrix_resolves_per_role():
    data = {
        "agents": {
            "defaults": {
                "backend": "codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "medium",
            },
            "roles": {
                role: {"model": "gpt-6-astra", "reasoning_effort": "ultra"}
                for role in ("projector", "analyzer")
            },
        }
    }
    for role in ROLES:
        runtime = resolve_agent_runtime(
            data,
            role,
            legacy_backend="claude-code",
            legacy_model="claude-legacy",
        )
        assert runtime.backend == "codex"
        if role in ("projector", "analyzer"):
            assert (runtime.model, runtime.reasoning_effort) == ("gpt-6-astra", "ultra")
        else:
            assert (runtime.model, runtime.reasoning_effort) == ("gpt-5.6-sol", "medium")


def test_validate_agents_checks_roles_and_portable_mcp_shape():
    data = {
        "agents": {
            "defaults": {
                "extra_mcp_servers": {
                    "agent-tools": {"type": "http", "url": "https://example.test"},
                    "legacy": {"type": "sse", "url": "https://example.test"},
                }
            },
            "roles": {"optimizer": {}},
        }
    }
    errors = validate_agents(data, known_roles=ROLES)
    assert any("reserved" in error for error in errors)
    assert any("stdio' or 'http" in error for error in errors)
    assert any("optimizer" in error for error in errors)


def test_structural_validation_defers_role_names_to_outer_workflow():
    data = {"agents": {"roles": {"optimizer": {"backend": "codex"}}}}
    assert validate_agents(data, known_roles=None) == []
