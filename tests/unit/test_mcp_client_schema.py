from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.agent.mcp_client import (
    MCPTool,
    _normalize_mcp_input_schema,
    _prepare_config,
    validate_mcp_tool_prefix,
)


class TestNormalizeMcpInputSchema:
    def test_lifts_property_level_required_booleans_to_parent_required_array(self):
        schema = {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "required": True},
                "market": {"type": "string", "required": False},
            },
        }

        normalized = _normalize_mcp_input_schema(schema)

        assert normalized["required"] == ["stock_code"]
        assert "required" not in normalized["properties"]["stock_code"]
        assert "required" not in normalized["properties"]["market"]
        assert schema["properties"]["stock_code"]["required"] is True

    def test_preserves_existing_required_arrays_while_fixing_nested_objects(self):
        schema = {
            "type": "object",
            "required": ["server"],
            "properties": {
                "server": {
                    "type": "object",
                    "required": ["transport"],
                    "properties": {
                        "transport": {"type": "string"},
                        "stock_code": {"type": "string", "required": True},
                        "market": {"type": "string", "required": False},
                    },
                }
            },
        }

        normalized = _normalize_mcp_input_schema(schema)

        assert normalized["required"] == ["server"]
        assert normalized["properties"]["server"]["required"] == [
            "transport",
            "stock_code",
        ]
        assert (
            "required"
            not in normalized["properties"]["server"]["properties"]["stock_code"]
        )
        assert (
            "required" not in normalized["properties"]["server"]["properties"]["market"]
        )

    def test_preserves_parent_required_flag_for_nested_object_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "required": True,
                    "properties": {
                        "transport": {"type": "string", "required": True},
                    },
                }
            },
        }

        normalized = _normalize_mcp_input_schema(schema)

        assert normalized["required"] == ["server"]
        assert normalized["properties"]["server"]["required"] == ["transport"]
        assert (
            "required"
            not in normalized["properties"]["server"]["properties"]["transport"]
        )

    def test_ignores_non_boolean_required_values_and_non_dict_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "server": "invalid-property-schema",
                "market": {"type": "string", "required": "yes"},
                "stock_code": {"type": "string", "required": True},
            },
        }

        normalized = _normalize_mcp_input_schema(schema)

        assert normalized["required"] == ["stock_code"]
        assert normalized["properties"]["server"] == "invalid-property-schema"
        assert normalized["properties"]["market"]["required"] == "yes"
        assert "required" not in normalized["properties"]["stock_code"]
        assert schema["properties"]["server"] == "invalid-property-schema"
        assert schema["properties"]["market"]["required"] == "yes"


class TestMCPToolSchemaNormalization:
    def test_mcp_tool_accepts_property_level_required_booleans(self):
        mcp_tool = SimpleNamespace(
            name="quote_lookup",
            description="Lookup a quote",
            inputSchema={
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string", "required": True},
                    "market": {"type": "string", "required": False},
                },
            },
        )

        tool = MCPTool(mcp_tool, MagicMock(), "gf-securities")

        assert tool.parameters["required"] == ["stock_code"]
        assert "required" not in tool.parameters["properties"]["stock_code"]
        assert "required" not in tool.parameters["properties"]["market"]

    @pytest.mark.asyncio
    async def test_mcp_tool_uses_prefixed_public_name_and_original_call_name(self):
        mcp_tool = SimpleNamespace(
            name="quote_lookup",
            description="Lookup a quote",
            inputSchema={"type": "object", "properties": {}},
        )
        mcp_client = MagicMock()
        mcp_client.call_tool_with_reconnect = AsyncMock(return_value="ok")
        context = SimpleNamespace(tool_call_timeout=12)

        tool = MCPTool(mcp_tool, mcp_client, "gf-securities", tool_prefix="gf_")

        assert tool.name == "gf_quote_lookup"
        result = await tool.call(context, stock_code="600000")
        assert result == "ok"
        mcp_client.call_tool_with_reconnect.assert_called_once_with(
            tool_name="quote_lookup",
            arguments={"stock_code": "600000"},
            read_timeout_seconds=timedelta(seconds=12),
        )

    def test_mcp_tool_default_prefix_preserves_original_public_name(self):
        mcp_tool = SimpleNamespace(
            name="quote_lookup",
            description="Lookup a quote",
            inputSchema={"type": "object", "properties": {}},
        )

        tool = MCPTool(mcp_tool, MagicMock(), "gf-securities")

        assert tool.name == "quote_lookup"


class TestMCPConfigMetadata:
    def test_prepare_config_strips_tool_prefix_and_active_metadata(self):
        prepared = _prepare_config(
            {
                "command": "python",
                "args": ["-m", "server"],
                "active": True,
                "tool_prefix": "gf_",
            }
        )

        assert prepared == {"command": "python", "args": ["-m", "server"]}

    def test_prepare_nested_config_strips_tool_prefix_and_active_metadata(self):
        prepared = _prepare_config(
            {
                "mcpServers": {
                    "gf": {
                        "url": "https://example.test/mcp",
                        "transport": "sse",
                        "active": False,
                        "tool_prefix": "gf_",
                    }
                }
            }
        )

        assert prepared == {
            "url": "https://example.test/mcp",
            "transport": "sse",
        }

    @pytest.mark.parametrize("prefix", ["", "abc", "abc_123-", "A-B_9"])
    def test_validate_mcp_tool_prefix_accepts_valid_values(self, prefix):
        assert validate_mcp_tool_prefix(prefix) == prefix

    @pytest.mark.parametrize("prefix", [123, "bad prefix", "bad.prefix", "a" * 65])
    def test_validate_mcp_tool_prefix_rejects_invalid_values(self, prefix):
        with pytest.raises(ValueError):
            validate_mcp_tool_prefix(prefix)


class TestMCPToolPrefixRouteMetadata:
    def _build_service(self, tool_mgr):
        from astrbot.dashboard.services.tools_service import ToolsService

        lifecycle = SimpleNamespace(
            provider_manager=SimpleNamespace(llm_tools=tool_mgr)
        )
        return ToolsService(lifecycle)

    @pytest.mark.asyncio
    async def test_add_mcp_server_ignores_nested_tool_prefix(self):
        tool_mgr = MagicMock()
        tool_mgr.load_mcp_config.return_value = {"mcpServers": {}}
        tool_mgr.save_mcp_config.return_value = True
        tool_mgr.test_mcp_server_connection = AsyncMock(return_value=[])
        tool_mgr.enable_mcp_server = AsyncMock()

        service = self._build_service(tool_mgr)
        result = await service.add_mcp_server(
            {
                "name": "nested",
                "active": True,
                "mcpServers": {
                    "nested": {
                        "url": "https://example.test/mcp",
                        "transport": "sse",
                        "tool_prefix": "nested_",
                    }
                },
            },
        )

        assert result == "Successfully added MCP server nested"
        saved_config = tool_mgr.save_mcp_config.call_args.args[0]
        assert saved_config["mcpServers"]["nested"]["tool_prefix"] == ""

    @pytest.mark.asyncio
    async def test_add_mcp_server_uses_top_level_tool_prefix(self):
        tool_mgr = MagicMock()
        tool_mgr.load_mcp_config.return_value = {"mcpServers": {}}
        tool_mgr.save_mcp_config.return_value = True
        tool_mgr.test_mcp_server_connection = AsyncMock(return_value=[])
        tool_mgr.enable_mcp_server = AsyncMock()

        service = self._build_service(tool_mgr)
        result = await service.add_mcp_server(
            {
                "name": "nested",
                "active": True,
                "tool_prefix": "top_",
                "mcpServers": {
                    "nested": {
                        "url": "https://example.test/mcp",
                        "transport": "sse",
                        "tool_prefix": "nested_",
                    }
                },
            },
        )

        assert result == "Successfully added MCP server nested"
        saved_config = tool_mgr.save_mcp_config.call_args.args[0]
        assert saved_config["mcpServers"]["nested"]["tool_prefix"] == "top_"

    @pytest.mark.asyncio
    async def test_update_mcp_server_preserves_prefix_when_top_level_omitted(self):
        tool_mgr = MagicMock()
        tool_mgr.load_mcp_config.return_value = {
            "mcpServers": {
                "nested": {
                    "url": "https://old.example.test/mcp",
                    "transport": "sse",
                    "active": False,
                    "tool_prefix": "old_",
                }
            }
        }
        tool_mgr.save_mcp_config.return_value = True
        tool_mgr.mcp_server_runtime_view = {}
        tool_mgr.disable_mcp_server = AsyncMock()
        tool_mgr.enable_mcp_server = AsyncMock()

        service = self._build_service(tool_mgr)
        result = await service.update_mcp_server(
            {
                "name": "nested",
                "active": False,
                "mcpServers": {
                    "nested": {
                        "url": "https://new.example.test/mcp",
                        "transport": "sse",
                        "tool_prefix": "nested_",
                    }
                },
            },
        )

        assert result == "Successfully updated MCP server nested"
        saved_config = tool_mgr.save_mcp_config.call_args.args[0]
        assert saved_config["mcpServers"]["nested"]["tool_prefix"] == "old_"
