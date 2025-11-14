"""Register custom tools for the agent server."""

from openhands.sdk.tool import register_tool
from openhands.sdk.logger import get_logger

logger = get_logger(__name__)


def npc_tool_factory(**kwargs):
    """Factory function that creates the NPC tool on demand.
    
    Must return a Sequence[ToolDefinition] (list of tools).
    """
    try:
        from custom_tools.npc_tool import create_npc_tool
        
        # Create tool and return as a list
        tool = create_npc_tool()
        logger.info("NPC tool factory called successfully")
        return [tool]  # ← Return as a LIST!
    except Exception as e:
        logger.error(f"Error in npc_tool_factory: {e}")
        import traceback
        traceback.print_exc()
        raise


def register_custom_tools():
    """Register all custom tools."""
    logger.info("Starting tool registration...")
    print("=" * 60)
    print("REGISTERING CUSTOM TOOLS")
    print("=" * 60)
    
    try:
        print("Registering tool with name 'chat_with_npc'...")
        register_tool("chat_with_npc", npc_tool_factory)
        
        print("✓ Successfully registered chat_with_npc tool")
        logger.info("Successfully registered chat_with_npc tool")
        return True
    except Exception as e:
        print(f"✗ Failed to register custom tools: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to register custom tools: {e}")
        return False


# Auto-register on import
print("custom_tools.register module loaded")