from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
async def get_by_tag(tag: str, limit: int = 50) -> str:
    """Get blog content by its tag.

    Args:
        tag: the tag associated with content
        limit: the number of results to include
    """
    return f"Tried to retrieve results for {tag}!"

@mcp.tool()
async def get_by_text(query: str, limit: int = 50) -> str:
    """Get blog content by text in content.

    Args:
        query: text for an exact match
        limit: the number of results to include
    """
    return f"Tried to retrieve results via text matches for {query}!"

@mcp.tool()
async def rebuild() -> bool:
    """Rebuild text index. Useful for when contents have changed on disk"""
    return True


if __name__ == "__main__":
    mcp.run(transport='stdio')
