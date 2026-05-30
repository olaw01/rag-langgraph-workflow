from mcp.server.fastmcp import FastMCP # import FastMCP - the simplest way to create an MCP server in python

#---------------------------------------------------------------------------------------------------------------------
# This file starts a local MCP server that exposes tools over stdio
#---------------------------------------------------------------------------------------------------------------------

# create an MCP server named "math-server" (server identity/name)
mcp = FastMCP("math-server")

# decorator registers the function as an MCP tool (callable by clients)
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    # tool description - the client/LLM may use it for understanding
    return a + b

# register a second tool
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    # tool description - the client/LLM may use it for understanding
    return a * b


if __name__ == "__main__":
    # Run the MCP server using stdio (stdin/stdout) transport
    mcp.run(transport="stdio")