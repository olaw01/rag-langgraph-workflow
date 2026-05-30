#---------------------------------------------------------------------------------------------------------------------
# python -m pip install "mcp>=1.9.2" "langchain-mcp-adapters>=0.2.1" "langchain-openai>=0.1"
#---------------------------------------------------------------------------------------------------------------------
from pathlib import Path
from dotenv import load_dotenv
import os

import asyncio # asyncio is required because MCP runs in an async style (sessions/IO)

from mcp import ClientSession, StdioServerParameters # ClientSession = MCP protocol session; StdioServerParameters = how to launch a stdio server
from mcp.client.stdio import stdio_client # stdio_client starts the server as a subprocess and provides read/write streams

from langchain_mcp_adapters.tools import load_mcp_tools # loads MCP tools and converts them into LangChain Tools
from langchain.chat_models import init_chat_model # init_chat_model creates a LangChain chat model
from langchain.agents import create_agent # create_agent creates an agent that can decide to call tools

#---------------------------------------------------------------------------------------------------------------------
# MCP vs. FUNCTION CALLING
#
# Function calling: the "model -> tool" mechanism; typically, a tool is defined and maintained IN YOUR CODE.
# MCP: the "application -> tool server" standard. A TOOL LIVES OUTSIDE the application, is shared, and versioned.
#
# The best way to talk:
# Function calling is the MODEL -> TOOL INTERFACE, and
# MCP is the standard for integrating APP -> TOOL SERVERS.
#
# What happens in this project?
# 1. Start an MCP stdio server with add and multiply tools.
# 2. Client opens an MCP session over stdio.
# 3. langchain-mcp-adapters converts MCP tools into LangChain Tool objects.
# 4. A LangChain agent uses those tools to answer math queries.
#---------------------------------------------------------------------------------------------------------------------

# main is async because we will await MCP session operations and agent calls
async def main() -> None:

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    # 1) Define how to launch the MCP server process (stdio transport).
    server_script = Path(__file__).parent / "server_math.py"
    server_params = StdioServerParameters(
        command="python",                                       # Run python
        args=[str(server_script)],    # ...and execute server_math.py (our MCP server)
    )


    # 2) Start the server subprocess and get two streams: read and write
    # read = server responses; write = requests we send to the server
    async with stdio_client(server_params) as (read, write):

        # 3) Create an MCP protocol session over those stdio streams (client-server connection)
        async with ClientSession(read, write) as session:

            # 4) Handshake / initialize the MCP session (initialize protocol and negotiate capabilities)
            await session.initialize()

            # 5) Load MCP tools and adapt them into LangChain Tool objects.
            # after this, the agent sees them as normal LangChain tools
            tools = await load_mcp_tools(session)

            # 6) Create an LLM wrapper and an agent that can call tools
            # create an LLM (OpenAI) - it will decide whether to call tools
            model = init_chat_model("openai:gpt-4o-mini")

            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt="Use tools when math is required. Be concise.",
            )

            # 7) Run the agent with a question that should trigger tool use.
            # run the agent asynchronously
            out = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "What's (3 + 5) * 12?"}]}
            )
            print(out)
            final_text = out["messages"][-1].content
            print(final_text)


#  Run async main via asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())