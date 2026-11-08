# Part 4 : Resilience & Interoperability 
# Task 14 : Expose your lookup tool via MCP.
import asyncio
import json

from fastmcp import Client


# connect to the mcp server

MCP_URL = "http://127.0.0.1:8001/mcp"


async def main():

    async with Client(MCP_URL) as client:

        print("connected to mcp server")

        # show available tools

        tools = await client.list_tools()

        print("\navailable tools:")

        for tool in tools:
            print(tool.name)

        # call tool for two record ids

        record_ids = ["5", "10"]

        for record_id in record_ids:

            print("\nchecking record:", record_id)

            result = await client.call_tool(
                "get_application_status",
                {"record_id": record_id}
            )

            # print standardized mcp response

            if hasattr(result, "model_dump"):

                print(
                    json.dumps(
                        result.model_dump(mode="json"),
                        indent=4,
                        default=str
                    )
                )

            else:

                print(result)


if __name__ == "__main__":

    asyncio.run(main())