# Part 4 : Resilience & Interoperability
# Task 14 : Expose your lookup tool via MCP
from fastmcp import FastMCP

from job_status_tool import check_job_application_status


# create private mcp server

mcp = FastMCP("job application status server")


@mcp.tool()
def get_application_status(record_id: str) -> dict:
    """
    Check the status and details of a job application using its record ID.

    Args:
        record_id: The ID of the job application to look up.

    Returns:
        A dictionary containing the application status information.
    """

    result = check_job_application_status(record_id)

    return {
        "record_id": record_id,
        "result": result
    }


if __name__ == "__main__":

    print("starting private mcp server")
    print("mcp endpoint: http://127.0.0.1:8001/mcp")

    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001
    )