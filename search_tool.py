from typing import Type

from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The web search query."
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the internet using DuckDuckGo. "
        "Use this tool to find relevant and recent "
        "information about the research topic."
    )

    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """Search DuckDuckGo and return formatted results."""

        query = query.strip()

        if not query:
            return "No search query was provided."

        try:
            search_client = DDGS()

            results = search_client.text(
                query,
                max_results=8,
            )

            if not results:
                return "No search results were found."

            output = []

            for index, result in enumerate(results, start=1):

                title = result.get(
                    "title",
                    "Untitled",
                )

                description = result.get(
                    "body",
                    "No description available.",
                )

                url = result.get(
                    "href",
                    "",
                )

                output.append(
                    f"""
SOURCE {index}
Title: {title}
Description: {description}
URL: {url}
""".strip()
                )

            return "\n\n".join(output)

        except Exception as error:

            return (
                "DuckDuckGo search failed. "
                f"Error: {str(error)}"
            )
