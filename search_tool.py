from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS


class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The search query to find relevant information on the web."
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the web using DuckDuckGo. "
        "Use this tool to find current and relevant information "
        "for the research topic."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(
                query,
                max_results=8
            )

            if not results:
                return "No search results were found."

            formatted_results = []

            for i, result in enumerate(results, start=1):
                title = result.get("title", "No title")
                body = result.get("body", "No description")
                url = result.get("href", "")

                formatted_results.append(
                    f"""
SOURCE {i}
Title: {title}
Description: {body}
URL: {url}
"""
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Search error: {str(e)}"
