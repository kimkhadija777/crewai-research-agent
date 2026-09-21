from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., description="The web search query")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the web with DuckDuckGo and return "
        "titles, snippets, and URLs."
    )
    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(
                query,
                max_results=8
            )

            if not results:
                return "No search results found."

            output = []

            for i, item in enumerate(results, start=1):
                title = item.get("title", "Untitled")
                body = item.get(
                    "body",
                    "No description available"
                )
                url = item.get("href", "")

                output.append(
                    f"[{i}] {title}\n"
                    f"{body}\n"
                    f"URL: {url}"
                )

            return "\n\n".join(output)

        except Exception as exc:
            return f"Web search failed: {exc}"
