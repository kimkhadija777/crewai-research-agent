from ddgs import DDGS


def search_web(query: str, max_results: int = 8) -> list:
    """
    Search DuckDuckGo and return clean search results.
    """

    try:
        results = DDGS().text(
            query,
            max_results=max_results
        )

        if not results:
            return []

        cleaned_results = []

        for result in results:
            cleaned_results.append({
                "title": result.get("title", ""),
                "body": result.get("body", ""),
                "url": result.get("href", "")
            })

        return cleaned_results

    except Exception as exc:
        raise RuntimeError(
            f"DuckDuckGo search failed: {exc}"
        )
