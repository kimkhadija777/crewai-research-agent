import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


def search_web(query: str, max_results: int = 8) -> list:
    """
    Search DuckDuckGo HTML results directly.

    This avoids the ddgs package and its
    browser impersonation dependency.
    """

    url = (
        "https://html.duckduckgo.com/html/"
        f"?q={quote(query)}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.select(
            ".result"
        ):

            title_element = result.select_one(
                ".result__title"
            )

            link_element = result.select_one(
                ".result__a"
            )

            snippet_element = result.select_one(
                ".result__snippet"
            )

            if not link_element:
                continue

            title = (
                title_element.get_text(
                    " ",
                    strip=True
                )
                if title_element
                else link_element.get_text(
                    " ",
                    strip=True
                )
            )

            link = link_element.get(
                "href",
                ""
            )

            snippet = (
                snippet_element.get_text(
                    " ",
                    strip=True
                )
                if snippet_element
                else ""
            )

            if link:

                results.append({
                    "title": title,
                    "body": snippet,
                    "url": link
                })

            if len(results) >= max_results:
                break

        return results

    except Exception as exc:

        raise RuntimeError(
            f"DuckDuckGo search failed: {exc}"
        )
