import os

from openai import OpenAI
from crewai import Agent, Crew, Process, Task

from search_tool import search_web


# =========================================================
# GROQ MODELS
# =========================================================

SUPPORTED_MODELS = {
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "Qwen3.6 27B": "qwen/qwen3.6-27b",
}


# =========================================================
# SEARCH
# =========================================================

def collect_web_research(
    topic: str,
    max_results_per_query: int = 6
) -> list:

    queries = [
        topic,
        f"{topic} latest developments",
        f"{topic} benefits challenges",
    ]

    all_results = []
    seen_urls = set()

    for query in queries:

        results = search_web(
            query=query,
            max_results=max_results_per_query
        )

        for result in results:

            url = result.get(
                "url",
                ""
            ).strip()

            if url and url not in seen_urls:

                seen_urls.add(url)

                all_results.append(
                    result
                )

    return all_results[:15]


# =========================================================
# FORMAT SOURCES
# =========================================================

def format_sources(
    results: list
) -> str:

    if not results:
        return (
            "No web sources were found."
        )

    output = []

    for index, result in enumerate(
        results,
        start=1
    ):

        output.append(
            f"""
SOURCE {index}

Title:
{result.get("title", "")}

Description:
{result.get("body", "")}

URL:
{result.get("url", "")}
"""
        )

    return "\n".join(output)


# =========================================================
# DIRECT GROQ
# =========================================================

def call_groq(
    model_name: str,
    topic: str,
    research: str,
    crew_analysis: str
) -> str:

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=(
            "https://api.groq.com/openai/v1"
        )
    )

    system_prompt = """
You are an expert research report writer.

Create accurate, clear and structured reports.

Use ONLY the supplied research sources.

Never invent:
- facts
- statistics
- citations
- URLs

If the supplied sources are insufficient,
clearly state that limitation.

Return Markdown.
"""

    user_prompt = f"""
RESEARCH TOPIC:

{topic}


WEB RESEARCH:

{research}


CREWAI ANALYSIS:

{crew_analysis}


Create the final research report.

Use exactly these sections:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Current Developments

# Challenges and Limitations

# Conclusion

# Sources

Under Sources, list the actual URLs from
the supplied web research.
"""

    response = client.chat.completions.create(
        model=model_name,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,

        max_tokens=6000
    )

    result = response.choices[0].message.content

    if not result:

        raise RuntimeError(
            "Groq returned an empty response."
        )

    return result


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_report(
    model_name: str,
    topic: str
) -> str:

    # -----------------------------------------------------
    # API key
    # -----------------------------------------------------

    if not os.getenv(
        "GROQ_API_KEY"
    ):

        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    if model_name not in (
        SUPPORTED_MODELS.values()
    ):

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # -----------------------------------------------------
    # STEP 1 — DuckDuckGo
    # -----------------------------------------------------

    web_results = collect_web_research(
        topic
    )

    if not web_results:

        raise RuntimeError(
            "DuckDuckGo returned no search results. "
            "Please try another topic."
        )

    research_context = format_sources(
        web_results
    )

    # -----------------------------------------------------
    # STEP 2 — CrewAI Single Agent
    # -----------------------------------------------------

    researcher = Agent(

        role="Research Analyst",

        goal=(
            "Analyze the provided web research "
            "and identify the most important "
            "facts and findings."
        ),

        backstory=(
            "You are a careful research analyst. "
            "You analyze supplied sources without "
            "inventing information."
        ),

        tools=[],

        allow_delegation=False,

        max_iter=1,

        verbose=False
    )

    task = Task(

        description=f"""
Analyze this research topic:

{topic}

Here are the web sources already collected:

{research_context}

Identify:

1. Main findings
2. Important facts
3. Relevant developments
4. Benefits
5. Challenges
6. Important source information

Do not search the web yourself.
Do not invent information.
Use only the supplied sources.
""",

        expected_output=(
            "A concise factual analysis "
            "of the supplied web research."
        ),

        agent=researcher
    )

    crew = Crew(

        agents=[researcher],

        tasks=[task],

        process=Process.sequential,

        verbose=False
    )

    crew_result = crew.kickoff()

    crew_analysis = getattr(
        crew_result,
        "raw",
        str(crew_result)
    )

    # -----------------------------------------------------
    # STEP 3 — Direct Groq final report
    # -----------------------------------------------------

    final_report = call_groq(
        model_name=model_name,
        topic=topic,
        research=research_context,
        crew_analysis=crew_analysis
    )

    return final_report
