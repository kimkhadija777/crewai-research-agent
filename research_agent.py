import os

from crewai import Agent, Crew, Process, Task, LLM
from ddgs import DDGS


# ---------------------------------------------------------
# Current Groq models
# ---------------------------------------------------------

SUPPORTED_MODELS = {
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "Qwen 3.8 27B": "qwen/qwen3.8-27b",
}


# ---------------------------------------------------------
# DuckDuckGo search
# ---------------------------------------------------------

def search_web(query: str, max_results: int = 8) -> str:
    """
    Search DuckDuckGo and return formatted web results.

    The search is executed by Python BEFORE CrewAI runs.
    Therefore CrewAI/Groq does not need to call a tool.
    """

    try:
        results = DDGS().text(
            query,
            max_results=max_results
        )

        if not results:
            return "No web results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):

            title = result.get(
                "title",
                "Untitled"
            )

            body = result.get(
                "body",
                "No description available."
            )

            url = result.get(
                "href",
                ""
            )

            formatted_results.append(
                f"""
SOURCE {index}
Title: {title}
Description: {body}
URL: {url}
"""
            )

        return "\n".join(formatted_results)

    except Exception as exc:

        return (
            "DuckDuckGo search failed.\n"
            f"Error: {exc}"
        )


# ---------------------------------------------------------
# Create CrewAI Research Crew
# ---------------------------------------------------------

def create_research_crew(
    model_name: str,
    topic: str
):
    """
    Creates a single-agent CrewAI research system.

    Flow:

    User Topic
        ↓
    DuckDuckGo Search
        ↓
    Search Results
        ↓
    Single CrewAI Agent
        ↓
    Groq
        ↓
    Research Report
    """

    # -----------------------------------------------------
    # API key
    # -----------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to Streamlit Secrets."
        )

    # -----------------------------------------------------
    # Clean model name
    # -----------------------------------------------------

    model_name = str(model_name).strip()

    # Handle accidental short model name
    if model_name == "gpt-oss-120b":
        model_name = "openai/gpt-oss-120b"

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    if model_name not in SUPPORTED_MODELS.values():

        raise ValueError(
            f"Unsupported Groq model: {model_name}\n\n"
            f"Supported models:\n"
            f"{list(SUPPORTED_MODELS.values())}"
        )

    # -----------------------------------------------------
    # SEARCH FIRST
    # -----------------------------------------------------

    search_results = search_web(
        query=topic,
        max_results=8
    )

    # -----------------------------------------------------
    # Groq LLM
    # -----------------------------------------------------

    llm = LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=0.2,
    )

    # -----------------------------------------------------
    # Single Agent
    #
    # IMPORTANT:
    # No tools are attached to the agent.
    #
    # This prevents the previous:
    # "Tool choice is none, but model called a tool"
    # error.
    # -----------------------------------------------------

    researcher = Agent(
        role="AI Research Analyst",

        goal=(
            "Analyze the provided web research and "
            "produce an accurate, structured research "
            "report about the requested topic."
        ),

        backstory=(
            "You are a careful research analyst. "
            "You analyze information gathered from "
            "web sources and create clear factual reports. "
            "You never invent sources, URLs, statistics, "
            "or facts."
        ),

        llm=llm,

        allow_delegation=False,

        max_iter=4,

        verbose=False,
    )

    # -----------------------------------------------------
    # Research Task
    # -----------------------------------------------------

    research_task = Task(

        description=f"""
Research Topic:

{topic}

The following information was collected from
DuckDuckGo before you started:

---------------- WEB SEARCH RESULTS ----------------

{search_results}

---------------- END WEB RESULTS ----------------

IMPORTANT INSTRUCTIONS:

1. Analyze the web search results provided above.

2. Do NOT attempt to call any external tool.

3. Do NOT invent URLs.

4. Do NOT invent statistics or facts.

5. Use the provided sources as evidence.

6. If the search results do not contain enough
   information for a claim, say that the available
   sources were insufficient.

7. Keep the report factual and easy to understand.

8. Mention the source URL next to important claims
   where appropriate.

9. At the end, provide a Sources section containing
   the URLs actually present in the search results.

Write the final report using this structure:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Current Developments

# Challenges and Limitations

# Conclusion

# Sources
""",

        expected_output=(
            "A complete, factual Markdown research report "
            "based on the supplied DuckDuckGo search results, "
            "including a Sources section with URLs."
        ),

        agent=researcher,
    )

    # -----------------------------------------------------
    # Single-Agent Crew
    # -----------------------------------------------------

    crew = Crew(

        agents=[researcher],

        tasks=[research_task],

        process=Process.sequential,

        verbose=False,
    )

    return crew
