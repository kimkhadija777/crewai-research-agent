import os

from crewai import Agent, Crew, LLM, Process, Task
from search_tool import DuckDuckGoSearchTool


def create_research_crew(model_name: str = "openai/gpt-oss-120b"):
    """
    Create a single-agent AI research crew using:
    - CrewAI
    - Groq OpenAI-compatible API
    - DuckDuckGo web search
    """

    # --------------------------------------------------
    # Get Groq API key
    # --------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to Streamlit Secrets."
        )

    # --------------------------------------------------
    # Clean model name
    # --------------------------------------------------

    model_name = model_name.strip()

    # Make sure the full Groq model ID is preserved
    if model_name == "gpt-oss-120b":
        model_name = "openai/gpt-oss-120b"

    # --------------------------------------------------
    # Groq LLM
    # --------------------------------------------------
    #
    # IMPORTANT:
    # Do NOT use:
    #     groq/openai/gpt-oss-120b
    #
    # We use Groq's OpenAI-compatible endpoint directly.
    #

    llm = LLM(
        model=model_name,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
    )

    # --------------------------------------------------
    # DuckDuckGo search tool
    # --------------------------------------------------

    search_tool = DuckDuckGoSearchTool()

    # --------------------------------------------------
    # Single Research Agent
    # --------------------------------------------------

    researcher = Agent(
        role="AI Research Analyst",

        goal=(
            "Research the user's topic using web search "
            "and create an accurate, structured and "
            "easy-to-understand research report."
        ),

        backstory=(
            "You are a careful AI research analyst. "
            "You search the web before writing your report. "
            "You use relevant and credible sources, "
            "avoid making up information, and provide "
            "source URLs for the information you use."
        ),

        tools=[search_tool],

        llm=llm,

        allow_delegation=False,

        max_iter=8,

        verbose=False,
    )

    # --------------------------------------------------
    # Research Task
    # --------------------------------------------------

    research_task = Task(
        description="""
Research the following topic:

{topic}

Follow these instructions carefully:

1. Search the web using the DuckDuckGo search tool
   before writing the report.

2. Use multiple searches when necessary.

3. Prefer recent, reliable and relevant sources.

4. Do not invent facts, statistics, citations or URLs.

5. Clearly separate established information from
   opinions or claims.

6. Use the information found through web search
   to prepare the report.

7. Include the URLs of the sources actually used.

8. Write the final answer in clear Markdown.

Use exactly this structure:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Current Developments

# Challenges and Limitations

# Conclusion

# Sources

Under # Sources, provide a numbered list containing
the title/name of each important source and its URL.
""",

        expected_output="""
A complete factual research report in Markdown format
with clear sections and source URLs.
""",

        agent=researcher,
    )

    # --------------------------------------------------
    # Single-Agent Crew
    # --------------------------------------------------

    crew = Crew(
        agents=[researcher],

        tasks=[research_task],

        process=Process.sequential,

        verbose=False,
    )

    return crew
