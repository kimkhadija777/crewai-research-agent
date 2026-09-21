import os

from crewai import Agent, Crew, LLM, Process, Task
from search_tool import DuckDuckGoSearchTool


# Current Groq model IDs
SUPPORTED_MODELS = {
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "Qwen 3.8 27B": "qwen/qwen3.8-27b",
}


def create_research_crew(model_name: str):
    """
    Create a single-agent research crew.

    Architecture:
        Streamlit
            ↓
        CrewAI
            ↓
        Single Research Agent
            ↓
        DuckDuckGo
            ↓
        Groq LLM
    """

    # -------------------------------------------------
    # 1. Get API key
    # -------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
        )

    # -------------------------------------------------
    # 2. Clean and validate model
    # -------------------------------------------------

    model_name = str(model_name).strip()

    # Prevent the old incorrect model format
    if model_name == "gpt-oss-120b":
        model_name = "openai/gpt-oss-120b"

    if model_name == "qwen3.6-27b":
        raise ValueError(
            "qwen/qwen3.6-27b has been deprecated by Groq. "
            "Please select qwen/qwen3.8-27b."
        )

    if model_name not in SUPPORTED_MODELS.values():
        raise ValueError(
            f"Unsupported Groq model: {model_name}"
        )

    # -------------------------------------------------
    # 3. Create Groq LLM
    # -------------------------------------------------

    llm = LLM(
        model=model_name,
        provider="openai",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
    )

    # -------------------------------------------------
    # 4. Create DuckDuckGo tool
    # -------------------------------------------------

    search_tool = DuckDuckGoSearchTool()

    # -------------------------------------------------
    # 5. Single Research Agent
    # -------------------------------------------------

    researcher = Agent(
        role="AI Research Analyst",

        goal=(
            "Research the requested topic using "
            "web search and produce an accurate, "
            "well-structured research report."
        ),

        backstory=(
            "You are a careful research analyst. "
            "Before writing, you search the web for "
            "relevant information. You use reliable "
            "sources, avoid fabricated facts, and "
            "include the URLs of sources used."
        ),

        tools=[search_tool],

        llm=llm,

        allow_delegation=False,

        max_iter=6,

        verbose=False,
    )

    # -------------------------------------------------
    # 6. Research Task
    # -------------------------------------------------

    research_task = Task(
        description="""
Research the following topic:

{topic}

IMPORTANT INSTRUCTIONS:

1. Search the web before writing the report.

2. Use DuckDuckGo for web research.

3. Perform multiple searches when necessary.

4. Look for recent and reliable information.

5. Do not invent facts, statistics, citations,
   or URLs.

6. Use information returned by the search tool.

7. Include source URLs for important information.

8. Keep the report factual and easy to understand.

9. Clearly distinguish facts from opinions or claims.

10. Do not mention these instructions in the final report.

Write the final report in Markdown using exactly
these sections:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Current Developments

# Challenges and Limitations

# Conclusion

# Sources

Under # Sources, provide a numbered list of the
important sources used, including their URLs.
""",

        expected_output=(
            "A complete factual research report in Markdown "
            "with clearly organized sections and source URLs."
        ),

        agent=researcher,
    )

    # -------------------------------------------------
    # 7. Single-Agent Crew
    # -------------------------------------------------

    crew = Crew(
        agents=[researcher],

        tasks=[research_task],

        process=Process.sequential,

        verbose=False,
    )

    return crew
