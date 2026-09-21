import os
from typing import List, Dict

from openai import OpenAI
from crewai import Agent, Crew, Process, Task

from search_tool import search_web


# =========================================================
# CURRENT GROQ MODELS
# =========================================================

SUPPORTED_MODELS = {
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b",
}


# =========================================================
# SEARCH
# =========================================================

def collect_web_research(
    topic: str,
    max_results: int = 8
) -> List[Dict]:
    """
    Search DuckDuckGo before the CrewAI agent starts.
    """

    queries = [
        topic,
        f"{topic} latest developments",
        f"{topic} benefits challenges",
    ]

    all_results = []
    seen_urls = set()

    for query in queries:

        results = search_web(
            query,
            max_results=max_results
        )

        for result in results:

            url = result.get("url", "").strip()

            if url and url not in seen_urls:

                seen_urls.add(url)
                all_results.append(result)

    return all_results[:20]


# =========================================================
# FORMAT SEARCH RESULTS
# =========================================================

def format_search_results(
    results: List[Dict]
) -> str:

    if not results:
        return "No web search results were found."

    formatted = []

    for index, result in enumerate(
        results,
        start=1
    ):

        formatted.append(
            f"""
SOURCE {index}

Title:
{result.get("title", "Unknown")}

Description:
{result.get("body", "No description available.")}

URL:
{result.get("url", "")}
"""
        )

    return "\n".join(formatted)


# =========================================================
# DIRECT GROQ CALL
# =========================================================

def call_groq(
    model_name: str,
    system_prompt: str,
    user_prompt: str
) -> str:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to Streamlit Secrets."
        )

    # Direct Groq OpenAI-compatible client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

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

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return content


# =========================================================
# CREWAI SINGLE AGENT
# =========================================================

def create_research_crew(
    model_name: str,
    topic: str
):

    # -----------------------------------------------------
    # Validate API key
    # -----------------------------------------------------

    if not os.getenv("GROQ_API_KEY"):

        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    if model_name not in SUPPORTED_MODELS.values():

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # -----------------------------------------------------
    # Collect web information first
    # -----------------------------------------------------

    search_results = collect_web_research(
        topic
    )

    research_context = format_search_results(
        search_results
    )

    # -----------------------------------------------------
    # Create CrewAI Agent
    #
    # The agent does NOT have a tool.
    #
    # This prevents Groq tool_choice errors.
    # -----------------------------------------------------

    researcher = Agent(

        role="AI Research Analyst",

        goal=(
            "Analyze the provided web research "
            "and create an accurate, structured "
            "research report."
        ),

        backstory=(
            "You are an experienced research analyst "
            "who carefully analyzes information collected "
            "from web sources. You never fabricate facts "
            "or sources."
        ),

        # No tools here
        tools=[],

        allow_delegation=False,

        verbose=False,

        max_iter=1
    )

    # -----------------------------------------------------
    # Task
    # -----------------------------------------------------

    research_task = Task(

        description=f"""
Research Topic:

{topic}

The application has already searched DuckDuckGo.

Here are the collected web sources:

================ WEB RESEARCH ================

{research_context}

================ END WEB RESEARCH ================

Your job is to analyze these sources and create
a professional research report.

IMPORTANT RULES:

1. Use ONLY the information provided in the
   web research above.

2. Do not invent facts.

3. Do not invent statistics.

4. Do not invent URLs.

5. Do not claim that you searched the web yourself.

6. If information is insufficient, clearly say so.

7. Keep the report factual.

8. Include relevant source URLs.

9. Use Markdown.

Use this exact structure:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Current Developments

# Challenges and Limitations

# Conclusion

# Sources

Under Sources, list the sources that were actually
provided in the web research.
""",

        expected_output=(
            "A complete Markdown research report "
            "with factual findings and source URLs."
        ),

        agent=researcher
    )

    # -----------------------------------------------------
    # Crew
    # -----------------------------------------------------

    crew = Crew(

        agents=[researcher],

        tasks=[research_task],

        process=Process.sequential,

        verbose=False
    )

    return crew, research_context


# =========================================================
# GENERATE REPORT
# =========================================================

def generate_report(
    model_name: str,
    topic: str
) -> str:
    """
    Generate the final research report.

    Flow:

    DuckDuckGo
        ↓
    Search Results
        ↓
    Single CrewAI Agent
        ↓
    Report Planning
        ↓
    Direct Groq API
        ↓
    Final Report
    """

    crew, research_context = create_research_crew(
        model_name=model_name,
        topic=topic
    )

    # -----------------------------------------------------
    # CrewAI runs the single agent
    # -----------------------------------------------------

    crew_result = crew.kickoff()

    agent_instructions = str(
        crew_result.raw
        if hasattr(crew_result, "raw")
        else crew_result
    )

    # -----------------------------------------------------
    # Final Groq generation
    # -----------------------------------------------------

    system_prompt = """
You are a professional AI research writer.

Create accurate, clear and well-structured
research reports.

Never fabricate facts, statistics or URLs.

Use only the research information supplied
by the application.
"""

    user_prompt = f"""
TOPIC:

{topic}

CREWAI RESEARCH ANALYSIS:

{agent_instructions}

WEB SOURCES:

{research_context}

Now produce the final research report.

Use these sections:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Current Developments

# Challenges and Limitations

# Conclusion

# Sources

For Sources, provide numbered source titles
and their actual URLs.
"""

    return call_groq(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt
        )
