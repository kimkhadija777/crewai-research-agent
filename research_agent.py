import os

from crewai import Agent, Crew, LLM, Process, Task
from search_tool import DuckDuckGoSearchTool


def create_research_crew(model_name: str):

    # Get Groq API key
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    # Groq model through LiteLLM
    llm = LLM(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
    )
        
        
    

    # DuckDuckGo search tool
    search_tool = DuckDuckGoSearchTool()

    # Single research agent
    researcher = Agent(
        role="AI Researcher",

        goal=(
            "Research the user's topic using current web "
            "sources and produce an accurate, "
            "well-structured report."
        ),

        backstory=(
            "You are a careful research analyst. "
            "You search the web before writing, "
            "cross-check important claims when possible, "
            "distinguish facts from opinions, and always "
            "include source URLs."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=8,
    )

    # Research task
    task = Task(
        description=(
            "Research this topic: {topic}\n\n"

            "Instructions:\n"

            "1. Use the DuckDuckGo search tool before "
            "writing the report.\n"

            "2. Search multiple queries when useful.\n"

            "3. Prefer recent, credible, and relevant "
            "sources.\n"

            "4. Do not invent facts, statistics, "
            "citations, or URLs.\n"

            "5. Clearly distinguish established facts "
            "from claims or opinions.\n"

            "6. Include source URLs for important "
            "information you use.\n\n"

            "Return the report using this structure:\n\n"

            "# Executive Summary\n"

            "# Introduction\n"

            "# Key Findings\n"

            "# Detailed Analysis\n"

            "# Current Developments\n"

            "# Challenges and Limitations\n"

            "# Conclusion\n"

            "# Sources\n\n"

            "For Sources, provide a numbered list of "
            "the URLs actually used."
        ),

        expected_output=(
            "A factual, structured research report "
            "with source URLs."
        ),

        agent=researcher,
    )

    # Single-agent Crew
    crew = Crew(
        agents=[researcher],

        tasks=[task],

        process=Process.sequential,

        verbose=False,
    )

    return crew
