import os

from crewai import Agent, Crew, LLM, Process, Task

from search_tool import DuckDuckGoSearchTool


def create_research_crew(model_name: str):
    """
    Creates a single-agent CrewAI research crew.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    llm = LLM(
        model=f"groq/{model_name}",
        api_key=groq_api_key,
        temperature=0.2,
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="AI Research Analyst",

        goal=(
            "Research the given topic carefully using web search, "
            "identify reliable and relevant information, "
            "cross-check important claims, and produce a clear "
            "well-structured research report."
        ),

        backstory=(
            "You are a careful research analyst who specializes in "
            "finding, organizing, and explaining information. "
            "You search the web before writing the report, prefer "
            "credible sources, avoid unsupported claims, and clearly "
            "separate established facts from uncertain information."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=8,
    )

    research_task = Task(
        description="""
        Research the following topic:

        {topic}

        Follow this process:

        1. Understand the research topic.
        2. Search the web using the DuckDuckGo search tool.
        3. Search using multiple relevant queries when necessary.
        4. Gather information from several sources.
        5. Prefer authoritative, reputable, and primary sources
           when available.
        6. Compare information across sources.
        7. Do not invent facts, statistics, quotations, or sources.
        8. If information is uncertain or conflicting, explicitly
           mention the uncertainty.
        9. Write a well-organized research report.

        The final report should contain:

        # Research Report

        ## 1. Executive Summary
        Give a concise overview of the topic.

        ## 2. Introduction
        Explain the topic and why it matters.

        ## 3. Key Findings
        Present the major findings in clear sections.

        ## 4. Detailed Analysis
        Explain the important concepts, evidence, examples,
        developments, advantages/disadvantages, or other relevant
        aspects depending on the topic.

        ## 5. Current Developments
        Include recent information when relevant.

        ## 6. Conclusion
        Summarize the main findings without adding unsupported claims.

        ## 7. Sources
        List the sources used.
        Include the source title and URL.

        Important:
        - Use only information supported by the search results.
        - Do not fabricate URLs.
        - Do not claim that you visited a page if the search tool
          only returned search-result information.
        - Keep the report factual and readable.
        """,

        expected_output=(
            "A comprehensive, factual, well-structured research report "
            "with an executive summary, introduction, key findings, "
            "detailed analysis, current developments, conclusion, "
            "and a list of source URLs."
        ),

        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew
