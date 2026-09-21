import os

from crewai import Agent, Crew, LLM, Process, Task

from search_tool import DuckDuckGoSearchTool


def create_research_crew(model_name: str):
    """
    Create a single-agent research crew.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    llm = LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=0.2,
    )

    search_tool = DuckDuckGoSearchTool()

    # --------------------------------------------------
    # SINGLE AGENT
    # --------------------------------------------------

    researcher = Agent(
        role="AI Research Analyst",

        goal=(
            "Research the user's topic using web search, "
            "analyze the available information, cross-check "
            "important claims, and produce a factual "
            "well-structured research report."
        ),

        backstory=(
            "You are an experienced research analyst. "
            "You carefully search for relevant information, "
            "prefer credible sources, compare information "
            "from multiple sources, and clearly identify "
            "uncertainty or conflicting information."
        ),

        tools=[
            search_tool
        ],

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=8,
    )

    # --------------------------------------------------
    # RESEARCH TASK
    # --------------------------------------------------

    research_task = Task(
        description="""
Research the following topic:

{topic}

Your job is to conduct web-based research and create
a useful research report.

Follow these steps:

1. Understand the topic.

2. Break the topic into useful search questions.

3. Use the DuckDuckGo Web Search tool to search for
   relevant information.

4. Perform multiple searches when necessary.

5. Look for information from several sources.

6. Prefer reliable sources such as:
   - official organizations
   - universities
   - research institutions
   - government websites
   - established publications
   - primary sources

7. Compare information from different sources.

8. Do not invent facts.

9. Do not invent statistics.

10. Do not invent quotations.

11. Do not invent source URLs.

12. If sources disagree, explain the disagreement.

13. If information cannot be verified from the search
    results, clearly say that it could not be verified.

14. Use recent information when the topic requires it.

15. Write the final report in clear Markdown.

The report must contain:

# Research Report

## Executive Summary

Provide a concise overview of the research.

## Introduction

Explain the topic and its importance.

## Key Findings

Present the most important findings.

## Detailed Analysis

Explain the topic in depth using the information
found during research.

Use appropriate subsections when useful.

## Current Developments

Discuss recent developments when relevant.

## Challenges and Limitations

Discuss important limitations, uncertainties,
or challenges.

## Conclusion

Summarize the main findings.

## Sources

List the sources used during research.

For every source include:

- Source title
- URL

IMPORTANT:

Only include URLs returned by the search tool.

Do not fabricate URLs.

Do not claim that you directly visited or verified
a webpage unless the available tool actually provided
that information.
""",

        expected_output=(
            "A factual, well-structured Markdown research "
            "report containing an executive summary, "
            "introduction, key findings, detailed analysis, "
            "current developments, challenges and limitations, "
            "conclusion, and source URLs."
        ),

        agent=researcher,
    )

    # --------------------------------------------------
    # CREW
    # --------------------------------------------------

    crew = Crew(
        agents=[
            researcher
        ],

        tasks=[
            research_task
        ],

        process=Process.sequential,

        verbose=False,
    )

    return crew
