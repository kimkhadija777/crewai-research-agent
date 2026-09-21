import os

import streamlit as st

from research_agent import create_research_crew


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🔎 AI Research Agent")

st.caption(
    "Research a topic using "
    "CrewAI + DuckDuckGo + Groq"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Choose Groq Model",
        [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
        ],
    )

    st.markdown("### 🧠 Architecture")

    st.markdown("🔹 Streamlit")
    st.markdown("🔹 CrewAI")
    st.markdown("🔹 Single Agent")
    st.markdown("🔹 DuckDuckGo")
    st.markdown("🔹 Groq")

    st.caption(
        "The agent searches the web before "
        "writing the research report."
    )


# --------------------------------------------------
# API Key
# --------------------------------------------------

api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)


if api_key:

    os.environ["GROQ_API_KEY"] = api_key

else:

    st.error(
        "GROQ_API_KEY is missing. "
        "Add it in Streamlit Secrets."
    )

    st.stop()


# --------------------------------------------------
# Research Topic
# --------------------------------------------------

topic = st.text_area(
    "📝 Research Topic",

    placeholder=(
        "What would you like me to research?"
    ),

    height=120,
)


# --------------------------------------------------
# Generate Report
# --------------------------------------------------

if st.button(
    "🔍 Generate Research Report",
    type="primary",
    use_container_width=True,
):

    topic = topic.strip()

    if not topic:

        st.warning(
            "Please enter a research topic first."
        )

        st.stop()

    with st.spinner(
        "🔎 Searching the web and preparing "
        "your report..."
    ):

        try:

            # Create CrewAI crew
            crew = create_research_crew(
                model_name
            )

            # Run research
            result = crew.kickoff(
                inputs={
                    "topic": topic
                }
            )

            # Get final report
            report = getattr(
                result,
                "raw",
                str(result)
            )

            st.success(
                "Research report generated successfully!"
            )

            # Display report
            st.markdown(report)

            # Download button
            st.download_button(
                "⬇️ Download Report",

                data=report,

                file_name="research_report.md",

                mime="text/markdown",

                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                "❌ The research agent encountered "
                "an error."
            )

            with st.expander(
                "Show technical error"
            ):

                st.code(str(exc))
