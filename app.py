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
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .source-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f7f7f7;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔎 AI Research Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Research any topic using CrewAI, DuckDuckGo and Groq."
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API key
# --------------------------------------------------

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")


if not groq_api_key:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Choose Groq Model",
        options=[
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
        ],
        index=0,
    )

    st.divider()

    st.markdown("### 🧠 Architecture")

    st.write("• Streamlit")
    st.write("• CrewAI")
    st.write("• Single AI Agent")
    st.write("• DuckDuckGo Search")
    st.write("• Groq LLM")

    st.divider()

    st.caption(
        "The research agent searches the web before "
        "generating the report."
    )


# --------------------------------------------------
# Research topic
# --------------------------------------------------

st.subheader("📝 Research Topic")

topic = st.text_area(
    "Enter the topic you want to research",
    placeholder=(
        "Example: How Generative AI is changing "
        "software development"
    ),
    height=120,
)


# --------------------------------------------------
# Research button
# --------------------------------------------------

research_button = st.button(
    "🔎 Start Research",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Run research
# --------------------------------------------------

if research_button:

    if not topic.strip():
        st.warning(
            "Please enter a research topic first."
        )
        st.stop()

    with st.status(
        "🔎 Research agent is working...",
        expanded=True,
    ) as status:

        st.write("Creating CrewAI research agent...")

        try:
            crew = create_research_crew(model_name)

            st.write(
                "Searching the web and analyzing sources..."
            )

            result = crew.kickoff(
                inputs={
                    "topic": topic.strip()
                }
            )

            status.update(
                label="✅ Research completed!",
                state="complete",
                expanded=False,
            )

        except Exception as e:

            status.update(
                label="❌ Research failed",
                state="error",
                expanded=True,
            )

            st.error(
                f"An error occurred:\n\n{str(e)}"
            )

            st.stop()


    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    st.divider()

    st.subheader("📑 Research Report")

    report_text = str(result)

    st.markdown(report_text)


    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    st.download_button(
        label="⬇️ Download Report",
        data=report_text,
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
          )
