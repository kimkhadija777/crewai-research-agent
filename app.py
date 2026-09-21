import os

import streamlit as st

from research_agent import create_research_crew


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)


# ==================================================
# CUSTOM CSS
# ==================================================

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
        color: #666666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🔎 AI Research Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Research a topic using CrewAI + DuckDuckGo + Groq"
    "</div>",
    unsafe_allow_html=True,
)


# ==================================================
# LOAD API KEY
# ==================================================

try:

    api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    api_key = os.getenv("GROQ_API_KEY")


if not api_key:

    st.error(
        "❌ GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY in Streamlit Cloud → "
        "Settings → Secrets."
    )

    st.stop()


# Make API key available to CrewAI
os.environ["GROQ_API_KEY"] = api_key


# ==================================================
# SIDEBAR
# ==================================================

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

    st.write("🔹 Streamlit")
    st.write("🔹 CrewAI")
    st.write("🔹 Single Agent")
    st.write("🔹 DuckDuckGo")
    st.write("🔹 Groq")

    st.divider()

    st.caption(
        "The agent searches the web before "
        "writing the research report."
    )


# ==================================================
# TOPIC INPUT
# ==================================================

st.subheader("📝 Research Topic")

topic = st.text_area(
    "What would you like me to research?",

    placeholder=(
        "Example: "
        "Impact of Generative AI on Software Development"
    ),

    height=120,
)


# ==================================================
# RESEARCH BUTTON
# ==================================================

start_research = st.button(
    "🔎 Start Research",

    type="primary",

    use_container_width=True,
)


# ==================================================
# RUN AGENT
# ==================================================

if start_research:

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a research topic."
        )

        st.stop()

    try:

        with st.spinner(
            "🔎 Research agent is working..."
        ):

            crew = create_research_crew(
                model_name=model_name
            )

            result = crew.kickoff(
                inputs={
                    "topic": topic.strip()
                }
            )

        # Convert CrewAI result to text
        report = str(result)

        # ==========================================
        # DISPLAY REPORT
        # ==========================================

        st.success(
            "✅ Research completed successfully!"
        )

        st.divider()

        st.subheader("📑 Research Report")

        st.markdown(report)

        # ==========================================
        # DOWNLOAD
        # ==========================================

        st.download_button(
            label="⬇️ Download Report",

            data=report,

            file_name="research_report.md",

            mime="text/markdown",

            use_container_width=True,
        )

    except Exception as error:

        st.error(
            "❌ The research agent encountered an error."
        )

        with st.expander(
            "Show technical error"
        ):

            st.code(
                str(error)
    )
