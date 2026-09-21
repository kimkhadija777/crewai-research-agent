import os

import streamlit as st

from research_agent import (
    SUPPORTED_MODELS,
    generate_report,
)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🔎 AI Research Agent")

st.caption(
    "Research a topic using "
    "CrewAI + DuckDuckGo + Groq"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    model_label = st.selectbox(
        "Choose Groq Model",
        list(SUPPORTED_MODELS.keys())
    )

    model_name = SUPPORTED_MODELS[
        model_label
    ]

    st.markdown("### 🧠 Architecture")

    st.markdown("🔹 Streamlit")
    st.markdown("🔹 CrewAI")
    st.markdown("🔹 Single Agent")
    st.markdown("🔹 DuckDuckGo")
    st.markdown("🔹 Groq")

    st.info(
        "DuckDuckGo collects web sources first. "
        "The single CrewAI agent analyzes them, "
        "and Groq generates the final report."
    )


# =========================================================
# GROQ API KEY
# =========================================================

api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)


if not api_key:

    st.error(
        "❌ GROQ_API_KEY is missing."
    )

    st.info(
        "Add GROQ_API_KEY in "
        "Streamlit → Settings → Secrets."
    )

    st.stop()


os.environ["GROQ_API_KEY"] = api_key


# =========================================================
# TOPIC
# =========================================================

topic = st.text_area(
    "📝 Research Topic",

    placeholder=(
        "What would you like me to research?"
    ),

    height=120
)


# =========================================================
# BUTTON
# =========================================================

if st.button(
    "🔍 Generate Research Report",

    type="primary",

    use_container_width=True
):

    topic = topic.strip()

    if not topic:

        st.warning(
            "Please enter a research topic."
        )

        st.stop()

    try:

        # ---------------------------------------------
        # Search
        # ---------------------------------------------

        with st.spinner(
            "🔎 Searching DuckDuckGo..."
        ):

            # Search + CrewAI + Groq
            report = generate_report(
                model_name=model_name,
                topic=topic
            )

        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        st.success(
            "✅ Research report generated!"
        )

        st.markdown(report)

        # ---------------------------------------------
        # Download
        # ---------------------------------------------

        st.download_button(
            label="⬇️ Download Report",

            data=report,

            file_name="research_report.md",

            mime="text/markdown",

            use_container_width=True
        )

    except Exception as exc:

        st.error(
            "❌ The research agent encountered an error."
        )

        with st.expander(
            "Show technical error"
        ):

            st.code(
                str(exc),
                language="text"
            )
