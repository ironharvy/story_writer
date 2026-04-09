import streamlit as st
import dspy
import os
import logging
from dotenv import load_dotenv

from story_modules import (
    QuestionGenerator,
    CorePremiseGenerator,
    SpineTemplateGenerator,
    StoryGenerator,
)
from world_bible_modules import WorldBibleGenerator, WorldBibleQuestionGenerator
from logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Story Writer", page_icon="📖", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEPS = [
    "Story Idea",
    "Answer Questions",
    "Core Premise",
    "Spine Template",
    "World Bible Questions",
    "World Bible",
    "Generate Story",
    "Download",
]


def _step_index() -> int:
    return st.session_state.get("step", 0)


def _set_step(idx: int):
    st.session_state["step"] = idx


def _configure_dspy(model_name: str, api_base: str, api_key: str, max_tokens: int):
    kwargs = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            kwargs["api_key"] = env_key

    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)
    lm = dspy.LM(model_name, max_tokens=max_tokens, cache=True, **kwargs)
    dspy.configure(lm=lm)


def _init_generators():
    if "generators" not in st.session_state:
        st.session_state["generators"] = {
            "QuestionGenerator": QuestionGenerator(),
            "CorePremiseGenerator": CorePremiseGenerator(),
            "SpineTemplateGenerator": SpineTemplateGenerator(),
            "WorldBibleQuestionGenerator": WorldBibleQuestionGenerator(),
            "WorldBibleGenerator": WorldBibleGenerator(),
            "StoryGenerator": StoryGenerator(),
        }
    return st.session_state["generators"]


def _build_markdown_output() -> str:
    """Assemble the final story output as a markdown string."""
    ss = st.session_state
    parts = ["# Story Output\n"]
    parts.append(f"## Core Premise\n{ss.get('core_premise', '')}\n")
    parts.append(f"## Spine Template\n{ss.get('spine_template', '')}\n")
    parts.append(f"## World Bible\n{ss.get('world_bible', '')}\n")
    if ss.get("chapter_plan"):
        parts.append(f"## Chapter Plan\n{ss['chapter_plan']}\n")
    if ss.get("enhancers_guide"):
        parts.append(f"## Enhancers Guide\n{ss['enhancers_guide']}\n")
    if ss.get("story"):
        parts.append(f"## Final Story\n{ss['story']}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input(
        "Model",
        value=os.environ.get("MODEL", "openai/gpt-4o-mini"),
        help="e.g. openai/gpt-4o-mini, ollama_chat/llama3",
    )
    api_key = st.text_input(
        "API Key",
        value=os.environ.get("API_KEY", ""),
        type="password",
        help="Leave blank to use OPENAI_API_KEY env var.",
    )
    api_base = st.text_input(
        "LLM URL (optional)",
        value=os.environ.get("LLM_URL", ""),
        help="Custom API base, e.g. http://localhost:11434 for Ollama.",
    )
    max_tokens = st.number_input("Max tokens", value=8000, min_value=256, step=256)

    st.divider()
    st.caption("Progress")
    current = _step_index()
    for i, name in enumerate(STEPS):
        if i < current:
            st.markdown(f"~~{name}~~")
        elif i == current:
            st.markdown(f"**-> {name}**")
        else:
            st.markdown(f"{name}")

    st.divider()
    if st.button("Reset / Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Ensure DSPy is configured before any generation step
# ---------------------------------------------------------------------------
def _ensure_dspy():
    if not st.session_state.get("dspy_configured"):
        _configure_dspy(model_name, api_base, api_key, max_tokens)
        setup_logging(verbosity=0)
        st.session_state["dspy_configured"] = True


# ---------------------------------------------------------------------------
# Step 0 — Story Idea
# ---------------------------------------------------------------------------
if _step_index() == 0:
    st.title("AI Story Writer")
    st.markdown("Enter your initial story idea to get started.")

    with st.form("idea_form"):
        idea = st.text_area(
            "What is your story idea?",
            height=150,
            placeholder="A detective in a steampunk city discovers that the clockwork machines are developing consciousness...",
        )
        submitted = st.form_submit_button("Begin")

    if submitted and idea.strip():
        st.session_state["idea"] = idea.strip()
        _set_step(1)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 1 — Answer Questions
# ---------------------------------------------------------------------------
elif _step_index() == 1:
    st.title("Interrogating Your Idea")
    _ensure_dspy()
    gens = _init_generators()

    # Generate questions once
    if "questions" not in st.session_state:
        with st.spinner("Generating questions about your idea..."):
            q_result = gens["QuestionGenerator"](idea=st.session_state["idea"])
            st.session_state["questions"] = q_result.questions_with_answers

    questions = st.session_state["questions"]

    with st.form("qa_form"):
        st.markdown("Review each proposed answer. Edit any you'd like to change.")
        answers = []
        for i, qa in enumerate(questions):
            st.markdown(f"**Q{i+1}: {qa.question}**")
            answer = st.text_area(
                f"Answer {i+1}",
                value=qa.proposed_answer,
                key=f"qa_{i}",
                label_visibility="collapsed",
            )
            answers.append((qa.question, answer))

        submitted = st.form_submit_button("Accept Answers & Continue")

    if submitted:
        qa_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in answers)
        st.session_state["qa_text"] = qa_text
        _set_step(2)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 2 — Core Premise
# ---------------------------------------------------------------------------
elif _step_index() == 2:
    st.title("Core Premise")
    _ensure_dspy()
    gens = _init_generators()

    if "core_premise" not in st.session_state:
        with st.spinner("Generating Core Premise..."):
            cp_result = gens["CorePremiseGenerator"](
                idea=st.session_state["idea"],
                qa_pairs=st.session_state["qa_text"],
            )
            st.session_state["core_premise"] = cp_result.core_premise

    st.markdown(st.session_state["core_premise"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept & Continue"):
            _set_step(3)
            st.rerun()
    with col2:
        with st.popover("Refine Premise"):
            refinement = st.text_area("What would you like to change?")
            if st.button("Regenerate"):
                st.session_state["idea"] = (
                    f"Original idea: {st.session_state['idea']}\n"
                    f"Refinements: {refinement}\n"
                    f"Current Core Premise: {st.session_state['core_premise']}"
                )
                # Clear downstream state so everything regenerates
                for key in ["core_premise", "questions", "qa_text",
                            "spine_template", "wb_questions", "wb_qa_text",
                            "world_bible", "story", "chapter_plan",
                            "enhancers_guide"]:
                    st.session_state.pop(key, None)
                _set_step(1)
                st.rerun()

# ---------------------------------------------------------------------------
# Step 3 — Spine Template
# ---------------------------------------------------------------------------
elif _step_index() == 3:
    st.title("Spine Template")
    _ensure_dspy()
    gens = _init_generators()

    if "spine_template" not in st.session_state:
        with st.spinner("Generating Spine Template..."):
            st_result = gens["SpineTemplateGenerator"](
                core_premise=st.session_state["core_premise"],
            )
            st.session_state["spine_template"] = st_result.spine_template

    st.markdown(st.session_state["spine_template"])

    if st.button("Continue to World Bible"):
        _set_step(4)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 4 — World Bible Questions
# ---------------------------------------------------------------------------
elif _step_index() == 4:
    st.title("World Bible Questions")
    _ensure_dspy()
    gens = _init_generators()

    if "wb_questions" not in st.session_state:
        with st.spinner("Generating World Bible questions..."):
            wb_q_result = gens["WorldBibleQuestionGenerator"](
                core_premise=st.session_state["core_premise"],
                spine_template=st.session_state["spine_template"],
            )
            st.session_state["wb_questions"] = wb_q_result.questions_with_answers

    wb_questions = st.session_state["wb_questions"]

    with st.form("wb_qa_form"):
        st.markdown("These questions help flesh out the world. Edit any answers you'd like to change.")
        wb_answers = []
        for i, qa in enumerate(wb_questions):
            st.markdown(f"**Q{i+1}: {qa.question}**")
            answer = st.text_area(
                f"WB Answer {i+1}",
                value=qa.proposed_answer,
                key=f"wb_qa_{i}",
                label_visibility="collapsed",
            )
            wb_answers.append((qa.question, answer))

        submitted = st.form_submit_button("Accept & Generate World Bible")

    if submitted:
        wb_qa_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in wb_answers)
        st.session_state["wb_qa_text"] = wb_qa_text
        _set_step(5)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 5 — World Bible
# ---------------------------------------------------------------------------
elif _step_index() == 5:
    st.title("World Bible")
    _ensure_dspy()
    gens = _init_generators()

    if "world_bible" not in st.session_state:
        with st.spinner("Generating World Bible (rules, characters, locations, timeline)..."):
            wb_result = gens["WorldBibleGenerator"](
                core_premise=st.session_state["core_premise"],
                spine_template=st.session_state["spine_template"],
                user_additions=st.session_state["wb_qa_text"],
            )
            st.session_state["world_bible"] = wb_result.world_bible

    st.markdown(st.session_state["world_bible"])

    if st.button("Continue to Story Generation"):
        _set_step(6)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 6 — Generate Story
# ---------------------------------------------------------------------------
elif _step_index() == 6:
    st.title("Story Generation")
    _ensure_dspy()
    gens = _init_generators()

    if "story" not in st.session_state:
        with st.spinner("Generating story (this may take a few minutes)..."):
            story_result = gens["StoryGenerator"](
                core_premise=st.session_state["core_premise"],
                spine_template=st.session_state["spine_template"],
                world_bible=st.session_state["world_bible"],
            )
            st.session_state["chapter_plan"] = story_result.chapter_plan
            st.session_state["enhancers_guide"] = story_result.enhancers_guide
            st.session_state["story"] = story_result.story

    tab_story, tab_plan, tab_enhancers = st.tabs(
        ["Final Story", "Chapter Plan", "Enhancers Guide"]
    )

    with tab_story:
        st.markdown(st.session_state["story"])
    with tab_plan:
        st.markdown(st.session_state["chapter_plan"])
    with tab_enhancers:
        st.markdown(st.session_state["enhancers_guide"])

    if st.button("Continue to Download"):
        _set_step(7)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 7 — Download
# ---------------------------------------------------------------------------
elif _step_index() == 7:
    st.title("Your Story is Ready!")

    md_output = _build_markdown_output()
    st.download_button(
        label="Download as Markdown",
        data=md_output,
        file_name="story_output.md",
        mime="text/markdown",
    )

    with st.expander("Preview"):
        st.markdown(md_output)
