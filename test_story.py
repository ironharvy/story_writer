import dspy
import os
import argparse
from story_modules import (
    QuestionGenerator,
    CorePremiseGenerator,
    SpineTemplateGenerator,
    WorldBibleGenerator,
    PlotGenerator
)

# A mock LM to avoid needing an API key for automated testing
class MockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock")

    def __call__(self, prompt=None, messages=None, **kwargs):
        # We need to return JSON since dspy expects it
        return ["""```json
{
  "questions_with_answers": [{"question": "q", "proposed_answer": "a"}],
  "core_premise": "core",
  "spine_template": "spine",
  "world_bible": "world",
  "arc_outline": "arc",
  "chapter_plan": "chapter",
  "scenes": "scenes"
}
```"""]

def test_pipeline(model_name="ollama_chat/llama3", api_base="http://localhost:11434", api_key=None):
    kwargs = {"max_tokens": 1000}
    if api_base:
        kwargs["api_base"] = api_base

    if api_key is not None:
        kwargs["api_key"] = api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            kwargs["api_key"] = env_key
    elif "ollama" in model_name.lower():
        # litellm requires api_key to not be empty string or None for some providers, but actually for ollama we can just pass none but without api_key=""
        pass

    print(f"Testing pipeline with model: {model_name}...")

    # We assume OPENAI_API_KEY is available in the run_in_bash_session, if not, we skip the actual test
    if "openai" in model_name.lower() and not kwargs.get("api_key"):
        print("OPENAI_API_KEY not found. Skipping full integration test to avoid errors.")
        return

    if model_name == "mock":
        lm = MockLM()
    else:
        lm = dspy.LM(model_name, **kwargs)
    dspy.configure(lm=lm)

    idea = "A story about a space pirate who finds a map to the center of the universe."

    # 1. Questions
    q_gen = QuestionGenerator()
    q_result = q_gen(idea=idea)
    print(f"Generated {len(q_result.questions_with_answers)} questions.")

    # Fake answers
    qa_text = ""
    for q in q_result.questions_with_answers:
        qa_text += f"Q: {q.question}\nA: {q.proposed_answer}\n\n"

    # 2. Core Premise
    cp_gen = CorePremiseGenerator()
    cp_result = cp_gen(idea=idea, qa_pairs=qa_text)
    print("Core Premise generated.")

    # 3. Spine Template
    st_gen = SpineTemplateGenerator()
    st_result = st_gen(core_premise=cp_result.core_premise)
    print("Spine Template generated.")

    # 4. World Bible
    wb_gen = WorldBibleGenerator()
    wb_result = wb_gen(core_premise=cp_result.core_premise, spine_template=st_result.spine_template)
    print("World Bible generated.")

    # 5. Plot
    plot_gen = PlotGenerator()
    plot_result = plot_gen(core_premise=cp_result.core_premise, spine_template=st_result.spine_template, world_bible=wb_result.world_bible)
    print("Plot generated.")
    print("Test passed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AI DSPy Story Writer")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "ollama_chat/llama3"), help="The language model to use (e.g., openai/gpt-4o-mini, ollama_chat/llama3). Defaults to MODEL env var or ollama_chat/llama3.")
    parser.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL", "http://localhost:11434"), help="The custom API base URL (e.g., http://localhost:11434 for Ollama). Defaults to LLM_URL env var or http://localhost:11434.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY"), help="The API key for the model. Defaults to API_KEY env var.")

    args = parser.parse_args()

    test_pipeline(model_name=args.model, api_base=args.llm_url, api_key=args.api_key)
