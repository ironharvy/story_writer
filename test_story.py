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

    def __call__(self, prompt, **kwargs):
        return ["Mock response"]

def test_pipeline(use_ollama=False, ollama_model="llama3", ollama_base_url="http://localhost:11434"):
    if use_ollama:
        print(f"Testing pipeline with Ollama local model: {ollama_model} at {ollama_base_url}...")
        lm = dspy.LM(f'ollama_chat/{ollama_model}', api_base=ollama_base_url, api_key='')
    else:
        # Attempt to load OpenAI key
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000)
            print("Testing pipeline with OpenAI API...")
        else:
            # Note: TypedPredictor doesn't easily work with mock strings, so we require an API key to test fully,
            # or we just rely on the API key being set in the environment.
            # For this test script, if we are in an environment without a key, we'll try to use LiteLLM mock
            pass

        # We assume OPENAI_API_KEY is available in the run_in_bash_session, if not, we skip the actual test
        if not api_key:
            print("OPENAI_API_KEY not found and USE_OLLAMA not set. Skipping full integration test to avoid errors.")
            return

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
    parser.add_argument("--use-ollama", action="store_true", help="Use a local Ollama model instead of OpenAI. Overrides USE_OLLAMA env var.")
    parser.add_argument("--ollama-model", type=str, default=os.environ.get("OLLAMA_MODEL", "llama3"), help="The Ollama model to use. Defaults to OLLAMA_MODEL env var or 'llama3'.")
    parser.add_argument("--ollama-base-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), help="The Ollama base URL. Defaults to OLLAMA_BASE_URL env var or 'http://localhost:11434'.")

    args = parser.parse_args()

    use_ollama = args.use_ollama or os.environ.get("USE_OLLAMA", "").lower() in ("1", "true", "yes")

    test_pipeline(use_ollama=use_ollama, ollama_model=args.ollama_model, ollama_base_url=args.ollama_base_url)
