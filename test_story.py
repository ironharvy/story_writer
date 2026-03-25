import dspy
import os
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

def test_pipeline():
    # Attempt to load OpenAI key, or use Mock
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000)
    else:
        # Note: TypedPredictor doesn't easily work with mock strings, so we require an API key to test fully,
        # or we just rely on the API key being set in the environment.
        # For this test script, if we are in an environment without a key, we'll try to use LiteLLM mock
        pass

    # We assume OPENAI_API_KEY is available in the run_in_bash_session, if not, we skip the actual test
    if not api_key:
        print("OPENAI_API_KEY not found. Skipping full integration test to avoid errors.")
        return

    dspy.configure(lm=lm)
    print("Testing pipeline with OpenAI API...")

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
    test_pipeline()
