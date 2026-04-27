import sys
import dspy


class WriteParagraph(dspy.Signature):
    """Write a paragraph based on the given input."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    act: str = dspy.InputField(desc="Act description")
    story_so_far: str = dspy.InputField(desc="The story so far.")
    chapter: str = dspy.InputField(desc="Chapter description")
    previous_paragraph: str = dspy.InputField(desc="The previous paragraph.")
    paragraph: str = dspy.OutputField(desc="The generated paragraph.")
    end_of_chapter: bool = dspy.OutputField(desc="Whether this is the end of the chapter.")
    updated_story_so_far: str = dspy.OutputField(desc="The updated story so far.")

if __name__ == "__main__":
    print("configuring lm...")
    lm = dspy.LM("ollama/qwen3", max_tokens=2000)
    dspy.configure(lm=lm)

    print("loading data...")
    with open(".tmp/chapter_plan.md", "r") as f:
        chapter_plan = f.read()
    print("loading world bible...")
    with open(".tmp/world_bible.md", "r") as f:
        world_bible = f.read()
    
    chapters = chapter_plan.split("\n")
    
    write_paragraph = dspy.ChainOfThought(WriteParagraph)
    story = "# Story\n\n"
    story_so_far = ""
    print("starting story generation...")
    for act in ["Act 1 - Setup", "Act 2 - Confrontation", "Act 3 - Resolution"]:
        print(f"Act: {act}")
        for chapter in chapters[1:]:
            chapter_title = chapter.split("-")[0].strip()
            story += f"## {chapter_title}\n\n"
            print(f"Chapter: {chapter}")
            end_of_chapter = False
            chapter_text = ""
            previous_paragraph = ""
            while not end_of_chapter:
                result = write_paragraph(
                    world_bible=world_bible,
                    act=act,
                    story_so_far=story_so_far,
                    chapter=chapter,
                    previous_paragraph=previous_paragraph,
                )

                chapter_text += result.paragraph + "\n"

                end_of_chapter = result.end_of_chapter
                previous_paragraph = result.paragraph
                story_so_far = result.updated_story_so_far

                print("## " + result.paragraph)
            with open(f".tmp/story.md", "w") as f:
                f.write(story)
            sys.exit(0)

