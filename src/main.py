import os
import json

import dotenv
import argparse
import logging
import coloredlogs

from writer import Writer
from data import QuestionWithAnswer

dotenv.load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, default="config.json")
    parser.add_argument("--level", type=str, default="INFO")
    
    return parser.parse_args()

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper()))
    coloredlogs.install(level=level.upper())


def read_multiline_idea(end_marker: str = "END") -> str:
    print("Paste your story idea or lyrics.")
    print(f"When done, type {end_marker!r} on a new line or press Ctrl+D.")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == end_marker:
            break
        lines.append(line)

    return "\n".join(lines).strip()

def load_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        return json.load(f)

def ask_questions(entries: list[QuestionWithAnswer]) -> list[QuestionWithAnswer]:
    """Display each question, let user accept or override the answer."""
    answered: list[QuestionWithAnswer] = []
    for i, entry in enumerate(entries, 1):
        print(f"\nQuestion {i}: {entry.question}")
        print(f"Proposed Answer: {entry.answer}")
        user_input = input("Accept? (Enter to accept, or type your answer): ").strip()
        answer = user_input if user_input else entry.answer
        answered.append(QuestionWithAnswer(question=entry.question, answer=answer))
    return answered


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.level)
    idea = read_multiline_idea()

    config = load_config(args.config)
    writer = Writer(config)
    qa_text = writer.compose(idea, ask_questions=ask_questions)
    print("\n--- Answered Q&A ---")
    print(qa_text)
    
    

