import os


def initialize_artifact(fname: str, title: str = "# Story") -> None:
    with open(fname, "w") as f:
        f.write(f"{title}\n\n")


def update_artifact(fname: str, section: str, value: str, level: int = 2):
    if not os.path.exists(fname):
        initialize_artifact(fname)

    with open(fname, "a") as f:
        f.write(f"{'#' * level} {section}\n\n{value}\n\n")
