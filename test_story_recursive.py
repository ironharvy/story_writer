from story_recursive import RecursiveWriteStory, StoryNode


def _leaf(title: str, prose: str) -> StoryNode:
    return StoryNode(title=title, beats="", depth=2, prose=prose)


def test_leaf_node_assembles_to_its_own_prose():
    leaf = _leaf("Scene", "Just this.")
    assert leaf.is_leaf is True
    assert leaf.leaf_count() == 1
    assert leaf.assembled_prose() == "Just this."


def test_internal_node_assembles_children_depth_first():
    parent = StoryNode(
        title="Chapter 1",
        beats="Mara leaves; the door closes.",
        depth=1,
        children=[_leaf("Scene A", "Alpha."), _leaf("Scene B", "Beta.")],
    )
    assert parent.is_leaf is False
    assert parent.leaf_count() == 2
    assert parent.assembled_prose() == "Alpha.\n\nBeta."


def test_nested_tree_flattens_left_to_right():
    root = StoryNode(
        title="Story",
        beats="",
        depth=0,
        children=[
            StoryNode("Ch1", "", 1, children=[_leaf("1a", "A"), _leaf("1b", "B")]),
            StoryNode("Ch2", "", 1, children=[_leaf("2a", "C")]),
        ],
    )
    assert root.leaf_count() == 3
    assert root.assembled_prose() == "A\n\nB\n\nC"


def test_recursive_write_story_builds_its_stages():
    module = RecursiveWriteStory()
    assert hasattr(module, "build_outline")
    assert hasattr(module, "expand_unit")
    assert hasattr(module, "project_bible")
    assert hasattr(module, "draft_chapter")
