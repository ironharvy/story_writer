"""Foundation package shared by all chapter-drafting generators.

`core.types`        — dataclasses passed between stages (WorldBible, PlanEntry,
                      Character, WorldState + render_world_state).
`core.artifact`     — incremental markdown writer (initialize/update).
`core.foundation`   — the idea → premise → spine → world-bible → chapter-plan
                      stages every variant shares, plus the act/spine slicer
                      and the cross-variant sanity check.

The per-variant drafting layer (rolling-summary baseline / world-state /
dspy.Module) lives elsewhere and only depends on these three modules.
"""
