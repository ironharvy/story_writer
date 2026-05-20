# Craft Vocabulary

Eric Evans calls it knowledge crunching: build the model with the domain experts and speak their language. For fiction, the "domain experts" are centuries of writing craft. This file collects the established terms we adopt so we don't invent worse versions of concepts that already have names. These terms are part of our ubiquitous language — use them in code, in prompts, and in `concept.md`.

Each entry: a one-line definition in our voice, plus the source it comes from.

## Story-shape frameworks

- **Three-act structure** — setup / confrontation / resolution; the default dramatic skeleton. *(Classical; Aristotle's* Poetics*; Syd Field for screen.)*
- **Five-act structure / Freytag's Pyramid** — exposition → rising action → climax → falling action → dénouement. *(Gustav Freytag,* Die Technik des Dramas*, 1863.)*
- **Hero's Journey (monomyth)** — ordinary world → call → refusal → mentor → threshold → trials → ordeal → reward → road back → resurrection → return with the elixir. *(Joseph Campbell,* The Hero with a Thousand Faces*; Christopher Vogler,* The Writer's Journey*.)*
- **Save the Cat beat sheet** — 15 beats at fixed proportional positions: opening image, theme stated, setup, catalyst, debate, break into two, B story, fun and games, midpoint, bad guys close in, all is lost, dark night of the soul, break into three, finale, final image. *(Blake Snyder,* Save the Cat!*.)*
- **Story Spine** — "Once upon a time… every day… but one day… because of that… because of that… until finally… and ever since." A compact dramatic shape. *(Kenn Adams; popularized via Pixar / Emma Coats. Already in* concept.md *§4.)*

## Meaning and theme

- **Controlling idea / theme** — the single sentence of meaning the whole story proves (e.g. "love endures when it stops trying to possess"). *(Robert McKee,* Story*.)*
- **Premise** — the compressed dramatic situation: protagonist + goal + opposition + stakes. *(General craft; cf. Lajos Egri,* The Art of Dramatic Writing*.)*
- **Story design spectrum: archplot / miniplot / antiplot** — classical closed design, minimalist open design, or anti-structure. Tells us how tightly to enforce structure. *(McKee,* Story*.)*

## Structure and momentum

- **Inciting incident** — the event that knocks the protagonist's life out of balance and starts the story's central question. *(McKee.)*
- **Turning point / reversal** — a beat where the situation flips from one charged state to its opposite. *(McKee; classical peripeteia.)*
- **The gap** — the rift between what a character expects from an action and what actually happens; the engine of escalation. *(McKee.)*
- **Try/fail cycles ("but / therefore", not "and then")** — scenes connected by causal consequence and complication, never mere sequence. *(South Park's Trey Parker & Matt Stone; widely adopted.)*
- **Setup & payoff / Chekhov's gun** — anything emphasized must later matter; anything that later matters should be earlier seeded. *(Anton Chekhov; general craft.)* The Outline must track these explicitly.
- **Stakes** — what is gained or lost if the protagonist succeeds or fails; must escalate. *(General craft.)*

## Conflict and character

- **Three levels of conflict** — inner (within the self), personal (between intimates), extra-personal (with society/environment). *(McKee,* Story*.)*
- **Character arc** — the protagonist's internal change (positive, negative, or flat) tracked across the story. *(General craft; K.M. Weiland,* Creating Character Arcs*.)*
- **Want vs. need** — the conscious external desire vs. the unconscious internal lesson; their tension drives arc. *(John Truby,* The Anatomy of Story*; McKee.)*
- **Flaw / wound / ghost** — the past injury and resulting misbelief the arc must resolve. *(Truby; general craft.)*
- **Agency** — the protagonist *drives* events by choice, rather than passively having things happen to them. *(General craft; a frequent failure mode — see the rubric.)*
- **Status** — the shifting power relationship between characters within a scene. *(Keith Johnstone,* Impro*.)*

## Scene-level craft

- **Scene-and-sequel** — a *scene* (goal → conflict → disaster) followed by a *sequel* (reaction → dilemma → decision); the alternating proactive/reactive rhythm. *(Dwight Swain,* Techniques of the Selling Writer*; Jack Bickham,* Scene & Structure*.)*
- **Motivation-Reaction Unit (MRU)** — at the sentence level, an external motivating stimulus followed by the character's reaction in the order felt → reflex → action/speech. *(Dwight Swain.)*
- **Scene value / scene turn** — every scene must shift a value charge (e.g. safe→endangered, trust→betrayal); a scene that ends on the same charge it began is inert. *(McKee,* Story*.)*

## Narration and voice

- **Point of view (POV)** — first / second / third person. *(Standard narratology.)*
- **Narrative access** — limited (one character's interiority) vs. omniscient (any). *(Standard narratology.)*
- **Narrative distance** — how close the narration sits to a character's consciousness, from distant report to deep interiority. *(Standard narratology; cf. John Gardner,* The Art of Fiction*.)*
- **Free indirect discourse** — rendering a character's thoughts in third person without tag, blending narrator and character voice. *(Literary technique.)*
- **Tense** — past vs. present; a voice decision held constant. *(General craft.)*

## Technique

- **Show, don't tell** — dramatize through action and sensory detail rather than summarizing or asserting. *(General craft; oft-attributed to Chekhov.)*
- **Dramatic irony** — the reader knows something a character does not, creating tension. *(Classical.)*
- **Foreshadowing** — seeding later developments so payoffs feel earned, not arbitrary. *(General craft.)*
- **Motif** — a recurring concrete image/object carrying thematic weight (e.g. the button-eyed doll that recurs across chapters in the prior bake-off's best run). *(General craft.)*

## How this maps onto our pipeline

This vocabulary tells each pipeline stage what it must be able to represent (see `concept.md` §6–8):

- **Bible decides world & character** → must represent: controlling idea/theme, character (arc, want/need, flaw/wound, voice, status relationships), world rules, POV/access/distance/tense, story-design type (archplot/miniplot/antiplot), name uniqueness.
- **Outline decides plot** → must represent: act/sequence/scene structure mapped to a chosen framework (three-act, Save the Cat, Hero's Journey, spine), inciting incident, turning points, escalating stakes, three-level conflict, and explicit setups & payoffs per scene.
- **Scene Generator decides prose** → must honor: scene-and-sequel rhythm, scene value/turn, MRUs, narrative distance, show-don't-tell, dramatic irony, motifs.

When we design the Bible schema and Outline format, these are the concepts the types must be able to hold. If a craft concept above has no home in our schema, that's a gap to question.
