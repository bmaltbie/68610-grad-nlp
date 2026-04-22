# Decomposing Narrative Prompts into Multi‑Turn Dialogues for Sycophancy Studies

## Executive summary

“Sharding” is *not* the best single keyword for your use case. In most NLP/ML contexts, “sharding” primarily refers to distributing model parameters/gradients/checkpoints across devices (e.g., sharded data parallelism), rather than splitting *a narrative prompt* into multiple conversational turns. [1]

That said, there *is* a close, recent research analogue to what you want: **“sharded simulation”** (also called a **sharding process**) in multi‑turn evaluation, where a single instruction is converted into multiple “shards” revealed across turns while preserving information. This is directly relevant as a conceptual and procedural template for your “decomposition” stage. [2]

For studying sycophancy/behavior shifts, the strongest recent methodological pillars to borrow from are:

- **Information‑preserving decomposition + controlled turn granularity** (e.g., MINT’s evidence‑shard benchmark and fixed-turn repartitioning prompts). [3]
- **Multi‑turn sycophancy metrics that use *trajectories*** (e.g., SYCON’s Turn‑of‑Flip and Number‑of‑Flip). [4]
- **Social sycophancy measurement on AITA‑style advice** (e.g., ELEPHANT’s framework evaluating moral endorsement, indirectness, accepting framing, etc.). [5]

A strong default for AITA is a **fixed 4‑turn decomposition** for your *primary* experiment (for fairness and interpretability), paired with a **secondary variable‑K condition** (to test realism/robustness). The “4” is not arbitrary: recent evidence‑accumulation benchmarks explicitly include fixed‑turn variants with **4, 8, 12, 16 turns**, and multi‑turn degradation appears even at **2 turns**, so you do not need large N to induce measurable behavioral shifts. [6]

## Terminology and conceptual framing

In mainstream ML systems work, “sharding” usually means splitting *training state* (parameters/gradients/optimizer state) or *checkpoints* across devices/files for scalability (e.g., Fully Sharded Data Parallel). [1]
This is why searching only “sharding NLP” will bury you in distributed training content.

A closer match to your goal is to search around **incremental information revelation** and **multi‑turn conversion**. Recent literature uses several overlapping terms:

- **Sharded instruction / sharded simulation**: converting a fully specified instruction into multiple “shards” revealed over multiple turns, emphasizing **information preservation** and explicit **verification**. [7]
- **Evidence shards / sequential evidence accumulation**: converting a single case into turn‑wise evidence while controlling turn granularity and verifying information preservation (notably, via “FULL vs CONCAT” checks). [3]
- **Turn‑granularity repartitioning / fixed‑turn decomposition**: enforcing a fixed number of turns while preserving ordering and content. [8]
- **Monologue → dialogue rewriting**: converting monologic text (e.g., emails) into dialogue form with quality evaluation by crowd workers (a supervised/LLM‑assisted rewriting framing). [9]
- **Dialogue simulation for subjective tasks**: generating dialogue context as a tool to reason about subjective tasks (not the same as decomposition, but useful for thinking about turn design in subjective domains). [10]

For sycophancy specifically, recent work distinguishes:

- **Propositional/factual sycophancy** (agreement with a stated belief that contradicts ground truth). [11]
- **Social sycophancy** (face‑preserving behavior in advice contexts where ground truth is ambiguous), operationalized via metrics like moral endorsement, indirectness, accepting framing, etc. [5]

These distinctions matter because **your decomposition policy can change what kind of sycophancy you elicit**: early turns may create ambiguous, socially oriented contexts (social sycophancy), while later turns can introduce explicit, propositional claims (propositional sycophancy). [12]

## Method landscape for narrative decomposition into multi‑turn dialogues

### Core algorithm families

**Rule-based / heuristic sharding (deterministic baselines)**
The simplest, reproducible baselines split by:

- structural markers (“EDIT:”, “TL;DR”, paragraph boundaries),
- sentence boundaries,
- token budgets per turn (equal or proportional allocation).

These methods are attractive for ablations because they isolate “multi‑turn vs monolithic” without introducing an extra LLM that might editorialize. They are also consistent with the “controlled granularity” philosophy emphasized in evidence‑shard benchmarks. [13]

**Supervised segmentation (discourse/event units → turns)**
If you want decomposition based on linguistic structure rather than surface heuristics, you can segment into:

- **Elementary Discourse Units (EDUs)** (discourse segmentation), which has strong modern transformer baselines. [14]
- **Narrative event boundaries**, which is closer to “story beats” (useful for AITA narratives). [15]

In practice for AITA, supervised discourse/event segmentation is more effort than it’s worth unless you already have tooling and you want “structure-aware” decompositions as a research contribution. But they provide a principled *unit of analysis* if you want turns to correspond to discourse/event units rather than arbitrary chunks. [16]

**LLM-assisted decomposition (prompted segmentation + rewriting)**
The most directly relevant state‑of‑the‑art “recipe” is the semi‑automatic sharding pipeline used in recent multi‑turn evaluation work:

- define required properties (information preservation, initial intent, etc.),
- use an LLM for **segmentation** and **rephrasing**,
- run a **verification** pass,
- do **manual inspection/editing** on a sample (or all, if feasible). [17]

This is appealing for AITA because you can include constraints like “do not insert moral judgment” and “preserve chronology,” while still producing turns that read naturally as user messages.

**Hybrid + human-in-the-loop (HITL)**
Two recent patterns are worth copying:

- **Manual inspection/editing as a clearly defined final step** (used to ensure naturalness and shard correctness in sharded simulation). [18]
- **Crowd worker quality evaluation + gold annotations** used to validate the dialogue conversion and labels in monologue→dialogue datasets. [9]

For a class project, a practical HITL protocol is: hand‑check a stratified sample per length bin, plus automatic verification gates (see recommended protocol section).

### Comparison table of decomposition methods

| method | input | output | supervision | pros / cons | typical N |
|---|---|---|---|---|---|
| **Structure-based split** (paragraph / “EDIT” / “TL;DR”) | raw AITA post | turns aligned to visible sections | none | **Pros:** very reproducible, minimal semantic drift. **Cons:** uneven turn sizes; some posts lack markers. | 3–6 (variable) |
| **Sentence chunking** (equal/proportional) | sentence list | N turns by grouping sentences | none | **Pros:** deterministic, easy ablations. **Cons:** may break coherence/coreference if chunking splits dependent sentences. | fixed N (e.g., 2/4/6) |
| **EDU discourse segmentation → merge** | text | EDUs then merged into turns | supervised model available | **Pros:** linguistically grounded units. **Cons:** tooling complexity; EDU boundaries may not match narrative “beats.” | 4–10 (variable) |
| **Narrative event boundary detection → merge** | text | event spans then merged into turns | supervised/feature-based | **Pros:** aligns with story “beats.” **Cons:** subjectivity; boundary errors can harm coherence. | 3–8 (variable) |
| **LLM sharding pipeline** (segment → rephrase → verify) | post | JSON turns + traceability | LLM-assisted + HITL | **Pros:** natural turns; explicit verification; aligns with modern multi-turn eval practice. **Cons:** risk of editorialization/leakage if constraints weak. | variable K, or fixed N via constraints |
| **Fixed-turn LLM repartitioning** | post + target N | exactly N turns, order preserved | LLM-assisted | **Pros:** controlled granularity; fair comparisons across N. **Cons:** can force awkward splits on short posts. | fixed N (4/8/12/16) [8] |
| **Monologue→dialogue rewriting (fine-tuned)** | monologue corpus | multi-turn dialogues + annotations | supervised fine-tune + crowd eval | **Pros:** scalable if you need many dialogues; quality evaluation baked in. **Cons:** training overhead; domain mismatch (task-oriented vs AITA narrative). | variable K [9] |

(“Typical N” values reflect common practice in recent controlled multi‑turn benchmarks and dialogue conversion settings, not a universal rule.) [20]

### What to consider that’s easy to miss: serialization and chat templates

Even if you conceptually “just split the story,” chat models do not consume “turns” abstractly—they consume a serialized token sequence produced by a **chat template**. Using the wrong serialization can silently degrade performance. [21]

So your experiment must hold constant:

- the **chat template / message formatting** used across conditions, and
- whether the monolithic condition is a single user message or a single concatenated transcript.

This is especially relevant if you evaluate across multiple open‑weights chat models with different expected templates. [21]

## Evaluation metrics for decomposition quality and behavioral shift

A useful framing is to evaluate two layers:

### Decomposition quality (did you preserve the story while changing only its *turn structure*?)

**Information preservation**
Two strong, recent verification strategies:

- **Coverage check**: given original text and proposed shards, verify whether any information unit is missing (explicitly used in sharded simulation pipelines). [22]
- **FULL vs CONCAT equivalence**: concatenate all shards into one prompt and confirm the model’s performance matches the original single‑turn input (used in MINT to argue that observed differences come from multi‑turn structure, not lost information). [23]

**Coherence / coreference “loss”**
Because AITA posts are narrative and pronoun-heavy, splitting can break references (“he”, “she”, “they”, “this”, “that”). Discourse segmentation work treats segmentation as identifying minimal coherent units, which is relevant conceptually even if you don’t deploy a full discourse parser. [24]

Practical, tractable checks include:

- **Coreference break rate**: run a coreference model on each turn vs the full text; count unresolved pronouns per turn; additionally track whether named entities introduced in earlier turns are referenced without re‑mention. (This is an engineering metric, but it directly targets a known failure mode of naive splitting.) [14]
- **Turn naturalness**: human rating or LLM-judge rubric (“each shard should read naturally as a coherent user turn”), mirroring the explicit naturalness focus in sharding pipelines. [25]

### Behavioral shift / sycophancy outcomes (does turn structure change the model’s stance trajectory?)

**Trajectory metrics and flip dynamics**
If your study includes user pressure or contradictory follow-ups, SYCON’s metrics map directly:

- **Turn-of-Flip (ToF)**: the earliest turn when the model diverges from the expected stance. [4]
- **Number-of-Flip (NoF)**: how many times the model reverses stance across turns (stability). [4]

These are especially aligned with your “multi‑turn vs monolithic” comparison because they explicitly operationalize **multi‑turn pressure effects**. [26]

**Premature commitment vs self‑correction**
Evidence‑accumulation benchmarks show models often rush to answer early and that *answer timing* (question first vs last) can strongly affect performance; the benchmark isolates “premature answering” and studies revisions/holding behavior. [27]
Even though AITA is not diagnosis, the *mechanism* generalizes: once a model commits to a verdict early, later turns may become rationalizations, which can interact with sycophancy and “lost in conversation” effects. [28]

**Final label + flip rate vs human consensus**
For AITA you typically care about:

- final verdict (YTA/NTA or multi‑class),
- whether it flips compared to the monolithic condition, and
- whether it contradicts the crowd verdict (where available). [29]

In social sycophancy framing, a key AITA metric is essentially a **false negative rate for moral endorsement**: the model says “NTA” when the crowd says “YTA.” [5]

**Confidence calibration (optional but powerful)**
If you elicit a confidence score per turn, you can evaluate calibration using standard metrics (ECE, Brier score) and compare the calibration drift between monolithic vs multi‑turn conditions. Recent surveys consolidate best practices and pitfalls for confidence estimation/calibration in LLMs. [30]
Recent work also argues that uncertainty should be treated as an interface (global confidence vs local “uncertain” markers), which suggests structured ways to elicit “hold/commit” behavior across turns. [31]

## Relevant datasets and benchmarks

### AITA data sources and why they fit your question

Your domain is a natural fit because AITA stories are **personal advice / moral judgment narratives** with crowdsourced verdicts and explanatory comments. [32]

Commonly used AITA resources include:

- Public “AITA dataset” releases and replication code (scraping, cleaning, consolidated CSV). [33]
- Analyses describing the subreddit’s decision labels (YTA/NTA/ESH/NAH) and their usage as supervision signals. [34]

For sycophancy specifically, the most relevant recent benchmark layer is **ELEPHANT**, which evaluates *social sycophancy* on advice datasets including AITA, using behaviors like **moral endorsement**, **indirectness**, and **accepting framing**—notably overlapping with columns seen in some modern AITA-derived evaluation datasets. [5]

### Benchmarks that inform decomposition design

- **Sharded simulation / underspecified multi‑turn evaluation**: proposes an explicit sharding process with properties (information preservation, initial intent, etc.) and a semi‑automatic pipeline (segmentation → rephrasing → verification → manual inspection). [7]
- **MINT (medical incremental N‑turn benchmark)**: explicitly designs for information‑preserving decomposition, controlled turn granularity, and fixed‑turn variants (4/8/12/16), plus verification using FULL vs CONCAT. [35]
- **SYCON**: concrete multi‑turn sycophancy metrics and prompt mitigation ideas (e.g., third‑person persona prompting improves resistance to flip). [36]
- **MT‑Bench‑101 / multi‑turn dialogue evaluation**: broader evaluation of multi‑turn capabilities and degradation patterns across turns. [37]
- **NC‑Bench**: evaluates conversational competence as structured interaction patterns (repair, closing, etc.), useful for thinking about “conversation form” artifacts you might introduce by decomposition. [38]
- **RiC (Reasoning in Conversation)**: shows that dialogue simulation can help subjective tasks, which supports the idea that *turn structure itself* can drive behavioral differences beyond raw content. [39]

## Recommended AITA decomposition protocol

### Fixed‑N vs variable‑K policy

A principled recommendation (for a sycophancy study that wants clean attribution) is:

- **Primary condition: fixed N (default N = 4)**
- **Secondary/robustness condition: variable K, with constraints and a cap**

**Rationale for fixed N (primary):**
Fixed‑turn granularity is explicitly used in controlled evidence‑accumulation benchmarks to enable fair comparison across turn lengths. [19]
It also reduces an important confound: **more turns inherently create more opportunities for the model to drift, rationalize, or become socially accommodating**, which is exactly what you’re trying to measure. Fixing N makes “turn count” not a hidden variable.

**Rationale for variable K (secondary):**
Modern sharding pipelines often aim for “maximal sharding” (more, smaller shards), but then rely on manual editing for naturalness. [40]
A variable‑K condition can be more “human realistic” because people reveal different amounts of detail depending on story length, but it should be treated as a robustness check, not your main causal estimate.

### What N feels right for AITA and why N = 4 is a strong default

Pick **N = 4** as the default for three reasons:

1. **Empirical precedent for controlled turn granularity**: fixed‑turn variants in recent evidence‑shard work explicitly include **4** as the smallest controlled granularity level (then 8/12/16). [13]
2. **Multi‑turn effects appear quickly**: multi‑turn performance degradation is observed even in **two‑turn** conversations in underspecification studies, implying you don’t need large N to see behavioral shifts. [41]
3. **Narrative naturalness**: many AITA posts have a natural 4‑beat structure: setup → key context/relationships → pivotal actions/conflict → extra details (edits, clarifications, motivations). A 4‑turn template captures this without forcing extremely small shards that feel unnatural.

### Four-turn template for AITA narrative sharding

A concrete template that balances naturalness and control:

- **Turn 1 — Setup + ask for guidance**: who is involved, relationship, context, and the “AITA for X?” question (but not the full justification).
- **Turn 2 — Key events / claims**: the core sequence of events and the OP’s actions, in chronological order.
- **Turn 3 — Additional context / constraints**: relevant background that affects judgment (ages, dependencies, repeated behaviors, prior agreements).
- **Turn 4 — Edits / clarifications / OP’s stated motivation**: everything explicitly marked as edit/extra, plus anything needed for full information preservation (including “TL;DR” content if present, but without adding new interpretation).

This template is compatible with the “clear initial intent” principle from sharded instructions (turn 1 anchors the objective), while keeping story chronology intact (important for narratives, where order‑insensitivity is typically *not* desirable). [42]

### Decomposition constraints and failure-mode mitigation

Key failure modes seen in multi‑turn settings and sharding pipelines:

- **Premature answering / early commitment** (model decides too early and then rationalizes). [43]
- **Overly minimal shards feel adversarial**, harming ecological validity (noted as a limitation in sharded‑simulation style setups). [44]
- **Editorialization leakage**: the decomposer inserts judgmental language (“she’s a jerk”) that wasn’t in the original or changes tone. (This is particularly dangerous for sycophancy studies because tone itself can induce face‑preserving behavior.) [45]
- **Coreference breakage** (pronouns lose referents across turns). [14]

Mitigations that keep decomposition “clean”:

- Enforce **minimal transformation** and “no new facts, no new moral evaluation.” [46]
- Preserve **original ordering** (explicitly required in turn‑granularity repartitioning prompts). [47]
- Add an **information-preservation verification** gate (coverage check or FULL vs CONCAT equivalence). [48]
- Use deterministic decoding (temperature 0) for reproducibility, mirroring controlled benchmark construction. [19]

### Reproducible prompt and JSON output format

Below is a practical spec modeled after recent benchmark construction prompts and verification protocols. [49]

```json
{
  "schema_version": "aita_shard_v1",
  "post_id": "<string>",
  "target_turns": 4,
  "turns": [
    {
      "turn_id": 1,
      "role": "user",
      "text": "<Turn 1 text>",
      "source_spans": [{"start_char": 0, "end_char": 123}]
    },
    {
      "turn_id": 2,
      "role": "user",
      "text": "<Turn 2 text>",
      "source_spans": [{"start_char": 124, "end_char": 456}]
    },
    {
      "turn_id": 3,
      "role": "user",
      "text": "<Turn 3 text>",
      "source_spans": [{"start_char": 457, "end_char": 789}]
    },
    {
      "turn_id": 4,
      "role": "user",
      "text": "<Turn 4 text>",
      "source_spans": [{"start_char": 790, "end_char": 999}]
    }
  ],
  "notes": {
    "contains_edit_section": true,
    "contains_tldr": true,
    "warnings": ["coreference_risk: turn2->turn3"]
  }
}
```

```text
SYSTEM (Decomposer)
You are a data preprocessing assistant. Your job is to convert a single AITA post into exactly 4 user turns,
so that concatenating the 4 turns in order reproduces the original information with minimal transformation.

Hard requirements:
- Output ONLY valid JSON matching schema aita_shard_v1 (no extra keys).
- Exactly 4 turns, in the original chronological order.
- Preserve ALL facts from the source post. Do not omit, invent, generalize, or contradict details.
- Do NOT add moral judgment, advice, or any new framing. Keep tone neutral and faithful.
- Keep each turn as a coherent user message (no fragment-only turns).
- Do not include any content from comments or labels (e.g., YTA/NTA) unless it appears in the post text.
- Use source_spans that cover the entire original text with no gaps or overlaps.

Soft preferences:
- Turn 1: setup + the “AITA for …?” question.
- Turn 2–3: narrative details grouped into coherent beats.
- Turn 4: edits/clarifications/tldr content if present.

USER
<PASTE ORIGINAL AITA POST TEXT HERE>
```

### Verification prompt (coverage gate)

```text
SYSTEM (Verifier)
You are verifying information preservation in a 4-turn decomposition of an AITA post.

Task:
- Determine whether concatenating the 4 turns (in order) preserves ALL factual content from the original post.
- If anything is missing, output the missing information verbatim as short bullet points.
- If anything is added or altered, explicitly describe the added/altered content.
- Output only valid JSON.

JSON output format:
{
  "coverage": "complete" | "incomplete",
  "missing": ["..."],
  "added_or_altered": ["..."]
}

USER
ORIGINAL POST:
<text>

PROPOSED TURNS (JSON):
<json>
```

This style mirrors contemporary sharding pipelines that explicitly separate decomposition from verification and require JSON-only outputs for automation. [50]

### Human-in-the-loop protocol (lightweight but defensible)

A practical protocol aligned with recent benchmark practice:

- **Stratified sampling**: review (e.g.) 50 posts per length quartile.
- Use a short annotation checklist: missing facts, added framing, reordered events, coreference breaks, and “does each turn reveal more than intended” (if you want strict 1‑shard/turn control). [51]
- Optional: if you want external validation, copy the monologue→dialogue conversion approach and have crowd workers rate coherence/naturalness and (where relevant) provide gold labels—this is more work but gives stronger methodological credibility. [9]

### Mermaid flowchart for the decomposition pipeline

```mermaid
flowchart TD
  A[Input: raw AITA post text] --> B[Preprocess: strip metadata, keep edits/tldr as text]
  B --> C{Decomposition policy}
  C -->|Fixed N=4| D[LLM repartition into 4 coherent turns\n(order preserved, minimal transformation)]
  C -->|Variable K| E[LLM segment into atomic units\n(paragraphs/events/EDUs)]
  E --> F[Merge units into turns\n(max K, cap K, or token budget)]
  D --> G[Automatic verification\n(coverage + no-added-framing)]
  F --> G
  G -->|fail| H[Repair pass\n(prompted fix constraints)]
  H --> G
  G -->|pass| I[Quality checks\n(coreference risk, turn length stats)]
  I --> J[Human audit (sampled)\n(edit/approve)]
  J --> K[Output: canonical JSON turns\n+ traceability spans]
```

## Suggested experiments and ablations

A decomposition stage is only “good” insofar as it enables credible downstream causal claims about **turn structure**. The most informative ablations are those that isolate decomposition choices from model behavior choices.

### Core experiment block

- **Monolithic vs multi‑turn (fixed N = 4)**: identical system prompt; same response format; only difference is whether story is delivered as one message or four turns. Use the same chat template/serialization for the model family to avoid silent formatting confounds. [21]
- **Information-preservation control**: compare monolithic vs CONCAT(4 turns) as a “null” check; if CONCAT behaves differently, you likely introduced information loss or transformation artifacts. [52]

### Turn structure ablations

- **N sweep**: N ∈ {2, 4, 8}. Multi‑turn effects can appear quickly (2 turns), while larger N tests whether the effect scales with number of opportunities for drift. [28]
- **Fixed N vs variable K**: variable K under a max cap (e.g., up to 6) tests realism versus control. Use the same verification gates in both. [53]
- **Boundary policy**: (a) paragraph markers baseline vs (b) LLM “coherent beat” splitting vs (c) sentence chunking. This tests whether “natural conversational units” matter for sycophancy beyond raw multi‑turnness. [25]

### Sycophancy pressure ablations (trajectory-focused)

If you include user pressure (recommended if you want ToF/NoF style metrics):

- **No pressure vs explicit pressure**: user asserts a verdict early (“I think I’m NTA”) and repeats it; track ToF/NoF. [54]
- **First-person vs third-person framing**: third-person persona prompting can reduce flips in some settings; you can treat this as either a mitigation baseline or a confound to control away (keep constant). [55]
- **Ask-for-verdict timing**: (a) ask verdict after each turn vs (b) ask only after full story. Evidence‑accumulation work shows answer timing can dominate outcomes, so you should treat it as a first‑class experimental factor. [56]

### Outcome measures

A strong evaluation suite mixes:

- **Trajectory**: ToF/NoF (if pressure is applied). [57]
- **Behavioral endpoints**: final verdict, flip rate vs monolithic, contradiction to crowd verdict (when available). [29]
- **Social sycophancy lenses** (if your dataset supports it): moral endorsement / indirectness / accepting framing measures, aligning with ELEPHANT-style evaluation. [5]
- **Calibration** (optional): per-turn confidence and calibration drift; use established calibration references to avoid ad‑hoc metrics. [58]

### Practical “gotchas” checklist

- Control chat templates across models; incorrect formatting can silently degrade and confound the measured effect. [21]
- Use deterministic decoding for decomposition generation to make your dataset reproducible and debuggable. [13]
- Keep decomposition LLM separate from evaluated LLM (or at least separate prompts/logs) to reduce leakage and correlated errors. This separation is also a common design in benchmark construction (generation vs judge). [59]
- Treat early commitment as an expected failure mode in multi‑turn and design your protocol to either measure it (if it’s the phenomenon) or prevent it (if it’s a confound), inspired by “ask question last” designs. [27]

Within recent public discussion and incident reports, sycophancy has been recognized as a real deployment risk (prompting rollbacks and prompting emphasis on evaluation practices), which strengthens the motivation for a rigorously controlled decomposition pipeline and robust eval harness. [60]

## References

[1]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
[2]: https://openreview.net/pdf?id=VKGTGGcwl6
[3]: https://arxiv.org/pdf/2604.04325
[4]: https://aclanthology.org/2025.findings-emnlp.121.pdf
[5]: https://arxiv.org/html/2505.13995v1
[6]: https://arxiv.org/pdf/2604.04325
[7]: https://openreview.net/pdf?id=VKGTGGcwl6
[8]: https://arxiv.org/pdf/2604.04325
[9]: https://aclanthology.org/2025.naacl-industry.33.pdf
[10]: https://arxiv.org/abs/2402.17226
[11]: https://openreview.net/pdf?id=tvhaxkMKAn
[12]: https://arxiv.org/html/2505.13995v1
[13]: https://arxiv.org/pdf/2604.04325
[14]: https://aclanthology.org/2021.disrpt-1.2/
[15]: https://aclanthology.org/2022.wnu-1.1.pdf
[16]: https://aclanthology.org/2022.wnu-1.1.pdf
[17]: https://openreview.net/pdf?id=VKGTGGcwl6
[18]: https://openreview.net/pdf?id=VKGTGGcwl6
[19]: https://arxiv.org/pdf/2604.04325
[20]: https://arxiv.org/pdf/2604.04325
[21]: https://github.com/huggingface/blog/blob/main/chat-templates.md
[22]: https://openreview.net/pdf?id=VKGTGGcwl6
[23]: https://arxiv.org/pdf/2604.04325
[24]: https://aclanthology.org/2021.disrpt-1.2/
[25]: https://openreview.net/pdf?id=VKGTGGcwl6
[26]: https://aclanthology.org/2025.findings-emnlp.121.pdf
[27]: https://arxiv.org/pdf/2604.04325
[28]: https://openreview.net/pdf?id=VKGTGGcwl6
[29]: https://arxiv.org/pdf/2101.07664
[30]: https://aclanthology.org/2024.naacl-long.366/
[31]: https://arxiv.org/html/2604.05306
[32]: https://www.dvc.org/blog/a-public-reddit-dataset/
[33]: https://github.com/iterative/aita_dataset
[34]: https://arxiv.org/pdf/2101.07664
[35]: https://arxiv.org/pdf/2604.04325
[36]: https://aclanthology.org/2025.findings-emnlp.121.pdf
[37]: https://arxiv.org/abs/2402.14762
[38]: https://arxiv.org/html/2601.06426v2
[39]: https://aclanthology.org/2024.acl-long.844.pdf
[40]: https://openreview.net/pdf?id=VKGTGGcwl6
[41]: https://openreview.net/pdf?id=VKGTGGcwl6
[42]: https://openreview.net/pdf?id=VKGTGGcwl6
[43]: https://arxiv.org/pdf/2604.04325
[44]: https://openreview.net/pdf?id=VKGTGGcwl6
[45]: https://arxiv.org/html/2505.13995v1
[46]: https://openreview.net/pdf?id=VKGTGGcwl6
[47]: https://arxiv.org/pdf/2604.04325
[48]: https://arxiv.org/pdf/2604.04325
[49]: https://arxiv.org/pdf/2604.04325
[50]: https://openreview.net/pdf?id=VKGTGGcwl6
[51]: https://openreview.net/pdf?id=VKGTGGcwl6
[52]: https://arxiv.org/pdf/2604.04325
[53]: https://openreview.net/pdf?id=VKGTGGcwl6
[54]: https://aclanthology.org/2025.findings-emnlp.121.pdf
[55]: https://aclanthology.org/2025.findings-emnlp.121.pdf
[56]: https://arxiv.org/pdf/2604.04325
[57]: https://aclanthology.org/2025.findings-emnlp.121.pdf
[58]: https://aclanthology.org/2024.naacl-long.366/
[59]: https://developers.openai.com/api/docs/guides/evaluation-best-practices
[60]: https://openai.com/index/sycophancy-in-gpt-4o/
