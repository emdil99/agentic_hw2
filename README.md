# Supplement Label Agent

An agent that answers questions about what is actually printed on dietary supplement labels, for people who buy fitness supplements: creatine, whey protein, electrolytes and collagen. Ask it how much of an ingredient a product contains, what its warnings or directions say, or which of two products has more of something. When a label hides an amount inside a proprietary blend, does not list an ingredient, or the product is not in the database, the agent should say so instead of guessing.

Path: **Evals + simple RAG**.

> **Status:** in progress. The data pipeline, agent and eval suite are done, and the iteration log below is being written as the work happens. The web front end, Cloud Run deployment and the sample-query comparison are still to come.

## Data

All label data comes from the NIH Office of Dietary Supplements' Dietary Supplement Label Database (DSLD), through its API at `https://api.ods.od.nih.gov/dsld/v9`, which needs no key. Each label can be viewed at `https://dsld.od.nih.gov/label/<id>`.

`scripts/build_corpus.py` searches DSLD for `creatine`, `whey protein`, `electrolyte` / `electrolytes` and `collagen`, keeps up to 300 on-market labels per category, and downloads each full label to `data/raw/dsld/`. It then renders each label as a readable text document in `data/corpus.jsonl`, with brand, serving size, every ingredient amount, other ingredients and label statements. Undisclosed amounts, which DSLD records as unit `NP`, are written as "amount not disclosed on label" so they cannot be read as zero.

DSLD files a separate label for each package size and label image of the same product, so labels with the same brand and product name are de-duplicated, keeping the newest.

The pull on 2026-11-09 selected 1,199 labels, dropped 264 duplicates, and produced **935 documents**: 237 creatine, 235 collagen, 232 electrolytes and 231 whey protein. `data/manifest.json` records the search totals and counts. To rebuild:

```
uv run scripts/build_corpus.py
```

The API rate-limits clients. The script waits as long as the API's `Retry-After` header asks, and re-runs only fetch labels that are missing.

## The agent

`agent.py` is deliberately simple. It has one tool, `search_labels`, which runs BM25 keyword search over the 935 documents and returns the top 5, plus a tool-calling loop over Gemini 3.5 Flash Lite through LiteLLM and Vertex AI. Its baseline system prompt is one sentence. Every tool call is recorded and returned with the answer.

```
uv run --no-sync agent.py "How much creatine is in one serving of MTN OPS Creatine Unflavored?"
```

## Eval set

`scripts/build_dataset.py` generates the questions from the structured label data, so every gold answer is computed from the label rather than written by hand. Each sample records its question type, the DSLD label IDs it depends on (`gold_label_ids`) and a `source` link to each label. The script uses a fixed random seed, so it produces the same file every time. No eval question is used as a few-shot example in the agent's prompt.

| Type | Count | Example | Gold answer comes from |
|---|---|---|---|
| `direct_dose` | 32 | How much creatine is in one serving of MTN OPS Creatine Unflavored? | The one unambiguous ingredient row |
| `container_total` | 12 | ...in one whole container of...? | Dose × servings per container |
| `undisclosed` | 16 | How much Di-Creatine Malate is in...? | A row whose amount is `NP` |
| `not_on_label` | 8 | Does the label of ... list sucralose? | No mention anywhere on the label |
| `unknown_product` | 4 | How much creatine is in Northbridge Labs CreaPure Max Tropical? | An invented product, recorded as `origin: invented_product_name` |
| `label_statement` | 16 | What does the label say about pregnancy, nursing, or medications? | The label's precaution or directions text |
| `comparison` | 12 | Which has more sodium per serving, A or B? | Two clean ingredient rows |
| `partial_name_same` | 12 | How much creatine is in CytoSport Monster Creatine? | Every flavor of the product has the same dose |
| `partial_name_varies` | 7 | How much protein is in NOW Sports Whey Protein Creamy? | Flavors differ, so the answer is "it depends" |

A question is only generated when the label supports exactly one answer. For example, a label listing creatine twice, an amount like "less than 1 g", or a dose given per scoop on a two-scoop serving is skipped.

The 119 samples are split into **90 dev** and **29 holdout**, with every fifth question of each type held out. Hill-climbing uses dev only. The holdout is run once, at the end.

## Scorers

`task.py` defines an [Inspect](https://inspect.aisi.org.uk/) task that runs the real agent on each question and scores it four ways:

- **Judge with a golden reference** (headline score): an LLM compares the answer to the gold answer, with type-specific rules. For example, "not disclosed" is only correct if no amount is guessed.
- **Judge with a rubric:** `leads_with_answer`, `cites_label` (gives the DSLD ID or link), and `separates_label_from_advice`.
- **Deterministic:** `retrieved_gold_labels`, meaning search showed the agent every label the answer needs.
- **Deterministic:** `numeric_match`, meaning the answer states the gold amount in any weight unit, within 0.5%.

To run the evals, after `gcloud auth application-default login`:

```
uv run --no-sync inspect eval task.py                     # dev split
uv run --no-sync inspect eval task.py -T split=holdout    # final check only
uv run --no-sync inspect view                             # browse every answer and score
```

The logs of each run in the log below are saved in `evals/results/` and open in `inspect view`.

## Iteration log

The margin of error is ±1/√n: ±11.5 points on the 76 v1 dev questions, and ±10.5 on the 90 v2 dev questions.

### 1. Baseline on dataset v1: 96.1%

Log: `evals/results/01-baseline-dataset-v1-dev.eval`. Dev split, 76 questions.

| Score | Result |
|---|---|
| Reference judge (headline) | **96.1%** (73/76) |
| Rubric: leads with answer | 90.8% |
| Rubric: cites label | 5.3% |
| Rubric: separates label from advice | 98.7% |
| Search retrieved the gold labels | 97.3% |
| Numeric match | 100% |

All three misses involved look-alike products. Search returned MET-Rx Creatine 4200 capsules instead of MET-Rx Creatine powder; two flavors of one brand were mixed up in a comparison; and search could not find "Only Protein Only Protein Protein Powder Vanilla".

**The problem:** correctness can rise by at most 3.9 points, which is less than the ±11.5-point margin, so no change could ever show a real gain on this dataset. The questions were too easy for the agent, and that had to be fixed before choosing changes to make.

### 2. Dataset v2: product names as shoppers say them

Since every miss involved look-alike products, the obvious hypothesis was that ambiguous product names are the agent's weakness. Dataset v2 adds 19 questions that name a product without its flavor, the way shoppers do, such as "NOW Sports Whey Protein Creamy". The gold answer is either the shared amount, or "it depends on the version" when flavors differ. The new questions use their own random generator, so the 100 v1 questions come out byte-for-byte unchanged. This was checked by comparing the files.

Log: `evals/results/02-baseline-dataset-v2-dev.eval`. Dev split, 90 questions.

| Score | v1 baseline | v2 baseline |
|---|---|---|
| Reference judge (headline) | 96.1% | **97.8%** (88/90) |
| Rubric: leads with answer | 90.8% | 94.4% |
| Rubric: cites label | 5.3% | 8.9% |
| Rubric: separates label from advice | 98.7% | 100% |
| Search retrieved the gold labels | 97.3% | 97.7% |
| Numeric match | 100% | 100% |

**The hypothesis was wrong.** The agent got all 14 new dev questions right, and on "it depends" questions it listed each flavor's amount without being asked to. The two remaining misses are the same retrieval failures as in v1.

This run also shows the noise directly. With nothing changed, the same 76 v1 questions scored 74/76, up from 73/76, because the flavor-confusion comparison flipped to correct. That one-question swing is why the margin of error matters.

**Takeaway:** correctness is still at the ceiling. The clearest room for improvement is in behaviour rather than facts: the agent almost never cites the label it used (8.9%), even though a shopper needs that to check the answer.

## Known limitations

- **The rubric judge never sees the label,** so `separates_label_from_advice` checks how an answer is worded, not whether its facts are true. Factual correctness is covered by the reference judge.
- **`numeric_match` passes if the gold amount appears anywhere in the answer,** even if the answer concludes something else. It is a supporting check, not the headline.
- **By default the grader is the same Gemini model as the agent.** A different grader can be set with `--model-role grader=<model>`.
- **Gold answers reflect DSLD's transcription of each label** as of the pull date. A typo in DSLD becomes a typo in the gold answer.
- **The NIH ODS fact sheets were planned as a second source,** but `ods.od.nih.gov` blocks automated clients, so this agent uses DSLD only.
