# Supplement Label Agent

An agent that answers questions about what is actually printed on dietary supplement labels, for people who buy fitness supplements: creatine, whey protein, electrolytes and collagen. Ask it how much of an ingredient a product contains, what its warnings or directions say, or which of two products has more of something. When a label hides an amount inside a proprietary blend, does not list an ingredient, or the product is not in the database, the agent should say so instead of guessing.

Path: **Evals + simple RAG**.

## Try it

**Live app: https://agentic-hw2-801623024658.europe-west1.run.app**

The app sits behind Identity-Aware Proxy, so sign in with a `columbia.edu` Google account. It deploys continuously from `main` through Cloud Build. The front end, "What's in the Scoop?", explains what the agent covers and who it is for, offers example questions, and shows every tool call behind an answer under **Show work**. `POST /chat` takes `{"message", "session_id"}` and returns `{"response", "session_id", "tool_calls"}`, the same shape as Project 1. Each tool call carries an `agent` field.

To run it locally, you need a Google Cloud project with Vertex AI enabled:

```
gcloud auth application-default login
uv run app.py          # then open http://127.0.0.1:8000
```

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

`agent.py` runs a tool-calling loop over Gemini 3.5 Flash Lite through LiteLLM and Vertex AI. Every tool call is recorded and returned with the answer. It has two tools:

- **`search_labels`:** BM25 keyword search over the 935 documents, returning the top 5. This was the only tool in the baseline.
- **`query_database`:** added in change 2 of the iteration log. It looks across every label in a category at once, so it can rank products by an ingredient's amount per serving, or count the products whose ingredient list mentions something. It says how many labels list the ingredient without one clear amount and so could not be ranked, and it never lets its result limit hide a tie.

The system prompt started as one sentence. Changes 1 and 2 below each added to it.

```
uv run agent.py "How much creatine is in one serving of MTN OPS Creatine Unflavored?"
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
| `database_most` | 3 | Which electrolyte product in the database has the most magnesium per serving? | Every label in the category, ranked |
| `database_least` | 4 | Which whey protein product in the database has the least protein per serving? | Every label in the category, ranked |
| `database_count` | 16 | How many collagen products in the database list biotin as an ingredient? | Every label in the category, counted |

The first seven types are dataset v1, the two `partial_name` types were added in v2, and the three `database` types in v3. Each version adds questions without changing earlier ones.

A question is only generated when the labels support exactly one answer. For example, a label listing creatine twice, an amount like "less than 1 g", or a dose given per scoop on a two-scoop serving is skipped. A most-or-least question is skipped if the winner is tied, or if a label that lists the ingredient without one clear amount discloses an amount that could beat the winner. This rules out both creatine questions, for example.

The 142 samples are split into **107 dev** and **35 holdout**, with every fifth question of each type held out. Hill-climbing uses dev only. The holdout is run once, at the end.

## Scorers

`task.py` defines an [Inspect](https://inspect.aisi.org.uk/) task that runs the real agent on each question and scores it four ways:

- **Judge with a golden reference** (headline score): an LLM compares the answer to the gold answer, with type-specific rules. For example, "not disclosed" is only correct if no amount is guessed.
- **Judge with a rubric:** `leads_with_answer`, `cites_label` (gives the DSLD ID or link), and `separates_label_from_advice`.
- **Deterministic:** `retrieved_gold_labels`, meaning search showed the agent every label the answer needs.
- **Deterministic:** `numeric_match`, meaning the answer states the gold amount in any weight unit, within 0.5%.

To run the evals, you need a Google Cloud project with Vertex AI and access to Gemini 3.5 Flash Lite. `uv run` installs the dependencies from `uv.lock` on first use.

```
gcloud auth application-default login
uv run inspect eval task.py                     # dev split, about 3 minutes
uv run inspect eval task.py -T split=holdout    # final check only
uv run inspect view                             # browse every answer and score
```

By default the grader model is the same Gemini model as the agent, so it grades its own answers. To use a different judge, add `--model-role grader=<model>`.

The logs of each run in the log below are saved in `evals/results/` and open in `inspect view`.

## Iteration log

The margin of error is ±1/√n: ±11.5 points on the 76 v1 dev questions, ±10.5 on the 90 v2 dev questions, and **±9.7 on the 107 v3 dev questions**, which is the final dataset.

### Summary

| Run | Dataset | Agent | Correct (headline) | Cites label |
|---|---|---|---|---|
| 1 | v1 | Baseline | 96.1% | 5.3% |
| 2 | v2 | Baseline | 97.8% | 8.9% |
| 3 | v3 | Baseline | **82.2%** | **8.4%** |
| 4 | v3 | + Change 1: cite labels | 80.4% | **95.3%** |
| 5 | v3 | + Change 2: `query_database` tool | **97.2%** | 87.9% |

On dataset v3, measured against a ±9.7-point margin:
- **Change 1 raised citations by 86.9 points** (8.4% → 95.3%), well beyond the margin. Its effect on correctness, −1.8 points, is within it.
- **Change 2 raised correctness by 16.8 points** (80.4% → 97.2%), beyond the margin, and by 15.0 points over the v3 baseline. Its effect on citations, −7.4 points, is within it.

Runs 1 and 2 are the story of finding a dataset hard enough to measure anything on.

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

### 3. Dataset v3, questions that need the whole database: 82.2%

Some questions a shopper asks need every label at once: which electrolyte has the most magnesium, or how many whey proteins use stevia. Top-5 search cannot answer these by design. Dataset v3 adds 23 of them, with gold answers computed over every label in the category. Each count was also re-checked independently against the documents the agent reads.

Log: `evals/results/03-baseline-dataset-v3-dev.eval`. Same agent as before. Dev split, 107 questions.

The agent got **0 of the 17 dev whole-database questions** right, while scoring 88/90 on the rest. Some answers admitted the limit ("the search tool... does not provide a total count"), but several were confidently wrong. For example, "there are **5** electrolyte products that list taurine", when the answer is 29, because it counted its 5 search results as if they were the whole database. That is the failure this dataset was built to expose, and it gives the eval room to measure a real gain.

### 4. Change 1, cite the labels used: citations 8.4% → 95.3%

One sentence added to the system prompt: *"End every answer with the labels you used: brand and product name, DSLD ID, and link."*

Log: `evals/results/04-change1-cite-labels-dev.eval`.

Citations rose by 86.9 points. Correctness moved from 82.2% to 80.4%, which is two questions. One was a look-alike product: the agent answered from Vital Proteins Collagen Peptides *Lemon* (20 g) instead of the plain product (3.6 g). The other was "Only Protein", the search failure seen in every run, which happened to pass in run 3. Both are within the margin and unrelated to citing.

### 5. Change 2, a whole-database tool: correctness 80.4% → 97.2%

Added the `query_database` tool and told the agent to use it for most, least and how-many questions, and never to count or rank from search results.

Log: `evals/results/05-change2-database-tool-dev.eval`.

Whole-database questions went from **0/17 to 17/17**. Three questions are still wrong:
- **MET-Rx Creatine and "Only Protein":** the same two retrieval failures as in every run.
- **Invented product "Halcyon Foundry Whey Isolate Maple":** the agent correctly said it was not found, then offered a similar product's protein amount, which the judge counted as guessing.

While testing the tool, the agent was asked which whey protein has the most protein. It requested 1 result and presented one product as *the* answer, though seven products tie at 30 g. The tool now always returns every product tied with the last one listed, and the agent then reports the seven-way tie. This fix was made before run 5, and that question is not in the eval, since tied questions are excluded.

Citations fell from 95.3% to 87.9%, within the margin. Part of the drop is count answers, which give a number of products without naming 40 labels. Part is judge error: one answer marked as not citing ends with "DSLD ID: 316098" and its link.

## Known limitations

- **The rubric judge never sees the label,** so `separates_label_from_advice` checks how an answer is worded, not whether its facts are true. Factual correctness is covered by the reference judge. The rubric judge also makes mistakes: in run 5 it marked an answer with a DSLD ID and link as not citing.
- **`query_database` and the gold answers share their label-reading code** (`label_facts.py`). So the whole-database questions measure whether the agent knows to use the tool and reports what it returns, not whether the label parsing is right. The parsing itself was checked separately by recounting from the rendered documents.
- **`numeric_match` passes if the gold amount appears anywhere in the answer,** even if the answer concludes something else. It is a supporting check, not the headline.
- **By default the grader is the same Gemini model as the agent.** A different grader can be set with `--model-role grader=<model>`.
- **Gold answers reflect DSLD's transcription of each label** as of the pull date. A typo in DSLD becomes a typo in the gold answer.
- **The NIH ODS fact sheets were planned as a second source,** but `ods.od.nih.gov` blocks automated clients, so this agent uses DSLD only.
