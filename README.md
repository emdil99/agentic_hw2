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

## Sample queries

Each query was asked of the final agent, and of the same model (Gemini 3.5 Flash Lite) with no tools and no system prompt.

### 1. How much creatine is in one serving of MTN OPS Creatine Unflavored?

- **Expected:** 5 g of creatine per 5 g serving.
- **Source:** DSLD label 327384, at `https://dsld.od.nih.gov/label/327384`, or through the API at `https://api.ods.od.nih.gov/dsld/v9/label/327384`.
- **Agent:** "One serving (5 grams) of MTN OPS Creatine Unflavored contains 5 grams of micronized creatine monohydrate." It then cites DSLD ID 327384 with a link.
- **No tools:** "one serving contains **5 grams** of pure Creatine Monohydrate."

Both are right. A 5 g creatine scoop is common knowledge, so this query is a control: the agent adds a citation the shopper can check.

### 2. How much magnesium is in one serving of Muscle Feast Electrolytes++ Hydrate Unflavored?

- **Expected:** not disclosed. The label lists magnesium without an amount (DSLD unit `NP`), and magnesium sulfate appears among its other ingredients.
- **Source:** DSLD label 267783, at `https://dsld.od.nih.gov/label/267783`.
- **Agent:** "The label ... lists magnesium under its ingredients (as magnesium sulfate), but the exact amount per serving is not disclosed on the label." It cites DSLD ID 267783.
- **No tools:** "one serving typically provides **125 mg of magnesium** (usually sourced from magnesium malate or TRAACS magnesium bisglycinate chelate...)."

The model on its own invents a specific amount and a plausible-sounding source. The agent reports what the label actually says.

### 3. How many collagen products in the database list biotin as an ingredient?

- **Expected:** 21 products.
- **Source:** every collagen label in `data/corpus.jsonl`, counted by `query_database` (`label_facts.lists_ingredient`). Each counted product links to its `https://dsld.od.nih.gov/label/<id>` page.
- **Agent:** "There are 21 collagen products in the database that list biotin as an ingredient. Here are 10 of them: ..." Each listed product has its DSLD ID and link.
- **No tools:** "I do not have access to a database of collagen products, so I cannot tell you how many of them list biotin."

Only the agent's whole-database tool can answer this. The baseline agent, with search alone, got every question like this wrong.

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

`agent.py` runs a tool-calling loop over Gemini 3.5 Flash Lite through LiteLLM and Vertex AI, capped at 6 tool rounds. Every tool call is recorded and returned with the answer. It has two tools:

- **`search_labels`:** BM25 keyword search over the 935 documents, returning the top 5. This was the only tool in the baseline.
- **`query_database`:** added in change 2 of the iteration log. It looks across every label in a category at once, so it can rank products by an ingredient's amount per serving, or count the products whose ingredient list mentions something. It says how many labels list the ingredient without one clear amount and so could not be ranked, and it never lets its result limit hide a tie.

The system prompt started as one sentence. Changes 1 and 2 below each added a line, and change 3 was tried and reverted.

```
uv run agent.py "How much creatine is in one serving of MTN OPS Creatine Unflavored?"
```

## Eval set

`scripts/build_dataset.py` generates the questions from the structured label data, so every gold answer is computed from the label rather than written by hand. All 141 samples are **gold**. Each records its question type, its `origin`, the DSLD label IDs it depends on (`gold_label_ids`), and a `source` link to each label, or a note saying which labels were counted. The script uses a fixed random seed, so it produces the same file every time. No eval question is used as a few-shot example in the agent's prompt.

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
| `database_most` | 2 | Which collagen product in the database has the most collagen per serving? | Every label in the category, ranked |
| `database_least` | 4 | Which whey protein product in the database has the least protein per serving? | Every label in the category, ranked |
| `database_count` | 16 | How many collagen products in the database list biotin as an ingredient? | Every label in the category, counted |

The first seven types are dataset v1, the two `partial_name` types were added in v2, and the three `database` types in v3. Each version adds questions without changing earlier ones.

A question is only generated when the labels support exactly one answer:
- **Single amounts:** a label listing creatine twice, an amount like "less than 1 g", or a dose given per scoop on a two-scoop serving is skipped.
- **Daily-value check:** an amount is ignored when its own exact % Daily Value says it is more than 10 times off. This catches DSLD unit errors such as magnesium "10 g" at 2% DV, which is really about 10 mg.
- **Most and least:** a question is skipped if the winner is tied, or if a label that lists the ingredient without one clear amount discloses an amount that could beat the winner. Its gold answer says "among the N labels that state one clear amount".
- **Counts:** counts include the synonyms labels use. Stevia also matches Reb A, rebaudioside and steviol glycosides; monk fruit also matches luo han guo.

The samples are split into **106 dev** and **35 holdout**, with every fifth question of each type held out. A question's split is frozen once assigned, so changing the dataset can never move a dev question into the holdout. Hill-climbing used dev only. The holdout was run once, on the final agent.

## Scorers

`task.py` defines an [Inspect](https://inspect.aisi.org.uk/) task that runs the real agent on each question and scores it four ways:

- **Judge with a golden reference** (headline score): an LLM compares the answer to the gold answer, with type-specific rules. For example, "not disclosed" is only correct if no amount is guessed.
- **Judge with a rubric:** `leads_with_answer`, `cites_label` (gives the DSLD ID or link), and `separates_label_from_advice`.
- **Deterministic:** `retrieved_gold_labels`, meaning search showed the agent every label the answer needs.
- **Deterministic:** `numeric_match`, meaning the answer states the gold amount in any weight unit, within 0.5%.

To run the evals, you need a Google Cloud project with Vertex AI and access to Gemini 3.5 Flash Lite. `uv run` installs the dependencies from `uv.lock` on first use. `--retry-on-error` re-runs a question that hits a temporary Vertex AI error (502 or 429) instead of aborting the whole eval.

```
gcloud auth application-default login
uv run inspect eval task.py --max-samples 3 --retry-on-error=3                     # dev split
uv run inspect eval task.py -T split=holdout --max-samples 3 --retry-on-error=3    # final check only
uv run inspect view                                                                # browse every answer and score
```

By default the grader model is the same Gemini model as the agent, so it grades its own answers. To use a different judge, add `--model-role grader=<model>`.

Every run in the log below is saved in `evals/results/` and opens in `inspect view`.

## Iteration log

The margin of error is ±1/√n. On the final dev set of 106 questions it is **±9.7 points**. On the 35-question holdout it is ±16.9 points.

### Summary

| Run | Dataset | Agent | Correct (headline) | Cites label |
|---|---|---|---|---|
| 1 | v1 | Baseline | 96.1% | 5.3% |
| 2 | v2 | Baseline | 97.8% | 8.9% |
| 6 | v3.1 | Baseline | **84.0%** | **8.5%** |
| 7 | v3.1 | + Change 1: cite labels | 84.0% | **97.2%** |
| 8 | v3.1 | + Change 2: `query_database` tool | **98.1%** | 86.8% |
| 9 | v3.1 | + Change 3: honesty prompt (reverted) | 96.2% | 84.9% |
| 10 | holdout | Final agent (change 2) | **100%** (35/35) | 91.4% |

On dataset v3.1, measured against the ±9.7-point margin:
- **Change 1 raised citations by 88.7 points** (8.5% → 97.2%), well beyond the margin. Correctness did not move.
- **Change 2 raised correctness by 14.2 points** (84.0% → 98.1%), beyond the margin. It also **lowered citations by 10.4 points** (97.2% → 86.8%), which is also beyond the margin, so it is a real regression and is explained below.
- **Change 3 moved correctness and citations by −1.9 points each,** both within the margin. It did not help, so it was reverted.
- **The final agent scored 35/35 on the holdout,** so the gains hold on questions that were never tuned against.

Runs 1 and 2 are the story of finding a dataset hard enough to measure anything on. Runs 3–5 were the first attempt on dataset v3. They were superseded when a code review found errors in some gold answers; their logs are kept in `evals/results/` for transparency.

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

All three misses involved look-alike products:
- Search returned MET-Rx Creatine 4200 capsules instead of MET-Rx Creatine powder.
- Two flavors of one brand were mixed up in a comparison.
- Search could not find "Only Protein Only Protein Protein Powder Vanilla".

**The problem:** correctness can rise by at most 3.9 points, which is less than the ±11.5-point margin on 76 questions, so no change could ever show a real gain on this dataset. The questions were too easy for the agent, and that had to be fixed before choosing changes to make.

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

### 3. Dataset v3, questions that need the whole database

Some questions a shopper asks need every label at once: which electrolyte has the most sodium, or how many whey proteins use stevia. Top-5 search cannot answer these by design. Dataset v3 added such questions, with gold answers computed over every label in the category.

On the first attempt (runs 3–5), the baseline got 0 of 17 dev whole-database questions right. Some answers admitted the limit ("the search tool... does not provide a total count"), but several were confidently wrong. For example, "there are **5** electrolyte products that list taurine", when the answer is 29, because it counted its 5 search results as if they were the whole database.

**Code review, and dataset v3.1.** A review of the code then found two kinds of errors in the v3 gold answers. The fixes are in commit `6a5faa2`:
- **Missed synonyms:** counts matched ingredient names literally, so 17 whey labels that list stevia only as Reb A or steviol glycosides were missed. Whey/stevia was 71 and is really 90. Four counts changed.
- **A DSLD unit error:** "most magnesium" rewarded a label recorded as "10 g" of magnesium at 2% Daily Value, which is really about 10 mg. The daily-value check now rejects it. With it gone, no winner can be proven, so the question was dropped.

The fixes changed only whole-database questions. Splits were frozen so no dev question could move into the holdout. **Runs 6–9 repeat every step on the corrected dataset.** Each uses the exact `agent.py` from that step's original commit (`8f6d217`, `127c47e`, `1055292`), so the only difference from runs 3–5 is the corrected gold answers.

### 6. Baseline on dataset v3.1: 84.0%

Log: `evals/results/06-baseline-dataset-v3.1-dev.eval`. Dev split, 106 questions.

The agent answered **0 of 16** whole-database questions correctly and 89 of the other 90. It cited its label on 8.5% of answers.

### 7. Change 1, cite the labels used: citations 8.5% → 97.2%

One sentence added to the system prompt: *"End every answer with the labels you used: brand and product name, DSLD ID, and link."*

Log: `evals/results/07-change1-cite-labels-dataset-v3.1-dev.eval`.

Citations rose by 88.7 points, from 9 to 103 of 106 answers. Correctness stayed at 84.0%.

### 8. Change 2, a whole-database tool: correctness 84.0% → 98.1%

Added the `query_database` tool, and told the agent to use it for most, least and how-many questions and never to count or rank from search results.

Log: `evals/results/08-change2-database-tool-dataset-v3.1-dev.eval`.

Whole-database questions went from **1/16 to 16/16**. The two remaining misses are the retrieval failures seen in every run: MET-Rx Creatine directions, and "Only Protein".

**The regression:** citations fell by 10.4 points, beyond the margin. On whole-database questions, citations fell from 16 of 16 to 8 of 16, because count answers give a number of products without naming the labels. Citations on the other questions fell from 87 to 84 of 90.

While testing the tool, the agent was asked which whey protein has the most protein. It requested 1 result and presented one product as *the* answer, though seven products tie at 30 g. The tool now always returns every product tied with the last one listed, and the agent then reports the seven-way tie. This fix was made before the tool was first scored. That question is not in the eval, since tied questions are excluded.

### 9. Change 3, an honesty prompt: no gain, reverted

Change 3 targeted the citation regression and one earlier miss, where the agent could not find an invented product and then offered a similar product's numbers. It added two lines: list the products `query_database` returns, and when a product cannot be found, say so without giving amounts from a different product.

Log: `evals/results/09-change3-honesty-prompt-reverted-dev.eval`.

| Score | Change 2 | Change 3 |
|---|---|---|
| Correct | 98.1% | 96.2% |
| Cites label | 86.8% | 84.9% |

Both moved −1.9 points, which is within the margin. Citations on whole-database answers improved from 8 to 12 of 16, but fell from 84 to 78 of 90 on the rest. The new wording also made the agent overcautious: it said it "could not find" Vital Proteins Collagen Peptides, which is in the database. **The change was reverted,** so the final agent uses the change 2 prompt.

### 10. Holdout: 35/35

Log: `evals/results/10-final-agent-holdout.eval`. The 35 holdout questions, run once on the final agent.

| Score | Result |
|---|---|
| Reference judge (headline) | **100%** (35/35) |
| Rubric: cites label | 91.4% |
| Rubric: leads with answer | 100% |
| Whole-database questions | 6/6 |

The holdout score is consistent with the 98.1% dev score. Its ±16.9-point margin is wide, but no question in it was ever run during development.

## Known limitations

- **The rubric judge never sees the label,** so `separates_label_from_advice` checks how an answer is worded, not whether its facts are true. Factual correctness is covered by the reference judge.
- **Both judges make mistakes:**
  - In run 5, the rubric judge marked an answer that gave a DSLD ID and link as not citing.
  - In run 9, the reference judge marked "12 whey protein products list maltodextrin" wrong, though 12 is the gold answer.
  - Single-question swings like these are why gains are only claimed beyond the margin of error.
- **`query_database` and the gold answers share their label-reading code** (`label_facts.py`). So the whole-database questions measure whether the agent knows to use the tool and reports what it returns, not whether the label parsing is right. The parsing was checked separately, by recounting from the rendered documents and by the code review that found the synonym and unit errors.
- **`numeric_match` passes if the gold amount appears anywhere in the answer,** even if the answer concludes something else. It is a supporting check, not the headline.
- **By default the grader is the same Gemini model as the agent.** A different grader can be set with `--model-role grader=<model>`.
- **Gold answers reflect DSLD's transcription of each label** as of the pull date. The daily-value check catches unit errors only for sodium, potassium, magnesium and protein.
- **Conversations are kept in memory.** The Cloud Run service runs a single instance so that follow-up questions reach the same conversation, and conversations reset when the service restarts.
- **The NIH ODS fact sheets were planned as a second source,** but `ods.od.nih.gov` blocks automated clients, so this agent uses DSLD only.
