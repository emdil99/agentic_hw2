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

### How to read this

Think of the eval as a **practice test**: 106 questions whose right answers we already know (the dev split). Hill climbing means running the agent on the test, changing one thing, running the exact same test again, and only counting an improvement if it is too big to be luck.

**What "too big to be luck" means.** The model is a little random. We ran the identical agent on identical questions twice and got 73 right, then 74, with nothing changed. The assignment's rule of thumb for this noise is ±1/√n. With 106 questions that is **±9.7 points**, so a change has to move a score by more than about 10 points before we call it real. On the 35-question final exam it is ±16.9 points.

### The results at a glance

| Step | What we changed | Correct answers | Answers that cite their label |
|---|---|---|---|
| Starting point | Search tool only, one-sentence prompt | 84.0% | 8.5% |
| Change 1 | "Show your sources" | 84.0% (no change) | **97.2%** (+88.7: real) |
| Change 2 | A tool that sees the whole database | **98.1%** (+14.2: real) | 86.8% (−10.4: a real side effect) |
| Change 3 | Extra honesty instructions | 96.2% (−1.9: luck) | 84.9% (−1.9: luck), **so it was undone** |
| Final exam | The final agent on 35 questions nobody had seen | **100%** (35/35) | 91.4% |

Every step except the final exam uses the same 106 questions (dataset v3.1). The final agent is the one from change 2.

### Before the starting point: finding a test hard enough to learn from

**The first test was too easy (96.1%).** On the first version of the questions, the agent got 73 of 76 right. That sounds great, but it is a problem: the most it could ever improve was 4 points, and the margin of error on 76 questions is ±11.5 points. No change could ever prove itself, like trying to show a tutor helped a student who already gets A's. Log: `evals/results/01-baseline-dataset-v1-dev.eval`.

**We guessed at a weakness, and guessed wrong (97.8%).** All three mistakes involved look-alike products, such as MET-Rx Creatine *powder* versus MET-Rx Creatine *4200 capsules*. So we added 19 questions that name a product without its flavor, the way shoppers do: *"How much protein is in NOW Sports Whey Protein Creamy?"* We expected the agent to get confused. Instead it got all 14 dev questions right, and listed each flavor's amount without being asked to. Wrong guess, so the test was still too easy. This is also the run where the identical questions scored 74 instead of 73. Log: `evals/results/02-baseline-dataset-v2-dev.eval`.

**We found the real weakness: questions about the whole database.** Imagine a shopper asks *"How many collagen products list biotin?"* The agent's only tool was a search that shows it **5 labels at a time**, like asking someone to count every red book in a library when they can only ever see 5 books. Some answers admitted they could not count. Others were confidently wrong: asked how many electrolyte products contain taurine, the agent said **"there are 5"**. It had counted the 5 search results in front of it, and the real answer is 29. We added 23 questions like this (dataset v3, later corrected to 22, as described next), and the agent got every dev one wrong. Now the test had room to show improvement.

**We fixed our own answer key, then redid every step.** A code review found mistakes in a few of *our* correct answers:
- We had counted 71 whey proteins with stevia, but missed labels that call it "Reb A" or "steviol glycosides". The real number is 90.
- One "most magnesium" answer came from an NIH typo: a label recorded as "10 g" of magnesium, which its own 2% Daily Value shows is really about 10 mg. With the typo excluded, no clear winner can be proven, so that question was dropped.

We fixed the answer key (dataset v3.1, commit `6a5faa2`) and re-ran the starting point and every change against it, each using that step's original code (commits `8f6d217`, `127c47e`, `1055292`). That way all the scores compare fairly, like regrading every exam after finding typos in the answer key. The earlier attempt's logs are kept as `evals/results/03-…`, `04-…` and `05-…`.

### Starting point: 84.0% correct, 8.5% citing

The agent got 89 of the 90 regular questions right and **0 of the 16** whole-database questions. It named the label behind its answer only 8.5% of the time. Log: `evals/results/06-baseline-dataset-v3.1-dev.eval`.

### Change 1: "Show your sources." Citations 8.5% → 97.2%

We added one sentence to the agent's instructions: *"End every answer with the labels you used: brand and product name, DSLD ID, and link."*

For example, in a simplified illustration:

- **Before:** "One serving contains 5 grams of creatine."
- **After:** "One serving contains 5 grams of creatine. Source: MTN OPS Creatine Unflavored, DSLD ID 327384, https://dsld.od.nih.gov/label/327384"

Citations jumped by 88.7 points, from 9 answers to 103. That is far more than the 10-point bar, so the gain is **real**. Correctness did not move, which is what we expected: this change makes answers checkable, not smarter. A shopper can now click through to the label and see for themselves. Log: `evals/results/07-change1-cite-labels-dataset-v3.1-dev.eval`.

### Change 2: a catalog, not just a search. Correct 84.0% → 98.1%

We gave the agent a second tool, `query_database`, that looks at **every** label in a category at once, like handing the person in the library the full catalog instead of letting them peek at 5 books. We also told it to use that tool for "which has the most", "which has the least" and "how many" questions, and never to count search results.

- **Before:** "There are 5 electrolyte products that list taurine."
- **After:** "There are 21 collagen products in the database that list biotin as an ingredient. Here are 10 of them: …"

Whole-database questions went from **1 of 16 to 16 of 16**, and overall correctness rose 14.2 points. That is more than the 10-point bar, so the gain is **real**. The two questions it still misses are the same ones it misses in every run, because search cannot find the right label: MET-Rx Creatine (it finds the 4200 capsules), and a product oddly named "Only Protein Only Protein Protein Powder Vanilla".

**A side effect.** On "how many" questions, the agent now often answers "21 products" without naming the labels, so it cites its sources less: citations fell from 97.2% to 86.8%. That 10.4-point drop is also beyond the bar, so it is a **real cost** of this change, and we report it rather than hide it.

**A bug found along the way.** While testing the new tool, we asked *"Which whey protein has the most protein?"* The agent asked for the single top result and named one product, but seven products are tied at 30 g. We fixed the tool so it always shows every product tied for a spot, and now the agent reports the seven-way tie. This fix was made before change 2 was scored, and that question is not in the test, since tied questions are left out.

Log: `evals/results/08-change2-database-tool-dataset-v3.1-dev.eval`.

### Change 3: trying to fix the side effect. No gain, so it was undone

We added instructions to list the products behind every whole-database answer, and, when a product cannot be found, to say so without giving a different product's numbers. (That second line targeted an earlier mistake, where the agent could not find an invented product and then offered a similar product's protein amount.)

| | After change 2 | After change 3 |
|---|---|---|
| Correct answers | 98.1% | 96.2% |
| Citing their label | 86.8% | 84.9% |

Both moved by 1.9 points, well inside the 10-point noise, so **nothing real happened**. Looking closer, citations on the 16 whole-database answers did improve, from 8 to 12, but they got worse on everything else, from 84 to 78 of 90. The new wording also made the agent overcautious: it claimed it "could not find" Vital Proteins Collagen Peptides, which is in the database.

So we **undid change 3**, and the final agent uses the change 2 instructions. Not every change helps, and keeping only the ones that do is the point of hill climbing. Log: `evals/results/09-change3-honesty-prompt-reverted-dev.eval`.

### Final exam: 35 out of 35

From the very start, 35 questions were set aside. Nobody looked at them, and the agent was never run on them while we made changes. At the end, we ran the final agent on them **once**:

| | Result |
|---|---|
| Correct answers | **35 of 35 (100%)** |
| Whole-database questions | 6 of 6 |
| Citing their label | 91.4% |
| Answering in the first sentence | 100% |

With only 35 questions the margin of error is wide (±16.9 points), but the score agrees with the 98.1% on the practice questions. That is evidence the improvements are real, not just tuned to the questions we practiced on. Log: `evals/results/10-final-agent-holdout.eval`.

### How this meets the requirement

- **Baseline score:** 84.0% correct and 8.5% citing, on the final 106 dev questions.
- **Two or more changes, each with its new score:**
  - change 1 (citations): 97.2% citing
  - change 2 (the whole-database tool): 98.1% correct
  - change 3 (tried and undone): 96.2% correct
- **Which gains beat the ±1/√n margin of ±9.7 points:** change 1's citation gain (+88.7) and change 2's correctness gain (+14.2) do. Change 2's citation drop (−10.4) is also beyond the margin, so it is reported as a real cost. Change 3's moves (−1.9) do not.

## Known limitations

- **The rubric judge never sees the label,** so `separates_label_from_advice` checks how an answer is worded, not whether its facts are true. Factual correctness is covered by the reference judge.
- **Both judges make mistakes:**
  - On the first, superseded attempt at dataset v3, the rubric judge marked an answer that gave a DSLD ID and link as not citing.
  - In change 3's run, the reference judge marked "12 whey protein products list maltodextrin" wrong, though 12 is the gold answer.
  - Single-question swings like these are why gains are only claimed beyond the margin of error.
- **`query_database` and the gold answers share their label-reading code** (`label_facts.py`). So the whole-database questions measure whether the agent knows to use the tool and reports what it returns, not whether the label parsing is right. The parsing was checked separately, by recounting from the rendered documents and by the code review that found the synonym and unit errors.
- **`numeric_match` passes if the gold amount appears anywhere in the answer,** even if the answer concludes something else. It is a supporting check, not the headline.
- **By default the grader is the same Gemini model as the agent.** A different grader can be set with `--model-role grader=<model>`.
- **Gold answers reflect DSLD's transcription of each label** as of the pull date. The daily-value check catches unit errors only for sodium, potassium, magnesium and protein.
- **Conversations are kept in memory.** The Cloud Run service runs a single instance so that follow-up questions reach the same conversation, and conversations reset when the service restarts.
- **The NIH ODS fact sheets were planned as a second source,** but `ods.od.nih.gov` blocks automated clients, so this agent uses DSLD only.
