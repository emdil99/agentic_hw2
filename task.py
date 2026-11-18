"""Inspect eval for the supplement label agent.

    uv run --no-sync inspect eval task.py
    uv run --no-sync inspect eval task.py -T split=holdout
    uv run --no-sync inspect view
"""
import asyncio
import json
import math
import os
import re

import google.auth
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import GenerateConfig, ModelOutput, ResponseSchema, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, model_graded_qa, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field, ValidationError

import agent

# Inspect's Vertex provider reads the project and location from the environment.
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = google.auth.default()[1] or ""
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

NUMERIC_TYPES = {"direct_dose", "container_total", "partial_name_same"}
NUMBER_WITH_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(mcg|micrograms?|mg|milligrams?|grams?|g)\b", re.IGNORECASE)
NUMERIC_TOLERANCE = 0.005


@solver
def supplement_agent():
    """Run the real agent on the question and record its answer and tool calls."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        answer, tool_calls = await asyncio.to_thread(agent.ask, state.input_text)
        state.output = ModelOutput.from_content(model=agent.MODEL, content=answer)
        state.messages.append(state.output.message)
        state.store.set("tool_calls", tool_calls)
        return state

    return solve


def retrieved_ids(state: TaskState) -> set[str]:
    """Return every label ID that any search showed the agent."""
    ids = set()
    for call in state.store.get("tool_calls", []):
        results = json.loads(call["result"])
        if isinstance(results, list):
            ids.update(result["id"] for result in results)
    return ids


def amounts_in_mg(text: str) -> list[float]:
    """Return every amount with a weight unit in the text, converted to milligrams."""
    amounts = []
    for number, unit in NUMBER_WITH_UNIT.findall(text):
        value = float(number.replace(",", ""))
        unit = unit.lower()
        if unit.startswith(("mcg", "micro")):
            amounts.append(value / 1000)
        elif unit.startswith(("mg", "milli")):
            amounts.append(value)
        else:
            amounts.append(value * 1000)
    return amounts


@scorer(metrics=[accuracy(), stderr()])
def retrieved_gold_labels():
    """Deterministic: did search show the agent every label needed to answer?"""

    async def score(state: TaskState, target: Target) -> Score:
        gold_ids = state.metadata["gold_label_ids"]
        if not gold_ids:
            return Score.unscored(reason="question has no gold label")
        missing = set(gold_ids) - retrieved_ids(state)
        return Score(
            value=INCORRECT if missing else CORRECT,
            answer=state.output.completion,
            explanation=f"missing labels: {sorted(missing)}" if missing else "all gold labels retrieved",
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def numeric_match():
    """Deterministic: does the answer state the gold amount, in any weight unit?"""

    async def score(state: TaskState, target: Target) -> Score:
        if state.metadata["type"] not in NUMERIC_TYPES:
            return Score.unscored(reason="not a numeric question")
        expected = state.metadata["value_mg"]
        found = amounts_in_mg(state.output.completion)
        matched = any(math.isclose(amount, expected, rel_tol=NUMERIC_TOLERANCE) for amount in found)
        return Score(
            value=CORRECT if matched else INCORRECT,
            answer=state.output.completion,
            explanation=f"expected {expected} mg, found {found} mg",
        )

    return score


# ---------- Model-graded: golden reference ----------

# Inspect fills in {question}, {answer}, {criterion} (the gold target) and {instructions}.
REFERENCE_TEMPLATE = """You are grading an answer to a question about a dietary supplement label by comparing it to the gold answer, which was taken from the label itself.

[BEGIN DATA]
***
[Question]: {question}
***
[Gold answer]: {criterion}
***
[Answer]: {answer}
***
[END DATA]

How to judge, depending on the gold answer:
- An amount: correct if the answer states the same amount in any unit (1.5 g is 1500 mg). Extra detail is fine.
- "The label does not disclose this amount": correct only if the answer says the amount is not given and does not state or estimate one.
- "This product is not in the database": correct only if the answer says the product could not be found and does not give an amount.
- "No, the label does not list ...": correct only if the answer says the ingredient is not listed.
- An amount followed by "the same in all N versions": correct if the answer states that amount.
- "It depends on the version", followed by each version's amount: correct only if the answer says the amount differs between versions, or asks which version is meant, and does not give one amount as the answer for the whole product. Listing each version's amount is fine.
- A product name (from a which-has-more question): correct if the answer picks that same product.
- A product name followed by "with ... per serving" (from a which-product-in-the-database question): correct only if the answer names that product as the one with the most or least. Naming another product is incorrect, even if the answer hedges.
- A number of products, such as "17 products.": correct only if the answer gives exactly that number.
- Label text such as warnings or directions: correct if the answer conveys the main points. It does not need to quote them.

{instructions}"""

model_graded_reference = model_graded_qa(template=REFERENCE_TEMPLATE)


# ---------- Model-graded: rubric ----------

RUBRIC_PROMPT = """You are grading an answer to a shopper's question about a dietary supplement label against a rubric. Judge each criterion on its own, and do not judge whether the answer is correct.

Question: {question}

Answer: {answer}

Rubric:
- leads_with_answer: The first sentence answers the question directly: an amount, a yes or no, a product name, or a clear statement that the information is not available.
- cites_label: Gives the DSLD ID or the link to the label the answer came from. If no product was found, passes when it says what it searched for.
- separates_label_from_advice: Anything a label would not say, such as dosing advice, health effects or general nutrition facts, is clearly marked as not coming from the label. Passes if the answer contains no such content."""


class Criterion(BaseModel):
    reason: str
    passed: bool = Field(alias="pass")


class RubricGrades(BaseModel):
    leads_with_answer: Criterion
    cites_label: Criterion
    separates_label_from_advice: Criterion


async def judge(prompt: str, schema: type[BaseModel]) -> BaseModel | None:
    """Ask the grader model for JSON matching the schema. None if it does not parse."""
    # The "grader" role (--model-role grader=...), or the task's model when unset.
    result = await get_model(role="grader").generate(
        prompt,
        config=GenerateConfig(response_schema=ResponseSchema(name=schema.__name__, json_schema=json_schema(schema))),
    )
    try:
        return schema.model_validate_json(result.completion)
    except ValidationError:
        return None


@scorer(metrics={"*": [accuracy(), stderr()]})
def model_graded_rubric():
    async def score(state: TaskState, target: Target) -> Score:
        prompt = RUBRIC_PROMPT.format(question=state.input_text, answer=state.output.completion)
        grades = await judge(prompt, RubricGrades)
        if grades is None:
            return Score.unscored(reason="grader_failed")
        return Score(
            value={name: CORRECT if criterion.passed else INCORRECT for name, criterion in grades},
            answer=state.output.completion,
            explanation=grades.model_dump_json(by_alias=True, indent=2),
        )

    return score


@task
def supplement_labels(split: str = "dev") -> Task:
    dataset = json_dataset("evals/dataset.jsonl").filter(lambda sample: sample.metadata["split"] == split)
    return Task(
        dataset=dataset,
        solver=supplement_agent(),
        # The first scorer's first metric is the headline number: the reference judge's accuracy.
        scorer=[model_graded_reference, model_graded_rubric(), retrieved_gold_labels(), numeric_match()],
        model="google/vertex/gemini-3.5-flash-lite",
    )
