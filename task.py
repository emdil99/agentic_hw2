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
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

import agent

# Inspect's Vertex provider reads the project and location from the environment.
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = google.auth.default()[1] or ""
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

NUMERIC_TYPES = {"direct_dose", "container_total"}
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


@task
def supplement_labels(split: str = "dev") -> Task:
    dataset = json_dataset("evals/dataset.jsonl").filter(lambda sample: sample.metadata["split"] == split)
    return Task(
        dataset=dataset,
        solver=supplement_agent(),
        scorer=[retrieved_gold_labels(), numeric_match()],
        model="google/vertex/gemini-3.5-flash-lite",
    )
