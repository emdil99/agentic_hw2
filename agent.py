"""A RAG agent that answers questions about dietary supplement labels."""
import json
import re
import sys
from pathlib import Path

import litellm
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
SEARCH_RESULT_COUNT = 5

MODEL = "vertex_ai/gemini-3.5-flash-lite"
MAX_TOOL_ROUNDS = 6
SYSTEM_PROMPT = "You answer questions about dietary supplement labels. Use the search_labels tool to look up products."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_labels",
            "description": "Search the database of dietary supplement labels. Returns the best matching label documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words to search for, such as a brand and product name.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus():
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


DOCUMENTS = load_corpus()
INDEX = BM25Okapi([tokenize(document["text"]) for document in DOCUMENTS])


def search_labels(query, limit=SEARCH_RESULT_COUNT):
    """Return the supplement label documents that best match a search query."""
    scores = INDEX.get_scores(tokenize(query))
    ranked = sorted(range(len(DOCUMENTS)), key=lambda i: scores[i], reverse=True)[:limit]
    return [
        {
            "id": DOCUMENTS[i]["id"],
            "brand": DOCUMENTS[i]["metadata"]["brand"],
            "product": DOCUMENTS[i]["metadata"]["product"],
            "url": DOCUMENTS[i]["metadata"]["url"],
            "text": DOCUMENTS[i]["text"],
        }
        for i in ranked
        if scores[i] > 0
    ]


def run_tool(name, args):
    if name != "search_labels":
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(search_labels(args["query"]), ensure_ascii=False)


def run_agent(messages):
    """Call the model until it answers without requesting a tool. Returns (answer, tool_calls)."""
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model=MODEL,
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
        ).choices[0].message

        # A plain dict: the raw object has provider-specific fields that break re-serialization next round.
        messages.append(reply.model_dump())

        if not reply.tool_calls:
            return reply.content or "(The model returned an empty answer.)", tool_calls

        # Every tool call must get a result message, or the next model call is rejected.
        for call in reply.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                result = run_tool(call.function.name, args)
            except Exception as error:
                args = call.function.arguments
                result = json.dumps({"error": f"{type(error).__name__}: {error}"})
            tool_calls.append({"name": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


def ask(question):
    """Answer one question in a fresh conversation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return run_agent(messages)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    answer, calls = ask(question)
    for call in calls:
        print(f"TOOL: {call['name']}({call['args']})")
    print(f"ANSWER: {answer}")
