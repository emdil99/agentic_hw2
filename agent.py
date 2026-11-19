"""A RAG agent that answers questions about dietary supplement labels."""
import json
import re
import sys
from pathlib import Path

import litellm
from rank_bm25 import BM25Okapi

from label_facts import amount_mg, format_mg, has_ingredient, lists_ingredient

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
RAW_DIR = ROOT / "data" / "raw" / "dsld"
SEARCH_RESULT_COUNT = 5
CATEGORIES = ["creatine", "whey protein", "electrolytes", "collagen"]
QUERY_RESULT_LIMIT = 10

MODEL = "vertex_ai/gemini-3.5-flash-lite"
MAX_TOOL_ROUNDS = 6
SYSTEM_PROMPT = (
    "You answer questions about dietary supplement labels. Use the search_labels tool to look up products.\n"
    "For questions about the whole database, such as which product has the most or least of an ingredient "
    "or how many products list an ingredient, use query_database. Search shows only 5 labels, so never "
    "count or rank products from search results.\n"
    "End every answer with the labels you used: brand and product name, DSLD ID, and link."
)

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
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Look across every label in one product category at once. Ranks products by the amount of an "
                "ingredient per serving, counts the products whose ingredient list mentions something, or both."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "ingredient": {
                        "type": "string",
                        "description": "Rank by this ingredient's amount per serving, such as protein, creatine, sodium, potassium, magnesium or collagen.",
                    },
                    "sort": {"type": "string", "enum": ["most", "least"], "description": "Ranking order. Defaults to most."},
                    "contains": {
                        "type": "string",
                        "description": "Only include products whose ingredient list mentions this, such as stevia or caffeine.",
                    },
                    "limit": {"type": "integer", "description": f"How many products to list. Defaults to {QUERY_RESULT_LIMIT}."},
                },
                "required": ["category"],
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


RAW_LABELS = {
    document["id"]: json.loads((RAW_DIR / f"{document['id']}.json").read_text(encoding="utf-8"))
    for document in DOCUMENTS
}


def describe(document, amount=None):
    product = {
        "id": document["id"],
        "product": f"{document['metadata']['brand']} {document['metadata']['product']}",
        "url": document["metadata"]["url"],
    }
    if amount is not None:
        product["amount_per_serving"] = format_mg(amount)
    return product


def query_database(category, ingredient=None, sort="most", contains=None, limit=QUERY_RESULT_LIMIT):
    """Rank or count every label in a category, which search alone cannot do."""
    # The model fills these arguments, so normalise them instead of trusting their types or casing.
    category = (category or "").strip().lower()
    if category not in CATEGORIES:
        return {"error": f"Unknown category {category!r}. Choose one of {CATEGORIES}."}
    sort = (sort or "most").strip().lower()
    if sort not in ("most", "least"):
        return {"error": f"Unknown sort {sort!r}. Use 'most' or 'least'."}
    try:
        limit = QUERY_RESULT_LIMIT if limit is None else max(1, int(limit))
    except (TypeError, ValueError):
        return {"error": f"limit must be a whole number, not {limit!r}."}
    ingredient = (ingredient or "").strip() or None
    contains = (contains or "").strip() or None

    documents = [d for d in DOCUMENTS if category in d["metadata"]["categories"]]
    if contains:
        documents = [d for d in documents if lists_ingredient(RAW_LABELS[d["id"]], contains)]
    result = {"category": category, "contains": contains, "matching_products": len(documents)}

    if not ingredient:
        result["products"] = [describe(d) for d in documents[:limit]]
        return result

    amounts = [(amount_mg(RAW_LABELS[d["id"]], ingredient), d) for d in documents]
    ranked = sorted([pair for pair in amounts if pair[0] is not None], key=lambda pair: pair[0], reverse=sort != "least")
    result["ingredient"] = ingredient
    result["sort"] = sort
    result["products_ranked"] = len(ranked)
    result["products_listing_it_without_one_clear_amount"] = sum(
        1 for amount, d in amounts if amount is None and has_ingredient(RAW_LABELS[d["id"]], ingredient)
    )
    shown = ranked[:limit]
    # Never let the limit hide a tie: keep listing products with the same amount as the last one shown.
    while shown and len(shown) < len(ranked) and ranked[len(shown)][0] == shown[-1][0]:
        shown.append(ranked[len(shown)])
    result["products"] = [describe(d, amount) for amount, d in shown]
    return result


def run_tool(name, args):
    if name == "search_labels":
        return json.dumps(search_labels(args["query"]), ensure_ascii=False)
    if name == "query_database":
        return json.dumps(query_database(**args), ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


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

    answer = "Sorry, I hit my tool-call limit before finishing."
    # Close the turn with an assistant message, or the next turn would follow raw tool results.
    messages.append({"role": "assistant", "content": answer})
    return answer, tool_calls


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
