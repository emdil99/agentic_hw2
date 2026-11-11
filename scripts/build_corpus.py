"""Pull supplement labels from the NIH DSLD API and render them as RAG documents.

    uv run scripts/build_corpus.py
    uv run scripts/build_corpus.py --limit 20

Raw labels are cached in data/raw/dsld/, so a re-run only downloads what is missing.
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://api.ods.od.nih.gov/dsld/v9"
LABEL_PAGE_URL = "https://dsld.od.nih.gov/label"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "dsld"
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

# DSLD search does not stem: "electrolyte" and "electrolytes" return different products.
CATEGORIES = {
    "creatine": ["creatine"],
    "whey protein": ["whey protein"],
    "electrolytes": ["electrolyte", "electrolytes"],
    "collagen": ["collagen"],
}
DEFAULT_LIMIT = 300
PAGE_SIZE = 50
REQUEST_DELAY_SECONDS = 0.2
TIMEOUT_SECONDS = 30
MAX_RATE_LIMIT_WAITS = 4
DEFAULT_RETRY_AFTER_SECONDS = 60

NOT_DISCLOSED = "amount not disclosed on label"
EXCLUDED_STATEMENT_TYPES = {
    "FDA Disclaimer Statement",
    "Brand IP Statement(s) re: (c), (TM), (SM)",
    "General: Product/Version Code",
}


def polite_get(session: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    """GET a URL at a steady pace, waiting as long as the API asks whenever it rate limits us."""
    for _ in range(MAX_RATE_LIMIT_WAITS):
        time.sleep(REQUEST_DELAY_SECONDS)
        response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait_seconds = int(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_SECONDS)) + 1
        print(f"Rate limited; waiting {wait_seconds} seconds as the API requested", flush=True)
        time.sleep(wait_seconds)
    raise RuntimeError(f"still rate limited after {MAX_RATE_LIMIT_WAITS} waits: {url}")


def collect_ids(session: requests.Session, term: str, limit: int) -> tuple[list[str], int]:
    """Return up to `limit` on-market label IDs for a search term, and the term's total hits."""
    ids: list[str] = []
    total = 0
    offset = 0
    while len(ids) < limit:
        response = polite_get(
            session,
            f"{BASE_URL}/search-filter",
            params={"q": term, "size": PAGE_SIZE, "from": offset},
        )
        data = response.json()
        total = data["stats"]["count"]
        hits = data["hits"]
        for hit in hits:
            if not hit["_source"].get("offMarket") and hit["_id"] not in ids:
                ids.append(hit["_id"])
        if len(hits) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return ids[:limit], total


def fetch_label(session: requests.Session, label_id: str) -> None:
    """Download one label to the raw cache, unless it is already there."""
    path = RAW_DIR / f"{label_id}.json"
    if path.exists():
        return
    response = polite_get(session, f"{BASE_URL}/label/{label_id}")
    if "ingredientRows" not in response.json():
        raise ValueError(f"label {label_id}: response is not a label")
    # A crash during a plain write would leave a truncated file that later runs treat as cached.
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(response.text, encoding="utf-8")
    temporary_path.replace(path)


def clean(text: str | None) -> str:
    """Collapse the label's line breaks and repeated spaces."""
    return " ".join((text or "").split())


def format_number(value: float | int) -> str:
    return f"{value:g}"


def format_daily_value(quantity: dict) -> str:
    groups = [g for g in quantity.get("dailyValueTargetGroup") or [] if g.get("percent") is not None]
    parts = []
    for group in groups:
        operator = group.get("operator") or ""
        operator = "" if operator == "=" else operator
        part = f"{operator}{format_number(group['percent'])}% DV"
        if len(groups) > 1:
            part += f" for {group['name']}"
        parts.append(part)
    return ", ".join(parts)


def format_quantity(quantity: dict) -> str:
    operator = "" if quantity.get("operator") in (None, "=") else f"{quantity['operator']} "
    unit = quantity["unit"].strip("{}")
    amount = f"{operator}{format_number(quantity['quantity'])} {unit}"
    serving_quantity = quantity.get("servingSizeQuantity")
    basis = "serving" if serving_quantity is None else f"{format_number(serving_quantity)} {quantity['servingSizeUnit']}"
    daily_value = format_daily_value(quantity)
    return f"{amount} per {basis}" + (f" ({daily_value})" if daily_value else "")


def format_amounts(quantities: list[dict]) -> str:
    # DSLD records an undisclosed amount as quantity 0 with unit "NP", which must not read as zero.
    disclosed = [q for q in quantities if q.get("unit") != "NP"]
    if not disclosed:
        return NOT_DISCLOSED
    return "; ".join(format_quantity(q) for q in disclosed)


def render_ingredient(row: dict, depth: int = 0) -> list[str]:
    """Render an ingredient row and its nested rows as indented bullet lines."""
    name = clean(row.get("name"))
    group = clean(row.get("ingredientGroup"))
    if group and group.casefold() != name.casefold() and row.get("category") != "blend":
        name = f"{name} ({group})"

    notes = clean(row.get("notes"))
    forms = [clean(form.get("name")) for form in row.get("forms") or []]
    missing_forms = [form for form in forms if form and form.casefold() not in notes.casefold()]

    amounts = format_amounts(row.get("quantity") or [])
    if row.get("category") == "blend":
        amounts = "blend total not disclosed on label" if amounts == NOT_DISCLOSED else f"blend total {amounts}"
    line = f"{'  ' * depth}- {name}: {amounts}"
    if missing_forms:
        line += f" | form: {', '.join(missing_forms)}"
    if notes:
        line += f" | note: {notes}"

    lines = [line]
    for nested in row.get("nestedRows") or []:
        lines.extend(render_ingredient(nested, depth + 1))
    return lines


def format_serving(serving: dict) -> str:
    low, high = serving.get("minQuantity"), serving.get("maxQuantity")
    amount = format_number(low) if low == high or high is None else f"{format_number(low)}-{format_number(high)}"
    text = f"{amount} {serving.get('unit')}"
    per_day_low, per_day_high = serving.get("minDailyServings"), serving.get("maxDailyServings")
    if per_day_low is not None:
        per_day = (
            format_number(per_day_low)
            if per_day_low == per_day_high or per_day_high is None
            else f"{format_number(per_day_low)}-{format_number(per_day_high)}"
        )
        text += f", {per_day} time(s) per day"
    return text


def label_to_text(label: dict, categories: list[str]) -> str:
    """Render a raw DSLD label as one readable document."""
    lines = [f"{clean(label.get('brandName'))} - {clean(label.get('fullName'))}"]
    lines.append(f"Categories: {', '.join(categories)}")

    physical_state = (label.get("physicalState") or {}).get("langualCodeDescription")
    if physical_state:
        lines.append(f"Form: {physical_state}")
    net_contents = [item["display"] for item in label.get("netContents") or [] if item.get("display")]
    if net_contents:
        lines.append(f"Net contents: {'; '.join(net_contents)}")
    servings = [format_serving(serving) for serving in label.get("servingSizes") or []]
    if servings:
        lines.append(f"Serving size: {' or '.join(servings)}")
    if label.get("servingsPerContainer"):
        lines.append(f"Servings per container: {label['servingsPerContainer']}")

    lines.append("Ingredients per serving:")
    for row in label.get("ingredientRows") or []:
        lines.extend(render_ingredient(row))

    other = label.get("otheringredients") or {}
    other_names = [clean(item.get("name")) for item in other.get("ingredients") or []]
    other_text = ", ".join(name for name in other_names if name) or clean(other.get("text"))
    if other_text:
        lines.append(f"Other ingredients: {other_text}")

    claims = [c["langualCodeDescription"] for c in label.get("claims") or [] if c.get("langualCodeDescription")]
    if claims:
        lines.append(f"Claim types: {', '.join(claims)}")

    statements = [
        f"- {statement['type']}: {clean(statement.get('notes'))}"
        for statement in label.get("statements") or []
        if statement.get("type") not in EXCLUDED_STATEMENT_TYPES and clean(statement.get("notes"))
    ]
    if statements:
        lines.append("Label statements:")
        lines.extend(statements)

    lines.append(f"DSLD ID: {label['id']}")
    return "\n".join(lines)


def build_corpus(label_categories: dict[str, list[str]]) -> tuple[int, int]:
    """Write corpus.jsonl and return (documents written, duplicate labels dropped)."""
    newest: dict[tuple[str, str], tuple[dict, list[str]]] = {}
    label_count = 0
    for label_id, categories in label_categories.items():
        label = json.loads((RAW_DIR / f"{label_id}.json").read_text(encoding="utf-8"))
        label_count += 1
        # DSLD files a new label for each package size or image of the same product.
        key = (clean(label.get("brandName")).casefold(), clean(label.get("fullName")).casefold())
        if key not in newest:
            newest[key] = (label, list(categories))
            continue
        kept, kept_categories = newest[key]
        merged = sorted(set(kept_categories) | set(categories))
        winner = label if label.get("entryDate", "") > kept.get("entryDate", "") else kept
        newest[key] = (winner, merged)

    with CORPUS_PATH.open("w", encoding="utf-8", newline="\n") as corpus:
        for label, categories in newest.values():
            record = {
                "id": str(label["id"]),
                "text": label_to_text(label, categories),
                "metadata": {
                    "brand": clean(label.get("brandName")),
                    "product": clean(label.get("fullName")),
                    "categories": categories,
                    "entry_date": label.get("entryDate"),
                    "url": f"{LABEL_PAGE_URL}/{label['id']}",
                },
            }
            corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(newest), label_count - len(newest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="maximum labels per category")
    limit = parser.parse_args().limit

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    label_categories: dict[str, list[str]] = {}
    manifest_categories = {}
    for category, terms in CATEGORIES.items():
        term_totals = {}
        category_ids: list[str] = []
        for term in terms:
            ids, total = collect_ids(session, term, limit // len(terms))
            term_totals[term] = total
            category_ids.extend(label_id for label_id in ids if label_id not in category_ids)
        for label_id in category_ids:
            label_categories.setdefault(label_id, []).append(category)
        manifest_categories[category] = {"search_terms": term_totals, "labels_selected": len(category_ids)}
        print(f"{category}: selected {len(category_ids)} labels, search totals {term_totals}")

    print(f"Fetching {len(label_categories)} unique labels")
    failed: set[str] = set()
    for position, label_id in enumerate(label_categories, start=1):
        try:
            fetch_label(session, label_id)
        except (requests.RequestException, ValueError) as error:
            failed.add(label_id)
            print(f"FAILED {label_id}: {error}", flush=True)
        if position % 50 == 0:
            print(f"  {position}/{len(label_categories)}", flush=True)

    if failed:
        raise SystemExit(f"{len(failed)} labels failed; corpus not written. Re-run to fetch only the missing labels.")

    documents, duplicates = build_corpus(label_categories)
    manifest = {
        "pulled_on": date.today().isoformat(),
        "source": BASE_URL,
        "limit_per_category": limit,
        "excludes": "labels marked offMarket",
        "categories": manifest_categories,
        "unique_labels_selected": len(label_categories),
        "labels_in_multiple_categories": sum(len(c) > 1 for c in label_categories.values()),
        "duplicate_labels_dropped": duplicates,
        "corpus_documents": documents,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {documents} documents to {CORPUS_PATH} ({duplicates} duplicates dropped)")


if __name__ == "__main__":
    main()
