"""Build the gold eval set from the structured DSLD labels.

    uv run scripts/build_dataset.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
RAW_DIR = ROOT / "data" / "raw" / "dsld"
OUTPUT_PATH = ROOT / "evals" / "dataset.jsonl"

SEED = 4570
LABEL_PAGE_URL = "https://dsld.od.nih.gov/label"

TARGET_GROUPS = {
    "creatine": ["Creatine"],
    "whey protein": ["Protein", "Protein (unspecified)"],
    "electrolytes": ["Magnesium", "Potassium", "Sodium"],
    "collagen": ["Collagen", "Collagen Peptides"],
}

# Rarest types come first so they get first pick of the labels that support them.
QUOTAS = {
    "undisclosed": 4,
    "comparison": 3,
    "container_total": 3,
    "label_statement": 4,
    "not_on_label": 2,
    "unknown_product": 1,
    "direct_dose": 8,
}

ABSENT_INGREDIENTS = ["sucralose", "aspartame", "stevia", "caffeine", "titanium dioxide"]
INVENTED_PRODUCTS = {
    "creatine": ("Northbridge Labs CreaPure Max Tropical", "creatine"),
    "whey protein": ("Halcyon Foundry Whey Isolate Maple", "protein"),
    "electrolytes": ("Saltmarsh Athletics Hydra Tabs Yuzu", "sodium"),
    "collagen": ("Lumen Harbor Marine Collagen Fig", "collagen"),
}
STATEMENT_QUESTIONS = {
    "Precautions re: Pregnant or Nursing or Prescription Medications": "What does the label of {name} say about pregnancy, nursing, or medications?",
    "Precautions re: Allergies": "What allergen or allergy warnings does the label of {name} give?",
    "Suggested/Recommended/Usage/Directions": "What directions for use does the label of {name} give?",
}
MG_PER_UNIT = {"mg": 1, "g": 1000, "Gram(s)": 1000, "mcg": 0.001}


def load_labels():
    labels = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        raw_path = RAW_DIR / f"{record['id']}.json"
        label = json.loads(raw_path.read_text(encoding="utf-8"))
        label["category"] = record["metadata"]["categories"][0]
        label["display_name"] = f"{record['metadata']['brand']} {record['metadata']['product']}"
        labels.append(label)
    return labels


def all_rows(rows):
    """Return every ingredient row, including rows nested inside blends."""
    found = []
    for row in rows:
        found.append(row)
        found.extend(all_rows(row.get("nestedRows") or []))
    return found


def is_undisclosed(row):
    quantities = row.get("quantity") or []
    return len(quantities) > 0 and all(q["unit"] == "NP" for q in quantities)


def ingredient_word(group):
    return group.replace(" (unspecified)", "").lower()


def format_mg(mg):
    if mg >= 1000:
        return f"{mg / 1000:g} g"
    return f"{mg:g} mg"


def single_serving_quantity(label):
    """Return the label's serving quantity, or None unless it has exactly one fixed serving size."""
    servings = label.get("servingSizes") or []
    if len(servings) != 1 or servings[0]["minQuantity"] != servings[0]["maxQuantity"]:
        return None
    return servings[0]["minQuantity"]


def find_dose(label, group):
    """Return milligrams per serving for an ingredient group, or None if there is no single clear dose."""
    matches = [r for r in all_rows(label["ingredientRows"]) if r.get("ingredientGroup") == group]
    if len(matches) != 1:
        return None
    quantities = [q for q in matches[0].get("quantity") or [] if q["unit"] != "NP"]
    if len(quantities) != 1:
        return None
    quantity = quantities[0]
    if quantity["operator"] != "=" or quantity["unit"] not in MG_PER_UNIT or quantity["quantity"] <= 0:
        return None
    if quantity["servingSizeQuantity"] != single_serving_quantity(label):
        return None
    return quantity["quantity"] * MG_PER_UNIT[quantity["unit"]]


def dose_options(label, rng):
    """Return (group, milligrams) for each target ingredient with a clear dose, in random order."""
    groups = TARGET_GROUPS[label["category"]][:]
    rng.shuffle(groups)
    return [(group, find_dose(label, group)) for group in groups if find_dose(label, group) is not None]


def make_sample(label, question_type, question, target, **extra):
    label_id = str(label["id"])
    return {
        "id": f"{label['category'].replace(' ', '-')}-{question_type}-{label_id}",
        "input": question,
        "target": target,
        "metadata": {
            "type": question_type,
            "category": label["category"],
            "gold_label_ids": [label_id],
            "origin": "dsld_label",
            "source": f"{LABEL_PAGE_URL}/{label_id}",
            **extra,
        },
    }


def make_direct_dose(label, rng):
    options = dose_options(label, rng)
    if not options:
        return None
    group, mg = options[0]
    question = f"How much {ingredient_word(group)} is in one serving of {label['display_name']}?"
    return make_sample(label, "direct_dose", question, format_mg(mg), value_mg=mg)


def make_container_total(label, rng):
    options = dose_options(label, rng)
    try:
        servings = float(label.get("servingsPerContainer"))
    except (TypeError, ValueError):
        return None
    if not options or servings <= 0:
        return None
    group, mg = options[0]
    total = mg * servings
    question = f"How much {ingredient_word(group)} is in one whole container of {label['display_name']}?"
    return make_sample(label, "container_total", question, format_mg(total), value_mg=total)


def make_undisclosed(label, rng):
    rows = all_rows(label["ingredientRows"])
    candidates = [r for r in rows if is_undisclosed(r) and r.get("category") != "blend" and not r.get("nestedRows")]
    candidates = [r for r in candidates if sum(o.get("ingredientGroup") == r.get("ingredientGroup") for o in rows) == 1]
    if not candidates:
        return None
    row = rng.choice(candidates)
    question = f"How much {row['name']} is in one serving of {label['display_name']}?"
    return make_sample(label, "undisclosed", question, "The label does not disclose this amount.", ingredient=row["name"])


def make_not_on_label(label, rng):
    label_text = json.dumps(label).casefold()
    absent = [term for term in ABSENT_INGREDIENTS if term not in label_text]
    if not absent:
        return None
    term = rng.choice(absent)
    question = f"Does the label of {label['display_name']} list {term}?"
    return make_sample(label, "not_on_label", question, f"No, the label does not list {term}.", ingredient=term)


def make_label_statement(label, rng):
    available = [t for t in STATEMENT_QUESTIONS if any(s["type"] == t and s.get("notes") for s in label.get("statements") or [])]
    if not available:
        return None
    statement_type = rng.choice(available)
    notes = [" ".join(s["notes"].split()) for s in label["statements"] if s["type"] == statement_type and s.get("notes")]
    question = STATEMENT_QUESTIONS[statement_type].format(name=label["display_name"])
    return make_sample(label, "label_statement", question, " ".join(notes), statement_type=statement_type)


def make_comparison(label_a, label_b, rng):
    for group, mg_a in dose_options(label_a, rng):
        mg_b = find_dose(label_b, group)
        if mg_b is None or mg_a == mg_b:
            continue
        winner = label_a if mg_a > mg_b else label_b
        question = (
            f"Which has more {ingredient_word(group)} per serving, "
            f"{label_a['display_name']} or {label_b['display_name']}?"
        )
        sample = make_sample(label_a, "comparison", question, winner["display_name"], value_mg={str(label_a["id"]): mg_a, str(label_b["id"]): mg_b})
        sample["id"] = f"{label_a['category'].replace(' ', '-')}-comparison-{label_a['id']}-{label_b['id']}"
        sample["metadata"]["gold_label_ids"] = [str(label_a["id"]), str(label_b["id"])]
        return sample
    return None


def make_unknown_product(category):
    product, ingredient = INVENTED_PRODUCTS[category]
    return {
        "id": f"{category.replace(' ', '-')}-unknown_product",
        "input": f"How much {ingredient} is in one serving of {product}?",
        "target": "This product is not in the database.",
        "metadata": {
            "type": "unknown_product",
            "category": category,
            "gold_label_ids": [],
            "origin": "invented_product_name",
            "source": "scripts/build_dataset.py",
        },
    }


def build_category(category, labels, rng):
    labels = labels[:]
    rng.shuffle(labels)
    used_ids = set()
    samples = []
    makers = {
        "direct_dose": make_direct_dose,
        "container_total": make_container_total,
        "undisclosed": make_undisclosed,
        "not_on_label": make_not_on_label,
        "label_statement": make_label_statement,
    }

    for question_type, quota in QUOTAS.items():
        made = 0
        if question_type == "unknown_product":
            samples.append(make_unknown_product(category))
            made = 1
        elif question_type == "comparison":
            for label_a in labels:
                for label_b in labels:
                    if made >= quota:
                        break
                    if label_a["id"] >= label_b["id"] or {label_a["id"], label_b["id"]} & used_ids:
                        continue
                    sample = make_comparison(label_a, label_b, rng)
                    if sample:
                        samples.append(sample)
                        used_ids.update([label_a["id"], label_b["id"]])
                        made += 1
        else:
            for label in labels:
                if made >= quota:
                    break
                if label["id"] in used_ids:
                    continue
                sample = makers[question_type](label, rng)
                if sample:
                    samples.append(sample)
                    used_ids.add(label["id"])
                    made += 1
        if made < quota:
            print(f"SHORT: {category} / {question_type}: made {made} of {quota}")
    return samples


def assign_splits(samples, rng):
    """Put every fifth question of each type in the holdout, so the split is even across types."""
    by_type = defaultdict(list)
    for sample in samples:
        by_type[sample["metadata"]["type"]].append(sample)
    for question_type in QUOTAS:
        group = by_type[question_type]
        rng.shuffle(group)
        for position, sample in enumerate(group):
            sample["metadata"]["split"] = "holdout" if position % 5 == 0 else "dev"


def main():
    rng = random.Random(SEED)
    labels = load_labels()
    samples = []
    for category in TARGET_GROUPS:
        category_labels = [label for label in labels if label["category"] == category]
        samples.extend(build_category(category, category_labels, rng))
    assign_splits(samples, rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    holdout = sum(sample["metadata"]["split"] == "holdout" for sample in samples)
    print(f"Wrote {len(samples)} samples ({holdout} holdout)")


if __name__ == "__main__":
    main()
