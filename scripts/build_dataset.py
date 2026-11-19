"""Build the gold eval set from the structured DSLD labels.

    uv run scripts/build_dataset.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# label_facts.py lives at the repo root, which is not on the path when this script runs.
sys.path.insert(0, str(ROOT))
from label_facts import (  # noqa: E402
    all_rows,
    amount_mg,
    disclosed_amounts_mg,
    find_dose,
    format_mg,
    has_ingredient,
    ingredient_word,
    is_undisclosed,
    lists_ingredient,
)

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

# Added in dataset v2, after the v1 baseline scored near the ceiling. These use their own random
# generator and run after everything else, so the 100 v1 questions stay identical.
PARTIAL_NAME_QUOTAS = {
    "partial_name_same": 3,
    "partial_name_varies": 3,
}
PARTIAL_NAME_SEED = SEED + 1
# A family must fit in one page of search results, so search can show the agent every version.
MIN_FAMILY_SIZE = 2
MAX_FAMILY_SIZE = 5

# Added in dataset v3: questions that need every label in a category at once, which top-5 search
# cannot answer. They involve no random choices, so the v1 and v2 questions stay identical.
DATABASE_TYPES = ["database_most", "database_least", "database_count"]
EXTREME_INGREDIENTS = {
    "creatine": ["creatine"],
    "whey protein": ["protein"],
    "electrolytes": ["sodium", "potassium", "magnesium"],
    "collagen": ["collagen"],
}
COUNT_TERMS = {
    "creatine": ["caffeine", "beta-alanine", "taurine", "maltodextrin"],
    "whey protein": ["monk fruit", "stevia", "vitamin c", "maltodextrin"],
    "electrolytes": ["coconut water", "caffeine", "monk fruit", "taurine"],
    "collagen": ["hyaluronic acid", "biotin", "stevia", "citric acid"],
}
CATEGORY_NOUNS = {"creatine": "creatine", "whey protein": "whey protein", "electrolytes": "electrolyte", "collagen": "collagen"}

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


def load_labels():
    labels = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        raw_path = RAW_DIR / f"{record['id']}.json"
        label = json.loads(raw_path.read_text(encoding="utf-8"))
        label["category"] = record["metadata"]["categories"][0]
        label["brand_name"] = record["metadata"]["brand"]
        label["product_name"] = record["metadata"]["product"]
        label["display_name"] = f"{record['metadata']['brand']} {record['metadata']['product']}"
        labels.append(label)
    return labels


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


def common_prefix(name_a, name_b):
    """Return the leading words two product names share, such as 'Best Creatine'."""
    shared = []
    for word_a, word_b in zip(name_a.split(), name_b.split()):
        if word_a.casefold() != word_b.casefold():
            break
        shared.append(word_a)
    return " ".join(shared)


def product_families(labels):
    """Group labels one brand sells under a shared name, like the flavors of one product.

    Returns {(brand, shared name): [labels]}. A shared name must be at least two words and must not
    be the complete name of any member, so a question using it really is ambiguous.
    """
    by_brand = defaultdict(list)
    for label in labels:
        by_brand[(label["category"], label["brand_name"])].append(label)

    families = {}
    for (_, brand), members in by_brand.items():
        for label_a in members:
            for label_b in members:
                if label_a["id"] >= label_b["id"]:
                    continue
                prefix = common_prefix(label_a["product_name"], label_b["product_name"])
                if len(prefix.split()) < 2 or (brand, prefix) in families:
                    continue
                family = [m for m in members if m["product_name"].casefold().startswith(prefix.casefold() + " ")]
                complete_name = any(m["product_name"].casefold() == prefix.casefold() for m in members)
                if MIN_FAMILY_SIZE <= len(family) <= MAX_FAMILY_SIZE and not complete_name:
                    families[(brand, prefix)] = family
    return families


def make_partial_name(brand, prefix, family, rng):
    """Ask about a product by its shared name. The answer depends on whether its versions agree."""
    category = family[0]["category"]
    groups = TARGET_GROUPS[category][:]
    rng.shuffle(groups)
    for group in groups:
        doses = [find_dose(label, group) for label in family]
        if any(dose is None for dose in doses):
            continue
        ids = [str(label["id"]) for label in family]
        question = f"How much {ingredient_word(group)} is in one serving of {brand} {prefix}?"
        if len(set(doses)) == 1:
            question_type = "partial_name_same"
            target = f"{format_mg(doses[0])}, the same in all {len(family)} versions."
            value = doses[0]
        else:
            question_type = "partial_name_varies"
            versions = "; ".join(f"{label['display_name']}: {format_mg(dose)}" for label, dose in zip(family, doses))
            target = f"It depends on the version. {versions}"
            value = dict(zip(ids, doses))
        return {
            "id": f"{category.replace(' ', '-')}-{question_type}-{ids[0]}",
            "input": question,
            "target": target,
            "metadata": {
                "type": question_type,
                "category": category,
                "gold_label_ids": ids,
                "value_mg": value,
                "origin": "dsld_label_family",
                "source": " ".join(f"{LABEL_PAGE_URL}/{label_id}" for label_id in ids),
            },
        }
    return None


def build_partial_name_questions(labels, used_ids, rng):
    families = list(product_families(labels).items())
    rng.shuffle(families)
    samples = []
    for category in TARGET_GROUPS:
        made = defaultdict(int)
        for (brand, prefix), family in families:
            if family[0]["category"] != category or any(str(label["id"]) in used_ids for label in family):
                continue
            sample = make_partial_name(brand, prefix, family, rng)
            if sample is None or made[sample["metadata"]["type"]] >= PARTIAL_NAME_QUOTAS[sample["metadata"]["type"]]:
                continue
            samples.append(sample)
            made[sample["metadata"]["type"]] += 1
            used_ids.update(sample["metadata"]["gold_label_ids"])
        for question_type, quota in PARTIAL_NAME_QUOTAS.items():
            if made[question_type] < quota:
                print(f"SHORT: {category} / {question_type}: made {made[question_type]} of {quota}")
    return samples


def database_sample(category, question_type, key, question, target, **extra):
    return {
        "id": f"{category.replace(' ', '-')}-{question_type}-{key.replace(' ', '-')}",
        "input": question,
        "target": target,
        "metadata": {
            "type": question_type,
            "category": category,
            "gold_label_ids": [],
            "origin": "dsld_all_labels",
            "source": f"computed from every {category} label in data/corpus.jsonl",
            **extra,
        },
    }


def make_extreme(category, labels, ingredient, direction):
    """Ask which product has the most or least of an ingredient. None unless the answer is provable.

    Labels that list the ingredient without one clear amount cannot be ranked, so the question is
    skipped if any amount they do disclose could beat the winner, or if the winner is tied.
    """
    ranked = sorted(
        ((amount_mg(label, ingredient), label) for label in labels if amount_mg(label, ingredient) is not None),
        key=lambda pair: pair[0],
        reverse=direction == "most",
    )
    unranked = [label for label in labels if has_ingredient(label, ingredient) and amount_mg(label, ingredient) is None]
    unranked_amounts = [amount for label in unranked for amount in disclosed_amounts_mg(label, ingredient) if amount > 0]
    if len(ranked) < 2 or ranked[0][0] == ranked[1][0]:
        print(f"SKIP: {category} / {direction} {ingredient}: tied")
        return None
    best_amount, best = ranked[0]
    beaten = [a for a in unranked_amounts if (a > best_amount if direction == "most" else a < best_amount)]
    if beaten:
        print(f"SKIP: {category} / {direction} {ingredient}: an unranked label discloses {format_mg(max(beaten) if direction == 'most' else min(beaten))}")
        return None
    question = f"Which {CATEGORY_NOUNS[category]} product in the database has the {direction} {ingredient} per serving?"
    target = (
        f"{best['display_name']}, with {format_mg(best_amount)} per serving, "
        f"among the {len(ranked)} labels that state one clear amount."
    )
    return database_sample(
        category, f"database_{direction}", ingredient, question, target,
        value_mg=best_amount, answer_label_id=str(best["id"]), labels_ranked=len(ranked),
    )


def make_count(category, labels, term):
    matching = [str(label["id"]) for label in labels if lists_ingredient(label, term)]
    question = f"How many {CATEGORY_NOUNS[category]} products in the database list {term} as an ingredient?"
    return database_sample(
        category, "database_count", term, question, f"{len(matching)} products.",
        count=len(matching), matching_label_ids=matching,
    )


def build_database_questions(labels):
    samples = []
    for category in TARGET_GROUPS:
        category_labels = [label for label in labels if label["category"] == category]
        for ingredient in EXTREME_INGREDIENTS[category]:
            for direction in ["most", "least"]:
                sample = make_extreme(category, category_labels, ingredient, direction)
                if sample:
                    samples.append(sample)
        samples.extend(make_count(category, category_labels, term) for term in COUNT_TERMS[category])
    return samples


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
    for question_type in [*QUOTAS, *PARTIAL_NAME_QUOTAS, *DATABASE_TYPES]:
        group = by_type[question_type]
        rng.shuffle(group)
        for position, sample in enumerate(group):
            sample["metadata"]["split"] = "holdout" if position % 5 == 0 else "dev"


def previous_splits():
    """Return {id: split} from the committed dataset, so a question never moves between dev and holdout.

    Dropping or adding a question reshuffles the split of every later question type, which would move
    questions already tuned against in dev into the holdout.
    """
    if not OUTPUT_PATH.exists():
        return {}
    records = [json.loads(line) for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines()]
    return {record["id"]: record["metadata"]["split"] for record in records}


def main():
    frozen_splits = previous_splits()
    rng = random.Random(SEED)
    labels = load_labels()
    samples = []
    for category in TARGET_GROUPS:
        category_labels = [label for label in labels if label["category"] == category]
        samples.extend(build_category(category, category_labels, rng))
    samples.extend(build_partial_name_questions(labels, set(), random.Random(PARTIAL_NAME_SEED)))
    samples.extend(build_database_questions(labels))
    assign_splits(samples, rng)
    for sample in samples:
        if sample["id"] in frozen_splits:
            sample["metadata"]["split"] = frozen_splits[sample["id"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    holdout = sum(sample["metadata"]["split"] == "holdout" for sample in samples)
    print(f"Wrote {len(samples)} samples ({holdout} holdout)")


if __name__ == "__main__":
    main()
