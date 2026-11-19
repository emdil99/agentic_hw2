"""Read structured facts from raw DSLD label JSON.

Shared by scripts/build_dataset.py, which computes gold answers, and agent.py, whose lookup tool
answers questions that need every label at once.
"""
import re

MG_PER_UNIT ={"mg": 1, "g": 1000, "Gram(s)": 1000, "mcg": 0.001}

# Words a shopper uses that DSLD files under more than one ingredient group.
INGREDIENT_ALIASES = {
    "protein": ["Protein", "Protein (unspecified)"],
    "collagen": ["Collagen", "Collagen Peptides"],
}

# Other names labels use for an ingredient, beyond what DSLD's ingredient groups already cover.
TERM_SYNONYMS = {
    "stevia": ["stevia", "rebaudioside", "reb a", "reb m", "steviol"],
    "monk fruit": ["monk fruit", "monkfruit", "luo han guo", "lo han guo", "lo han", "siraitia", "mogroside"],
}

# FDA Daily Values for adults, in milligrams, used to catch amounts DSLD transcribed in the wrong unit.
DAILY_VALUE_MG = {
    "Sodium": 2300,
    "Potassium": 4700,
    "Magnesium": 420,
    "Protein": 50000,
    "Protein (unspecified)": 50000,
}
MAX_DAILY_VALUE_MISMATCH = 10


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
    amount = quantity["quantity"] * MG_PER_UNIT[quantity["unit"]]
    if not matches_daily_value(group, quantity, amount):
        return None
    return amount


def matches_daily_value(group, quantity, amount):
    """False when the label's own exact %DV contradicts the amount, e.g. magnesium '10 g' at 2% DV."""
    if group not in DAILY_VALUE_MG:
        return True
    for target in quantity.get("dailyValueTargetGroup") or []:
        # "<1%" style values are rounded down on purpose, so only an exact percentage can contradict.
        if target.get("operator") in ("=", None) and target.get("percent"):
            implied = target["percent"] / 100 * DAILY_VALUE_MG[group]
            if not implied / MAX_DAILY_VALUE_MISMATCH <= amount <= implied * MAX_DAILY_VALUE_MISMATCH:
                return False
    return True


def groups_for(ingredient):
    """Return the DSLD ingredient groups that a plain ingredient word refers to."""
    word = ingredient.strip().lower()
    return INGREDIENT_ALIASES.get(word, [word])


def has_ingredient(label, ingredient):
    groups = {group.lower() for group in groups_for(ingredient)}
    return any((row.get("ingredientGroup") or "").lower() in groups for row in all_rows(label["ingredientRows"]))


def amount_mg(label, ingredient):
    """Return milligrams per serving of an ingredient, or None unless the label states one clear amount."""
    groups = {group.lower() for group in groups_for(ingredient)}
    present = {
        row["ingredientGroup"]
        for row in all_rows(label["ingredientRows"])
        if (row.get("ingredientGroup") or "").lower() in groups
    }
    if len(present) != 1:
        return None
    return find_dose(label, present.pop())


def disclosed_amounts_mg(label, ingredient):
    """Return every disclosed amount of an ingredient on the label, in milligrams, however it is listed."""
    groups = {group.lower() for group in groups_for(ingredient)}
    amounts = []
    for row in all_rows(label["ingredientRows"]):
        if (row.get("ingredientGroup") or "").lower() not in groups:
            continue
        for quantity in row.get("quantity") or []:
            if quantity["unit"] in MG_PER_UNIT:
                amount = quantity["quantity"] * MG_PER_UNIT[quantity["unit"]]
                if matches_daily_value(row["ingredientGroup"], quantity, amount):
                    amounts.append(amount)
    return amounts


def listed_ingredient_text(label):
    """Return the label's ingredient names, groups, forms and notes, and its other ingredients, as lowercase text."""
    parts = []
    for row in all_rows(label["ingredientRows"]):
        parts += [row.get("name") or "", row.get("ingredientGroup") or "", row.get("notes") or ""]
        parts += [form.get("name") or "" for form in row.get("forms") or []]
    other = label.get("otheringredients") or {}
    for item in other.get("ingredients") or []:
        parts += [item.get("name") or "", item.get("ingredientGroup") or ""]
    parts.append(other.get("text") or "")
    return " ".join(parts).lower()


def lists_ingredient(label, term):
    """True if the term or a known synonym appears as whole words, so 'vitamin c' does not match 'vitamin complex'."""
    text = listed_ingredient_text(label)
    names = TERM_SYNONYMS.get(term.strip().lower(), [term])
    return any(re.search(rf"\b{re.escape(name.lower())}\b", text) for name in names)
