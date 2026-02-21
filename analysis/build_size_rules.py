import pandas as pd

# =========================
# LOAD DATASET
# =========================
DATASET_PATH = "../data/processed/dataset_ai_tailor_public.csv"
df = pd.read_csv(DATASET_PATH)

# =========================
# CHEST AS ANCHOR
# =========================
chest = df["chest_cm"]

# =========================
# QUANTILES
# =========================
q25 = chest.quantile(0.25)
q50 = chest.quantile(0.50)
q75 = chest.quantile(0.75)

# =========================
# TAIL CONFIG
# =========================
TAIL_STEP = 8  # cm (standar industri)

# =========================
# BOUNDARIES (STRICT, NO OVERLAP)
# =========================
s_max = round(q25, 1)
m_max = round(q50, 1)
l_max = round(q75, 1)

xl_max = round(l_max + TAIL_STEP, 1)
xxl_max = round(l_max + 2 * TAIL_STEP, 1)

# =========================
# SIZE RULES
# =========================
size_rules = {
    "S": f"≤ {s_max} cm",
    "M": f"{round(s_max+0.1,1)} – {m_max} cm",
    "L": f"{round(m_max+0.1,1)} – {l_max} cm",
    "XL": f"{round(l_max+0.1,1)} – {xl_max} cm",
    "XXL": f"{round(xl_max+0.1,1)} – {xxl_max} cm",
    "XXXL": f"≥ {round(xxl_max+0.1,1)} cm",
}

# =========================
# OUTPUT
# =========================
print("=== RULE UKURAN BAJU (S–XXXL) ===")
for size, rule in size_rules.items():
    print(f"{size}: {rule}")

rules_df = pd.DataFrame.from_dict(
    size_rules,
    orient="index",
    columns=["chest_cm_range"]
)

rules_df.to_csv("../analysis/size_rules.csv")

print("\n📁 Rule ukuran disimpan di analysis/size_rules.csv")
