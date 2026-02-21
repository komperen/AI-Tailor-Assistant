import pandas as pd

# =========================
# PATH
# =========================
RAW_CSV = "../data/raw/Body Measurements Image Dataset.csv"
OUT_CSV = "../data/processed/dataset_ai_tailor_public.csv"

# =========================
# LOAD DATASET (SEMICOLON!)
# =========================
df = pd.read_csv(RAW_CSV, sep=";")

print("=== KOLOM ASLI CSV (SETELAH FIX) ===")
print(df.columns.tolist())

# =========================
# NORMALIZE COLUMN NAMES
# =========================
df.columns = df.columns.str.strip().str.lower()

# =========================
# SELECT & RENAME
# =========================
COLUMN_MAP = {
    "id": "id",
    "gender": "gender",
    "height_cm": "height_cm",
    "chest_cm": "chest_cm",
    "waist_cm": "waist_cm",
    "hip_cm": "hip_cm",
    "arm_circumference_cm": "arm_circumference_cm",
    "arm_length_cm": "arm_length_cm",
    "thigh_cm": "thigh_cm",
    "calf_cm": "calf_cm",
}

df_final = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

# =========================
# CLEAN tbr
# =========================
for col in ["height_cm", "chest_cm", "waist_cm", "hip_cm"]:
    df_final[col] = (
        df_final[col]
        .astype(str)
        .str.replace("_tbr", "", regex=False)
    )
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

df_final = df_final.dropna(subset=["height_cm", "chest_cm", "waist_cm", "hip_cm"])

# =========================
# NORMALIZE GENDER
# =========================
df_final["gender"] = df_final["gender"].str.lower()

# =========================
# ADD METADATA
# =========================
df_final["source"] = "public_body_measurements"
df_final["notes"] = ""

# =========================
# SAVE
# =========================
df_final.to_csv(OUT_CSV, index=False)

print("\n✅ DATASET BERHASIL DINORMALISASI")
print("📁 Output:", OUT_CSV)
print("📊 Jumlah data:", len(df_final))
print(df_final.head())
