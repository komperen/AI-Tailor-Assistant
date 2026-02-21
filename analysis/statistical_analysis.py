import pandas as pd

# =========================
# LOAD DATASET FINAL
# =========================
DATASET_PATH = "../data/processed/dataset_ai_tailor_public.csv"
df = pd.read_csv(DATASET_PATH)

# =========================
# PILIH KOLOM NUMERIK
# =========================
NUMERIC_COLS = [
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "arm_circumference_cm",
    "arm_length_cm",
    "thigh_cm",
    "calf_cm",
]

df_num = df[NUMERIC_COLS]

# =========================
# HITUNG STATISTIK
# =========================
stats = df_num.describe().loc[["mean", "std", "min", "max"]]

# BULATKAN AGAR RAPI
stats = stats.round(2)

# =========================
# SIMPAN KE CSV
# =========================
OUT_STATS = "../analysis/statistical_summary.csv"
stats.to_csv(OUT_STATS)

print("✅ ANALISIS STATISTIK SELESAI")
print("📁 Output:", OUT_STATS)
print("\n=== RINGKASAN STATISTIK ===")
print(stats)
