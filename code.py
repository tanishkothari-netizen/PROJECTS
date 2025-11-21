# =====================================================
# Datathon: Data Driven Policy Innovation – IIIT Bangalore
# Full Pipeline: EDA • Modelling • Policy Insights (robust I/O)
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_23 = os.path.join(BASE_DIR, "UDISE Education Dataset", "UDISE 2023-24")
BASE_24 = os.path.join(BASE_DIR, "UDISE Education Dataset", "UDISE 2024-25")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _warn(m):
    print(f"[WARN] {m}")


def _require(df, col, file_hint):
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in {file_hint}. "
                         f"Available columns: {list(df.columns)[:15]}{' ...' if df.shape[1] > 15 else ''}")


def _select_if_exists(df, cols, file_hint):
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _warn(f"{missing} not found in {file_hint}; continuing without them.")
    return df[existing]


# ---------------- LOAD ------------------
def load_year(base):
    prof_path = os.path.join(base, "100_prof1.csv")
    fac_path = os.path.join(base, "100_fac.csv")
    tch_path = os.path.join(base, "100_tch.csv")
    enr_path = os.path.join(base, "100_enr1.csv")

    # Read with low_memory=False to avoid dtype guessing issues
    prof = pd.read_csv(prof_path, low_memory=False)
    fac = pd.read_csv(fac_path, low_memory=False)
    tch = pd.read_csv(tch_path, low_memory=False)
    enr = pd.read_csv(enr_path, low_memory=False)

    # Normalize column names
    for df in (prof, fac, tch, enr):
        df.columns = df.columns.str.lower().str.strip()

    # Ensure key
    _require(prof, "pseudocode", prof_path)

    # Profile: select what exists
    prof_wanted = ["pseudocode", "state", "district", "rural_urban", "management"]
    prof = _select_if_exists(prof, prof_wanted, prof_path)

    # Facilities: optional columns are handled gracefully
    fac_wanted = [
        "pseudocode",
        "electricity_availability",
        "library_availability",
        "comp_ict_lab_yn",
        "internet",
        "total_boys_func_toilet",
        "total_girls_func_toilet",
    ]
    fac = _select_if_exists(fac, fac_wanted, fac_path)

    # Teachers
    tch_wanted = ["pseudocode", "total_tch"]
    tch = _select_if_exists(tch, tch_wanted, tch_path)
    _require(tch, "pseudocode", tch_path)

    # Enrolment
    enr_wanted = ["pseudocode"] + [f"c{i}_b" for i in range(1, 13)] + [f"c{i}_g" for i in range(1, 13)]
    enr = _select_if_exists(enr, enr_wanted, enr_path)
    _require(enr, "pseudocode", enr_path)

    # Merge
    df = prof.merge(fac, on="pseudocode", how="left") \
             .merge(tch, on="pseudocode", how="left") \
             .merge(enr, on="pseudocode", how="left")
    return df


print("Loading data...")
udise23 = load_year(BASE_23)
udise24 = load_year(BASE_24)


# ---------------- FEATURE ENGINEERING ----------------
def derive(df, year):
    boys = [c for c in df.columns if c.endswith("_b")]
    girls = [c for c in df.columns if c.endswith("_g")]

    if not boys or not girls:
        _warn("Enrolment columns c*_b / c*_g missing; students_total/gpi may be NaN.")

    df["boys_total"] = df[boys].sum(axis=1, min_count=1) if boys else np.nan
    df["girls_total"] = df[girls].sum(axis=1, min_count=1) if girls else np.nan
    df["students_total"] = df["boys_total"] + df["girls_total"]

    if "total_tch" not in df.columns:
        _warn("'total_tch' missing; PTR will be NaN.")
        df["total_tch"] = np.nan

    # Ratios
    df["gpi"] = df["girls_total"] / df["boys_total"].replace(0, np.nan)
    df["ptr"] = df["students_total"] / df["total_tch"].replace(0, np.nan)

    # Binary recodes if present
    for col in ["electricity_availability", "library_availability", "comp_ict_lab_yn", "internet"]:
        if col in df.columns:
            df[col + "_b"] = df[col].apply(lambda x: 1 if x == 1 else 0)
        else:
            _warn(f"'{col}' not found; not included in infra_index.")

    bin_cols = [c for c in df.columns if c.endswith("_b") and c not in boys + girls]
    df["infra_index"] = df[bin_cols].mean(axis=1) if bin_cols else np.nan

    df["year"] = year
    return df


udise23 = derive(udise23, 2023)
udise24 = derive(udise24, 2024)


# ---------------- DISTRICT AGGREGATION (FIXED) ----------------
def aggregate_correct(df):
    g = df.groupby(["state","district"], dropna=False, as_index=False).agg(
        students_total=("students_total","sum"),
        girls_total=("girls_total","sum"),
        boys_total=("boys_total","sum"),
        total_tch=("total_tch","sum"),
        elec=("electricity_availability_b","mean"),
        lib=("library_availability_b","mean"),
        ict=("comp_ict_lab_yn_b","mean"),
        net=("internet_b","mean"),
        toilets_b=("total_boys_func_toilet","sum"),
        toilets_g=("total_girls_func_toilet","sum")
    )
    # recompute ratios from sums
    g["gpi"] = g["girls_total"] / g["boys_total"].replace(0, np.nan)
    g["ptr"] = g["students_total"] / g["total_tch"].replace(0, np.nan)
    # infra index from district means of binaries
    g["infra_index"] = g[["elec","lib","ict","net"]].mean(axis=1)
    # convenience per-100-students toilet metrics
    g["toilet_b_per100"] = 100 * g["toilets_b"] / g["boys_total"].replace(0, np.nan)
    g["toilet_g_per100"] = 100 * g["toilets_g"] / g["girls_total"].replace(0, np.nan)
    return g

agg23 = aggregate_correct(udise23)
agg24 = aggregate_correct(udise24)
df = agg24.copy()

# basic cleaning
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ptr","infra_index","gpi"])
df["ptr"] = df["ptr"].clip(0, 120)  # clip extreme outliers

# ---------------- EDA (unchanged) ----------------
# ... your plots ...

# ---------------- MODELLING (weighted) ----------------
X = df[["infra_index","gpi","toilet_g_per100","toilet_b_per100"]].fillna(0)
y = df["ptr"]
w = df["students_total"].clip(lower=1)  # weights by size

# Weighted OLS via sample_weight
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y, sample_weight=w)
pred_lin = lin.predict(X)
from sklearn.metrics import r2_score
r2_lin = r2_score(y, pred_lin, sample_weight=w)

# Random Forest (train/test split)
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=400, random_state=42)
rf.fit(Xtr, ytr)
pred_rf = rf.predict(Xte)
r2_rf = r2_score(yte, pred_rf)

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# ---------------- POLICY INSIGHTS (thresholds now meaningful) ----------------
high_ptr = df[df["ptr"] > 40]          # teacher shortage
low_infra = df[df["infra_index"] < 0.5]
low_gpi = df[df["gpi"] < 0.90]

policy = [
    f"Teacher shortage: {len(high_ptr)} districts with PTR > 40 — targeted hiring/redeployment + hardship incentives.",
    f"Infrastructure deficit: {len(low_infra)} districts with Infra Index < 0.5 — electricity, toilets, ICT labs first.",
    f"Gender gap: {len(low_gpi)} districts with GPI < 0.90 — scholarships, bicycles/transport, community outreach."
]

with open(os.path.join(OUT_DIR,"model_summary.txt"),"w") as f:
    f.write("=== MODEL PERFORMANCE (district, weighted where applicable) ===\n")
    f.write(f"Weighted OLS R2: {r2_lin:.3f}\n")
    f.write(f"Random Forest R2: {r2_rf:.3f}\n\n")
    f.write("Feature Importance (RF):\n")
    for k,v in imp.items(): f.write(f"- {k}: {v:.3f}\n")
    f.write("\n=== POLICY INSIGHTS ===\n")
    for p in policy: f.write(f"- {p}\n")

df.to_csv(os.path.join(OUT_DIR,"district_level_final.csv"), index=False)

print("✅ Done. Key Results:")
print(f"Weighted OLS R² = {r2_lin:.3f}")
print(f"Random Forest R² = {r2_rf:.3f}")
print("Insights:")
for p in policy: print("-", p)
print("Exported:", os.path.join(OUT_DIR,"district_level_final.csv"))

# ---------- Tail diagnostics & policy target lists ----------
# Where is the right tail?
q90 = df["ptr"].quantile(0.90)
q95 = df["ptr"].quantile(0.95)
mx  = df["ptr"].max()
print(f"PTR tail — P90={q90:.1f}, P95={q95:.1f}, Max={mx:.1f}")

# Use policy-meaningful thresholds
THRESH_STRICT = 35.0   # teacher-shortage flag (use 30 if you want to be stricter)
high_ptr = df[df["ptr"] > THRESH_STRICT].copy()

# Top-20 highest PTR (sanity check)
top20 = df.sort_values("ptr", ascending=False).head(20)[["state","district","ptr","students_total","total_tch"]]
print("\nTop 20 PTR districts:")
print(top20.to_string(index=False))

# Export flagged lists for the note/dashboard
needs_teacher = df[df["ptr"] > THRESH_STRICT].sort_values("ptr", ascending=False)
needs_infra   = df[df["infra_index"] < 0.50].sort_values("infra_index")
needs_gpi     = df[df["gpi"] < 0.90].sort_values("gpi")

needs_teacher.to_csv(os.path.join(OUT_DIR,"flag_teacher_shortage_PTR_gt35.csv"), index=False)
needs_infra.to_csv(os.path.join(OUT_DIR,"flag_infrastructure_lt_0p50.csv"), index=False)
needs_gpi.to_csv(os.path.join(OUT_DIR,"flag_gender_parity_lt_0p90.csv"), index=False)

# Update printed insights with the new threshold
print(f"\nDistricts with PTR > {THRESH_STRICT:.0f}: {len(needs_teacher)}")


# ---------------- EXPORT ----------------
df.to_csv(os.path.join(OUT_DIR, "district_level_final.csv"), index=False)
print(f"\nExported district-level data to: {os.path.join(OUT_DIR, 'district_level_final.csv')}")
