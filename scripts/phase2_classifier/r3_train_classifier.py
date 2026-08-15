import os, sys, json, numpy as np, pandas as pd
import xgboost as xgb, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(r"C:\Users\Apple\deep_sea_pulse")
OUT_DIR = ROOT / "outputs" / "results" / "phase2" / "R3"
FIG_DIR = ROOT / "outputs" / "figures" / "R3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("  PHASE 2 - XGBoost Weathering State Classifier")
print("  Region R3: Western Mediterranean Sea")
print("  Deep Sea Plastic Pulse | Tianjin University")
print("="*60)

# R3 oligotrophic parameters — slow biofouling, aged gyre plastic
# W1: dominant class, virgin/lightly weathered (gyre accumulation)
# W2: moderate — surface pitting, sparse biofilm
# W3: near-neutral buoyancy — less common in oligotrophic environment
# W4: critical threshold — rare but present at gyre periphery/eddy ejection zones
print("\n[1/5] Generating R3 training data (oligotrophic parameterisation)...")
rng = np.random.default_rng(42)
records = []

# (FDI_mean, FDI_std, SWI_mean, SWI_std, NDVI_mean, NDVI_std, SNR_mean, SNR_std, n_samples)
# R3: W1 and W2 dominate (1600 each), W3/W4 less common (1100 each) — reflects gyre reality
params = {
    "W1": (0.09, 0.010, -0.06, 0.010, 0.13, 0.020, 3.6, 0.3, 1600),
    "W2": (0.07, 0.012, 0.01,  0.010, 0.10, 0.020, 2.9, 0.4, 1600),
    "W3": (0.04, 0.012, 0.09,  0.012, 0.05, 0.018, 2.0, 0.4, 1100),
    "W4": (0.02, 0.010, 0.17,  0.018, 0.02, 0.012, 1.3, 0.3, 1100),
}

for label, (fm,fs,sm,ss,nm,ns,rm,rs,n) in params.items():
    for _ in range(n):
        records.append({
            "FDI":  rng.normal(fm, fs),
            "SWI":  rng.normal(sm, ss),
            "NDVI": rng.normal(nm, ns),
            "SNR":  rng.normal(rm, rs),
            "label": label
        })

df = pd.DataFrame(records).sample(frac=1, random_state=43).reset_index(drop=True)
print(f"   {len(df)} samples: W1=1600, W2=1600, W3=1100, W4=1100")
print(f"   Reflects oligotrophic gyre — W1/W2 dominant, W4 at periphery")

# 2. Train classifier
print("\n[2/5] Training XGBoost classifier...")
le = LabelEncoder()
X = df[["FDI","SWI","NDVI","SNR"]].values
y = le.fit_transform(df["label"].values)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=43, n_jobs=1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
print(f"   5-fold CV F1 (macro): {scores.mean():.4f} +/- {scores.std():.4f}")
model.fit(X, y)
y_pred = model.predict(X)
print(classification_report(y, y_pred, target_names=le.classes_))

# 3. SHAP
print("\n[3/5] Running SHAP explainability...")
try:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    w4i = list(le.classes_).index("W4")
    plt.figure(figsize=(8,5))
    shap.summary_plot(sv[w4i], X,
        feature_names=["FDI","SWI","NDVI","SNR"],
        show=False, plot_type="bar")
    plt.title("SHAP Feature Importance - W4 Critical Sinking State (R3 Gyre)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_w4_importance.png", dpi=150)
    plt.close()
    print("   SHAP plot saved")
except Exception as e:
    print(f"   SHAP skipped: {e}")

# 4. PtS scores
print("\n[4/5] Computing PtS scores...")
pts = model.predict_proba(X)[:,list(le.classes_).index("W4")]
print(f"   PtS mean: {pts.mean():.4f} | W4 fraction (>0.7): {(pts>=0.7).mean():.3f}")

# 5. Save outputs
print("\n[5/5] Saving outputs...")
model.save_model(str(OUT_DIR / "xgboost_weathering_model_R3.json"))
np.save(OUT_DIR / "pts_scores_training.npy", pts)

cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(le.classes_)
ax.set_yticks(range(4)); ax.set_yticklabels(le.classes_)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix W1-W4 | R3 North Pacific Gyre")
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_w1w4.png", dpi=150)
plt.close()

summary = {
    "timestamp": datetime.now().isoformat(),
    "region": "R3_Mediterranean",
    "cv_f1_macro": float(scores.mean()),
    "cv_f1_std": float(scores.std()),
    "n_samples": int(len(df)),
    "class_distribution": {"W1":1600,"W2":1600,"W3":1100,"W4":1100},
    "pts_mean": float(pts.mean()),
    "pts_w4_fraction": float((pts>=0.7).mean()),
    "classes": list(le.classes_),
    "mediterranean_parameterisation": True,
    "chl_a_ref_mg_m3": 0.50,
    "rho_sw_kg_m3": 1026.0
}
with open(OUT_DIR / "phase2_summary.json","w") as f:
    json.dump(summary, f, indent=2)

print("\nPhase 2 Summary:")
for k,v in summary.items():
    print(f"   {k}: {v}")
print("\n✅ Phase 2 R3 complete.")
print(f"   Model  → {OUT_DIR}")
print(f"   Figures → {FIG_DIR}")
print("="*60)