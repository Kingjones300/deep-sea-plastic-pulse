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
OUT_DIR = ROOT / "outputs" / "results" / "phase2_R1"
FIG_DIR = ROOT / "outputs" / "figures" / "R1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("="*60)
print("  PHASE 2 - XGBoost Weathering State Classifier - R1")
print("  Region: Strait of Malacca | Tianjin University")
print("="*60)
print("\n[1/5] Generating training data (R1 tropical tuning)...")
rng = np.random.default_rng(42)
records = []
params = {
    "W1": (0.08,0.01,-0.05,0.01,0.12,0.02,3.5,0.3),
    "W2": (0.06,0.015,0.02,0.01,0.09,0.02,2.8,0.4),
    "W3": (0.04,0.015,0.08,0.015,0.06,0.02,2.1,0.4),
    "W4": (0.02,0.01,0.15,0.02,0.03,0.015,1.4,0.3),
}
for label,(fm,fs,sm,ss,nm,ns,rm,rs) in params.items():
    for _ in range(1250):
        records.append({
            "FDI":rng.normal(fm,fs),
            "SWI":rng.normal(sm,ss),
            "NDVI":rng.normal(nm,ns),
            "SNR":rng.normal(rm,rs),
            "label":label
        })
df = pd.DataFrame(records).sample(frac=1,random_state=42).reset_index(drop=True)
print(f"   {len(df)} samples generated, 1250 per class")
print("\n[2/5] Training XGBoost classifier...")
le = LabelEncoder()
X = df[["FDI","SWI","NDVI","SNR"]].values
y = le.fit_transform(df["label"].values)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42, n_jobs=1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
print(f"   5-fold CV F1 (macro): {scores.mean():.4f} +/- {scores.std():.4f}")
model.fit(X, y)
y_pred = model.predict(X)
print(classification_report(y, y_pred, target_names=le.classes_))
print("\n[3/5] Running SHAP explainability...")
try:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    w4i = list(le.classes_).index("W4")
    plt.figure(figsize=(8,5))
    shap.summary_plot(sv[w4i], X,
        feature_names=["FDI","SWI","NDVI","SNR"],
        show=False, plot_type="bar")
    plt.title("SHAP Feature Importance - W4 Critical Sinking State - R1 Malacca")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_w4_importance_R1.png", dpi=150)
    plt.close()
    print("   SHAP plot saved")
except Exception as e:
    print(f"   SHAP skipped: {e}")
print("\n[4/5] Computing PtS scores...")
pts = model.predict_proba(X)[:,3]
print(f"   PtS mean: {pts.mean():.4f} | W4 fraction (>0.7): {(pts>=0.7).mean():.3f}")
print("\n[5/5] Saving outputs...")
model.save_model(str(OUT_DIR / "xgboost_weathering_model_R1.json"))
np.save(OUT_DIR / "pts_scores_training_R1.npy", pts)
cm = confusion_matrix(y, y_pred)
fig,ax = plt.subplots(figsize=(6,5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(le.classes_)
ax.set_yticks(range(4)); ax.set_yticklabels(le.classes_)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix W1-W4 - R1 Malacca")
for i in range(4):
    for j in range(4):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_w1w4_R1.png", dpi=150)
plt.close()
summary = {
    "timestamp": datetime.now().isoformat(),
    "region": "R1_Malacca",
    "cv_f1_macro": float(scores.mean()),
    "cv_f1_std": float(scores.std()),
    "n_samples": 5000,
    "pts_mean": float(pts.mean()),
    "pts_w4_fraction": float((pts>=0.7).mean()),
    "classes": list(le.classes_)
}
with open(OUT_DIR / "phase2_R1_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print("\nPhase 2 R1 Summary:")
for k,v in summary.items():
    print(f"   {k}: {v}")
print("\n✅ Phase 2 R1 complete.")
print(f"   Model: {OUT_DIR}")
print(f"   Figures: {FIG_DIR}")
print("="*60)