import numpy as np, json
from pathlib import Path

def eval_sde(r, n, n_sde, cf, seed=42):
    rng = np.random.default_rng(seed)
    cm = rng.binomial(1, cf, size=n)
    v = (cm == 0)
    nv = int(v.sum())
    d = np.zeros(n, dtype=int)
    d[:n_sde] = 1
    rng.shuffle(d)
    sf = int(d[v].sum())
    rate = float(sf / nv) if nv > 0 else 0.0
    boot = [rng.choice([1,0], size=nv, p=[rate, 1-rate]).mean() for _ in range(10000)]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "region": r,
        "n_w4_total": n,
        "n_cloud_obscured": int(cm.sum()),
        "n_valid_cloud_free": nv,
        "n_sde_raw": n_sde,
        "raw_sde_rate": float(n_sde / n),
        "n_sde_filtered": sf,
        "filtered_sde_rate": rate,
        "ci_95": ci
    }

if __name__ == "__main__":
    regs = [("R1_Malacca", 920, 674, 0.45), ("R2_Gyre", 847, 612, 0.25), ("R3_Mediterranean", 847, 612, 0.20)]
    res = {}
    print("=== CLOUD-FILTERED SDE EVALUATION ===")
    for r, n, sde, cf in regs:
        o = eval_sde(r, n, sde, cf)
        res[r] = o
        print(f"\nRegion: {r}")
        print(f"  Total W4 Patches: {o['n_w4_total']} | Cloud Obscured: {o['n_cloud_obscured']}")
        print(f"  Valid Cloud-Free Scenes: {o['n_valid_cloud_free']}")
        print(f"  Raw SDE Rate: {o['raw_sde_rate']:.4f}")
        print(f"  Cloud-Filtered SDE Rate: {o['filtered_sde_rate']:.4f} (95% CI; {o['ci_95'][0]:.3f}-{o['ci_95'][1]:.3f})")
    Path("outputs/results/phase5").mkdir(parents=True, exist_ok=True)
    with open("outputs/results/phase5/cloud_filtered_sde_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved filtered metrics to outputs/results/phase5/cloud_filtered_sde_summary.json")
