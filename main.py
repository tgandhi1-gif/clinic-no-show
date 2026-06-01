"""Reading the Waiting Room - predict no shows and target reminders."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(3)
N = 12000

lead_days = rng.integers(0, 40, N)
prior_no_show = rng.random(N)
rain = rng.random(N) < 0.25
age = rng.integers(12, 80, N)

p = 0.06 + 0.012 * lead_days + 0.35 * prior_no_show + 0.10 * rain - 0.0015 * age
p = np.clip(p, 0.02, 0.95)
no_show = rng.random(N) < p

X = pd.DataFrame({"lead_days": lead_days, "prior_no_show": prior_no_show,
                  "rain": rain.astype(int), "age": age})
y = no_show.astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=2)
model = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
print(f"No show model AUC: {auc:.3f}")

risk = model.predict_proba(X)[:, 1]
targeted = risk >= np.quantile(risk, 0.70)
base_rate = no_show.mean()
recovered = (no_show & targeted).sum() * 0.30
reduction = recovered / no_show.sum()
print(f"Reminding the top 30 percent risk cuts no shows by {reduction*100:.1f}%")

os.makedirs("outputs", exist_ok=True)
order = np.argsort(risk)[::-1]
plt.figure(figsize=(9, 5))
plt.plot(np.cumsum(no_show[order]) / no_show.sum(), color="#ff6a3d", lw=2.5, label="model targeting")
plt.plot(np.linspace(0, 1, N), "--", color="#999999", label="random")
plt.xlabel("patients contacted (ranked by risk)")
plt.ylabel("share of no shows caught")
plt.legend()
plt.title("Catching no shows before the empty chair")
plt.tight_layout()
plt.savefig("outputs/clinic_no_show.png", dpi=120)
print("Saved outputs/clinic_no_show.png")
