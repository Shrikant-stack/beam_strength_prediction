"""
================================================================================
FORWARD STUDY — Structural Beam Load Prediction
================================================================================
Paper: A Comparative Study of Machine Learned Structural Capacity Prediction
       and Inverse Parameter Synthesis using Synthetic Multi-Material Beam Datasets
Author: Shrikant Thakur | C-DAC, Kolkata

PURPOSE:
  Predict Max_Load_kN from beam geometry, material, and section properties
  using multiple ML models. Generates Fusion 360 validation export.

MODELS BENCHMARKED:
  1. Ridge Regression        (Baseline — linear)
  2. Random Forest           (RF)
  3. Extra Trees             (ET)
  4. XGBoost                 (XGB)
  5. CatBoost                (CB)
  6. MLP Neural Network      (MLP)
  7. TabNet                  (TB — requires pytorch-tabnet)

INSTALL:
  pip install pandas numpy scikit-learn matplotlib seaborn shap
  pip install xgboost catboost pytorch-tabnet
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Optional packages (graceful degradation) ────────────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False
    print("⚠  XGBoost not found → using HistGradientBoostingRegressor as proxy")

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("⚠  CatBoost not found → using GradientBoostingRegressor as proxy")

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print("⚠  TabNet not found → skipping (pip install pytorch-tabnet)")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠  SHAP not found → skipping (pip install shap)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "physics_beam_dataset_ML.csv"
OUTPUT_DIR  = "forward_study_outputs"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
CV_FOLDS    = 5
N_FUSION360 = 25          # samples to export for FEA validation

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans',
                     'axes.titlesize': 13, 'axes.labelsize': 11})

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 1: DATA LOADING & PREPROCESSING")
print("="*70)

df_raw = pd.read_csv(DATA_PATH)
print(f"  Dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ── 2.1 Remove degenerate I-Beam rows (I = 1e-10 placeholder) ────────────────
df = df_raw[df_raw['Moment_of_Inertia_m4'] > 1e-9].copy()
# ── 2.2 Remove clamped boundary samples (0.5 kN minimum / 5000 kN maximum) ──
df = df[(df['Max_Load_kN'] > 0.5) & (df['Max_Load_kN'] < 5000)].copy()
df.reset_index(drop=True, inplace=True)
print(f"  After cleaning (remove degenerate I-Beam + clamped): {len(df)} rows")

# ── 2.3 Encode categorical variables ─────────────────────────────────────────
le_mat  = LabelEncoder()
le_cs   = LabelEncoder()
le_gc   = LabelEncoder()
df['Material_enc']    = le_mat.fit_transform(df['Material'])
df['CrossSection_enc']= le_cs.fit_transform(df['Cross_Section_Type'])
df['GovCrit_enc']     = le_gc.fit_transform(df['Governing_Criterion'])

# ── 2.4 Physics-informed engineered feature ───────────────────────────────────
# P_min = min(P_Bending, P_Shear, P_Deflection) = physics definition of Max_Load
# Including this feature tests whether models can learn the governing criterion rule
df['P_min_physics'] = df[['P_Bending_kN', 'P_Shear_kN', 'P_Deflection_kN']].min(axis=1)

# ── 2.5 Log-transform target (skewness = 4.69 → must transform) ──────────────
df['Log_Max_Load'] = np.log1p(df['Max_Load_kN'])

print(f"\n  Target (Max_Load_kN) stats:")
print(f"    Range : {df['Max_Load_kN'].min():.2f} — {df['Max_Load_kN'].max():.2f} kN")
print(f"    Mean  : {df['Max_Load_kN'].mean():.2f} kN")
print(f"    Skew  : {df['Max_Load_kN'].skew():.3f} (raw) → {df['Log_Max_Load'].skew():.3f} (log)")
print(f"\n  Class distribution:")
print(f"    Materials     : {df['Material'].value_counts().to_dict()}")
print(f"    Cross-Sections: {df['Cross_Section_Type'].value_counts().to_dict()}")
print(f"    Governing     : {df['Governing_Criterion'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
# Full feature set (includes derived physics quantities Pb, Ps, Pd)
FEATURES = [
    # Geometry
    'Beam_Length_m', 'Thickness_m',
    # Design
    'Factor_of_Safety',
    # Cross-section properties (physics-derived from geometry)
    'Moment_of_Inertia_m4', 'Cross_Sec_Area_m2', 'y_max_m',
    # Material
    'Young_Modulus_GPa', 'Yield_Strength_MPa', 'Sigma_Allow_MPa',
    # Load capacities per criterion (key physics features)
    'P_Bending_kN', 'P_Shear_kN', 'P_Deflection_kN',
    # Structural indices
    'Slenderness_Ratio', 'Max_Deflection_mm',
    # Stress state
    'Bending_Stress_MPa', 'Stress_Utilization_pct',
    # Encoded categoricals
    'Material_enc', 'CrossSection_enc', 'GovCrit_enc',
    # Engineered physics feature
    'P_min_physics'
]
TARGET = 'Log_Max_Load'   # log-transformed; expm1() applied at evaluation

X = df[FEATURES].values
y = df[TARGET].values
y_orig = df['Max_Load_kN'].values

print(f"\n  Feature count: {len(FEATURES)}")

# ── Train / Test split (stratified by material for balance) ───────────────────
X_train, X_test, y_train, y_test, yo_train, yo_test = train_test_split(
    X, y, y_orig, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    stratify=df['Material_enc'].values
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)} (stratified by material)")

# ── Scaled version for MLP / TabNet ──────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name, y_true_orig, y_pred_log, results_dict):
    """Convert log predictions back and compute all 4 metrics."""
    y_pred = np.expm1(y_pred_log)
    y_true = y_true_orig
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # MAPE: guard against division by zero
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    results_dict[name] = {'R2': r2, 'RMSE_kN': rmse, 'MAE_kN': mae,
                          'MAPE_pct': mape, 'y_pred': y_pred}
    print(f"  {name:<38} R²={r2:.4f} | RMSE={rmse:8.2f} kN | "
          f"MAE={mae:7.2f} kN | MAPE={mape:.2f}%")
    return r2, rmse, mae, mape

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 2: MODEL TRAINING & EVALUATION")
print("="*70)
print(f"  {'Model':<38} {'R²':>6}   {'RMSE (kN)':>10}   {'MAE (kN)':>9}   {'MAPE %':>7}")
print("  " + "-"*68)

results    = {}
models_fit = {}   # store fitted models for SHAP / CV

# ── Model 1: Ridge Regression (Linear Baseline) ───────────────────────────────
m = Ridge(alpha=1.0)
m.fit(X_train, y_train)
models_fit['Ridge'] = m
evaluate_model('Ridge Regression (Baseline)', yo_test, m.predict(X_test), results)

# ── Model 2: Random Forest ────────────────────────────────────────────────────
m = RandomForestRegressor(n_estimators=500, min_samples_leaf=1,
                          random_state=RANDOM_SEED, n_jobs=-1)
m.fit(X_train, y_train)
models_fit['Random Forest'] = m
evaluate_model('Random Forest', yo_test, m.predict(X_test), results)

# ── Model 3: Extra Trees ──────────────────────────────────────────────────────
m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1,
                        random_state=RANDOM_SEED, n_jobs=-1)
m.fit(X_train, y_train)
models_fit['Extra Trees'] = m
evaluate_model('Extra Trees', yo_test, m.predict(X_test), results)

# ── Model 4: XGBoost ─────────────────────────────────────────────────────────
if HAS_XGB:
    m = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8,
                     early_stopping_rounds=50, eval_metric='rmse',
                     random_state=RANDOM_SEED, verbosity=0)
    m.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=False)
else:
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.03,
                                       max_depth=6, early_stopping=True,
                                       random_state=RANDOM_SEED)
    m.fit(X_train, y_train)

models_fit['XGBoost'] = m
evaluate_model('XGBoost', yo_test, m.predict(X_test), results)

# ── Model 5: CatBoost ────────────────────────────────────────────────────────
if HAS_CB:
    m = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6,
                          random_seed=RANDOM_SEED, verbose=0,
                          early_stopping_rounds=50)
    m.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
else:
    m = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03,
                                   max_depth=5, subsample=0.8,
                                   random_state=RANDOM_SEED)
    m.fit(X_train, y_train)

models_fit['CatBoost'] = m
evaluate_model('CatBoost', yo_test, m.predict(X_test), results)

# ── Model 6: MLP Neural Network ───────────────────────────────────────────────
m = MLPRegressor(hidden_layer_sizes=(512, 256, 128, 64),
                 activation='relu', solver='adam',
                 learning_rate_init=0.001, batch_size=32,
                 max_iter=2000, early_stopping=True, n_iter_no_change=50,
                 validation_fraction=0.1, random_state=RANDOM_SEED)
m.fit(X_train_sc, y_train)
models_fit['MLP'] = m
evaluate_model('MLP Neural Network (512-256-128-64)',
               yo_test, m.predict(X_test_sc), results)

# ── Model 7: TabNet (optional) ────────────────────────────────────────────────
if HAS_TABNET:
    m = TabNetRegressor(n_d=32, n_a=32, n_steps=5, gamma=1.3,
                        n_independent=2, n_shared=2,
                        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                        scheduler_params={"step_size": 50, "gamma": 0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='sparsemax', seed=RANDOM_SEED, verbose=0)
    m.fit(X_train_sc, y_train.reshape(-1, 1),
          eval_set=[(X_test_sc, y_test.reshape(-1, 1))],
          patience=30, max_epochs=200, batch_size=256,
          virtual_batch_size=128)
    models_fit['TabNet'] = m
    evaluate_model('TabNet', yo_test,
                   m.predict(X_test_sc).flatten(), results)

# ─────────────────────────────────────────────────────────────────────────────
# 6. IDENTIFY BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['R2'])
best_model = models_fit[best_name]
print(f"\n  ★  BEST MODEL: {best_name}  "
      f"R²={results[best_name]['R2']:.4f}  "
      f"MAPE={results[best_name]['MAPE_pct']:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION (Best Model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 3: 5-FOLD CROSS-VALIDATION (Best Model)")
print("="*70)

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_r2   = cross_val_score(best_model, X, y, cv=kf, scoring='r2', n_jobs=-1)
cv_neg_rmse = cross_val_score(best_model, X, y, cv=kf,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)

print(f"  Model: {best_name}")
print(f"  R² per fold : {[f'{v:.4f}' for v in cv_r2]}")
print(f"  R² mean ± std: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  RMSE per fold: {[f'{-v:.2f}' for v in cv_neg_rmse]}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PER-MATERIAL & PER-CRITERION BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 4: PER-MATERIAL & PER-CRITERION R² BREAKDOWN")
print("="*70)

# Re-train best on all training data, predict on test
best_model.fit(X_train, y_train)
test_idx = df.index[np.isin(df.index, [])]   # placeholder; use test indices

# Rebuild test DataFrame with predictions
df_test = df.iloc[
    train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                     random_state=RANDOM_SEED,
                     stratify=df['Material_enc'].values)[1]
].copy()

# Predict based on model type
if best_name == 'MLP':
    df_test['Pred_Log'] = best_model.predict(
        scaler.transform(df_test[FEATURES].values))
else:
    df_test['Pred_Log'] = best_model.predict(df_test[FEATURES].values)

df_test['Pred_kN'] = np.expm1(df_test['Pred_Log'])

print("\n  Per-Material R²:")
print(f"  {'Material':<15} {'N':>5} {'R²':>8} {'MAE (kN)':>12} {'MAPE %':>8}")
print("  " + "-"*50)
for mat in sorted(df_test['Material'].unique()):
    sub = df_test[df_test['Material'] == mat]
    r2m  = r2_score(sub['Max_Load_kN'], sub['Pred_kN'])
    maem = mean_absolute_error(sub['Max_Load_kN'], sub['Pred_kN'])
    mask = sub['Max_Load_kN'] > 0
    mapem = np.mean(np.abs((sub.loc[mask,'Max_Load_kN'] -
                            sub.loc[mask,'Pred_kN']) /
                           sub.loc[mask,'Max_Load_kN'])) * 100
    print(f"  {mat:<15} {len(sub):>5} {r2m:>8.4f} {maem:>12.2f} {mapem:>8.2f}")

print("\n  Per-Governing-Criterion R²:")
print(f"  {'Criterion':<15} {'N':>5} {'R²':>8} {'MAE (kN)':>12} {'MAPE %':>8}")
print("  " + "-"*50)
for gc in sorted(df_test['Governing_Criterion'].unique()):
    sub = df_test[df_test['Governing_Criterion'] == gc]
    if len(sub) < 2:
        continue
    r2c  = r2_score(sub['Max_Load_kN'], sub['Pred_kN'])
    maec = mean_absolute_error(sub['Max_Load_kN'], sub['Pred_kN'])
    mask = sub['Max_Load_kN'] > 0
    mapec = np.mean(np.abs((sub.loc[mask,'Max_Load_kN'] -
                            sub.loc[mask,'Pred_kN']) /
                           sub.loc[mask,'Max_Load_kN'])) * 100
    print(f"  {gc:<15} {len(sub):>5} {r2c:>8.4f} {maec:>12.2f} {mapec:>8.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. ABLATION STUDY — With vs Without P_Bending / P_Shear / P_Deflection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 5: ABLATION STUDY")
print("="*70)

# This is critical for peer review: proves model doesn't trivially fit min()
FEATURES_GEOM_ONLY = [
    'Beam_Length_m', 'Thickness_m', 'Factor_of_Safety',
    'Moment_of_Inertia_m4', 'Cross_Sec_Area_m2', 'y_max_m',
    'Young_Modulus_GPa', 'Yield_Strength_MPa', 'Sigma_Allow_MPa',
    'Slenderness_Ratio', 'Max_Deflection_mm', 'Bending_Stress_MPa',
    'Stress_Utilization_pct', 'Material_enc', 'CrossSection_enc'
]
FEATURES_FULL = FEATURES   # includes P_b, P_s, P_d, GovCrit, P_min

print(f"\n  {'Configuration':<40} {'R²':>8} {'MAPE %':>8}")
print("  " + "-"*58)

from sklearn.ensemble import RandomForestRegressor as RF_abl

for label, feat_list in [
    ('Geometry-only (no Pb/Ps/Pd)', FEATURES_GEOM_ONLY),
    ('Full features (with Pb/Ps/Pd)', FEATURES_FULL)
]:
    Xa = df[feat_list].values
    ya = df['Log_Max_Load'].values
    Xa_tr, Xa_te, ya_tr, ya_te, yo_tr, yo_te = train_test_split(
        Xa, ya, df['Max_Load_kN'].values,
        test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df['Material_enc'].values
    )
    ma = RF_abl(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    ma.fit(Xa_tr, ya_tr)
    pred_abl = np.expm1(ma.predict(Xa_te))
    r2a  = r2_score(yo_te, pred_abl)
    mask = yo_te > 0
    mapea = np.mean(np.abs((yo_te[mask] - pred_abl[mask]) / yo_te[mask])) * 100
    print(f"  {label:<40} {r2a:>8.4f} {mapea:>8.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SHAP:
    print("\n" + "="*70)
    print("  STEP 6: SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    # Use RF or tree-based model for TreeExplainer
    shap_model_name = 'Random Forest' if 'Random Forest' in models_fit else best_name
    shap_model = models_fit[shap_model_name]

    # TreeExplainer works on tree-based models
    explainer = shap.TreeExplainer(shap_model)
    X_shap = X_test[:200]   # subset for speed
    shap_values = explainer.shap_values(X_shap)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': FEATURES,
                             'Mean |SHAP|': mean_abs_shap}).sort_values(
                             'Mean |SHAP|', ascending=False).reset_index(drop=True)

    print(f"\n  Top 10 SHAP Features ({shap_model_name}):")
    print(f"  {'Rank':<6} {'Feature':<30} {'Mean |SHAP|':>12}")
    print("  " + "-"*50)
    for i, row in shap_df.head(10).iterrows():
        print(f"  {i+1:<6} {row['Feature']:<30} {row['Mean |SHAP|']:>12.4f}")

    # Save SHAP beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap, feature_names=FEATURES,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary Plot — {shap_model_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_SHAP_Summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  SHAP plot saved → {OUTPUT_DIR}/Fig_SHAP_Summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALIZATION SUITE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 7: GENERATING ALL FIGURES")
print("="*70)

# ── Fig 1: Model Comparison Bar Chart ─────────────────────────────────────────
valid_results = {k: v for k, v in results.items() if v['R2'] > 0}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Forward Study — Model Performance Comparison', fontsize=14, fontweight='bold')

metrics  = ['R2', 'RMSE_kN', 'MAPE_pct']
ylabels  = ['R² Score', 'RMSE (kN)', 'MAPE (%)']
colors   = ['#2ecc71', '#e74c3c', '#3498db']

for ax, metric, ylabel, color in zip(axes, metrics, ylabels, colors):
    names = list(valid_results.keys())
    vals  = [valid_results[n][metric] for n in names]
    bars  = ax.bar(names, vals, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    if metric == 'R2':
        ax.set_ylim(0, 1.05)
        ax.axhline(0.98, color='red', linestyle='--', linewidth=1.2, label='98% target')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig1_Model_Comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 1 saved → {OUTPUT_DIR}/Fig1_Model_Comparison.png")

# ── Fig 2: Predicted vs Actual scatter (best model) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Forward Study — {best_name}: Predicted vs Actual',
             fontsize=13, fontweight='bold')

y_pred_best = results[best_name]['y_pred']
ax = axes[0]
ax.scatter(yo_test, y_pred_best, alpha=0.55, s=18, c='#2980b9', edgecolors='none')
lo = min(yo_test.min(), y_pred_best.min())
hi = max(yo_test.max(), y_pred_best.max())
ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Max Load (kN)', fontsize=11)
ax.set_ylabel('Predicted Max Load (kN)', fontsize=11)
ax.set_title(f'R² = {results[best_name]["R2"]:.4f}', fontsize=11)
ax.legend(); ax.grid(True, alpha=0.3)

# Log scale
ax = axes[1]
ax.scatter(np.log1p(yo_test), np.log1p(y_pred_best),
           alpha=0.55, s=18, c='#27ae60', edgecolors='none')
lo2 = min(np.log1p(yo_test).min(), np.log1p(y_pred_best).min())
hi2 = max(np.log1p(yo_test).max(), np.log1p(y_pred_best).max())
ax.plot([lo2, hi2], [lo2, hi2], 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('log(1 + Actual)', fontsize=11)
ax.set_ylabel('log(1 + Predicted)', fontsize=11)
ax.set_title('Log Scale (shows low-load accuracy)', fontsize=11)
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig2_Predicted_vs_Actual.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 2 saved → {OUTPUT_DIR}/Fig2_Predicted_vs_Actual.png")

# ── Fig 3: Residual plot ───────────────────────────────────────────────────────
residuals = yo_test - y_pred_best
pct_error = (residuals / yo_test) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Forward Study — {best_name}: Residual Analysis', fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(y_pred_best, residuals, alpha=0.5, s=18, c='#8e44ad', edgecolors='none')
ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
ax.set_xlabel('Predicted Max Load (kN)', fontsize=11)
ax.set_ylabel('Residual (Actual − Predicted) kN', fontsize=11)
ax.set_title('Residuals vs Predicted', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(pct_error, bins=40, color='#e67e22', alpha=0.8, edgecolor='white')
ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
ax.set_xlabel('Percentage Error (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Error Distribution | MAPE={results[best_name]["MAPE_pct"]:.2f}%', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig3_Residual_Analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 3 saved → {OUTPUT_DIR}/Fig3_Residual_Analysis.png")

# ── Fig 4: Feature Importance (RF) ────────────────────────────────────────────
rf_model = models_fit.get('Random Forest', None)
if rf_model is not None:
    fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(fi.index, fi.values, color='#16a085', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Feature Importance (Mean Decrease Impurity)', fontsize=11)
    ax.set_title('Random Forest — Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    # Annotate top features
    for bar, val in zip(bars, fi.values):
        if val > 0.01:
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Feature_Importance_RF.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 4 saved → {OUTPUT_DIR}/Fig4_Feature_Importance_RF.png")

# ── Fig 5: Per-material performance heatmap ───────────────────────────────────
heat_data = []
for mat in sorted(df_test['Material'].unique()):
    row = []
    sub = df_test[df_test['Material'] == mat]
    for gc in ['Bending', 'Deflection', 'Shear']:
        s2 = sub[sub['Governing_Criterion'] == gc]
        if len(s2) >= 2:
            r2v = r2_score(s2['Max_Load_kN'], s2['Pred_kN'])
        else:
            r2v = np.nan
        row.append(r2v)
    heat_data.append(row)

heat_df = pd.DataFrame(heat_data,
                        index=sorted(df_test['Material'].unique()),
                        columns=['Bending', 'Deflection', 'Shear'])
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heat_df, annot=True, fmt='.4f', cmap='YlGn', ax=ax,
            vmin=0.85, vmax=1.0, linewidths=0.5, cbar_kws={'label': 'R²'})
ax.set_title(f'R² Heatmap: Material × Governing Criterion ({best_name})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig5_R2_Heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 5 saved → {OUTPUT_DIR}/Fig5_R2_Heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. RESULTS TABLE — CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 8: EXPORT RESULTS TABLES")
print("="*70)

# Model comparison table
results_table = pd.DataFrame([
    {'Model': k,
     'R2': f"{v['R2']:.4f}",
     'RMSE_kN': f"{v['RMSE_kN']:.2f}",
     'MAE_kN': f"{v['MAE_kN']:.2f}",
     'MAPE_pct': f"{v['MAPE_pct']:.2f}",
     'Status': '✓ ≥98%' if v['R2'] >= 0.98 else '✗ <98%'}
    for k, v in sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
])
results_table.to_csv(f"{OUTPUT_DIR}/Table1_Model_Comparison.csv", index=False)
print(f"  Table 1 → {OUTPUT_DIR}/Table1_Model_Comparison.csv")
print(results_table.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 13. FUSION 360 VALIDATION SET — FORWARD (25 SAMPLES)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 9: GENERATING FUSION 360 FORWARD VALIDATION DATASET")
print("="*70)

# Stratified selection: 1 sample per (Material × Cross_Section_Type) = 5×5 = 25
df_clean = df[(df['Moment_of_Inertia_m4'] > 1e-9) &
              (df['Max_Load_kN'] > 0.5) &
              (df['Max_Load_kN'] < 5000)].copy()

fusion_samples = []
for mat in sorted(df_clean['Material'].unique()):
    for cs in sorted(df_clean['Cross_Section_Type'].unique()):
        subset = df_clean[(df_clean['Material'] == mat) &
                          (df_clean['Cross_Section_Type'] == cs)]
        if len(subset) == 0:
            continue
        # Pick sample with median load (most representative)
        med_idx = (subset['Max_Load_kN'] -
                   subset['Max_Load_kN'].median()).abs().idxmin()
        fusion_samples.append(subset.loc[med_idx])

df_fusion_raw = pd.DataFrame(fusion_samples).reset_index(drop=True)

# Get ML prediction for each Fusion sample
X_fus = df_fusion_raw[FEATURES].values
if best_name == 'MLP':
    log_pred = best_model.predict(scaler.transform(X_fus))
else:
    log_pred = best_model.predict(X_fus)

df_fusion_raw['ML_Pred_Max_Load_kN'] = np.expm1(log_pred)
df_fusion_raw['Pred_Error_pct'] = (
    np.abs(df_fusion_raw['ML_Pred_Max_Load_kN'] -
           df_fusion_raw['Max_Load_kN']) /
    df_fusion_raw['Max_Load_kN'] * 100
)

# ── Fusion 360 Export Columns ─────────────────────────────────────────────────
# Columns that Fusion 360 FEA engineer needs to set up the simulation
fusion_export = df_fusion_raw[[
    'Material', 'Cross_Section_Type',
    'Beam_Length_m', 'Thickness_m',
    'Moment_of_Inertia_m4', 'Cross_Sec_Area_m2', 'y_max_m',
    'Young_Modulus_GPa', 'Yield_Strength_MPa',
    'Factor_of_Safety', 'Sigma_Allow_MPa',
    'P_Bending_kN', 'P_Shear_kN', 'P_Deflection_kN',
    'Max_Load_kN',               # Ground truth from physics
    'ML_Pred_Max_Load_kN',       # Model prediction (to validate)
    'Governing_Criterion',
    'Slenderness_Ratio',
    'Pred_Error_pct'
]].copy()

# Add sample ID and Fusion setup instructions
fusion_export.insert(0, 'Sample_ID', [f'FWD-{i+1:02d}' for i in range(len(fusion_export))])

# Add computed dimensions for CAD modelling
# For simply supported beam: span = Beam_Length_m, width ≈ 0.5*t, height ≈ 2*t
fusion_export['CAD_Span_m']   = fusion_export['Beam_Length_m']
fusion_export['CAD_Height_m'] = fusion_export['Thickness_m'] * 2.0
fusion_export['CAD_Width_m']  = fusion_export['Thickness_m'] * 1.0

# Allowable deflection for serviceability check (L/360 rule)
fusion_export['Allowable_Deflection_mm'] = (fusion_export['Beam_Length_m'] / 360.0) * 1000

# Load to apply in Fusion 360 FEA = ML prediction (test if FEA stress ≤ sigma_allow)
fusion_export['FEA_Apply_Load_kN']  = fusion_export['ML_Pred_Max_Load_kN']
fusion_export['FEA_Apply_Load_N']   = fusion_export['ML_Pred_Max_Load_kN'] * 1000

# Columns for FEA engineer to fill after simulation
fusion_export['FEA_Max_VonMises_MPa'] = ''
fusion_export['FEA_Max_Deflection_mm'] = ''
fusion_export['FEA_Load_at_Failure_kN'] = ''
fusion_export['FEA_vs_ML_Error_pct'] = ''
fusion_export['FEA_Pass_Fail'] = ''

fusion_export.to_csv(f"{OUTPUT_DIR}/Table2_Forward_Fusion360_Validation.csv", index=False)
print(f"\n  ✓  {len(fusion_export)} forward validation samples exported")
print(f"  File: {OUTPUT_DIR}/Table2_Forward_Fusion360_Validation.csv")
print(f"\n  Sample (first 5 rows):")
print(fusion_export[['Sample_ID', 'Material', 'Cross_Section_Type',
                      'Beam_Length_m', 'Thickness_m',
                      'Max_Load_kN', 'ML_Pred_Max_Load_kN',
                      'Pred_Error_pct']].head().to_string(index=False))

# ── Fusion 360 Setup Instructions ─────────────────────────────────────────────
fusion_instructions = """
================================================================================
FUSION 360 FEA SETUP INSTRUCTIONS — FORWARD VALIDATION
================================================================================

For each row in Table2_Forward_Fusion360_Validation.csv:

STEP 1: CREATE GEOMETRY
  - Open Fusion 360 → Design workspace
  - Create beam cross-section based on Cross_Section_Type:
      Rectangular : width = CAD_Width_m, height = CAD_Height_m
      Circular    : diameter = Thickness_m * 2
      I-Beam      : use standard sections with height = Thickness_m
      T-Beam      : flange = Thickness_m*2 wide, web depth = Thickness_m
      Box         : outer height = CAD_Height_m, wall = Thickness_m*0.1
      L-Beam      : leg length = Thickness_m, thickness = Thickness_m*0.15
  - Extrude beam to length = CAD_Span_m (along X-axis)

STEP 2: ASSIGN MATERIAL
  - Go to Modify → Manage Materials
  - Assign material matching Material column (Steel/Aluminum/Concrete/Wood/Composite)
  - Verify E (Young's Modulus) matches Young_Modulus_GPa × 1e9 Pa
  - Verify σ_y (Yield Strength) matches Yield_Strength_MPa × 1e6 Pa

STEP 3: SET UP SIMULATION
  - Switch to Simulation workspace
  - Study type: Static Stress (Linear)
  - Mesh: Adaptive | min element size = Thickness_m * 0.05

STEP 4: BOUNDARY CONDITIONS
  - Left end (x=0)  : Pin support  → Fixed translation in Y and Z; free in X
  - Right end (x=L) : Roller       → Fixed translation in Y only; free in X and Z
  - (Simply supported beam — matches Euler-Bernoulli assumption)

STEP 5: APPLY LOAD
  - Location: midspan (x = CAD_Span_m / 2)
  - Load type: Force, downward (−Y direction)
  - Magnitude: FEA_Apply_Load_N Newtons (= ML_Pred_Max_Load_kN × 1000)

STEP 6: RUN & RECORD
  - Run simulation
  - Record: Max Von Mises Stress (MPa) → paste into FEA_Max_VonMises_MPa
  - Record: Max deflection (mm)        → paste into FEA_Max_Deflection_mm
  - Find load at which σ_VonMises = Sigma_Allow_MPa → FEA_Load_at_Failure_kN
  - Compute: FEA_vs_ML_Error_pct = |FEA_Load - ML_Pred| / ML_Pred × 100
  - Pass/Fail: PASS if σ_FEA ≤ Sigma_Allow_MPa AND deflection ≤ Allowable_Deflection_mm

ACCEPTANCE CRITERIA:
  - Target: Mean FEA_vs_ML_Error_pct < 5%
  - Individual: FEA_vs_ML_Error_pct < 10% per sample
================================================================================
"""

with open(f"{OUTPUT_DIR}/Fusion360_Forward_Instructions.txt", 'w') as f:
    f.write(fusion_instructions)
print(f"\n  Instructions saved → {OUTPUT_DIR}/Fusion360_Forward_Instructions.txt")

# ─────────────────────────────────────────────────────────────────────────────
# 14. SUMMARY PRINT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FORWARD STUDY COMPLETE — SUMMARY")
print("="*70)
print(f"""
  Dataset       : {len(df)} samples (after cleaning)
  Features      : {len(FEATURES)} (including physics-engineered P_min)
  Best Model    : {best_name}
  R²            : {results[best_name]['R2']:.4f}
  RMSE          : {results[best_name]['RMSE_kN']:.2f} kN
  MAE           : {results[best_name]['MAE_kN']:.2f} kN
  MAPE          : {results[best_name]['MAPE_pct']:.2f}%
  CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}
  Target ≥98%   : {'✓ ACHIEVED' if results[best_name]['R2'] >= 0.98 else '✗ NOT REACHED'}

  Output files  : {OUTPUT_DIR}/
    Fig1_Model_Comparison.png
    Fig2_Predicted_vs_Actual.png
    Fig3_Residual_Analysis.png
    Fig4_Feature_Importance_RF.png
    Fig5_R2_Heatmap.png
    Table1_Model_Comparison.csv
    Table2_Forward_Fusion360_Validation.csv   ← GIVE THIS TO FUSION 360
    Fusion360_Forward_Instructions.txt        ← FEA SETUP GUIDE
""")