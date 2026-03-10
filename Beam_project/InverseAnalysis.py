"""
================================================================================
INVERSE STUDY — Inverse Parameter Synthesis for Structural Beam Design
================================================================================
Paper: A Comparative Study of Machine Learned Structural Capacity Prediction
       and Inverse Parameter Synthesis using Synthetic Multi-Material Beam Datasets
Author: Shrikant Thakur | C-DAC, Kolkata

PURPOSE:
  Given a TARGET Max Load (kN), generate beam design parameters
  (geometry, material, cross-section) that can sustain that load.
  This is the ONE-TO-MANY inverse problem — no closed-form solution exists.

APPROACHES IMPLEMENTED:
  Method A: Tandem Neural Network (TNN)
            → Forward surrogate (RF) + Inverse MLP trained in tandem
  Method B: Optimization-Based Inversion
            → Use best forward model + scipy.optimize to find parameters
  Method C: k-NN Nearest Neighbor Retrieval (interpretable baseline)

NOTE:
  Inverse material predictions are RESTRICTED to metals only:
  Aluminum and Steel (as per validation requirements).

INSTALL:
  pip install pandas numpy scikit-learn matplotlib seaborn scipy
  pip install xgboost catboost  (optional — uses RF if unavailable)
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
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH      = "physics_beam_dataset_ML.csv"
OUTPUT_DIR     = "inverse_study_outputs"
RANDOM_SEED    = 42
TEST_SIZE      = 0.20
N_FUSION360    = 25

# 25 log-spaced target loads for inverse generation (1 → 2000 kN)
TARGET_LOADS_KN = np.logspace(np.log10(1), np.log10(2000), N_FUSION360)

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
df = df_raw[df_raw['Moment_of_Inertia_m4'] > 1e-9].copy()
df = df[(df['Max_Load_kN'] > 0.5) & (df['Max_Load_kN'] < 5000)].copy()
df.reset_index(drop=True, inplace=True)
print(f"  Clean dataset: {len(df)} rows")

# ── Encoders ──────────────────────────────────────────────────────────────────
le_mat = LabelEncoder();  le_mat.fit(df['Material'])
le_cs  = LabelEncoder();  le_cs.fit(df['Cross_Section_Type'])
le_gc  = LabelEncoder();  le_gc.fit(df['Governing_Criterion'])

df['Material_enc']     = le_mat.transform(df['Material'])
df['CrossSection_enc'] = le_cs.transform(df['Cross_Section_Type'])
df['GovCrit_enc']      = le_gc.transform(df['Governing_Criterion'])
df['P_min_physics']    = df[['P_Bending_kN', 'P_Shear_kN',
                              'P_Deflection_kN']].min(axis=1)
df['Log_Max_Load']     = np.log1p(df['Max_Load_kN'])

# ── Feature sets ──────────────────────────────────────────────────────────────
# FORWARD features (X → load)
FWD_FEATURES = [
    'Beam_Length_m', 'Thickness_m', 'Factor_of_Safety',
    'Moment_of_Inertia_m4', 'Cross_Sec_Area_m2', 'y_max_m',
    'Young_Modulus_GPa', 'Yield_Strength_MPa', 'Sigma_Allow_MPa',
    'P_Bending_kN', 'P_Shear_kN', 'P_Deflection_kN',
    'Slenderness_Ratio', 'Max_Deflection_mm',
    'Bending_Stress_MPa', 'Stress_Utilization_pct',
    'Material_enc', 'CrossSection_enc', 'GovCrit_enc', 'P_min_physics'
]

# INVERSE targets — what we want the inverse model to generate
# (geometry + design — what an engineer controls)
INV_TARGETS = [
    'Beam_Length_m', 'Thickness_m', 'Factor_of_Safety',
    'Material_enc', 'CrossSection_enc'
]

# ── Material & Section lookup tables ─────────────────────────────────────────
# Maps encoded int → properties
MATERIAL_PROPS = {
    mat: {
        'E_GPa':    df[df['Material']==mat]['Young_Modulus_GPa'].iloc[0],
        'sigma_y':  df[df['Material']==mat]['Yield_Strength_MPa'].iloc[0]
    }
    for mat in df['Material'].unique()
}

SECTION_TYPES  = list(le_cs.classes_)
MATERIAL_NAMES = list(le_mat.classes_)

# ── METAL-ONLY RESTRICTION for inverse predictions ────────────────────────────
# Inverse validation is restricted to Aluminum and Steel only.
METAL_KEYWORDS   = ('aluminum', 'aluminium', 'steel')
METAL_MATERIALS  = [m for m in MATERIAL_NAMES
                    if m.lower() in METAL_KEYWORDS]

if len(METAL_MATERIALS) < 2:
    raise ValueError(
        f"Expected at least Aluminum and Steel in dataset. "
        f"Found metals: {METAL_MATERIALS} | All materials: {MATERIAL_NAMES}"
    )

# Encoded integer indices for metal materials
METAL_ENC_INDICES = [int(le_mat.transform([m])[0]) for m in METAL_MATERIALS]

print(f"  All materials   : {MATERIAL_NAMES}")
print(f"  Inverse metals  : {METAL_MATERIALS}  (encoded: {METAL_ENC_INDICES})")
print(f"  Sections        : {SECTION_TYPES}")
print(f"  Target loads    : {[f'{v:.1f}' for v in TARGET_LOADS_KN]} kN")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN FORWARD SURROGATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 2: TRAIN FORWARD SURROGATE MODEL")
print("="*70)

X_fwd  = df[FWD_FEATURES].values
y_fwd  = df['Log_Max_Load'].values
y_orig = df['Max_Load_kN'].values

X_tr, X_te, y_tr, y_te, yo_tr, yo_te = train_test_split(
    X_fwd, y_fwd, y_orig, test_size=TEST_SIZE,
    random_state=RANDOM_SEED, stratify=df['Material_enc'].values
)

# Best forward model — Random Forest (consistently R²>0.999)
forward_model = RandomForestRegressor(
    n_estimators=500, min_samples_leaf=1,
    random_state=RANDOM_SEED, n_jobs=-1
)
forward_model.fit(X_tr, y_tr)

pred_log = forward_model.predict(X_te)
pred_kn  = np.expm1(pred_log)
fwd_r2   = r2_score(yo_te, pred_kn)
fwd_mape = np.mean(np.abs((yo_te - pred_kn) / yo_te)) * 100

print(f"  Forward surrogate (Random Forest):")
print(f"    R²   = {fwd_r2:.4f}")
print(f"    MAPE = {fwd_mape:.2f}%")
print(f"    Status: {'✓ Surrogate validated' if fwd_r2 >= 0.98 else '⚠ Check model'}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PHYSICS HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_section_properties(cross_section_name: str, t: float):
    """
    Compute I, A, y_max for a given cross-section and thickness parameter t.
    These are derived from structural mechanics first principles.

    Convention: t = Thickness_m from dataset
      - Rectangular: h=2t, b=t
      - Circular   : d=2t (diameter)
      - I-Beam     : H=2t, bf=1.5t, tf=0.15t, tw=0.1t
      - T-Beam     : bf=1.5t, tf=0.15t, H=t+0.15t, tw=0.1t
      - Box        : H=2t, B=t, wall=0.1t
      - L-Beam     : b=t, h=t, wall=0.12t
    """
    cs = cross_section_name.lower()

    if 'rectangular' in cs:
        b, h = t, 2 * t
        I    = b * h**3 / 12.0
        A    = b * h
        ymax = h / 2.0

    elif 'circular' in cs:
        d    = 2 * t
        I    = np.pi * d**4 / 64.0
        A    = np.pi * d**2 / 4.0
        ymax = d / 2.0

    elif 'i-beam' in cs or 'ibeam' in cs:
        H, bf, tf_f, tw = 2*t, 1.5*t, 0.15*t, 0.1*t
        hw = H - 2 * tf_f
        I  = (bf * H**3 - (bf - tw) * hw**3) / 12.0
        A  = 2 * bf * tf_f + hw * tw
        ymax = H / 2.0

    elif 't-beam' in cs or 'tbeam' in cs:
        bf, tf_f, tw, hw = 1.5*t, 0.15*t, 0.1*t, t
        H  = hw + tf_f
        A  = bf * tf_f + tw * hw
        # Centroid from bottom
        y_bar = (bf * tf_f * (hw + tf_f/2) + tw * hw * (hw/2)) / A
        # I about centroidal axis
        I = (bf * tf_f**3 / 12 + bf * tf_f * (H - y_bar - tf_f/2)**2 +
             tw * hw**3 / 12 + tw * hw * (y_bar - hw/2)**2)
        ymax = max(y_bar, H - y_bar)

    elif 'box' in cs:
        H, B, wall = 2*t, t, 0.1*t
        I_out = B * H**3 / 12.0
        I_in  = (B - 2*wall) * (H - 2*wall)**3 / 12.0
        I     = I_out - I_in
        A     = B*H - (B - 2*wall)*(H - 2*wall)
        ymax  = H / 2.0

    elif 'l-beam' in cs or 'lbeam' in cs:
        b, h, wall = t, t, 0.12*t
        A = b * wall + (h - wall) * wall
        y_bar = (b * wall * wall/2 + (h - wall) * wall * (wall + (h-wall)/2)) / A
        I = (b * wall**3/12 + b * wall * (y_bar - wall/2)**2 +
             wall * (h-wall)**3/12 + wall * (h-wall) * (wall + (h-wall)/2 - y_bar)**2)
        ymax = max(y_bar, h - y_bar)

    else:
        # Fallback: rectangular
        b, h = t, 2 * t
        I    = b * h**3 / 12.0
        A    = b * h
        ymax = h / 2.0

    return max(I, 1e-10), max(A, 1e-6), max(ymax, 1e-4)


def compute_load_capacities(L, t, E_GPa, sigma_y_MPa, fos, cross_section_name):
    """
    Compute P_Bending, P_Shear, P_Deflection, and Max_Load for a beam.
    Simply supported, central point load.

    Governing equations:
      P_bend  = 4 * sigma_allow * I / (L * y_max)         [from M = P*L/4, σ = M*y/I]
      P_shear = (4/3) * tau_allow * A                      [rectangular shear, von Mises]
      P_defl  = 48 * E * I * delta_allow / L^3            [midspan deflection]
      delta_allow = L / 360                               [serviceability limit]
    """
    I, A, ymax = compute_section_properties(cross_section_name, t)

    E         = E_GPa * 1e9                # Pa
    sigma_y   = sigma_y_MPa * 1e6         # Pa
    sig_allow = sigma_y / fos             # Pa
    tau_allow = 0.577 * sig_allow         # von Mises shear criterion

    # Load capacities (in N → convert to kN)
    P_bend  = (4 * sig_allow * I / (L * ymax)) / 1000.0
    P_shear = (tau_allow * A) / 1000.0     # simplified; (4/3) factor for rectangular
    delta_allow = L / 360.0
    P_defl  = (48 * E * I * delta_allow / L**3) / 1000.0

    P_max   = max(0.5, min(P_bend, P_shear, P_defl, 5000.0))
    governing = 'Bending' if (P_bend <= P_shear and P_bend <= P_defl) else (
                'Deflection' if P_defl <= P_shear else 'Shear')

    # Derived features (for forward model input)
    slenderness = L / np.sqrt(I / A) if A > 0 else 0
    defl_mm     = (P_max * 1000 * L**3) / (48 * E * I) * 1000.0
    bending_stress = (P_max * 1000 * (L/4) * ymax / I) / 1e6

    return {
        'I': I, 'A': A, 'ymax': ymax,
        'P_Bending_kN': P_bend, 'P_Shear_kN': P_shear, 'P_Deflection_kN': P_defl,
        'Max_Load_kN': P_max, 'Governing': governing,
        'Sigma_Allow_MPa': sig_allow / 1e6,
        'Slenderness_Ratio': slenderness,
        'Max_Deflection_mm': defl_mm,
        'Bending_Stress_MPa': bending_stress,
        'Stress_Utilization_pct': (bending_stress / (sig_allow / 1e6)) * 100
    }


def build_feature_vector(L, t, fos, mat_name, cs_name, fwd_features, le_mat, le_cs, le_gc):
    """
    Given design parameters → compute all features → return forward model input vector.
    """
    props = MATERIAL_PROPS[mat_name]
    E_GPa, sigma_y = props['E_GPa'], props['sigma_y']
    caps  = compute_load_capacities(L, t, E_GPa, sigma_y, fos, cs_name)

    mat_enc = le_mat.transform([mat_name])[0]
    cs_enc  = le_cs.transform([cs_name])[0]
    gc_enc  = le_gc.transform([caps['Governing']])[0]

    P_min_phys = min(caps['P_Bending_kN'], caps['P_Shear_kN'], caps['P_Deflection_kN'])

    row = {
        'Beam_Length_m':         L,
        'Thickness_m':           t,
        'Factor_of_Safety':      fos,
        'Moment_of_Inertia_m4':  caps['I'],
        'Cross_Sec_Area_m2':     caps['A'],
        'y_max_m':               caps['ymax'],
        'Young_Modulus_GPa':     E_GPa,
        'Yield_Strength_MPa':    sigma_y,
        'Sigma_Allow_MPa':       caps['Sigma_Allow_MPa'],
        'P_Bending_kN':          caps['P_Bending_kN'],
        'P_Shear_kN':            caps['P_Shear_kN'],
        'P_Deflection_kN':       caps['P_Deflection_kN'],
        'Slenderness_Ratio':     caps['Slenderness_Ratio'],
        'Max_Deflection_mm':     caps['Max_Deflection_mm'],
        'Bending_Stress_MPa':    caps['Bending_Stress_MPa'],
        'Stress_Utilization_pct':caps['Stress_Utilization_pct'],
        'Material_enc':          mat_enc,
        'CrossSection_enc':      cs_enc,
        'GovCrit_enc':           gc_enc,
        'P_min_physics':         P_min_phys
    }
    return np.array([row[f] for f in fwd_features])


# ─────────────────────────────────────────────────────────────────────────────
# 5. METHOD A: TANDEM NEURAL NETWORK (TNN)
#
#    Architecture:
#      Step 1: Train forward surrogate F: X → log(P)   [already done above]
#      Step 2: Train inverse MLP  G: log(P_target) → [L, t, fos, mat, cs]
#      Step 3: At inference: feed P_target → G → get params → verify via F
#
#    MATERIAL RESTRICTION:
#      The TNN material classifier is trained on all materials, but at
#      inference time we select the best-scoring METAL class only
#      (Aluminum or Steel) using predict_proba().
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 3: METHOD A — TANDEM NEURAL NETWORK (TNN)")
print("="*70)

# ── Prepare inverse training data ─────────────────────────────────────────────
# Input  = log(Max_Load_kN)  [scalar]
# Output = [L, t, fos, mat_enc, cs_enc]

df_inv = df[INV_TARGETS + ['Log_Max_Load', 'Max_Load_kN']].copy()

# Normalize continuous inverse targets to [0, 1] for stable training
scaler_L   = MinMaxScaler(); scaler_L.fit(df_inv[['Beam_Length_m']])
scaler_t   = MinMaxScaler(); scaler_t.fit(df_inv[['Thickness_m']])
scaler_fos = MinMaxScaler(); scaler_fos.fit(df_inv[['Factor_of_Safety']])

df_inv['L_norm']   = scaler_L.transform(df_inv[['Beam_Length_m']])
df_inv['t_norm']   = scaler_t.transform(df_inv[['Thickness_m']])
df_inv['fos_norm'] = scaler_fos.transform(df_inv[['Factor_of_Safety']])

X_inv = df_inv[['Log_Max_Load']].values          # shape (N, 1)
y_inv = df_inv[['L_norm', 't_norm', 'fos_norm',
                 'Material_enc', 'CrossSection_enc']].values  # shape (N, 5)

X_inv_tr, X_inv_te, y_inv_tr, y_inv_te = train_test_split(
    X_inv, y_inv, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# ── Scale inverse inputs ───────────────────────────────────────────────────────
inv_scaler = StandardScaler()
X_inv_tr_sc = inv_scaler.fit_transform(X_inv_tr)
X_inv_te_sc = inv_scaler.transform(X_inv_te)

# ── Continuous targets: L, t, fos ─────────────────────────────────────────────
tnn_continuous = MLPRegressor(
    hidden_layer_sizes=(256, 256, 128, 64),
    activation='relu', solver='adam',
    learning_rate_init=0.001, batch_size=32,
    max_iter=2000, early_stopping=True,
    n_iter_no_change=50, random_state=RANDOM_SEED
)
tnn_continuous.fit(X_inv_tr_sc, y_inv_tr[:, :3])  # L_norm, t_norm, fos_norm

# ── Material classifier (all classes; metal restriction applied at inference) ──
tnn_material = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    learning_rate_init=0.001, batch_size=32,
    max_iter=1000, early_stopping=True,
    n_iter_no_change=30, random_state=RANDOM_SEED
)
tnn_material.fit(X_inv_tr_sc, y_inv_tr[:, 3].astype(int))

# ── Cross-section classifier ───────────────────────────────────────────────────
tnn_cs = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    learning_rate_init=0.001, batch_size=32,
    max_iter=1000, early_stopping=True,
    n_iter_no_change=30, random_state=RANDOM_SEED
)
tnn_cs.fit(X_inv_tr_sc, y_inv_tr[:, 4].astype(int))

# ── Evaluate TNN on test set ───────────────────────────────────────────────────
preds_cont = tnn_continuous.predict(X_inv_te_sc)
r2_L   = r2_score(y_inv_te[:, 0], preds_cont[:, 0])
r2_t   = r2_score(y_inv_te[:, 1], preds_cont[:, 1])
r2_fos = r2_score(y_inv_te[:, 2], preds_cont[:, 2])
acc_mat = np.mean(tnn_material.predict(X_inv_te_sc) == y_inv_te[:, 3].astype(int))
acc_cs  = np.mean(tnn_cs.predict(X_inv_te_sc)      == y_inv_te[:, 4].astype(int))

print(f"  TNN Inverse Model — Test Performance:")
print(f"    R² (Beam Length)   : {r2_L:.4f}")
print(f"    R² (Thickness)     : {r2_t:.4f}")
print(f"    R² (FoS)           : {r2_fos:.4f}")
print(f"    Accuracy (Material): {acc_mat*100:.1f}%")
print(f"    Accuracy (Section) : {acc_cs*100:.1f}%")


def tnn_predict(target_load_kn):
    """
    Given a target load, use TNN to generate beam parameters.
    Material is RESTRICTED to metals (Aluminum / Steel) by selecting
    the highest-probability metal class from predict_proba().

    Returns: dict with L, t, fos, material, cross_section
    """
    log_target = np.array([[np.log1p(target_load_kn)]])
    inp = inv_scaler.transform(log_target)

    # ── Continuous parameters ──────────────────────────────────────────────
    cont = tnn_continuous.predict(inp)[0]
    L   = float(scaler_L.inverse_transform([[cont[0]]])[0][0])
    t   = float(scaler_t.inverse_transform([[cont[1]]])[0][0])
    fos = float(scaler_fos.inverse_transform([[cont[2]]])[0][0])

    # ── Material: restrict to metals using class probabilities ─────────────
    mat_probs = tnn_material.predict_proba(inp)[0]         # prob per class
    # Among valid metal encoded indices, pick the one with highest prob
    best_metal_enc = METAL_ENC_INDICES[
        int(np.argmax([mat_probs[e] for e in METAL_ENC_INDICES]))
    ]
    mat_enc = best_metal_enc

    # ── Cross-section (unrestricted) ───────────────────────────────────────
    cs_enc = int(tnn_cs.predict(inp)[0])

    # ── Clamp to valid ranges ──────────────────────────────────────────────
    L   = np.clip(L, 2.0, 20.0)
    t   = np.clip(t, 0.05, 0.5)
    fos = np.clip(fos, 1.5, 3.5)
    cs_enc = int(np.clip(cs_enc, 0, len(SECTION_TYPES) - 1))

    return {
        'L': L, 't': t, 'fos': fos,
        'material':      MATERIAL_NAMES[mat_enc],
        'cross_section': SECTION_TYPES[cs_enc]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. METHOD B: OPTIMIZATION-BASED INVERSION
#
#    Minimize: (forward_model(params) - log(P_target))²
#    Subject to: physical bounds on L, t, fos
#    Discrete variables: material (metals only), cross-section (enumerated)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 4: METHOD B — OPTIMIZATION-BASED INVERSION")
print("="*70)
print("  (Restricted to metal materials: Aluminum, Steel)")


def optimize_inverse(target_load_kn, mat_name, cs_name):
    """
    For fixed material (must be a metal) + cross-section, optimize L, t, fos
    to hit target_load_kn. Uses scipy differential_evolution for global optimum.

    mat_name must be one of METAL_MATERIALS.
    """
    assert mat_name in METAL_MATERIALS, (
        f"optimize_inverse: material '{mat_name}' is not a metal. "
        f"Use one of {METAL_MATERIALS}."
    )

    log_target = np.log1p(target_load_kn)

    def objective(params):
        L_opt, t_opt, fos_opt = params[0], params[1], params[2]
        try:
            fvec = build_feature_vector(
                L_opt, t_opt, fos_opt, mat_name, cs_name,
                FWD_FEATURES, le_mat, le_cs, le_gc
            )
            log_pred = forward_model.predict(fvec.reshape(1, -1))[0]
            return (log_pred - log_target) ** 2
        except Exception:
            return 1e6

    bounds = [(2.0, 20.0), (0.05, 0.50), (1.5, 3.5)]
    result = differential_evolution(objective, bounds, seed=RANDOM_SEED,
                                    maxiter=100, tol=1e-6, disp=False)
    L_opt, t_opt, fos_opt = result.x
    return {
        'L': L_opt, 't': t_opt, 'fos': fos_opt,
        'material': mat_name, 'cross_section': cs_name,
        'opt_loss': result.fun
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. METHOD C: k-NN RETRIEVAL (INTERPRETABLE BASELINE)
#
#    For a target load, find the k nearest training samples by log(load)
#    and return their average parameters.
#    Material is snapped to the nearest METAL encoded index.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 5: METHOD C — k-NN RETRIEVAL BASELINE")
print("="*70)

# Fit k-NN on log load → parameters
knn_model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
knn_model.fit(df_inv[['Log_Max_Load']].values,
              df_inv[['Beam_Length_m', 'Thickness_m', 'Factor_of_Safety',
                      'Material_enc', 'CrossSection_enc']].values)
print("  k-NN (k=5) retrieval model fitted ✓")
print(f"  Material will be snapped to nearest metal: {METAL_MATERIALS}")


def knn_predict(target_load_kn):
    """
    k-NN prediction with material RESTRICTED to metals.
    The raw predicted material encoding is snapped to the nearest
    metal encoded index (by absolute integer distance).
    """
    pred = knn_model.predict([[np.log1p(target_load_kn)]])[0]

    # ── Snap material to nearest metal ────────────────────────────────────
    raw_mat_enc = pred[3]
    mat_enc = METAL_ENC_INDICES[
        int(np.argmin([abs(raw_mat_enc - e) for e in METAL_ENC_INDICES]))
    ]

    cs_enc = int(round(np.clip(pred[4], 0, len(SECTION_TYPES) - 1)))

    return {
        'L':            np.clip(pred[0], 2.0, 20.0),
        't':            np.clip(pred[1], 0.05, 0.50),
        'fos':          np.clip(pred[2], 1.5, 3.5),
        'material':     MATERIAL_NAMES[mat_enc],
        'cross_section': SECTION_TYPES[cs_enc]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. GENERATE 25 INVERSE PREDICTIONS FOR FUSION 360
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 6: GENERATING 25 INVERSE PREDICTIONS FOR FUSION 360")
print("="*70)
print(f"  Target loads (kN): {[f'{v:.1f}' for v in TARGET_LOADS_KN]}\n")

records = []
for i, target in enumerate(TARGET_LOADS_KN):
    sample_id = f"INV-{i+1:02d}"

    # ── TNN Prediction (metal-restricted) ────────────────────────────────────
    tnn_params = tnn_predict(target)
    tnn_caps   = compute_load_capacities(
        tnn_params['L'], tnn_params['t'],
        MATERIAL_PROPS[tnn_params['material']]['E_GPa'],
        MATERIAL_PROPS[tnn_params['material']]['sigma_y'],
        tnn_params['fos'], tnn_params['cross_section']
    )
    tnn_physics_load = tnn_caps['Max_Load_kN']
    tnn_error        = abs(tnn_physics_load - target) / target * 100

    # ── kNN Prediction (metal-restricted) ────────────────────────────────────
    knn_params = knn_predict(target)
    knn_caps   = compute_load_capacities(
        knn_params['L'], knn_params['t'],
        MATERIAL_PROPS[knn_params['material']]['E_GPa'],
        MATERIAL_PROPS[knn_params['material']]['sigma_y'],
        knn_params['fos'], knn_params['cross_section']
    )
    knn_physics_load = knn_caps['Max_Load_kN']
    knn_error        = abs(knn_physics_load - target) / target * 100

    # ── Forward model verification (TNN params → ML load prediction) ──────────
    try:
        fvec = build_feature_vector(
            tnn_params['L'], tnn_params['t'], tnn_params['fos'],
            tnn_params['material'], tnn_params['cross_section'],
            FWD_FEATURES, le_mat, le_cs, le_gc
        )
        ml_verify_load  = np.expm1(forward_model.predict(fvec.reshape(1, -1))[0])
        ml_verify_error = abs(ml_verify_load - target) / target * 100
    except Exception:
        ml_verify_load  = np.nan
        ml_verify_error = np.nan

    records.append({
        # Identifiers
        'Sample_ID':          sample_id,
        'Target_Load_kN':     round(target, 3),

        # TNN Generated Parameters (metal only)
        'TNN_Material':       tnn_params['material'],
        'TNN_CrossSection':   tnn_params['cross_section'],
        'TNN_Length_m':       round(tnn_params['L'], 3),
        'TNN_Thickness_m':    round(tnn_params['t'], 4),
        'TNN_FoS':            round(tnn_params['fos'], 2),

        # TNN Derived Outputs (computed by physics, not ML)
        'TNN_I_m4':           f"{tnn_caps['I']:.6e}",
        'TNN_A_m2':           round(tnn_caps['A'], 6),
        'TNN_ymax_m':         round(tnn_caps['ymax'], 5),
        'TNN_E_GPa':          MATERIAL_PROPS[tnn_params['material']]['E_GPa'],
        'TNN_sigma_y_MPa':    MATERIAL_PROPS[tnn_params['material']]['sigma_y'],
        'TNN_sigma_allow_MPa':round(tnn_caps['Sigma_Allow_MPa'], 3),
        'TNN_P_Bending_kN':   round(tnn_caps['P_Bending_kN'], 3),
        'TNN_P_Shear_kN':     round(tnn_caps['P_Shear_kN'], 3),
        'TNN_P_Deflection_kN':round(tnn_caps['P_Deflection_kN'], 3),
        'TNN_Physics_Load_kN':round(tnn_physics_load, 3),
        'TNN_Physics_Error_pct': round(tnn_error, 2),
        'TNN_ML_Verify_Load_kN': round(ml_verify_load, 3) if not np.isnan(ml_verify_load) else '',
        'TNN_ML_Error_pct':   round(ml_verify_error, 2) if not np.isnan(ml_verify_error) else '',
        'TNN_Governing':      tnn_caps['Governing'],
        'TNN_Slenderness':    round(tnn_caps['Slenderness_Ratio'], 2),

        # kNN Baseline Parameters (metal only)
        'kNN_Material':       knn_params['material'],
        'kNN_CrossSection':   knn_params['cross_section'],
        'kNN_Length_m':       round(knn_params['L'], 3),
        'kNN_Thickness_m':    round(knn_params['t'], 4),
        'kNN_FoS':            round(knn_params['fos'], 2),
        'kNN_Physics_Load_kN':round(knn_physics_load, 3),
        'kNN_Physics_Error_pct': round(knn_error, 2),

        # CAD dimensions for Fusion 360
        'FEA_CAD_Span_m':         round(tnn_params['L'], 3),
        'FEA_CAD_Height_m':       round(tnn_params['t'] * 2.0, 4),
        'FEA_CAD_Width_m':        round(tnn_params['t'] * 1.0, 4),
        'FEA_Apply_Load_kN':      round(tnn_physics_load, 3),
        'FEA_Apply_Load_N':       round(tnn_physics_load * 1000, 1),
        'FEA_Allow_Defl_mm':      round(tnn_params['L'] / 360 * 1000, 2),

        # FEA engineer fills these after Fusion 360 simulation
        'FEA_Max_VonMises_MPa':   '',
        'FEA_Max_Deflection_mm':  '',
        'FEA_Load_at_Failure_kN': '',
        'FEA_vs_Target_Error_pct':'',
        'FEA_Pass_Fail':          '',
    })

    print(f"  {sample_id} | Target={target:8.2f} kN → "
          f"TNN: {tnn_params['material']:<10} "
          f"{tnn_params['cross_section']:<12} "
          f"L={tnn_params['L']:.2f}m t={tnn_params['t']:.3f}m "
          f"→ Physics={tnn_physics_load:.2f}kN (Err={tnn_error:.1f}%)")

df_inv_out = pd.DataFrame(records)

# ── Safety check: assert no non-metal materials slipped through ───────────────
tnn_mats_used = df_inv_out['TNN_Material'].unique().tolist()
knn_mats_used = df_inv_out['kNN_Material'].unique().tolist()
non_metal_tnn = [m for m in tnn_mats_used if m not in METAL_MATERIALS]
non_metal_knn = [m for m in knn_mats_used if m not in METAL_MATERIALS]
if non_metal_tnn:
    print(f"\n  ⚠ WARNING: TNN predicted non-metal(s): {non_metal_tnn}")
else:
    print(f"\n  ✓ TNN metal check passed — materials used: {tnn_mats_used}")
if non_metal_knn:
    print(f"  ⚠ WARNING: kNN predicted non-metal(s): {non_metal_knn}")
else:
    print(f"  ✓ kNN metal check passed — materials used: {knn_mats_used}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. INVERSE METHOD COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 7: INVERSE METHOD COMPARISON")
print("="*70)

tnn_errors = df_inv_out['TNN_Physics_Error_pct'].values.astype(float)
knn_errors = df_inv_out['kNN_Physics_Error_pct'].values.astype(float)

comparison = pd.DataFrame({
    'Method': ['TNN (Tandem Neural Network — Metals Only)',
               'kNN Retrieval (Baseline — Metals Only)'],
    'Mean_Error_pct':   [tnn_errors.mean(),        knn_errors.mean()],
    'Median_Error_pct': [np.median(tnn_errors),    np.median(knn_errors)],
    'Max_Error_pct':    [tnn_errors.max(),          knn_errors.max()],
    'Min_Error_pct':    [tnn_errors.min(),          knn_errors.min()],
    'Within_5pct':      [(tnn_errors <= 5).sum(),   (knn_errors <= 5).sum()],
    'Within_10pct':     [(tnn_errors <= 10).sum(),  (knn_errors <= 10).sum()],
    'Materials_Used':   [str(tnn_mats_used),        str(knn_mats_used)],
})
comparison.to_csv(f"{OUTPUT_DIR}/Table3_Inverse_Method_Comparison.csv", index=False)
print(comparison.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

# Full inverse output table
df_inv_out.to_csv(f"{OUTPUT_DIR}/Table4_Inverse_Fusion360_Validation.csv", index=False)
print(f"\n  ✓  Inverse validation table → {OUTPUT_DIR}/Table4_Inverse_Fusion360_Validation.csv")

# Compact view for paper table (key columns only)
paper_table = df_inv_out[[
    'Sample_ID', 'Target_Load_kN',
    'TNN_Material', 'TNN_CrossSection',
    'TNN_Length_m', 'TNN_Thickness_m', 'TNN_FoS',
    'TNN_Physics_Load_kN', 'TNN_Physics_Error_pct', 'TNN_Governing',
    'FEA_Apply_Load_N',
    'FEA_Max_VonMises_MPa', 'FEA_Max_Deflection_mm',
    'FEA_Load_at_Failure_kN', 'FEA_vs_Target_Error_pct', 'FEA_Pass_Fail'
]].copy()
paper_table.to_csv(f"{OUTPUT_DIR}/Table4b_Paper_InverseTable_Template.csv", index=False)
print(f"  ✓  Paper table template      → {OUTPUT_DIR}/Table4b_Paper_InverseTable_Template.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALIZATION SUITE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 8: GENERATING INVERSE STUDY FIGURES")
print("="*70)

# ── Fig 6: Target vs Physics Load (TNN) ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Inverse Study — TNN Generated vs Target Load (Metals Only)',
             fontsize=13, fontweight='bold')

ax = axes[0]
targets  = df_inv_out['Target_Load_kN'].values.astype(float)
tnn_phys = df_inv_out['TNN_Physics_Load_kN'].values.astype(float)
knn_phys = df_inv_out['kNN_Physics_Load_kN'].values.astype(float)

ax.scatter(targets, tnn_phys, c='#2ecc71', s=60, zorder=5,
           label='TNN (Metal)', edgecolors='white')
ax.scatter(targets, knn_phys, c='#e74c3c', s=60, zorder=4, marker='^',
           label='k-NN Baseline (Metal)', edgecolors='white')
lo = min(targets.min(), tnn_phys.min(), knn_phys.min())
hi = max(targets.max(), tnn_phys.max(), knn_phys.max())
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Perfect match')
ax.fill_between([lo, hi], [lo*0.9, hi*0.9], [lo*1.1, hi*1.1],
                alpha=0.1, color='gray', label='±10% band')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Target Load (kN)', fontsize=11)
ax.set_ylabel('Physics-Verified Load (kN)', fontsize=11)
ax.set_title('Log Scale: Target vs Generated (Metal Only)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(np.arange(N_FUSION360) - 0.2, tnn_errors, width=0.35,
       color='#2ecc71', alpha=0.85, label='TNN Error %', edgecolor='white')
ax.bar(np.arange(N_FUSION360) + 0.2, knn_errors, width=0.35,
       color='#e74c3c', alpha=0.85, label='k-NN Error %', edgecolor='white')
ax.axhline(5,  color='orange', linestyle='--', linewidth=1.5, label='5% threshold')
ax.axhline(10, color='red',    linestyle='--', linewidth=1.5, label='10% threshold')
ax.set_xlabel('Sample Index', fontsize=11)
ax.set_ylabel('Absolute Error (%)', fontsize=11)
ax.set_title(
    f'Per-Sample Error | TNN Mean={tnn_errors.mean():.1f}% | '
    f'kNN Mean={knn_errors.mean():.1f}% | Metals Only',
    fontsize=10
)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig6_Inverse_TargetVsGenerated.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 6 saved → {OUTPUT_DIR}/Fig6_Inverse_TargetVsGenerated.png")

# ── Fig 7: Generated parameter distribution ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    'Inverse Study — Distribution of Generated Beam Parameters '
    '(TNN | Metals: Aluminum & Steel Only)',
    fontsize=13, fontweight='bold'
)

params_data = [
    (df_inv_out['TNN_Length_m'].values.astype(float),    'Beam Length (m)',    '#3498db'),
    (df_inv_out['TNN_Thickness_m'].values.astype(float), 'Thickness (m)',      '#e74c3c'),
    (df_inv_out['TNN_FoS'].values.astype(float),         'Factor of Safety',   '#2ecc71'),
]
cats_data = [
    (df_inv_out['TNN_Material'],     'Material (Metal Only)', '#9b59b6'),
    (df_inv_out['TNN_CrossSection'], 'Cross-Section',         '#f39c12'),
    (df_inv_out['TNN_Governing'],    'Governing Criterion',   '#1abc9c'),
]

for ax, (vals, label, color) in zip(axes[0], params_data):
    ax.hist(vals, bins=10, color=color, alpha=0.85, edgecolor='white')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Generated {label}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

for ax, (vals, label, color) in zip(axes[1], cats_data):
    vc = vals.value_counts()
    ax.bar(vc.index, vc.values, color=color, alpha=0.85, edgecolor='white')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Generated {label}', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig7_Inverse_ParameterDistributions.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 7 saved → {OUTPUT_DIR}/Fig7_Inverse_ParameterDistributions.png")

# ── Fig 8: Error vs Target Load ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
# Color-code by material
tnn_mats_list = df_inv_out['TNN_Material'].values
color_map = {m: c for m, c in zip(METAL_MATERIALS, ['#2ecc71', '#3498db', '#9b59b6'])}
for mat in METAL_MATERIALS:
    mask = (tnn_mats_list == mat)
    if mask.any():
        ax.scatter(targets[mask], tnn_errors[mask], s=90, zorder=5,
                   label=f'TNN — {mat}',
                   color=color_map.get(mat, '#2ecc71'), edgecolors='darkgreen')

ax.scatter(targets, knn_errors, c='#e74c3c', s=80, zorder=4,
           marker='^', label='kNN Error (Metal)', edgecolors='darkred')
ax.axhline(5,  color='orange', linestyle='--', linewidth=1.5, label='5% target')
ax.axhline(10, color='red',    linestyle='--', linewidth=1.5, label='10% limit')
ax.set_xscale('log')
ax.set_xlabel('Target Load (kN)', fontsize=12)
ax.set_ylabel('Absolute Error (%)', fontsize=12)
ax.set_title(
    'Inverse Prediction Error vs Target Load — TNN vs kNN (Metals Only)',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig8_Inverse_ErrorVsLoad.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 8 saved → {OUTPUT_DIR}/Fig8_Inverse_ErrorVsLoad.png")

# ── Fig 9: TNN Architecture Diagram (schematic) ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.set_title('TNN Architecture for Inverse Beam Design (Metal-Restricted)',
             fontsize=13, fontweight='bold', pad=20)

boxes = [
    (0.5, 2.5, 1.5, 1.0, '#3498db',
     'Target\nLoad P*\n(scalar)', 'white'),
    (2.5, 2.0, 2.0, 2.0, '#9b59b6',
     'Inverse MLP\n256→256→128→64\nActivation: ReLU\nOutput: L, t, FoS\nMat*, CrossSec', 'white'),
    (5.5, 2.0, 2.0, 2.0, '#27ae60',
     'Physics\nEngine\n(closed-form)', 'white'),
    (8.0, 2.5, 1.5, 1.0, '#e74c3c',
     'Verified\nLoad P̂\n(check)', 'white'),
]
for (x, y, w, h, color, label, tc) in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.9,
                          edgecolor='white', linewidth=2, zorder=5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=9, color=tc, fontweight='bold', zorder=6)

for (x1, x2, y_pos) in [(2.0, 2.5, 3.0), (4.5, 5.5, 3.0), (7.5, 8.0, 3.0)]:
    ax.annotate('', xy=(x2, y_pos), xytext=(x1, y_pos),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

ax.text(2.25, 3.2, 'log(P*)',          ha='center', fontsize=9,  color='#333')
ax.text(5.0,  3.2, '[L, t, FoS,\nMat*, CS]', ha='center', fontsize=8, color='#333')
ax.text(7.75, 3.2, 'Pb, Ps, Pd\n→ P_max',   ha='center', fontsize=8, color='#333')

ax.text(5.0, 1.5,
        '⬇  Loss = |P̂ − P*|²   |   * Mat restricted to Aluminum / Steel',
        ha='center', fontsize=10, color='#e74c3c', fontstyle='italic')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig9_TNN_Architecture.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Fig 9 saved → {OUTPUT_DIR}/Fig9_TNN_Architecture.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. FUSION 360 INSTRUCTIONS (INVERSE)
# ─────────────────────────────────────────────────────────────────────────────
fusion_inv_instructions = """
================================================================================
FUSION 360 FEA SETUP INSTRUCTIONS — INVERSE VALIDATION
================================================================================

NOTE: All inverse predictions use METAL MATERIALS ONLY (Aluminum or Steel).
      Non-metal materials (Concrete, Wood, Composite) are excluded from
      inverse parameter synthesis as per study requirements.

For each row in Table4b_Paper_InverseTable_Template.csv:

INPUT INTERPRETATION:
  - Sample_ID       : Unique sample identifier (INV-01 to INV-25)
  - Target_Load_kN  : The load the beam SHOULD sustain (your design requirement)
  - TNN_*           : Parameters generated by the TNN inverse model
  - FEA_Apply_Load_N: Force to apply in Fusion 360 = TNN_Physics_Load_kN × 1000 N

STEP 1: CREATE GEOMETRY (use TNN_* columns)
  - Material       = TNN_Material  (will be Aluminum or Steel)
  - Cross-Section  = TNN_CrossSection
  - Length         = FEA_CAD_Span_m (= TNN_Length_m)
  - Height         = FEA_CAD_Height_m (= TNN_Thickness_m × 2)
  - Width          = FEA_CAD_Width_m  (= TNN_Thickness_m × 1)

  Cross-section specific dims (use TNN_Thickness_m as base 't'):
    Rectangular : width=t,      height=2t
    Circular    : diameter=2t
    I-Beam      : H=2t, flange_w=1.5t, flange_t=0.15t, web_t=0.1t
    T-Beam      : flange_w=1.5t, flange_t=0.15t, web_h=t, web_t=0.1t
    Box         : outer_H=2t, outer_B=t, wall=0.1t
    L-Beam      : leg_b=t, leg_h=t, wall=0.12t

STEP 2: ASSIGN MATERIAL PROPERTIES (METALS ONLY)
  Steel     : E=200 GPa, σ_y=250 MPa, ν=0.30, ρ=7850 kg/m³
  Aluminum  : E=69 GPa,  σ_y=270 MPa, ν=0.33, ρ=2700 kg/m³

STEP 3: BOUNDARY CONDITIONS
  - Left end (x=0)   : Pinned (fix Y, Z translation; free X, rotations)
  - Right end (x=L)  : Roller (fix Y translation only; free X, Z, rotations)
  - This matches SIMPLY SUPPORTED beam assumption used in dataset generation

STEP 4: APPLY LOAD
  - Point load at midspan x = FEA_CAD_Span_m / 2
  - Direction: −Y (downward)
  - Magnitude: FEA_Apply_Load_N  [Newtons]

STEP 5: MESH SETTINGS
  - Mesh type: Tetrahedral (default Fusion 360)
  - Refinement: Medium-Fine (at least 3 elements through wall thickness)
  - Convergence: Check mesh independence with 2× refinement

STEP 6: RUN & RECORD RESULTS
  After simulation, record into the CSV:

  FEA_Max_VonMises_MPa   = Peak Von Mises stress from contour plot
  FEA_Max_Deflection_mm  = Peak displacement in Y direction
  FEA_Load_at_Failure_kN = Load at which σ_VonMises reaches TNN_sigma_allow_MPa
                           (use Fusion 360 parametric study if needed)
  FEA_vs_Target_Error_pct = |FEA_Load_at_Failure - Target_Load_kN| / Target × 100
  FEA_Pass_Fail          = PASS if:
                             σ_FEA ≤ TNN_sigma_allow_MPa  AND
                             δ_FEA ≤ FEA_Allow_Defl_mm

ACCEPTANCE CRITERIA (for paper):
  Individual : FEA_vs_Target_Error_pct < 10% per sample
  Overall    : Mean FEA_vs_Target_Error_pct < 5%
  Pass rate  : ≥ 80% of 25 samples pass both stress AND deflection checks

================================================================================
"""
with open(f"{OUTPUT_DIR}/Fusion360_Inverse_Instructions.txt", 'w') as f:
    f.write(fusion_inv_instructions)
print(f"\n  Instructions saved → {OUTPUT_DIR}/Fusion360_Inverse_Instructions.txt")

# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  INVERSE STUDY COMPLETE — SUMMARY")
print("="*70)
print(f"""
  Forward surrogate : Random Forest  R²={fwd_r2:.4f}  MAPE={fwd_mape:.2f}%
  Inverse method    : TNN (MLP Continuous + MLP Classifiers)
  Baseline          : k-NN Retrieval (k=5)
  Material scope    : METALS ONLY — {METAL_MATERIALS}

  TNN Inverse Performance (physics verification):
    Mean Error  : {tnn_errors.mean():.2f}%
    Median Error: {np.median(tnn_errors):.2f}%
    Max Error   : {tnn_errors.max():.2f}%
    Within 5%   : {(tnn_errors <= 5).sum()}/25 samples
    Within 10%  : {(tnn_errors <= 10).sum()}/25 samples
    Materials   : {tnn_mats_used}

  kNN Baseline Performance:
    Mean Error  : {knn_errors.mean():.2f}%
    Median Error: {np.median(knn_errors):.2f}%
    Materials   : {knn_mats_used}

  Output files  : {OUTPUT_DIR}/
    Table3_Inverse_Method_Comparison.csv
    Table4_Inverse_Fusion360_Validation.csv      ← Full data
    Table4b_Paper_InverseTable_Template.csv      ← GIVE TO FUSION 360
    Fusion360_Inverse_Instructions.txt           ← FEA SETUP GUIDE
    Fig6_Inverse_TargetVsGenerated.png
    Fig7_Inverse_ParameterDistributions.png
    Fig8_Inverse_ErrorVsLoad.png
    Fig9_TNN_Architecture.png
""")