import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from thermo.chemical import Chemical
import pickle
import os
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="DES Solubility & Selectivity Predictor", layout="wide")

# --- LOAD CONSTANTS & CACHED DATA ---
R = 8.3144598 / 4184
AEFFPRIME = 7.5
c_hb = 85580.0
sigma_hb = 0.0084
q0, r0, z_coordination = 79.53, 66.69, 10
EPS = 3.667
EO = 2.395e-4
FPOL = (EPS - 1.0) / (EPS + 0.5)
ALPHA = (0.3 * AEFFPRIME ** (1.5)) / (EO)
alpha_prime = FPOL * ALPHA

sigma_tabulated = np.linspace(-0.025, 0.025, 51)
sigma_m = np.tile(sigma_tabulated, (len(sigma_tabulated), 1))
sigma_n = np.tile(np.array(sigma_tabulated, ndmin=2).T, (1, len(sigma_tabulated)))
sigma_acc = np.tril(sigma_n) + np.triu(sigma_m, 1)
sigma_don = np.tril(sigma_m) + np.triu(sigma_n, 1)
DELTAW = (alpha_prime / 2) * (sigma_m + sigma_n) ** 2 + c_hb * np.maximum(0, sigma_acc - sigma_hb) * np.minimum(0,
                                                                                                                sigma_don + sigma_hb)


@st.cache_data
def get_hbd_list():
    """Load unique HBD names from the COSMO index file."""
    df = pd.read_csv('profiles/VT2005/Sigma_Profile_Database_Index_v2.txt', sep='\t')
    return sorted(df['Compound Name'].unique().tolist())


@st.cache_data
def get_hbd_smiles_features(hbd_cosmo_name):
    """
    Improved retrieval using:
    1. Local hardcoded mapping (fastest & most accurate)
    2. thermo.Chemical lookup (robust name matching)
    3. PubChem (fallback)
    """
    # Step 1: Data Engineering - Clean the name
    clean_name = hbd_cosmo_name.replace('_', ' ')

    # Step 2: Local Mapping for common DES components
    # This fixes the "1,2-propanediol" and "Choline chloride" issues immediately
    local_smiles_db = {
        "1,2-propanediol": "CC(CO)O",
        "Furfuryl alcohol": "C1=COC(=C1)CO",
        "Triethylene glycol": "C(COCCOCCO)O",
        "Levulinic acid": "CC(=O)CCC(=O)O",
        "Diethylene glycol": "C(COCCO)O",
        "Ethylene glycol": "C(CO)O",
        "Glycerol": "OCC(O)CO",
        "Urea": "C(=O)(N)N",
        "Guaiacol": "COC1=CC=CC=C1O",
        "Phenol": "C1=CC=C(C=C1)O",
        "Choline chloride": "C[N+](C)(C)CCO.[Cl-]",
        "2,3-butanediol": "CC(C(C)O)O",
        "1,4-butanediol": "C(CCO)CO",
        "1,3-propanediol": "C(CO)CO"
    }

    smiles = None

    # Try Local DB first
    if clean_name in local_smiles_db:
        smiles = local_smiles_db[clean_name]

    # If not in local, try the Chemical library (already in your thermo import)
    if not smiles:
        try:
            chem = Chemical(clean_name)
            smiles = chem.smiles
        except:
            pass

    # Final Fallback: PubChem (only if the above two fail)
    if not smiles:
        try:
            import pubchempy as pcp
            results = pcp.get_compounds(clean_name, 'name')
            if results:
                smiles = results[0].isomeric_smiles
        except:
            smiles = None

    # Feature Engineering from the found SMILES
    if smiles:
        return {
            'SMILES_Length': float(len(smiles)),
            'SMILES_O_Count': float(smiles.count('O') + smiles.count('o')),  # count both cases
            'SMILES_C_Count': float(smiles.count('C') + smiles.count('c')),
            'SMILES_String': smiles
        }
    else:
        # If absolutely nothing is found, use a safe generic mean for DES
        return {
            'SMILES_Length': 10.0,
            'SMILES_O_Count': 2.0,
            'SMILES_C_Count': 3.0,
            'SMILES_String': "Not Found (Using Defaults)"
        }


@st.cache_resource
def load_models():
    """Load the Density (RF) and Solubility (XGB) models."""
    with open('best_rf_model.pkl', 'rb') as f:
        d_data = pickle.load(f)
        d_model = d_data['model'] if isinstance(d_data, dict) else d_data
    with open('XGB_model_Test_01.pkl', 'rb') as f:
        s_model = pickle.load(f)
    return d_model, s_model


# --- THERMO FUNCTIONS ---
def get_sigma_profile(name):
    df = pd.read_csv('profiles/VT2005/Sigma_Profile_Database_Index_v2.txt', sep='\t')
    mask = df['Compound Name'] == name
    index = int(df[mask]['Index No.'].iloc[0])
    v_cosmo = float(df[mask]['Vcosmo, A3'].iloc[0])
    path = f'profiles/VT2005/Sigma_Profiles_v2/VT2005-{index:04d}-PROF.txt'
    dd = pd.read_csv(path, names=['sigma', 'pA'], sep='\s+')
    dd['A'] = dd['pA'].sum()
    dd['p(sigma)'] = dd['pA'] / dd['A']
    return dd, v_cosmo


def get_Gamma(T, psigma):
    Gamma = np.ones_like(psigma)
    AA = np.exp(-DELTAW / (R * T)) * psigma
    for _ in range(50):
        Gammanew = 1 / np.sum(AA * Gamma, axis=1)
        Gamma = (Gammanew + Gamma) / 2
        if np.max(np.abs(Gammanew - Gamma)) < 1e-8: break
    return Gamma


def calculate_solubility(T, P, gas_name, hba_name, hbd_name, ratio1, ratio2, d_model, s_model, smiles_feats):
    # 1. Activity Coeff via COSMO-SAC
    gas_comp = 'Ammonia' if gas_name == 'NH3' else 'Carbon_dioxide'
    comps = [gas_comp, hba_name, hbd_name]

    profs, v_cosmos = [], []
    for c in comps:
        p, v = get_sigma_profile(c)
        profs.append(p);
        v_cosmos.append(v)

    x_gas = 1e-10
    x = np.array([x_gas, (1 - x_gas) * ratio1 / (ratio1 + ratio2), (1 - x_gas) * ratio2 / (ratio1 + ratio2)])
    As = np.array([p['pA'].sum() for p in profs])
    psigma_mix = sum(x[i] * profs[i]['pA'] for i in range(3)) / np.dot(x, As)

    lnG_mix = np.log(get_Gamma(T, np.array(psigma_mix)))
    psigma_i = np.array(profs[0]['p(sigma)'])
    lnGi = np.log(get_Gamma(T, psigma_i))
    lng_resid = profs[0]['A'].iloc[0] / AEFFPRIME * np.sum(psigma_i * (lnG_mix - lnGi))

    q, r = As / q0, np.array(v_cosmos) / r0
    theta, phi = (x * q) / np.dot(x, q), (x * r) / np.dot(x, r)
    l = z_coordination / 2 * (r - q) - (r - 1)
    lng_comb = np.log(phi[0] / x[0]) + z_coordination / 2 * q[0] * np.log(theta[0] / phi[0]) + l[0] - phi[0] / x[
        0] * np.dot(x, l)

    gamma = np.exp(lng_resid + lng_comb)

    # 2. Density Prediction using automated SMILES features
    X_den = pd.DataFrame([[
        smiles_feats['SMILES_Length'],
        smiles_feats['SMILES_O_Count'],
        smiles_feats['SMILES_C_Count'],
        ratio1 / ratio2,
        T
    ]], columns=['SMILES_Length', 'SMILES_O_Count', 'SMILES_C_Count', 'Ratio', 'T(K)'])
    rho = d_model.predict(X_den)[0]

    # 3. XGBoost Final Solubility
    gas_chem = Chemical(gas_name.lower())
    f_gas = gas_chem.VaporPressure(T) / 1000

    # Calculate MW for molar conversion
    mw_hba = 139.62
    try:
        mw_hbd = Chemical(hbd_name.replace('_', ' ')).MW
    except:
        mw_hbd = 100.0
    des_mw = (ratio1 / (ratio1 + ratio2)) * mw_hba + (ratio2 / (ratio1 + ratio2)) * mw_hbd

    s_calc = (P / (gamma * f_gas)) / (des_mw / 1000)

    X_s = pd.DataFrame([[P, s_calc, rho]], columns=['P', 'S_calc', 'Predicted_Density'])
    s_ml = abs(s_model.predict(X_s)[0])

    return s_ml


# --- UI LAYOUT ---
st.title("🧪 DES Gas Solubility & Selectivity Predictor")
st.markdown("Integrated **COSMO-SAC + Machine Learning** with automated SMILES retrieval. [Selectivity = S_NH3/S_CO2]")

with st.sidebar:
    st.header("1. DES Configuration")
    hbd_comp = st.selectbox("HBD Component (from COSMO DB)", get_hbd_list())
    hba_ratio = st.number_input("HBA Ratio (e.g. Choline Chloride)", value=1.0, step=0.1)
    hbd_ratio = st.number_input("HBD Ratio", value=2.0, step=0.1)
    hba_comp = "Choline_chloride"

    st.divider()
    # Fetch features immediately on selection
    smiles_data = get_hbd_smiles_features(hbd_comp)
    st.write(f"**Identified SMILES:** `{smiles_data['SMILES_String']}`")
    st.write(
        f"**Features:** L:{smiles_data['SMILES_Length']}, O:{smiles_data['SMILES_O_Count']}, C:{smiles_data['SMILES_C_Count']}")

# Load models
try:
    d_mod, s_mod = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Point Calculation", "Selectivity Map (Contour)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1: p_input = st.number_input("Pressure (kPa)", value=100.0, min_value=1.0)
    with col2: t_input = st.number_input("Temperature (K)", value=298.15, min_value=273.15)

    if st.button("Run Single Calculation"):
        with st.spinner("Processing thermodynamic cycles..."):
            s_nh3 = calculate_solubility(t_input, p_input, 'NH3', hba_comp, hbd_comp, hba_ratio, hbd_ratio, d_mod,
                                         s_mod, smiles_data)
            s_co2 = calculate_solubility(t_input, p_input, 'CO2', hba_comp, hbd_comp, hba_ratio, hbd_ratio, d_mod,
                                         s_mod, smiles_data)
            sel = s_nh3 / s_co2 if s_co2 > 0 else 0

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Solubility NH3", f"{s_nh3:.4f}", "mol/kg")
            m2.metric("Solubility CO2", f"{s_co2:.4f}", "mol/kg")
            m3.metric("Selectivity", f"{sel:.2f}")

with tab2:
    st.subheader("Pressure vs Temperature Selectivity Surface")
    if st.button("Generate Contour Plot"):
        with st.spinner("Calculating 100-point grid..."):
            p_range = np.linspace(100, 1000, 10)
            t_range = np.linspace(298, 353, 10)
            z_matrix = np.zeros((len(t_range), len(p_range)))

            for i, t in enumerate(t_range):
                for j, p in enumerate(p_range):
                    s_n = calculate_solubility(t, p, 'NH3', hba_comp, hbd_comp, hba_ratio, hbd_ratio, d_mod, s_mod,
                                               smiles_data)
                    s_c = calculate_solubility(t, p, 'CO2', hba_comp, hbd_comp, hba_ratio, hbd_ratio, d_mod, s_mod,
                                               smiles_data)
                    z_matrix[i, j] = s_n / s_c if s_c > 0 else 0

            fig = go.Figure(data=[go.Contour(
                z=z_matrix, x=p_range, y=t_range,
                colorscale='Viridis',
                colorbar=dict(title='Selectivity')
            )])
            fig.update_layout(
                xaxis_title="Pressure (kPa)",
                yaxis_title="Temperature (K)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

#streamlit run "20250930_Onozawa/CO2 and NH3 in ChCl DES/App for selectivity/App_01.py"