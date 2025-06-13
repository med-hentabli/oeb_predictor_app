# -*- coding: utf-8 -*-
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'  # Must be first

import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from urllib.parse import quote
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem

# Handle RDKit drawing import with fallback
try:
    from rdkit.Chem import Draw
    RDKIT_DRAW_ENABLED = True
except Exception:
    RDKIT_DRAW_ENABLED = False

from tensorflow.keras.models import load_model
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.special import softmax
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="OEB Prediction Pro", 
    layout="wide", 
    page_icon="🔬",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "OEB Prediction Pro - Predict Occupational Exposure Bands"
    }
)

# --- CONSTANTS ---
try:
    DESC_NAMES = [desc[0] for desc in Descriptors._descList]
except Exception:
    DESC_NAMES = []
    st.warning("Could not load RDKit descriptor names")

OEB_DESCRIPTIONS = {
    0: "No exposure limits: Minimal or no systemic toxicity.",
    1: "OEB 1: Low hazard (OEL: 1000 - 5000 µg/m³)",
    2: "OEB 2: Moderate hazard (OEL: 100 - 1000 µg/m³)",
    3: "OEB 3: High hazard (OEL: 10 - 100 µg/m³)",
    4: "OEB 4: Very high hazard (OEL: 1 - 10 µg/m³)",
    5: "OEB 5: Extremely high hazard (OEL: < 1 µg/m³)",
    6: "OEB 6: Extremely potent (OEL: < 0.1 µg/m³)"
}
MODEL_NAMES = ["MLP", "SVC", "XGBoost", "RandomForest", "DecisionTree"]
DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
MODEL_DIR = "models"

def get_model_path(filename):
    """Constructs an absolute path to the model file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, MODEL_DIR, filename)

@st.cache_resource
def load_models_and_scalers():
    """Loads all models and scalers with error handling."""
    try:
        scalers = {
            "desc": joblib.load(get_model_path("scaler_descriptors.pkl")),
            "cnn_input": joblib.load(get_model_path("scaler_features_cnn.pkl")),
            "cnn_output": joblib.load(get_model_path("scaler_features_cnn_output.pkl"))
        }
        classifiers = {
            name: joblib.load(get_model_path(f"model_{name}.pkl")) 
            for name in MODEL_NAMES
        }
        cnn_model = load_model(get_model_path("cnn_feature_extractor_model.h5"))
        return cnn_model, scalers, classifiers
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, {}, {}

def compute_cnn_ready_features(smiles, scalers, cnn_model):
    """Computes features from SMILES string with error handling."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error("Invalid SMILES string")
            return None

        # Calculate descriptors
        if not DESC_NAMES:
            st.error("Descriptor names not available")
            return None
            
        desc_calc = MolecularDescriptorCalculator(DESC_NAMES)
        descriptors = np.array(desc_calc.CalcDescriptors(mol))
        
        # Pad descriptors to 1024
        padded_desc = np.zeros(1024)
        padded_desc[:min(len(descriptors), 1024)] = descriptors[:min(len(descriptors), 1024)]
        norm_desc = scalers["desc"].transform([padded_desc])[0]

        # Calculate fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_array = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_array)

        # Prepare CNN input
        combined = np.stack((norm_desc, fp_array), axis=-1).reshape(32, 32, 2)
        norm_input = scalers["cnn_input"].transform(combined.reshape(1, -1)).reshape(1, 32, 32, 2)
        
        # Extract features
        features_raw = cnn_model.predict(norm_input)
        return scalers["cnn_output"].transform(features_raw)
        
    except Exception as e:
        st.error(f"Feature computation failed: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_pubchem_data(compound_name):
    """Fetches compound data from PubChem with error handling."""
    try:
        encoded_name = quote(compound_name)
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        cid_res = requests.get(cid_url, timeout=10).json()
        cid = cid_res.get("IdentifierList", {}).get("CID", [None])[0]
        
        if cid:
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            smiles_res = requests.get(smiles_url, timeout=10).json()
            smiles = smiles_res.get("PropertyTable", {}).get("Properties", [{}])[0].get("CanonicalSMILES")
            return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}", smiles
    except Exception:
        pass
    return None, None

def smiles_to_image(smiles, size=(300, 300)):
    """Converts SMILES to image with fallback handling."""
    if not RDKIT_DRAW_ENABLED or not smiles:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
            
        try:
            return Draw.MolToImage(mol, size=size)
        except Exception:
            return Draw.MolToImage(mol, size=size, kekulize=False)
    except Exception:
        return None

def main():
    st.title("🔬 OEB Prediction Pro")
    st.markdown("Predict Occupational Exposure Bands for chemical compounds using machine learning.")
    
    # Load models
    cnn_model, scalers, classifiers = load_models_and_scalers()
    if not cnn_model:
        st.error("Failed to load required models. Please check the logs.")
        return

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    selected_model = st.sidebar.selectbox("Choose Model", MODEL_NAMES, index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("OEB Reference")
    for oeb, desc in OEB_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**OEB {oeb}:** {desc.split(':')[1].split('(')[0].strip()}")
    
    # Main content
    input_col, vis_col = st.columns([0.6, 0.4])

    with input_col:
        st.subheader("🧪 Compound Input")
        
        # PubChem search
        pubchem_name = st.text_input("Search PubChem by Name", key="pubchem_name")
        pubchem_url = None
        retrieved_smiles = None
        
        if pubchem_name:
            with st.spinner(f"Searching PubChem for '{pubchem_name}'..."):
                pubchem_url, retrieved_smiles = get_pubchem_data(pubchem_name)
            
            if retrieved_smiles:
                st.success(f"Found '{pubchem_name}'")
                if st.button(f"Use SMILES for {pubchem_name}"):
                    st.session_state.smiles = retrieved_smiles
            elif pubchem_name:
                st.warning("Compound not found")

        # SMILES input
        smiles = st.text_input(
            "Or enter SMILES directly", 
            value=st.session_state.get("smiles", DEFAULT_SMILES),
            key="smiles_input"
        )
        st.session_state.smiles = smiles

        if st.button("🚀 Predict OEB", type="primary"):
            if not smiles:
                st.error("Please enter a SMILES string")
            else:
                with st.spinner("Analyzing molecule..."):
                    features = compute_cnn_ready_features(smiles, scalers, cnn_model)
                
                if features is not None:
                    model = classifiers[selected_model]
                    
                    try:
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(features)[0]
                        else:
                            scores = model.decision_function(features)
                            if scores.ndim == 1:
                                scores = scores.reshape(1, -1)
                            probs = softmax(scores, axis=1)[0]
                        
                        # Ensure probabilities match OEB classes
                        probs = probs[:len(OEB_DESCRIPTIONS)]
                        if len(probs) < len(OEB_DESCRIPTIONS):
                            probs = np.pad(probs, (0, len(OEB_DESCRIPTIONS)-len(probs)))
                        
                        pred_class = np.argmax(probs)
                        st.success(f"Predicted OEB: **{pred_class}**")
                        st.markdown(f"**{OEB_DESCRIPTIONS[pred_class]}**")
                        
                        # Display probabilities
                        prob_df = pd.DataFrame({
                            "OEB": list(OEB_DESCRIPTIONS.keys()),
                            "Description": [d.split(":")[0] for d in OEB_DESCRIPTIONS.values()],
                            "Probability": probs
                        }).set_index("OEB")
                        
                        st.dataframe(
                            prob_df.style.format({"Probability": "{:.1%}"})
                                .bar(subset=["Probability"], color='#5fba7d'),
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

    with vis_col:
        st.subheader("👁️ Molecule Viewer")
        smiles_to_display = st.session_state.get("smiles", "")
        
        if smiles_to_display:
            if RDKIT_DRAW_ENABLED:
                img = smiles_to_image(smiles_to_display)
                if img:
                    st.image(img, caption="Molecular Structure", use_column_width=True)
                else:
                    st.warning("Could not render molecule")
                    st.code(smiles_to_display)
            else:
                st.warning("Molecule rendering unavailable")
                st.code(smiles_to_display)
            
            if pubchem_url:
                st.markdown(f"[View on PubChem]({pubchem_url})")
        else:
            st.info("Enter a SMILES to view structure")

    st.markdown("---")
    st.caption("OEB Prediction Pro | Powered by AI and Cheminformatics")

if __name__ == "__main__":
    main()
