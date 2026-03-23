import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from transformers import pipeline
import torch

# Set page config
st.set_page_config(page_title="Turkish NLI Inference", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🧠 Turkish NLI Inference System")
st.markdown("Compare predictions across multiple models and ensemble configurations")

# ============================================================================
# SIDEBAR: Model Information and Configuration
# ============================================================================
st.sidebar.header("📊 Model Configuration")

# Ensemble weights configurations
ensemble_configs = {
    "Ensemble 1 (Balanced)": {
        "qwen":     0.40,
        "gemma":    0.20,
        "mdeberta": 0.10,
        "bert":     0.30,
    },
    "Ensemble 2 (Optimized)": {
        "gemma":    0.30,
        "qwen":     0.50,
        "mdeberta": 0.15,
        "bert":     0.05,
    },
    "Ensemble 3 (Current)": {
        "gemma":    0.30,
        "qwen":     0.45,
        "mdeberta": 0.15,
        "bert":     0.10,
    }
}

# Model configurations
models_config = {
    "mDeBERTa": {
        "model_id": "microsoft/mdeberta-v3-base",
        "hf_task_id": "microsoft/mdeberta-v3-base",
        "enabled": True
    },
    "BERT (Turkish)": {
        "model_id": "dbmdz/bert-base-turkish-cased",
        "hf_task_id": "dbmdz/bert-base-turkish-cased",
        "enabled": True
    },
    "Qwen2 7B": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "hf_task_id": "Qwen/Qwen2-7B-Instruct",
        "enabled": True
    },
    "Gemma3 27B": {
        "model_id": "google/gemma-3-27b-it",
        "hf_task_id": "google/gemma-3-27b-it",
        "enabled": True
    }
}

# Initialize session state for model selection
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = list(models_config.keys())

if 'selected_ensembles' not in st.session_state:
    st.session_state.selected_ensembles = list(ensemble_configs.keys())

st.sidebar.subheader("📋 Base Models")
selected_models = []
for model_name in models_config:
    if st.sidebar.checkbox(f"{model_name}", value=model_name in st.session_state.selected_models, key=f"model_{model_name}"):
        selected_models.append(model_name)
st.session_state.selected_models = selected_models

st.sidebar.subheader("🔗 Ensemble Configurations")
selected_ensembles = []
for config_name in ensemble_configs:
    if st.sidebar.checkbox(f"{config_name}", value=config_name in st.session_state.selected_ensembles, key=f"ensemble_{config_name}"):
        selected_ensembles.append(config_name)
        weights = ensemble_configs[config_name]
        for model, weight in weights.items():
            st.sidebar.caption(f"  {model}: {weight:.2f}")
st.session_state.selected_ensembles = selected_ensembles

# ============================================================================
# LABEL MAPPINGS
# ============================================================================
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_NAMES = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
LABEL_NAMES_REVERSE = {"Entailment": 0, "Neutral": 1, "Contradiction": 2}
# Map model output strings to display names
MODEL_LABEL_TO_DISPLAY = {
    "entailment": "Entailment",
    "neutral": "Neutral",
    "contradiction": "Contradiction"
}
LABEL_COLORS = {
    "Entailment": "#00CC96",
    "Neutral": "#636EFA",
    "Contradiction": "#EF553B"
}

# ============================================================================
# CACHE MODELS
# ============================================================================
@st.cache_resource
def load_model(model_id):
    """Load a model from Hugging Face with caching."""
    try:
        pipe = pipeline("zero-shot-classification", model=model_id, device=0 if torch.cuda.is_available() else -1)
        return pipe
    except Exception as e:
        st.error(f"Error loading {model_id}: {str(e)}")
        return None

@st.cache_resource
def load_nli_model(model_id):
    """Load NLI-specific model."""
    try:
        pipe = pipeline("zero-shot-classification", model=model_id, device=0 if torch.cuda.is_available() else -1)
        return pipe
    except Exception as e:
        st.error(f"Error loading {model_id}: {str(e)}")
        return None

# ============================================================================
# MAIN INTERFACE
# ============================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Input")
    premise = st.text_area(
        "**Premise:**",
        placeholder="Enter the premise here...",
        height=100,
        key="premise"
    )
    
    hypothesis = st.text_area(
        "**Hypothesis:**",
        placeholder="Enter the hypothesis here...",
        height=100,
        key="hypothesis"
    )
    
    run_inference = st.button("🚀 Run Inference", type="primary", use_container_width=True)

# ============================================================================
# INFERENCE LOGIC
# ============================================================================
if run_inference and premise.strip() and hypothesis.strip():
    with st.spinner("⏳ Loading models and running inference..."):
        # Auto-select base models if ensemble is selected but no base models are
        models_to_run = list(st.session_state.selected_models)
        
        # If ensemble is selected but no base models, automatically run all base models
        if st.session_state.selected_ensembles and not models_to_run:
            models_to_run = list(models_config.keys())
            st.info("📌 Auto-running all base models (required for ensemble calculation)")
        
        if not models_to_run and not st.session_state.selected_ensembles:
            st.error("❌ Please select at least one base model or ensemble configuration!")
        else:
            # Load models
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # For Turkish models, we'll use zero-shot classification
            candidate_labels = ["entailment", "contradiction", "neutral"]
            
            # Dictionary to store predictions
            predictions = {}
            probabilities = {}
            
            # Map display names to model IDs
            model_id_map = {
                "BERT (Turkish)": "dbmdz/bert-base-turkish-cased",
                "mDeBERTa": "microsoft/mdeberta-v3-base",
                "Qwen2 7B": "Qwen/Qwen2-7B-Instruct",
                "Gemma3 27B": "google/gemma-3-27b-it",
            }
            
            # Filter to only models that need to run
            model_list = [(name, model_id_map[name]) for name in models_to_run]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (model_name, model_id) in enumerate(model_list):
                try:
                    status_text.text(f"Loading {model_name}...")
                    
                    # Load model
                    classifier = load_nli_model(model_id)
                    
                    if classifier is not None:
                        status_text.text(f"Running inference with {model_name}...")
                        
                        # Run inference
                        result = classifier(
                            f"{premise} [SEP] {hypothesis}",
                            candidate_labels,
                            multi_class=False
                        )
                        
                        # Store prediction
                        pred_label = result["labels"][0]
                        scores = {label: score for label, score in zip(result["labels"], result["scores"])}
                        
                        predictions[model_name] = pred_label
                        probabilities[model_name] = scores
                        
                except Exception as e:
                    st.warning(f"⚠️ Error with {model_name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(model_list) if model_list else 1)
            
            status_text.empty()
            progress_bar.empty()
        
        # ========================================================================
        # DISPLAY RESULTS
        # ========================================================================
        
        # Only show tabs if we have models or ensembles selected
        if not predictions and not st.session_state.selected_ensembles:
            st.info("✅ Select base models to see their predictions")
        else:
            # Create tabs for different views
            tabs = []
            if models_to_run:  # Show base models tab if any models were run
                tabs.append("Base Models")
            if st.session_state.selected_ensembles:
                tabs.append("Ensemble Configurations")
            tabs.append("Detailed Analysis")
            
            tab_dict = {}
            tab_objs = st.tabs(tabs)
            for i, tab_name in enumerate(tabs):
                tab_dict[tab_name] = tab_objs[i]
            
            # ====== TAB 1: Base Models ======
            if "Base Models" in tab_dict:
                with tab_dict["Base Models"]:
                    st.subheader("🤖 Base Model Predictions")
                    
                    results_cols = st.columns(2)
                    
                    for idx, (model_name, pred_label) in enumerate(predictions.items()):
                        with results_cols[idx % 2]:
                            st.markdown(f"### {model_name}")
                            
                            # Display prediction
                            pred_display = MODEL_LABEL_TO_DISPLAY.get(pred_label, pred_label)
                            pred_color = LABEL_COLORS.get(pred_display, "#000000")
                            st.markdown(f"**Prediction:** <span style='color:{pred_color}; font-size:20px;'>{pred_display}</span>", unsafe_allow_html=True)
                            
                            # Pie chart for this model
                            model_probs = probabilities.get(model_name, {})
                            if model_probs:
                                fig = go.Figure(data=[go.Pie(
                                    labels=[MODEL_LABEL_TO_DISPLAY.get(k, k) for k in model_probs.keys()],
                                    values=[model_probs[k] for k in model_probs.keys()],
                                    marker=dict(colors=[LABEL_COLORS[MODEL_LABEL_TO_DISPLAY.get(k, k)] for k in model_probs.keys()]),
                                    textposition='inside',
                                    textinfo='label+percent',
                                    hovertemplate='<b>%{label}</b><br>%{value:.2%}<extra></extra>'
                                )])
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    showlegend=False,
                                    font=dict(size=12)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show probabilities as bar chart
                                st.write("**Confidence Scores:**")
                                prob_df = pd.DataFrame([
                                    {
                                        "Label": LABEL_NAMES.get(k, k),
                                        "Confidence": f"{model_probs[k]:.2%}"
                                    }
                                    for k in model_probs.keys()
                                ]).sort_values("Confidence", ascending=False)
                                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # ====== TAB 2: Ensemble Configurations ======
            if "Ensemble Configurations" in tab_dict:
                with tab_dict["Ensemble Configurations"]:
                    st.subheader("🔗 Ensemble Configurations")
                    
                    ensemble_results = {}
                    
                    for config_name, weights in [(name, ensemble_configs[name]) for name in st.session_state.selected_ensembles]:
                        # Calculate weighted soft voting
                        aggregated_scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                        
                        for model_name, model_key in [
                            ("BERT (Turkish)", "bert"),
                            ("mDeBERTa", "mdeberta"),
                            ("Qwen2 7B", "qwen"),
                            ("Gemma3 27B", "gemma"),
                        ]:
                            if model_name in probabilities:
                                weight = weights.get(model_key, 0)
                                for label, score in probabilities[model_name].items():
                                    aggregated_scores[label] += weight * score
                        
                        # Normalize
                        total = sum(aggregated_scores.values())
                        if total > 0:
                            aggregated_scores = {k: v / total for k, v in aggregated_scores.items()}
                        
                        # Get ensemble prediction
                        ensemble_pred = max(aggregated_scores.items(), key=lambda x: x[1])[0]
                        ensemble_results[config_name] = {
                            "prediction": ensemble_pred,
                            "probabilities": aggregated_scores
                        }
                    
                    # Display each ensemble
                    ensemble_cols = st.columns(len(st.session_state.selected_ensembles))
                    
                    for col_idx, config_name in enumerate(st.session_state.selected_ensembles):
                        weights = ensemble_configs[config_name]
                        with ensemble_cols[col_idx]:
                            st.markdown(f"#### {config_name.split('(')[0].strip()}")
                            
                            ensemble_pred = ensemble_results[config_name]["prediction"]
                            ensemble_probs = ensemble_results[config_name]["probabilities"]
                            
                            # Display prediction
                            pred_display = MODEL_LABEL_TO_DISPLAY.get(ensemble_pred, ensemble_pred)
                            pred_color = LABEL_COLORS.get(pred_display, "#000000")
                            st.markdown(f"**Prediction:** <span style='color:{pred_color}; font-size:18px;'>{pred_display}</span>", unsafe_allow_html=True)
                            
                            # Weights
                            with st.expander("View Weights"):
                                for model, weight in weights.items():
                                    st.caption(f"{model}: {weight:.2f}")
                            
                            # Pie chart
                            fig = go.Figure(data=[go.Pie(
                                labels=[MODEL_LABEL_TO_DISPLAY.get(k, k) for k in ensemble_probs.keys()],
                                values=[ensemble_probs[k] for k in ensemble_probs.keys()],
                                marker=dict(colors=[LABEL_COLORS[MODEL_LABEL_TO_DISPLAY.get(k, k)] for k in ensemble_probs.keys()]),
                                textposition='inside',
                                textinfo='label+percent',
                                hovertemplate='<b>%{label}</b><br>%{value:.2%}<extra></extra>'
                            )])
                            fig.update_layout(
                                height=350,
                                margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False,
                                font=dict(size=11)
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # ====== TAB 3: Detailed Analysis ======
            with tab_dict["Detailed Analysis"]:
                st.subheader("📈 Detailed Analysis")
                
                if predictions or st.session_state.selected_ensembles:
                    # Create comparison table
                    if predictions:
                        st.write("**All Model Predictions:**")
                        
                        comparison_data = []
                        for model_name in predictions.keys():
                            pred_label = predictions[model_name]
                            probs = probabilities.get(model_name, {})
                            comparison_data.append({
                                "Model": model_name,
                                "Prediction": MODEL_LABEL_TO_DISPLAY.get(pred_label, pred_label),
                                "Entailment": f"{probs.get('entailment', 0):.2%}",
                                "Neutral": f"{probs.get('neutral', 0):.2%}",
                                "Contradiction": f"{probs.get('contradiction', 0):.2%}",
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Ensemble comparison
                    if st.session_state.selected_ensembles:
                        st.write("\n**Ensemble Predictions:**")
                        
                        ensemble_data = []
                        for config_name in st.session_state.selected_ensembles:
                            result = ensemble_results.get(config_name, {})
                            if result:
                                pred_label = result["prediction"]
                                probs = result["probabilities"]
                                ensemble_data.append({
                                    "Configuration": config_name,
                                    "Prediction": MODEL_LABEL_TO_DISPLAY.get(pred_label, pred_label),
                                    "Entailment": f"{probs.get('entailment', 0):.2%}",
                                    "Neutral": f"{probs.get('neutral', 0):.2%}",
                                    "Contradiction": f"{probs.get('contradiction', 0):.2%}",
                                })
                        
                        if ensemble_data:
                            ensemble_df = pd.DataFrame(ensemble_data)
                            st.dataframe(ensemble_df, use_container_width=True, hide_index=True)
                    
                    # Model agreement
                    if predictions:
                        st.write("\n**Model Agreement Analysis:**")
                        
                        predictions_only = list(predictions.values())
                        unique_preds = set(predictions_only)
                        
                        if len(unique_preds) == 1:
                            st.success("✅ All models agree!")
                        else:
                            st.info(f"⚠️ Models disagree: {len(unique_preds)} different predictions")
                            
                            # Count agreement
                            agreement_counts = {}
                            for pred in predictions_only:
                                pred_name = MODEL_LABEL_TO_DISPLAY.get(pred, pred)
                                agreement_counts[pred_name] = agreement_counts.get(pred_name, 0) + 1
                            
                            agreement_df = pd.DataFrame([
                                {"Prediction": pred, "Count": count}
                                for pred, count in agreement_counts.items()
                            ]).sort_values("Count", ascending=False)
                            
                            st.dataframe(agreement_df, use_container_width=True, hide_index=True)
                        
                        # Confidence overview
                        st.write("\n**Confidence Overview:**")
                        
                        max_confidences = []
                        for model_name in predictions.keys():
                            probs = probabilities.get(model_name, {})
                            max_conf = max(probs.values()) if probs else 0
                            max_confidences.append({
                                "Model": model_name,
                                "Max Confidence": f"{max_conf:.2%}"
                            })
                        
                        conf_df = pd.DataFrame(max_confidences).sort_values("Max Confidence", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                        st.dataframe(conf_df, use_container_width=True, hide_index=True)

else:
    if run_inference:
        st.warning("⚠️ Please enter both premise and hypothesis to run inference.")
    else:
        st.info("👈 Enter a premise and hypothesis in the input fields and click 'Run Inference' to get started!")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Turkish NLI Inference System | Models from Hugging Face | Ensemble Comparison
    </div>
    """,
    unsafe_allow_html=True
)
