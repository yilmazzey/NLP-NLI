# Turkish NLI Frontend Application

A web-based interface for comparing Turkish Natural Language Inference (NLI) predictions across multiple models and ensemble configurations.

## Features

✨ **4 Base Models**
- BERT (Turkish): `dbmdz/bert-base-turkish-cased`
- mDeBERTa: `microsoft/mdeberta-v3-base`
- Qwen2 7B: `Qwen/Qwen2-7B-Instruct`
- Gemma3 27B: `google/gemma-3-27b-it`

✨ **3 Ensemble Configurations**
1. **Balanced Ensemble**: Qwen (0.40) | Gemma (0.20) | mDeBERTa (0.10) | BERT (0.30)
2. **Optimized Ensemble**: Gemma (0.30) | Qwen (0.50) | mDeBERTa (0.15) | BERT (0.05)
3. **Current Ensemble**: Gemma (0.30) | Qwen (0.45) | mDeBERTa (0.15) | BERT (0.10)

✨ **Visualizations**
- Pie charts for confidence distribution across labels (Entailment, Neutral, Contradiction)
- Side-by-side comparison of all models
- Model agreement analysis
- Detailed probability tables

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_frontend.txt
```

### 2. (Optional) Set Up GPU Support

For faster inference, install CUDA-enabled PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. (Optional) Hugging Face Authentication

If using gated models, authenticate with Hugging Face:

```bash
huggingface-cli login
```

## Usage

### Run the Application

```bash
streamlit run frontend_app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use

1. **Enter Input**:
   - Enter a **Premise** (the context sentence)
   - Enter a **Hypothesis** (the statement to verify)

2. **Run Inference**:
   - Click the "🚀 Run Inference" button
   - Wait for all models to load and process

3. **View Results**:
   - **Base Models Tab**: Individual predictions and confidence distributions for each model
   - **Ensemble Configurations Tab**: Predictions from the three ensemble setups with pie charts
   - **Detailed Analysis Tab**: Comparison tables, model agreement metrics, and confidence overview

### Example Input

**Premise:** "Bir kedi masanın üzerinde oturuyor."
(A cat is sitting on the table.)

**Hypothesis:** "Hayvan masanın üzerinde oturuyor."
(An animal is sitting on the table.)

**Expected:** Entailment (the hypothesis follows from the premise)

## Architecture

### File Structure

```
frontend_app.py              # Main Streamlit application
requirements_frontend.txt    # Python dependencies
```

### Key Components

1. **Model Loading**: Uses `transformers` library with Hugging Face models
2. **Inference**: Zero-shot classification for NLI task
3. **Ensemble**: Weighted soft voting across all models
4. **Visualization**: Plotly pie charts and Streamlit DataFrames

## Performance Notes

### Model Loading Time

First run will download models (~2-10GB total):
- BERT (Turkish): ~350MB
- mDeBERTa: ~500MB
- Qwen2 7B: ~15GB
- Gemma3 27B: ~50GB

**Tip**: Use smaller models first to test installation

### GPU Acceleration

- With CUDA GPU: ~5-15 seconds per inference
- On CPU: ~30-60 seconds per inference

### Model Caching

Streamlit caches loaded models, so subsequent runs are much faster.

## Customization

### Add or Modify Models

Edit the `models_config` dictionary in `frontend_app.py`:

```python
"Your Model Name": {
    "model_id": "your-org/your-model",
    "hf_task_id": "your-org/your-model",
    "enabled": True
}
```

### Modify Ensemble Weights

Edit the `ensemble_configs` dictionary:

```python
"My Ensemble": {
    "qwen":     0.35,
    "gemma":    0.35,
    "mdeberta": 0.20,
    "bert":     0.10,
}
```

### Change Label Colors

Modify `LABEL_COLORS` dictionary:

```python
LABEL_COLORS = {
    "Entailment": "#00CC96",      # Green
    "Neutral": "#636EFA",          # Blue
    "Contradiction": "#EF553B"     # Red
}
```

## Troubleshooting

### Models Won't Load

**Issue**: `RuntimeError: CUDA out of memory`

**Solution**: Use CPU or reduce batch size
```bash
CUDA_VISIBLE_DEVICES="" streamlit run frontend_app.py
```

### Slow Inference

**Issue**: Taking too long per inference

**Solution**: Reduce number of loaded models or use smaller models

### ModuleNotFoundError

**Issue**: Missing packages

**Solution**: Reinstall requirements
```bash
pip install --upgrade -r requirements_frontend.txt
```

### Hugging Face Authentication

**Issue**: `You have passed a `repo_id`... but not authenticated`

**Solution**: Login to Hugging Face
```bash
huggingface-cli login
```

## API Integration

For programmatic use without Streamlit UI, create a `predict.py`:

```python
from transformers import pipeline

def get_nli_prediction(premise, hypothesis, model_id):
    classifier = pipeline("zero-shot-classification", model=model_id)
    result = classifier(
        f"{premise} [SEP] {hypothesis}",
        ["entailment", "neutral", "contradiction"],
        multi_class=False
    )
    return result
```

## Performance Metrics

On a typical system:

| Model | Load Time | Inference Time |
|-------|-----------|-----------------|
| BERT | 30s | 2s |
| mDeBERTa | 45s | 3s |
| Qwen2 7B | 60s | 8s |
| Gemma3 27B | 120s | 15s |
| Ensemble | - | 30s (all 4 models) |

## License

This application uses Hugging Face models. Check individual model licenses:
- BERT: Apache 2.0
- mDeBERTa: MIT
- Qwen2: Tongyi Qianwen License
- Gemma: Gemma Terms of Use

## Support

For issues:
1. Check Streamlit logs: `streamlit run frontend_app.py --logger.level=debug`
2. Verify Hugging Face models are available
3. Check CUDA/PyTorch installation

## Future Enhancements

- [ ] Support for English NLI models
- [ ] Fine-tuning interface
- [ ] Batch inference
- [ ] Model performance comparison charts
- [ ] Export results to CSV/JSON
- [ ] API endpoint wrapper
- [ ] Dark mode theme
