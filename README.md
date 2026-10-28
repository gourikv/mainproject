# Stylometric-Transformer Hybrid AI Text Detector

An advanced, production-ready AI text detection system combining semantic deep learning representations (**RoBERTa-base**) with 16 hand-crafted **linguistic stylometric features**. 

The system distinguishes human-authored content from large language model (LLM / ChatGPT) generations with state-of-the-art accuracy, provides feature-level interpretability, supports long-text chunked analysis, and features an interactive Streamlit web application with drag-and-drop PDF ingestion.

---

## 📊 Key Evaluation Metrics

Evaluated on the held-out test split of the **HC3 (Human ChatGPT Comparison Corpus)** benchmark dataset. The Hybrid Classifier combines RoBERTa semantic embeddings with normalized stylometric vectors, outperforming pure transformer baselines:

| Metric | RoBERTa Baseline | Hybrid Model (RoBERTa + Stylometrics) | Delta |
|:---|:---:|:---:|:---:|
| **Accuracy** | 95.47% | **96.79%** | **+1.32%** |
| **Precision (AI Class)** | 94.85% | **96.14%** | **+1.29%** |
| **Recall (AI Class)** | 96.20% | **97.48%** | **+1.28%** |
| **F1-Score (AI Class)** | 95.52% | **96.81%** | **+1.29%** |

---

## 🧠 Architecture Overview

The system bridges deep contextual representations with classical computational linguistics:

1. **Transformer Encoder Branch**:
   - `roberta-base` extracts 768-dimensional contextual representations from tokenized text (`[CLS]` token embedding).
2. **Stylometric Feature Branch**:
   - 16 interpretable linguistic features extracted across four linguistic dimensions:
     - **Sentence Structure**: Average sentence length, standard deviation of sentence length (burstiness), total sentence count.
     - **Lexical Diversity**: Type-Token Ratio (TTR), average word length.
     - **Part-of-Speech (POS) Ratios**: Noun, verb, adjective, adverb, and pronoun distribution via spaCy.
     - **Punctuation & Readability**: Comma frequency, period frequency, question mark frequency, average paragraph length, Flesch-Kincaid Grade Level, Flesch Reading Ease.
   - Scaled using a fitted `StandardScaler` (`saved_models/scaler.pkl`) and projected through a dense layer (16 &rarr; 128 hidden units with ReLU and Dropout).
3. **Fusion Classifier**:
   - Concatenates the 768-dim RoBERTa representation with the 128-dim stylometric representation (896 dimensions total).
   - Passes through a multi-layer classification head (896 &rarr; 256 &rarr; 2 classes) with Softmax for calibrated probabilities.

---

## 📦 Dataset

**HC3 – Human ChatGPT Comparison Corpus** ([Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3))
- Covers multiple domains: Open QA, Reddit ELI5, Wikipedia Q&A, Medicine, Finance, and Law.
- Contains question-response pairs with verified human responses matched against ChatGPT responses.
- Partitioned into train, validation, and test splits with precomputed stylometric feature tables (`data/processed/`).

---

## 📁 Repository Structure & Main Modules

```
mainproject/
├── app/
│   └── streamlit_app.py      # Streamlit web dashboard (direct text input & PDF upload)
├── src/
│   ├── features.py           # 16-feature stylometric extraction engine (spaCy + textstat)
│   ├── preprocess.py         # Text normalization and dataset processing
│   └── load_data.py          # HC3 dataset ingestion from HuggingFace
├── analysis/
│   ├── evaluate_models.py    # Side-by-side benchmark evaluation on HC3 test split
│   ├── compute_feature_stats.py # Computes feature distribution benchmarks for explainability
│   ├── feature_distributions.py # Statistical significance and distribution analysis
│   └── plot_comparison.py    # Metric comparison charts and confusion matrices
├── saved_models/
│   ├── hybrid_model.pt       # PyTorch checkpoint for trained HybridClassifier
│   ├── scaler.pkl            # Fitted StandardScaler for 16 stylometric features
│   ├── feature_stats.json    # Reference training set statistics for explainability
│   └── roberta_baseline_final/ # Fine-tuned standalone RoBERTa baseline model
├── predict.py                # Core end-to-end inference script (single pass)
├── predict_long.py           # Chunked inference pipeline for long-form documents
├── explain.py                # Feature-level interpretability and heuristic attribution
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

### Module Descriptions
- **[`predict.py`](predict.py)**: Loads the trained hybrid model checkpoint, tokenizer, and fitted feature scaler. Extracts features from input text, normalizes them, and executes the forward pass to produce binary classification verdicts and probabilities.
- **[`predict_long.py`](predict_long.py)**: Handles long documents (>150 words) by partitioning text at natural sentence boundaries into ~150-word chunks. Computes individual chunk probabilities and aggregates them via a word-count-weighted average, preventing truncation loss.
- **[`explain.py`](explain.py)**: Evaluates extracted stylometrics against class-conditional reference averages (`feature_stats.json`) to determine directional pull (Human vs. AI) for each metric, powering the visual explanation panel.
- **[`app/streamlit_app.py`](app/streamlit_app.py)**: Interactive web application featuring a modern dark-mode UI, live sample presets, PDF document extraction (`pypdf`), Plotly probability gauges, chunk-by-chunk probability progression, and an expandable 16-feature diagnostic drawer.
- **[`analysis/`](analysis/)**: Reproducible evaluation scripts and plotting utilities used to benchmark model performance, analyze feature rankings, and generate report figures.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.12)
- Recommended: 8 GB+ RAM

### 1. Clone & Set Up Virtual Environment

```bash
# Clone repository and navigate to project root
cd mainproject

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download spaCy Language Model

The stylometric pipeline uses spaCy's English model (`en_core_web_sm`) for POS tagging and sentence boundary detection:

```bash
python -m spacy download en_core_web_sm
```

*(Note: If not manually downloaded, `predict.py` will automatically attempt to download it on first run.)*

---

## 💻 Running the Application

Launch the Streamlit web dashboard locally:

```bash
streamlit run app/streamlit_app.py
```

The app will start at `http://localhost:8501`.

### App Features:
- **Direct Text Input & Presets**: Test custom text or load pre-configured human and AI sample presets.
- **PDF Document Ingestion**: Upload multi-page `.pdf` files; text is automatically extracted, previewed, and analyzed.
- **Chunked Long-Text Analysis**: Visualizes chunk-by-chunk AI probability transitions across long essays or articles.
- **Linguistic Diagnostics Drawer**: Expandable breakdown of all 16 stylometric metrics comparing input text against typical human and AI baselines.

---

## 🔬 Running Model Evaluation

To reproduce the test benchmark metrics across the baseline and hybrid models:

```bash
python analysis/evaluate_models.py
```

To regenerate metric comparison visual plots:

```bash
python analysis/plot_comparison.py
```

---

## 📜 Citation & Attribution

This project utilizes the **Human ChatGPT Comparison Corpus (HC3)**:
```bibtex
@article{guo2023hc3,
  title   = {How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
  author  = {Biyang Guo and Xin Zhang and Ziyuan Wang and Minqi Jiang and Jinran Nie and Yuxuan Ding and Jianwei Yue and Yupeng Wu},
  journal = {arXiv preprint arXiv:2301.07597},
  year    = {2023}
}
```
