# Machine Learning & NLP Projects Portfolio

Two end-to-end machine learning projects solving distinct e-commerce challenges: **Logistics Risk Prediction** and  **Localized Sentiment Analysis** .

---

## Project 1: Amazon Order Delay Prediction

### Objective

Build a predictive engine that identifies high-risk orders (**Delayed** or  **Cancelled** ) using historical logistics data, enabling proactive supply chain management.

| Detail                | Value                                                    |
| --------------------- | -------------------------------------------------------- |
| **Dataset**     | 100,000+ historical records                              |
| **Model**       | `RandomForestClassifier`                               |
| **Accuracy**    | 96.06%                                                   |
| **Core Skills** | Feature Engineering, Label Encoding, Supervised Learning |

---

## Project 2: Shopee Review Sentiment Analysis (Manglish)

### Objective

Standard NLP libraries often fail on "Bahasa Rojak" — mixed Malay-English slang common in Malaysian e-commerce. This project features a custom pipeline specifically tuned for this linguistic landscape.

### Key Innovation: Localized Preprocessing

* **Slang Normalization:** A dictionary-based mapper for 9,000+ short-forms (e.g., `jg` → `juga`, `sdp` → `sedap`)
* **Vectorization:** TF-IDF with `ngram_range=(1, 2)` to capture local context and sentiment-heavy phrases
* **Deployment:** Integrated with **FastAPI** for real-time inference

### Model Performance

| Sentiment          | Precision | Recall | F1-Score |
| ------------------ | --------- | ------ | -------- |
| **Negative** | 75%       | 75%    | 75%      |
| **Neutral**  | 44%       | 54%    | 48%      |
| **Positive** | 86%       | 77%    | 81%      |

**Overall Accuracy: 72%**

> The model is highly effective at distinguishing polarized (Positive/Negative) feedback, which is critical for business escalation workflows.

---

## API Deployment in Action

The model is served via a **Uvicorn** ASGI server. Below is a live demonstration handling a Manglish review.

### 1. Request Interface

The API accepts a JSON payload through the interactive Swagger UI. Test phrase:

> *"Best gila! Item received safely and the packaging is damn secure."*

<p align="center">
  <img src="api_test_1.png" width="800" alt="FastAPI Input Interface">
</p>
### 2. Prediction Result

The backend processes the slang-heavy text and returns a sentiment classification with a confidence score. The model correctly identifies **Positive** sentiment with **47.18%** confidence.

<p align="center">
  <img src="api_test_2.png" width="800" alt="API Response Output">
</p>
---

## Tech Stack

| Category             | Tools                                   |
| -------------------- | --------------------------------------- |
| **ML Core**    | Python, Scikit-learn, Pandas, Joblib    |
| **NLP**        | TF-IDF, Regex-based Slang Normalization |
| **Deployment** | FastAPI, Uvicorn, JSON API              |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the API
uvicorn main:app --reload
```

Then navigate to `http://127.0.0.1:8000/docs` to access the interactive Swagger UI.
