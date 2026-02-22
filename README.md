# Machine Learning & NLP Projects Portfolio

This repository features two end-to-end machine learning projects solving distinct e-commerce challenges: **Logistics Risk Prediction** and **Localized Sentiment Analysis.**

---

## Project 1: Amazon Order Delay Prediction

### Objective

To build a predictive engine that identifies high-risk orders (**Delayed** or  **Cancelled** ) using historical logistics data, allowing for proactive supply chain management.

* **Dataset:** 100,000+ historical records.
* **Model:** `RandomForestClassifier` (Selected for its ability to handle non-linear tabular data).
* **Performance:**  **96.06% Accuracy** .
* **Core Skills:** Feature Engineering, Label Encoding, Supervised Learning.

---

## Project 2: Shopee Review Sentiment Analysis (Manglish)

### Objective

Standard NLP libraries often fail on "Bahasa Rojak" (mixed Malay-English slang). This project features a custom pipeline specifically tuned for the Malaysian e-commerce landscape.

### Key Innovation: Localized Preprocessing

* **Slang Normalization:** A dictionary-based mapper for 9,000+ short-forms (e.g., `jg` → `juga`, `sdp` → `sedap`).
* **Vectorization:** TF-IDF with `ngram_range=(1, 2)` to capture local context and sentiment-heavy phrases.
* **Deployment:** Integrated with **FastAPI** for real-time inference.

---

## API Deployment in Action

The model is served via a **Uvicorn** ASGI server. Below is a demonstration of the live API handling a "Manglish" review.

### 1. The Request Interface

The API accepts a JSON payload through the interactive Swagger UI. Here, we test the phrase:

> *"Best gila! Item received safely and the packaging is damn secure."*

<p align="center">

<img src="Shopee Review Sentiment Analysis/api_test_1.png" width="800" alt="FastAPI Input Interface">

</p>

### 2. The Prediction Result

The backend processes the slang-heavy text and returns a sentiment classification with a confidence score. As seen below, the model correctly identifies the **Positive** sentiment with a **47.18%** confidence level.

<p align="center">

<img src="Shopee Review Sentiment Analysis/api_test_2.png" width="800" alt="API Response Output">

</p>

---

## Model Performance Metrics

| **Sentiment** | **Precision** | **Recall** | **F1-Score** |
| ------------------- | ------------------- | ---------------- | ------------------ |
| **Negative**  | 75%                 | 75%              | 75%                |
| **Neutral**   | 44%                 | 54%              | 48%                |
| **Positive**  | 86%                 | 77%              | 81%                |

**Overall Accuracy: 72%** *Note: The model is highly effective at distinguishing polarized (Positive/Negative) feedback, which is critical for business escalation.*

---

## Tech Stack

* **ML Core:** Python, Scikit-learn, Pandas, Joblib
* **NLP:** TF-IDF, Regex-based Slang Normalization
* **Deployment:** FastAPI, Uvicorn, JSON API

---

### How to Run Locally

1. **Clone the repo:** `git clone <repo-url>`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Launch API:** `uvicorn main:app --reload`
4. **Test:** Navigate to `http://127.0.0.1:8000/docs` to use the interactive UI shown in the screenshots above.
