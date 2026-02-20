# Machine Learning & NLP Projects Portfolio

This repository showcases two end-to-end machine learning projects focused on real-world e-commerce use cases:

1. **Amazon Order Delay Prediction**
2. **Shopee Review Sentiment Analysis (Manglish / Bahasa Rojak)**

Both projects demonstrate practical ML engineering skills including data preprocessing, feature engineering, model training, evaluation, and API deployment.

---

# Project 1: Amazon Order Delay Prediction

## Objective

Develop a predictive model to identify whether an Amazon order will be **Delayed** or  **Cancelled** , enabling logistics teams to proactively manage delivery risks and improve customer satisfaction.

---

## Dataset Overview

* 100,000 historical Amazon order records
* Target variable: `OrderStatus` (converted into binary classification)

### Key Features

* Financial: `UnitPrice`, `Discount`, `ShippingCost`, `TotalAmount`
* Logistics: `Quantity`, `Category`, `PaymentMethod`
* Geography: `Country`, `State`, `City`

---

## Technical Workflow

### Data Preprocessing

* Removed null values and irrelevant identifiers
* Label encoding for categorical features
* 80/20 train-test split

### Model Selection

Model used: **Random Forest Classifier**

Chosen for:

* Robustness to outliers
* Strong performance on tabular e-commerce data
* Ability to capture non-linear relationships

<pre class="overflow-visible! px-0!" data-start="1569" data-end="1658"><div class="w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border corner-superellipse/1.1 border-token-border-light bg-token-bg-elevated-secondary rounded-3xl"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="pointer-events-none absolute inset-x-px top-0 bottom-96"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-bg-elevated-secondary"></div></div></div><div class="corner-superellipse/1.1 rounded-3xl bg-token-bg-elevated-secondary"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼt">model</span><span></span><span class="ͼn">=</span><span></span><span class="ͼt">RandomForestClassifier</span><span>(</span><span class="ͼt">random_state</span><span class="ͼn">=</span><span class="ͼq">42</span><span>)</span><br/><span class="ͼt">model</span><span class="ͼn">.</span><span>fit(</span><span class="ͼt">X_train</span><span>, </span><span class="ͼt">y_train</span><span>)</span></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

## Results

* **Accuracy: 96.06%**
* High precision in distinguishing successful deliveries from disruptions

---

## Skills Demonstrated

* Feature engineering
* Supervised learning
* Model evaluation
* Business-oriented ML problem framing

---

# Project 2: Shopee Review Sentiment Analysis (Manglish / Bahasa Rojak)

## Objective

Build an end-to-end NLP pipeline capable of understanding Malaysian e-commerce reviews written in  **Bahasa Rojak (mixed English-Malay slang)** .

Unlike standard NLP systems, this model is optimized for localized linguistic patterns.

---

## Key Innovation: Localized Preprocessing

### Custom Slang Normalization

Mapped 9,000+ Malaysian short-forms:

* `jg` → `juga`
* `tk` → `tidak`
* `sdp` → `sedap`

### Noise Reduction

* Regex-based cleaning
* Preserved contextual meaning
* Removed non-alphabetic noise

---

## Tech Stack

### Model

* Logistic Regression (`class_weight='balanced'`)
* Selected for interpretability and deployment efficiency

### Vectorization

* TF-IDF
* `ngram_range=(1, 2)` for contextual phrase detection

### Deployment

* **FastAPI**
* Uvicorn ASGI server
* Model serialization using Joblib

---

## Model Performance

* **Overall Accuracy: 72%**

### Classification Report

| Sentiment | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Negative  | 75%       | 75%    | 75%      |
| Neutral   | 44%       | 54%    | 48%      |
| Positive  | 86%       | 77%    | 81%      |

---

## API Deployment

The model is production-ready and deployed via FastAPI with interactive Swagger (OpenAPI) documentation.

Example API response:

<pre class="overflow-visible! px-0!" data-start="3393" data-end="3462"><div class="w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border corner-superellipse/1.1 border-token-border-light bg-token-bg-elevated-secondary rounded-3xl"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="pointer-events-none absolute inset-x-px top-0 bottom-96"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-bg-elevated-secondary"></div></div></div><div class="corner-superellipse/1.1 rounded-3xl bg-token-bg-elevated-secondary"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>{</span><br/><span>  "sentiment": </span><span class="ͼr">"Positive"</span><span>,</span><br/><span>  "confidence_score": </span><span class="ͼq">0.87</span><br/><span>}</span></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Engineering Highlights

* End-to-end ML lifecycle implementation
* Real-world e-commerce datasets
* API deployment experience
* Production-ready model serialization
* Localization-aware NLP preprocessing

---

# Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* FastAPI
* Uvicorn
* Joblib
* Jupyter Notebook
