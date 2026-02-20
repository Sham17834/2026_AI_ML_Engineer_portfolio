# Import necessary libraries
import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Malay Sentiment Analysis API")

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Define a mapping for common Malay slang to their formal equivalents
SLANG_MAP = {
    "tk": "tidak", "x": "tidak", "tak": "tidak", "xde": "tak ada",
    "tp": "tapi", "sgt": "sangat", "mcm": "macam", "jgk": "juga",
    "brg": "barang", "nk": "nak", "dh": "dah", "yg": "yang",
    "utk": "untuk", "sdp": "sedap", "giler": "gila"
}

# Function to clean and preprocess the input text
def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    words = text.split()
    cleaned_words = [SLANG_MAP.get(word, word) for word in words]
    
    return " ".join(cleaned_words)

# Define the input data model for the API
class Comment(BaseModel):
    text: str

# Define API endpoints
@app.get("/")
def home():
    return {"status": "online", "description": "Malay Sentiment API"}

# Endpoint to predict sentiment of a given comment
@app.post("/predict")
async def predict_sentiment(comment: Comment):
    
    cleaned_text = clean_text(comment.text)
    
    vectorized_input = tfidf.transform([cleaned_text])
    
    prediction = model.predict(vectorized_input)[0]
    probabilities = model.predict_proba(vectorized_input)[0]

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    result_label = label_map.get(int(prediction), "Unknown")
    
    return {
        "original_text": comment.text,
        "cleaned_text": cleaned_text,
        "sentiment": result_label,
        "confidence": f"{max(probabilities):.2%}"
    }