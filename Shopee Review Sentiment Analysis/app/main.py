# Load necessary libraries
import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the web server
app = FastAPI(
    title="Malay Sentiment Analysis API",
)

# Load the pre-trained model and vectorizer when the server starts
model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Load slang mapping 
slang = {
    "tk": "tidak", "x": "tidak", "tak": "tidak", "xde": "tak ada",
    "tp": "tapi", "sgt": "sangat", "mcm": "macam", "jgk": "juga",
    "brg": "barang", "nk": "nak", "dh": "dah", "yg": "yang",
    "utk": "untuk", "sdp": "sedap", "giler": "gila"
}

# Text cleaning function to prepare the review for the model
def clean_text(raw_text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())
    
    words = text.split()
    standardized_words = [slang.get(w, w) for w in words]
    
    return " ".join(standardized_words)

# Define the input data model for the API
class ReviewInput(BaseModel):
    text: str

# A simple home page to check if the server is running
@app.get("/")
def home():
    return {"message": "System is live! Use the /predict endpoint to analyze reviews"}

# The main endpoint to receive reviews and return sentiment predictions
@app.post("/predict")
async def predict_sentiment(review: ReviewInput):
    cleaned = clean_text(review.text)
    
    vector = tfidf.transform([cleaned])
    
    prediction_num = model.predict(vector)[0]
    all_probabilities = model.predict_proba(vector)[0]
    
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    final_sentiment = labels.get(int(prediction_num), "Unknown")
    
    # Format the response with the original review, cleaned text, predicted sentiment, and confidence score
    return {
        "status": "success",
        "data": {
            "input": review.text,
            "cleaned text": cleaned,
            "prediction": final_sentiment,
            "confidence_score": f"{max(all_probabilities):.2%}"
        }
    }