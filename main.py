import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of valid API keys
VALID_API_KEYS = ["actual_api_key_1", "actual_api_key_2"]

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to the pre-trained model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int

# Load the pre-trained model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Model and vectorizer files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the directory.")

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = vectorizer.transform([request.comment])
        prediction = model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/add_comment")
async def add_comment(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(DATASET_PATH, index=False)
        
        logger.info(f"Added new comment and retraining model: {request.comment}")
        
        return {"message": "Comment added and model retrained successfully."}
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise 