import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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
COMMENT_MODEL_PATH = "fake_comment_model.pkl"
COMMENT_VECTORIZER_PATH = "vectorizer.pkl"
SPAM_MODEL_PATH = "emailmodel.pkl"
SPAM_VECTORIZER_PATH = "spam_vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"
SPAM_DATASET_PATH = "spam_dataset.csv"

# Function to safely load models and vectorizers
def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path):
        logger.error(f"{model_name} Model file not found: {model_path}")
        raise FileNotFoundError(f"{model_name} Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        logger.error(f"{model_name} Vectorizer file not found: {vectorizer_path}")
        raise FileNotFoundError(f"{model_name} Vectorizer file not found: {vectorizer_path}")
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"{model_name} model and vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading {model_name} model or vectorizer: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name} model/vectorizer due to: {str(e)}")

# Load models and vectorizers safely
comment_model, comment_vectorizer = load_model_and_vectorizer(COMMENT_MODEL_PATH, COMMENT_VECTORIZER_PATH, "Comment Detection")

# Ensure spam vectorizer exists, otherwise train a new one
if not os.path.exists(SPAM_VECTORIZER_PATH):
    try:
        logger.warning("Spam vectorizer not found. Training a new one.")
        df = pd.read_csv(SPAM_DATASET_PATH)
        df.columns = df.columns.str.strip().str.lower()
        if "email_content" not in df.columns:
            raise ValueError(f"The dataset must contain an 'email_content' column. Found columns: {df.columns}")
        
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(df["email_content"])
        print("Feature Names:", vectorizer.get_feature_names_out())
        print("Number of Features:", len(vectorizer.get_feature_names_out()))
        joblib.dump(vectorizer, SPAM_VECTORIZER_PATH)
        logger.info("Spam vectorizer trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training spam vectorizer: {str(e)}")
        raise RuntimeError(f"Failed to train spam vectorizer: {str(e)}")

# Load spam model and vectorizer after ensuring vectorizer exists
spam_model, spam_vectorizer = load_model_and_vectorizer(SPAM_MODEL_PATH, SPAM_VECTORIZER_PATH, "Spam Detection")

# Debugging: Check vectorizer feature count
print("Vectorizer feature count:", len(spam_vectorizer.get_feature_names_out()))

# Request and Response Models
class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class EmailRequest(BaseModel):
    content: str

class EmailResponse(BaseModel):
    content: str
    is_spam: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = comment_vectorizer.transform([request.comment])
        prediction = comment_model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except Exception as e:
        logger.error(f"Error processing comment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/detect_spam_email", response_model=EmailResponse)
async def detect_spam_email(request: EmailRequest):
    try:
        if not request.content.strip():
            logger.warning("Empty email content received.")
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")
        
        if len(request.content.strip()) < 3:
            logger.warning("Short email content received, defaulting to not spam.")
            return EmailResponse(content=request.content, is_spam=False)
        
        processed_content = re.sub(r"[^\w\s]", "", request.content.lower())
        transformed_content = spam_vectorizer.transform([processed_content])
        
        expected_features = 5000
        if transformed_content.shape[1] == expected_features:
            logger.error(f"Feature mismatch: Expected {expected_features}, but got {transformed_content.shape[1]}")
            raise ValueError(f"Feature mismatch: Expected {expected_features}, but got {transformed_content.shape[1]}")
        
        prediction_proba = spam_model.predict_proba(transformed_content)[0]
        is_spam = prediction_proba[1] > 0.5  # Set threshold at 50%
        
        logger.info(f"Email: {request.content} -> Predicted: {'spam' if is_spam else 'not spam'} with probability {prediction_proba[1]:.4f}")
        return EmailResponse(content=request.content, is_spam=is_spam)
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
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
        logger.info(f"Added new comment: {request.comment}")
        return {"message": "Comment added successfully."}
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
