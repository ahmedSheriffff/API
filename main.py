import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
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
        train_model()
        
        return {"message": "Comment added and model retrained successfully."}
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Training function
def train_model():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

        if "comment" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'comment' and 'label' columns.")

        # Balance the dataset
        fake_comments = df[df["label"] == 1]
        real_comments = df[df["label"] == 0]

        min_samples = min(len(fake_comments), len(real_comments))

        df_balanced = pd.concat([
            fake_comments.sample(min_samples, random_state=42),
            real_comments.sample(min_samples, random_state=42)
        ])

        # Train the model with balanced data
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))  # Use n-grams
        X = vectorizer.fit_transform(df_balanced["comment"])
        y = df_balanced["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning using Grid Search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2'],  # Corrected values
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    else:
        raise FileNotFoundError("Dataset file not found. Please add 'fake_comments_dataset.csv'.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
