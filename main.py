import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = FastAPI()

DATASET_PATH = "fake_comments.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    import pandas as pd
except ModuleNotFoundError:
    import subprocess
    subprocess.run(["pip", "install", "pandas"])
    import pandas as pd

def train_model():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        
        if "comment" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'comment' and 'label' columns.")
        
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df["comment"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    else:
        raise FileNotFoundError("Dataset file not found. Please add 'fake_comments_dataset.csv'.")

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    train_model()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

@app.post("/detect_fake_comment", response_model=CommentResponse)
def detect_fake_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")

    transformed_comment = vectorizer.transform([request.comment])
    prediction = model.predict(transformed_comment)
    result = bool(prediction[0])
    
    return CommentResponse(comment=request.comment, is_fake=result)

@app.post("/add_comment")
def add_comment(comment: str, label: int):
    if not comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")
    
    new_data = pd.DataFrame({"comment": [comment], "label": [label]})
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_csv(DATASET_PATH, index=False)
    
    train_model()
    return {"message": "Comment added and model retrained successfully."}
