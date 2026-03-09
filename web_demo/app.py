import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from preprocessing.tokenizer import tokenize, normalize_tokens

# ================= APP =================
app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL =================
EXP_PATH = "experiments/exp_016_XGBoost_400k"

with open(f"{EXP_PATH}/model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{EXP_PATH}/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ================= SCHEMA =================
class CodePair(BaseModel):
    code1: str
    code2: str

# ================= PREPROCESS =================
def preprocess(code: str) -> str:
    tokens = tokenize(code)
    norm_tokens = normalize_tokens(tokens)
    return " ".join(norm_tokens)

# ================= ROUTES =================
@app.get("/", response_class=HTMLResponse)
def home():
    with open("web_demo/index.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict(pair: CodePair):

    code1 = preprocess(pair.code1)
    code2 = preprocess(pair.code2)

    X1 = vectorizer.transform([code1])
    X2 = vectorizer.transform([code2])

    X_pair = abs(X1 - X2)

    prob = model.predict_proba(X_pair)[0][1]

    return {
        "probability": float(prob),
        "prediction": "Duplicated" if prob > 0.98 else "Not Duplicated"
    }
