from fastapi import FastAPI
from api.routes.predict import router as predict_router
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="Loan Default Risk API",
    version="1.0.0"
)

app.include_router(predict_router)


app = FastAPI(
    title="Loan Default Risk API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)