import torch

# Python Imports
from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from typing import List

from fastapi import (
    FastAPI,
    Path,
    HTTPException,
    Body,
    Depends,
    status,
    Query,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware

from enum import Enum
from fastapi import APIRouter

from logger import get_logger_by_name

logger = get_logger_by_name("Hivata | Rare diseases | Drugs Predictor - Explainer ")

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available devices
    num_devices = torch.cuda.device_count()
    logger.info(f"Number of available CUDA devices: {num_devices}")

    # List all available devices
    for i in range(num_devices):
        logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    logger.info("CUDA is not available.")

from txgnn import TxData, TxGNN, TxEval

TxData = TxData(data_folder_path="./data")
TxData.prepare_split(split="complex_disease", seed=42)  # no_kg=False
TxGNN = TxGNN(
    data=TxData,
    weight_bias_track=False,
    proj_name="TxGNN",  # wandb project name
    exp_name="TxGNN",  # wandb experiment name
    device="cuda:0",  # define your cuda device
)
TxGNN.load_pretrained("/model")
logger.info(f"Initializing evaluation model for GNN ... {repr(TxGNN)}")
TxEval = TxEval(model=TxGNN)
logger.info(f"Evaluation model for GNN loaded successfully! {repr(TxEval)}")


# Enum for mode to limit to 'indication' or 'contradiction'
class ModeEnum(str, Enum):
    indication = "indication"
    contradiction = "contradiction"


class BodySizeLimiterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body_size = len(await request.body())
        if body_size > 512 * 1024 * 1024:  # 100 MB
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        response = await call_next(request)
        return response


load_dotenv()

example_project_id = str(uuid4())
router = APIRouter(prefix="/api/v1")
app = FastAPI(
    title="Hivata Drug Replacement Service",
    description="""Drug Replacement Service - Predictor and Explainer""",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def get_application_health():
    api_message = "Up and running!"
    return {
        "message": api_message,
    }


@router.post("/predict")
async def get_drug_replacement_prediction(
    disease_idxs: List[float] = Body(
        ...,
        title="Disease Indices",
        description="A list of disease indices for which drug replacement is requested.",
    ),
    mode: ModeEnum = Query(
        ...,
        title="Mode",
        description="Either 'indication' or 'contradiction' for the mode of evaluation.",
    ),
):
    """
    Get a drug replacement prediction based on the given list of disease indices.

    This endpoint returns a prediction or suggestion of alternative drugs
    that can be used for treating the specified diseases. The result will
    include drug names or any related treatment options.

    Args:
        disease_idxs (List[float]): A list of disease indices for which drug replacement
                          is being requested.
        mode (ModeEnum): Either 'indication' or 'contradiction'.

    Returns:
        JSONResponse: The predicted drug replacement(s) for the given disease indices.
    """
    # Use the provided disease indices and mode for evaluation

    results = TxEval.eval_disease_centric(
        disease_idxs=disease_idxs,
        show_plot=False,
        return_raw=True,
        # verbose=True,
        save_result=False,
        relation=mode.value,
    )
    result = results["result"]

    result["Prediction"] = None
    result["Labels"] = None
    print(result, " ... result")
    return JSONResponse(content=result)


@router.get("/explain")
async def get_drug_replacement_explanation(disease: str, drug: str):
    """
    Provide an explanation for the recommended drug replacement for a disease.

    This endpoint gives a detailed explanation of why a certain drug is
    suggested as a replacement for the given disease. The explanation may
    include the drug's efficacy, side effects, interactions, and other factors
    considered for the recommendation.

    Args:
        disease (str): The name of the disease for which the drug replacement
                       is being requested.
        drug (str): The name of the drug being recommended or explained as a
                    replacement.

    Returns:
        JSONResponse: A detailed explanation of the drug replacement for the
                      specified disease and drug.
    """

    print("disease", disease)
    print("drug", drug)
    return JSONResponse(content=disease)


app.include_router(router)
app.add_middleware(BodySizeLimiterMiddleware)
