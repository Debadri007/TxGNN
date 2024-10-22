from typing import Optional, Tuple
from txgnn import TxData, TxGNN, TxEval
from logger import get_logger_by_name

# Tests the SDK connection with the server

logger = get_logger_by_name("Hivata | Rare diseases | Knowledge Graph")


def get_gnn_model(
    data_folder_path: str = "./data",
    model_path: str = "/model",
    split: str = "full_graph",
    seed: int = 42,
    device: str = "cuda:0",
    return_data: bool = False,
) -> Tuple[Optional[TxData], TxGNN, TxEval]:
    global TxData, TxGNN, TxEval

    # Initialize TxData if not already done
    if TxData is None:
        logger.info(f"Initializing TxData with folder: {data_folder_path}")
        TxData = TxData(data_folder_path=data_folder_path)
        TxData.prepare_split(split=split, seed=seed)

    # Initialize TxGNN if not already done
    if TxGNN is None:
        logger.info(f"Initializing TxGNN model on device: {device}")
        TxGNN = TxGNN(
            data=TxData,
            proj_name="TxGNN",  # wandb project name
            exp_name="TxGNN",  # wandb experiment name
            device=device,  # define your cuda device
        )
        logger.info(
            f"Loading pre-trained GNN model from {model_path} ... {repr(TxGNN)}"
        )
        TxGNN.load_pretrained(model_path)
        logger.info(f"Pre-trained GNN model loaded successfully! {repr(TxGNN)}")

    # Initialize TxEval if not already done
    if TxEval is None:
        logger.info(f"Initializing evaluation model for GNN ... {repr(TxGNN)}")
        TxEval = TxEval(model=TxGNN)
        logger.info(f"Evaluation model for GNN loaded successfully! {repr(TxEval)}")

    # Return the tuple with optional TxData based on return_data flag
    if return_data:
        return TxData, TxGNN, TxEval
    else:
        return None, TxGNN, TxEval


# Example usage:
# _, TxGNN_model, TxEval_model = get_gnn_model()
# Or if you want to return the data object as well
