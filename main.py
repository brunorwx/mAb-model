from uvicorn import run

from src.model.train import train_all_models
from src.preprocessing.pipeline import PREPROCESSOR_ARTIFACT_PATH


def ensure_artifacts() -> None:
    """Train the model and save preprocessing artifacts if they are missing."""
    missing_preprocessor = not PREPROCESSOR_ARTIFACT_PATH.exists()
    if missing_preprocessor:
        print("Preprocessor artifact not found; training models before startup...")
        train_all_models()


def main() -> None:
    ensure_artifacts()
    run("src.api.app:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
