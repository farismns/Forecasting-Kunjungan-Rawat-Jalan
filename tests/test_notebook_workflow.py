import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/01_crisp_dm_forecasting.ipynb")


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_is_complete_crisp_dm_workflow():
    notebook = load_notebook()
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for phase in [
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Deployment",
    ]:
        assert phase in markdown

    for function in [
        "def preprocess_frames",
        "def calculate_metrics",
        "def create_baseline_forecasts",
        "def build_model",
    ]:
        assert function in code


def test_retraining_notebook_does_not_import_removed_training_modules():
    notebook = load_notebook()
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    forbidden_imports = [
        "from preprocessing import",
        "from train_models import",
        "from evaluation import",
    ]
    assert all(statement not in code for statement in forbidden_imports)
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs", [])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

