from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def datos():
    ruta = Path(__file__).resolve().parents[1] / "data" / "cloud_task_scheduling_dataset_20k.csv"
    return pd.read_csv(ruta)
