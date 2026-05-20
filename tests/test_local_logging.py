from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging import CSVLogger, JSONListLogger


def test_local_logging_file_creation(tmp_path):
    csv_path = tmp_path / "local_controller_decisions.csv"
    json_path = tmp_path / "local_actions.json"
    CSVLogger(csv_path).log({"cycle": 0, "accepted": True})
    JSONListLogger(json_path).log({"cycle": 0, "actions": []})
    assert csv_path.exists()
    assert json_path.exists()
    assert "accepted" in csv_path.read_text(encoding="utf-8")
