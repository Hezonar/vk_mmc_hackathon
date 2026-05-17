import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.profpath.ml_model import predict_with_candidate_model
from app.profpath.ml_stub import predict_with_ml_stub
from app.profpath.parsing import get_parsed_info, split_codes

client = TestClient(app)


def test_split_codes_plain_string():
    assert split_codes("a; b") == ["a", "b"]


def test_predict_stub_no_conclusions():
    row = pd.Series(
        {
            "specialist_conclusions": "",
            "assigned_harmful_factors": "",
        }
    )
    parsed = get_parsed_info(row)
    out = predict_with_ml_stub(row, parsed)
    assert out["risk_level"] == "no_data"
    assert out["has_contraindications_pred"] is False


def test_candidate_model_artifact_predicts_assigned_subset():
    row = pd.Series(
        {
            "exam_row_id": "1015884464",
            "patient_id": "8571",
            "consultation_date": "2026-01-13 15:24:41.000",
            "assigned_harmful_factors": "18.1",
            "specialist_conclusions": "[]",
        }
    )
    out = predict_with_candidate_model(row)
    assert out["model_version"] == "exp009_all_data_candidate_binary"
    assert set(out["contraindicated_factors_pred"]) <= {"18.1"}


def test_profpath_demo_rows():
    r = client.get("/v1/profpath/demo-rows", params={"patient_id": "123"})
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body
    assert len(body["rows"]) >= 1


def test_profpath_patient_summary():
    demo = client.get("/v1/profpath/demo-rows")
    assert demo.status_code == 200
    rows = demo.json()["rows"]
    r = client.post(
        "/v1/profpath/patient-summary",
        json={"patient_id": "123", "rows": rows, "use_ml_stub": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["patient_id"] == "123"
    assert "predictions" in data
    assert "summary" in data
    assert len(data["summary"]) >= 1
