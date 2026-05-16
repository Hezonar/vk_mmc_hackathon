import app.analyzers.example_rules  # noqa: F401

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_index():
    r = client.get("/")
    assert r.status_code == 200
    assert "ПрофАрбитр AI" in r.text


def test_index_prefills_from_query_string():
    r = client.get("/", params={"id": "11", "name": "2", "age": "33", "analyzer_id": ""})
    assert r.status_code == 200
    assert 'value="11"' in r.text
    assert 'value="2"' in r.text
    assert 'value="33"' in r.text


def test_favicon_ico_redirects_to_svg():
    r = client.get("/favicon.ico", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers.get("location") == "/static/favicon.svg"


def test_predict_form():
    r = client.post("/predict", data={"id": "p1", "age": "40"})
    assert r.status_code == 200
    assert "healthy" in r.text


def test_predict_htmx_validation_partial():
    r = client.post(
        "/predict",
        data={"age": "40"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 422
    assert "Проверьте" in r.text


def test_v1_predict_json():
    r = client.post(
        "/v1/predict",
        json={"patient": {"id": "p", "age": 70}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "unhealthy"
    assert "confidence" in body


def test_v1_predict_unknown_analyzer():
    r = client.post(
        "/v1/predict",
        json={"patient": {"id": "p"}, "analyzer_id": "missing"},
    )
    assert r.status_code == 404


def test_profpath_cards_ok():
    import io

    from app.profpath.constants import REQUIRED_COLUMNS

    header = ",".join(REQUIRED_COLUMNS) + "\n"
    row = "101,M1,123,16.05.2024,шум,заключение в норме\n"
    buf = io.BytesIO((header + row).encode("utf-8"))
    r = client.post(
        "/profpath/cards",
        files={"file": ("t.csv", buf, "text/csv")},
        data={"patient_id": "123"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert "card-" in r.text
    assert "Посещение" in r.text


def test_profpath_cards_missing_columns():
    import io

    buf = io.BytesIO(b"a,b\n1,2\n")
    r = client.post(
        "/profpath/cards",
        files={"file": ("x.csv", buf, "text/csv")},
        data={"patient_id": "1"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 422
    assert "колонок" in r.text or "Проверьте" in r.text


def test_profpath_cards_patient_not_found():
    import io

    from app.profpath.constants import REQUIRED_COLUMNS

    header = ",".join(REQUIRED_COLUMNS) + "\n"
    row = "101,M1,123,16.05.2024,шум,норма\n"
    buf = io.BytesIO((header + row).encode("utf-8"))
    r = client.post(
        "/profpath/cards",
        files={"file": ("t.csv", buf, "text/csv")},
        data={"patient_id": "999"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert "card-gray" in r.text
    assert "не найден" in r.text.lower()
