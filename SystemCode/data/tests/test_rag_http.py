"""HTTP contract tests; no live API key, database server or network needed."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import rag_http_api as api


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_question_and_vehicle_forwarded_with_exact_response(self):
        evidence = {"answer": "检查参考", "vehicle_analysis": [], "sources": [],
                    "retrieval_warnings": ["No evidence"], "retrieval_trace": []}
        with patch.object(api, "ask_fleetcare_detailed", return_value=evidence) as call:
            response = self.client.post("/api/rag/ask", json={"question": " 检查什么？ ", "vehicle_id": 7})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), evidence)
            call.assert_called_once_with("检查什么？", vehicle_id=7)

    def test_rejects_blank_coerced_ids_extra_fields_and_long_input(self):
        for body in [{"question": " "}, {"question": "x", "vehicle_id": "7"},
                     {"question": "x", "vehicle_id": True}, {"question": "x", "token": "secret"},
                     {"question": "x"*6001}]:
            with self.subTest(body_type=list(body)):
                self.assertEqual(self.client.post("/api/rag/ask", json=body).status_code, 422)

    def test_generation_error_does_not_expose_credentials(self):
        with patch.object(api, "ask_fleetcare_detailed", side_effect=RuntimeError("secret-password")):
            response = self.client.post("/api/rag/ask", json={"question": "check"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret-password", response.text)

    def test_search_filter_does_not_call_generation(self):
        from unittest.mock import Mock
        retriever = Mock()
        retriever.search.return_value = {"success": True, "hits": []}
        with patch.object(api, "_retriever", retriever), patch.object(api, "ask_fleetcare_detailed") as generation:
            response = self.client.post("/api/rag/search", json={"question": "brakes", "kind": "maintenance"})
            self.assertEqual(response.status_code, 200)
            retriever.search.assert_called_once_with("brakes", kind="maintenance")
            generation.assert_not_called()
        self.assertEqual(self.client.post("/api/rag/search", json={"question": "x", "kind": "unknown"}).status_code, 422)

    def test_search_error_sanitized(self):
        from unittest.mock import Mock
        retriever = Mock()
        retriever.search.side_effect = RuntimeError("secret-db-path")
        with patch.object(api, "_retriever", retriever):
            response = self.client.post("/api/rag/search", json={"question": "brakes"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret-db-path", response.text)

    def test_cors_allows_configured_local_frontend(self):
        response = self.client.options("/api/rag/ask", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")
        denied = self.client.options("/api/rag/ask", headers={"Origin": "https://unknown.example", "Access-Control-Request-Method": "POST"})
        self.assertEqual(denied.status_code, 400)
