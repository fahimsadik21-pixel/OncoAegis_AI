from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.railway_app import app


class TestRailwayApp(unittest.TestCase):
    def test_resource_safe_profile_exposes_product_health_and_model_routes(self):
        with TestClient(app) as client:
            health = client.get("/health")
            models = client.get("/models")
            information_status = client.get("/information/cancer/status")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(models.status_code, 200)
        self.assertTrue(models.json()["models"])
        self.assertEqual(information_status.status_code, 200)


if __name__ == "__main__":
    unittest.main()
