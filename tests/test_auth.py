import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.config import load_config
from src.dashboard_web import build_app
from src.engine import Engine
from src.mock_market import MockMarketDataProvider
from src.storage import Storage


def make_app(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = tmp_path
    provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
    storage = Storage(tmp_path)
    engine = Engine(cfg, provider, storage)
    return build_app(cfg, engine), engine


class NoPasswordSetTests(unittest.TestCase):
    """Existing no-auth behavior must be untouched when DASHBOARD_PASSWORD
    isn't set at all — this is the default for anyone who hasn't opted in."""

    def setUp(self):
        os.environ.pop("DASHBOARD_PASSWORD", None)

    def test_api_reachable_without_login(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)
            r = client.get("/api/state")
            self.assertEqual(r.status_code, 200)
            engine.storage.close()

    def test_root_reachable_without_login(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)
            r = client.get("/")
            self.assertEqual(r.status_code, 200)
            engine.storage.close()


class PasswordSetTests(unittest.TestCase):
    def setUp(self):
        os.environ["DASHBOARD_PASSWORD"] = "correct-horse"

    def tearDown(self):
        os.environ.pop("DASHBOARD_PASSWORD", None)

    def _client(self, tmp):
        app, engine = make_app(tmp)
        return TestClient(app), engine

    def test_api_blocked_without_cookie(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            r = client.get("/api/state")
            self.assertEqual(r.status_code, 401)
            engine.storage.close()

    def test_root_redirects_to_login_without_cookie(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            r = client.get("/", follow_redirects=False)
            self.assertEqual(r.status_code, 303)
            self.assertEqual(r.headers["location"], "/login")
            engine.storage.close()

    def test_login_page_itself_does_not_require_auth(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            r = client.get("/login")
            self.assertEqual(r.status_code, 200)
            engine.storage.close()

    def test_wrong_password_rejected_no_cookie_set(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            r = client.post("/login", json={"password": "nope"})
            self.assertEqual(r.status_code, 401)
            self.assertNotIn("okx_bot_session", r.cookies)
            engine.storage.close()

    def test_correct_password_sets_cookie_and_grants_access(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            r = client.post("/login", json={"password": "correct-horse"})
            self.assertEqual(r.status_code, 200)
            self.assertIn("okx_bot_session", client.cookies)

            # the persisted cookie now grants access to both the API and root
            r2 = client.get("/api/state")
            self.assertEqual(r2.status_code, 200)
            r3 = client.get("/")
            self.assertEqual(r3.status_code, 200)
            engine.storage.close()

    def test_tampered_cookie_rejected(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            client.cookies.set("okx_bot_session", "0" * 64)
            r = client.get("/api/state")
            self.assertEqual(r.status_code, 401)
            engine.storage.close()

    def test_logout_clears_cookie_and_reblocks(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            client, engine = self._client(Path(tmp))
            client.post("/login", json={"password": "correct-horse"})
            self.assertEqual(client.get("/api/state").status_code, 200)

            client.post("/logout")
            r = client.get("/api/state")
            self.assertEqual(r.status_code, 401)
            engine.storage.close()


if __name__ == "__main__":
    unittest.main()
