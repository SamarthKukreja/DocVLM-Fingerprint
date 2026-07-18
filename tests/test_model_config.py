"""Tests for custom model registry loading."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import parse_models  # noqa: E402
from vlm_clients import MockVLMClient, get_client, load_model_config  # noqa: E402


class ModelConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = ROOT_DIR / "tests" / ".tmp" / "model_config"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)
        self.config_path = self.tmp_dir / "custom.yaml"
        self.config_path.write_text(
            "models:\n"
            "  - name: local_mock\n"
            "    provider: mock\n"
            "    max_retries: 1\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(ROOT_DIR / "tests" / ".tmp", ignore_errors=True)

    def test_load_custom_config_and_construct_client(self) -> None:
        configs = load_model_config(self.config_path)
        self.assertIn("local_mock", configs)
        self.assertEqual(configs["local_mock"]["provider"], "mock")
        client = get_client("local_mock", config_path=self.config_path)
        self.assertIsInstance(client, MockVLMClient)

    def test_parse_models_defaults_to_mock_from_custom_config(self) -> None:
        self.assertEqual(parse_models(None, self.config_path), ["local_mock", "mock"])
        self.assertEqual(parse_models("local_mock", self.config_path), ["local_mock"])


if __name__ == "__main__":
    unittest.main()
