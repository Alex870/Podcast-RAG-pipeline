import os
import unittest
from pathlib import Path

from podcast_rag.config import PipelineConfig, apply_env_overrides, config_fingerprint, load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_returns_defaults_for_missing_file(self):
        config = load_config(Path("missing-podcast-rag-config.json"))
        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.input_dir, "data")

    def test_load_config_reports_invalid_json_location(self):
        path = Path(".test_tmp_invalid_config.json")
        try:
            path.write_text('{"input_dir": "data",}', encoding="utf-8")
            with self.assertRaises(SystemExit) as exc:
                load_config(path)
            self.assertIn("Invalid JSON in config file", str(exc.exception))
            self.assertIn("Line", str(exc.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_apply_env_overrides_updates_model_settings(self):
        config = PipelineConfig()
        previous = {key: os.environ.get(key) for key in ("EMBEDDING_MODEL", "LM_STUDIO_BASE_URL", "LM_STUDIO_API_KEY", "LM_STUDIO_MODEL")}
        try:
            os.environ["EMBEDDING_MODEL"] = "test-embedding"
            os.environ["LM_STUDIO_BASE_URL"] = "http://localhost:9999/v1"
            os.environ["LM_STUDIO_API_KEY"] = "test-key"
            os.environ["LM_STUDIO_MODEL"] = "test-model"
            updated = apply_env_overrides(config)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(updated.embedding_model, "test-embedding")
        self.assertEqual(updated.lm_studio_base_url, "http://localhost:9999/v1")
        self.assertEqual(updated.lm_studio_api_key, "test-key")
        self.assertEqual(updated.lm_studio_model, "test-model")

    def test_config_fingerprint_changes_with_relevant_values(self):
        base = PipelineConfig()
        changed = PipelineConfig(lm_studio_model="different-model")
        self.assertNotEqual(config_fingerprint(base), config_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
