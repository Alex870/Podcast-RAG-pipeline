import os
import tempfile
import unittest
from pathlib import Path

from podcast_rag.config import (
    PipelineConfig,
    apply_env_overrides,
    config_fingerprint,
    embedding_constructor_kwargs,
    generation_config_fingerprint,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def test_load_config_returns_defaults_for_missing_file(self):
        config = load_config(Path("missing-podcast-rag-config.json"))
        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.input_dir, "data")

    def test_load_config_reports_invalid_json_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_config.json"
            path.write_text('{"input_dir": "data",}', encoding="utf-8")
            with self.assertRaises(SystemExit) as exc:
                load_config(path)
            self.assertIn("Invalid JSON in config file", str(exc.exception))
            self.assertIn("Line", str(exc.exception))

    def test_apply_env_overrides_updates_model_settings(self):
        config = PipelineConfig()
        previous = {key: os.environ.get(key) for key in ("EMBEDDING_MODEL", "EMBEDDING_CACHE_DIR", "EMBEDDING_LOCAL_FILES_ONLY", "LM_STUDIO_BASE_URL", "LM_STUDIO_API_KEY", "LM_STUDIO_MODEL")}
        try:
            os.environ["EMBEDDING_MODEL"] = "test-embedding"
            os.environ["EMBEDDING_CACHE_DIR"] = "state/test-hf-cache"
            os.environ["EMBEDDING_LOCAL_FILES_ONLY"] = "false"
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
        self.assertEqual(updated.embedding_cache_dir, "state/test-hf-cache")
        self.assertFalse(updated.embedding_local_files_only)
        self.assertEqual(updated.lm_studio_base_url, "http://localhost:9999/v1")
        self.assertEqual(updated.lm_studio_api_key, "test-key")
        self.assertEqual(updated.lm_studio_model, "test-model")

    def test_config_fingerprint_changes_with_relevant_values(self):
        base = PipelineConfig()
        changed = PipelineConfig(lm_studio_model="different-model")
        self.assertNotEqual(config_fingerprint(base), config_fingerprint(changed))

    def test_embedding_cache_policy_does_not_invalidate_generated_cache(self):
        base = PipelineConfig()
        changed = PipelineConfig(embedding_local_files_only=False, embedding_cache_dir="cache")
        self.assertEqual(config_fingerprint(base), config_fingerprint(changed))

    def test_embedding_constructor_uses_local_cache_policy_and_resolves_cache_path(self):
        config = PipelineConfig(embedding_cache_dir="state/hf-cache")
        kwargs = embedding_constructor_kwargs(config, Path("C:/project"))
        self.assertEqual({"local_files_only": True}, kwargs["model_kwargs"])
        self.assertEqual(str(Path("C:/project/state/hf-cache")), kwargs["cache_folder"])

    def test_hierarchy_settings_invalidate_generation_fingerprint(self):
        base = PipelineConfig()
        changed = PipelineConfig(hierarchy_max_noise_rate=0.10)
        self.assertNotEqual(generation_config_fingerprint(base), generation_config_fingerprint(changed))

    def test_adaptive_hierarchy_defaults_are_explicit(self):
        config = PipelineConfig()
        self.assertEqual("adaptive-v2", config.hierarchy_algorithm_version)
        self.assertEqual(2, config.hierarchy_min_parent_docs)
        self.assertEqual(0.60, config.hierarchy_max_dominant_cluster_fraction)
        self.assertEqual(0.25, config.hierarchy_max_noise_rate)
        self.assertEqual(12, config.hierarchy_summary_cluster_size_divisor)
        self.assertEqual(3, config.hierarchy_summary_min_cluster_size)
        self.assertEqual(6, config.hierarchy_summary_max_cluster_size)
        self.assertEqual(2, config.hierarchy_summary_min_samples)
        self.assertEqual("chronological", config.hierarchy_fallback_mode)


if __name__ == "__main__":
    unittest.main()
