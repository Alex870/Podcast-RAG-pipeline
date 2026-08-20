import json, tempfile, unittest
from pathlib import Path
from podcast_rag.hardening import (
    create_backup,
    inspect_backup,
    inspect_resilience,
    restore_backup,
)
from podcast_rag.m6_preflight import build_preflight


class M6HardeningTests(unittest.TestCase):
    def test_preflight_and_offline_resilience_report(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as raw:
            root = Path(raw)
            (root / "processed_data").mkdir()
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "processed_data_dir": "processed_data",
                        "state_path": "state/state.json",
                    }
                )
            )
            value = inspect_resilience(root, config)
            self.assertEqual([], value["blockers"])
            self.assertIn(
                "inspect", value["execution"]["offline_deterministic_operations"]
            )
            self.assertFalse(value["execution"]["model_downloads_implicit"])
            pre = build_preflight(
                "rag",
                root,
                required_modules=("m6_missing_dependency",),
                minimum_free_bytes=1,
            )
            self.assertNotIn(str(root.resolve()), json.dumps(pre))
            self.assertGreater(pre["profile"]["total_memory_bytes"], 0)
            missing = next(
                item
                for item in pre["capabilities"]
                if item["capability"] == "m6_missing_dependency"
            )
            self.assertEqual(
                "python -m pip install -r podcast_rag_requirements.txt",
                missing["remediation_command"],
            )

    def test_checkpoint_backup_restore_is_approval_gated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as raw:
            root = Path(raw)
            (root / "state/file_checkpoints").mkdir(parents=True)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "state_path": "state/state.json",
                        "checkpoint_dir": "state/file_checkpoints",
                    }
                )
            )
            (root / "state/state.json").write_text('{"files":{}}')
            (root / "state/file_checkpoints/a.json").write_text('{"step":2}')
            backup = create_backup(root, config, root / "backup.zip")
            self.assertTrue(inspect_backup(root / "backup.zip")["valid"])
            (root / "state/state.json").write_text("changed")
            with self.assertRaises(PermissionError):
                restore_backup(root / "backup.zip", root, approved_backup_id="wrong")
            restore_backup(
                root / "backup.zip", root, approved_backup_id=backup["backup_id"]
            )
            self.assertEqual('{"files":{}}', (root / "state/state.json").read_text())


if __name__ == "__main__":
    unittest.main()
