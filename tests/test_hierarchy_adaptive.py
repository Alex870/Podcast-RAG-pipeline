import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import podcast_rag.pipeline as pipeline_module
import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig
from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.schema import validate_processed_documents
from podcast_rag.state import checkpoint_path


class StubDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class StubControl:
    @staticmethod
    def max_parallel_model_requests():
        return 1


def leaf(index):
    return StubDocument(
        f"Transcript evidence for segment {index}.",
        {
            "node_id": f"leaf_{index:03d}",
            "node_type": "leaf_chunk",
            "level": "leaf",
            "parent_id": None,
            "child_ids": [],
            "source": "episode.json",
            "episode_id": "episode-1",
            "episode_title": "Episode",
            "episode_date": "2026-04-29",
            "source_type": "json_transcript",
            "speaker_scope": "mixed",
            "start_time": float(index),
            "end_time": float(index) + 1.0,
            "source_segment_id": f"segment-{index:03d}",
        },
    )


def summary(index):
    document = leaf(index)
    document.metadata.update(
        {
            "node_id": f"summary_{index:03d}",
            "node_type": "cluster_summary",
            "level": "summary_1",
        }
    )
    return document


def hierarchy_stub(config=None):
    pipeline = PodcastRagPipeline.__new__(PodcastRagPipeline)
    pipeline.config = config or PipelineConfig(fake_llm=True)
    pipeline.control = StubControl()
    pipeline.cluster_telemetry = []
    pipeline.fallback_count = 0
    pipeline._file_fallback_count = 0
    pipeline._file_hierarchy_manifest = {}
    pipeline._file_hierarchy_levels = {}
    pipeline.performance = types.SimpleNamespace(
        current_file_max_total_tokens=0,
        run_max_total_tokens=0,
        maybe_report=lambda *args, **kwargs: None,
    )
    return pipeline


class FakeArray:
    shape = (20, 2)


class FakeNumpy:
    @staticmethod
    def array(values, dtype=None):
        return FakeArray()


class FakePCA:
    def __init__(self, **kwargs):
        pass

    def fit_transform(self, values):
        return values


class FakeHdbscan:
    labels = []
    kwargs = {}

    @classmethod
    def HDBSCAN(cls, **kwargs):
        cls.kwargs = kwargs
        return cls()

    def fit_predict(self, values):
        return list(self.labels)


class AdaptiveHierarchyTests(unittest.TestCase):
    def _semantic_groups(self, labels):
        config = PipelineConfig(fake_llm=True, min_docs_to_cluster=12, group_fallback_size=6)
        pipeline = hierarchy_stub(config)
        documents = [leaf(index) for index in range(len(labels))]
        FakeHdbscan.labels = labels
        pipeline.embed_in_batches = lambda texts: [[1.0, 0.0] for _ in texts]
        pipeline.record_diagnostic = lambda *args, **kwargs: None
        with patch.object(pipeline_module, "np", FakeNumpy), patch.object(pipeline_module, "normalize", lambda value: value), patch.object(pipeline_module, "PCA", FakePCA), patch.object(pipeline_module, "hdbscan", FakeHdbscan):
            groups = pipeline.cluster_documents(documents)
        return pipeline, groups

    def test_semantic_quality_gate_uses_chronological_fallback(self):
        config = PipelineConfig(
            fake_llm=True,
            min_docs_to_cluster=12,
            group_fallback_size=6,
            hierarchy_max_dominant_cluster_fraction=0.60,
            hierarchy_max_noise_rate=0.25,
        )
        pipeline = hierarchy_stub(config)
        documents = [leaf(index) for index in range(22)]
        FakeHdbscan.labels = [0] * 20 + [1] * 2
        pipeline.embed_in_batches = lambda texts: [[1.0, 0.0] for _ in texts]
        pipeline.record_diagnostic = lambda *args, **kwargs: None

        with patch.object(pipeline_module, "np", FakeNumpy), patch.object(pipeline_module, "normalize", lambda value: value), patch.object(pipeline_module, "PCA", FakePCA), patch.object(pipeline_module, "hdbscan", FakeHdbscan):
            groups = pipeline.cluster_documents(documents)

        self.assertEqual([6, 6, 6, 4], [len(group) for group in groups])
        self.assertEqual("chronological_fallback", pipeline.cluster_telemetry[-1]["strategy"])
        self.assertIn("dominant_cluster_fraction_exceeded", pipeline.cluster_telemetry[-1]["fallback_reasons"])
        self.assertEqual([f"segment-{index:03d}" for index in range(6)], [doc.metadata["source_segment_id"] for doc in groups[0]])

    def test_noise_quality_gate_uses_chronological_fallback(self):
        pipeline, groups = self._semantic_groups([-1] * 7 + [0] * 15)

        self.assertEqual([6, 6, 6, 4], [len(group) for group in groups])
        self.assertEqual("chronological_fallback", pipeline.cluster_telemetry[-1]["strategy"])
        self.assertIn("noise_rate_exceeded", pipeline.cluster_telemetry[-1]["fallback_reasons"])

    def test_healthy_semantic_groups_are_retained(self):
        pipeline, groups = self._semantic_groups([0] * 11 + [1] * 11)

        self.assertEqual([11, 11], [len(group) for group in groups])
        self.assertEqual("semantic", pipeline.cluster_telemetry[-1]["strategy"])

    def test_summary_levels_use_less_conservative_hdbscan_parameters(self):
        config = PipelineConfig(fake_llm=True, min_docs_to_cluster=12, group_fallback_size=6)
        pipeline = hierarchy_stub(config)
        documents = [summary(index) for index in range(41)]
        FakeHdbscan.labels = [0] * 20 + [1] * 21
        pipeline.embed_in_batches = lambda texts: [[1.0, 0.0] for _ in texts]
        pipeline.record_diagnostic = lambda *args, **kwargs: None

        with patch.object(pipeline_module, "np", FakeNumpy), patch.object(pipeline_module, "normalize", lambda value: value), patch.object(pipeline_module, "PCA", FakePCA), patch.object(pipeline_module, "hdbscan", FakeHdbscan):
            groups = pipeline.cluster_documents(documents)

        self.assertEqual([20, 21], [len(group) for group in groups])
        self.assertEqual({"min_cluster_size": 3, "min_samples": 2}, FakeHdbscan.kwargs)
        self.assertEqual(
            {
                "min_cluster_size": 3,
                "min_samples": 2,
                "summary_level_tuning": True,
            },
            pipeline.cluster_telemetry[-1]["clustering_parameters"],
        )

    def test_two_to_eleven_nodes_create_one_forced_parent(self):
        pipeline = hierarchy_stub(PipelineConfig(fake_llm=True, max_levels=4))

        def summarize(level, documents, source):
            node_id = f"summary_{level}"
            for document in documents:
                document.metadata["parent_id"] = node_id
            return StubDocument(
                f"Summary level {level} covering {len(documents)} nodes.",
                {
                    "node_id": node_id,
                    "node_type": "cluster_summary",
                    "level": f"summary_{level}",
                    "parent_id": None,
                    "child_ids": [doc.metadata["node_id"] for doc in documents],
                    "source": source,
                    "episode_id": "episode-1",
                    "episode_title": "Episode",
                    "episode_date": "2026-04-29",
                    "source_type": "json_transcript",
                    "speaker_scope": "mixed",
                    "start_time": documents[0].metadata["start_time"],
                    "end_time": documents[-1].metadata["end_time"],
                    "source_segment_id": documents[0].metadata["source_segment_id"],
                },
            )

        def cluster(documents):
            pipeline.cluster_telemetry.append(
                {
                    "strategy": "forced_parent" if len(documents) < 12 else "chronological_fallback",
                    "forced_parent": len(documents) < 12,
                    "fallback_reasons": [],
                    "quality_metrics": {},
                }
            )
            return [documents] if len(documents) < 12 else [documents[:6], documents[6:]]

        pipeline.summarize_cluster = summarize
        pipeline.cluster_documents = cluster
        with patch.object(pipeline_module, "Document", StubDocument):
            documents, thesis = pipeline.build_hierarchy([leaf(index) for index in range(5)], "episode.json")

        summaries = [doc for doc in documents if doc.metadata.get("node_type") == "cluster_summary"]
        self.assertEqual(["summary_1"], [doc.metadata["level"] for doc in summaries])
        self.assertEqual([1], pipeline._file_hierarchy_manifest["forced_rollup_levels"])
        self.assertEqual("fewer_than_two_current_nodes", pipeline._file_hierarchy_manifest["stop_reason"])
        self.assertEqual(["summary_1"], thesis.metadata["child_ids"])

    def test_large_input_continues_into_l2_and_preserves_generation_provenance(self):
        pipeline = hierarchy_stub(PipelineConfig(fake_llm=True, max_levels=4))

        def summarize(level, documents, source):
            node_id = f"summary_{level}_{documents[0].metadata['node_id']}"
            for document in documents:
                document.metadata["parent_id"] = node_id
            return StubDocument(
                f"Unique summary level {level} starting at {node_id}.",
                {
                    "node_id": node_id,
                    "node_type": "cluster_summary",
                    "level": f"summary_{level}",
                    "parent_id": None,
                    "child_ids": [doc.metadata["node_id"] for doc in documents],
                    "source": source,
                    "episode_id": "episode-1",
                    "episode_title": "Episode",
                    "episode_date": "2026-04-29",
                    "source_type": "json_transcript",
                    "speaker_scope": "mixed",
                    "start_time": documents[0].metadata["start_time"],
                    "end_time": documents[-1].metadata["end_time"],
                    "source_segment_id": documents[0].metadata.get("source_segment_id", node_id),
                },
            )

        def cluster(documents):
            forced = len(documents) < 12
            groups = [documents] if forced else [documents[index : index + 6] for index in range(0, len(documents), 6)]
            pipeline.cluster_telemetry.append(
                {
                    "strategy": "forced_parent" if forced else "chronological_fallback",
                    "forced_parent": forced,
                    "fallback_reasons": ["dominant_cluster_fraction_exceeded"] if not forced else [],
                    "quality_metrics": {},
                }
            )
            return groups

        pipeline.summarize_cluster = summarize
        pipeline.cluster_documents = cluster
        with patch.object(pipeline_module, "Document", StubDocument):
            documents, thesis = pipeline.build_hierarchy([leaf(index) for index in range(80)], "episode.json")

        self.assertGreaterEqual(sum(1 for doc in documents if doc.metadata.get("level") == "summary_2"), 2)
        self.assertTrue(thesis.metadata.get("generation_source_node_ids"))
        self.assertTrue(all(doc.metadata["level"] == "summary_2" for doc in documents if doc.metadata.get("node_id") in thesis.metadata["generation_source_node_ids"]))
        self.assertEqual("adaptive-v2", pipeline._file_hierarchy_manifest["algorithm_version"])
        self.assertEqual(3, len(pipeline._file_hierarchy_manifest["levels"]))
        self.assertTrue(pipeline._file_hierarchy_manifest["forced_rollup_levels"])

        result = validate_processed_documents(documents, require_provenance=True)
        self.assertTrue(result.valid, result.errors)

    def test_known_singleton_tail_shape_receives_l2_parent(self):
        pipeline = hierarchy_stub(PipelineConfig(fake_llm=True, max_levels=4))
        calls = []

        def summarize(level, documents, source):
            node_id = f"summary_{level}_{len(calls)}"
            calls.append(node_id)
            for document in documents:
                document.metadata["parent_id"] = node_id
            return StubDocument(
                f"Summary {node_id}.",
                {
                    "node_id": node_id,
                    "node_type": "cluster_summary",
                    "level": f"summary_{level}",
                    "parent_id": None,
                    "child_ids": [doc.metadata["node_id"] for doc in documents],
                    "source": source,
                    "episode_id": "episode-1",
                    "episode_title": "Episode",
                    "episode_date": "2026-04-29",
                    "source_type": "json_transcript",
                    "speaker_scope": "mixed",
                    "start_time": documents[0].metadata["start_time"],
                    "end_time": documents[-1].metadata["end_time"],
                    "source_segment_id": documents[0].metadata.get("source_segment_id", node_id),
                },
            )

        def cluster(documents):
            if len(documents) == 226:
                groups = [documents[:208], documents[208:223], *([documents[index : index + 1] for index in range(223, 226)])]
                strategy = "chronological_fallback"
                reasons = ["dominant_cluster_fraction_exceeded"]
            else:
                groups = [documents]
                strategy = "forced_parent"
                reasons = []
            pipeline.cluster_telemetry.append(
                {
                    "strategy": strategy,
                    "forced_parent": strategy == "forced_parent",
                    "fallback_reasons": reasons,
                    "quality_metrics": {},
                }
            )
            return groups

        pipeline.summarize_cluster = summarize
        pipeline.cluster_documents = cluster
        with patch.object(pipeline_module, "Document", StubDocument):
            documents, thesis = pipeline.build_hierarchy([leaf(index) for index in range(226)], "episode.json")

        self.assertEqual(5, sum(1 for doc in documents if doc.metadata.get("level") == "summary_1"))
        self.assertEqual(1, sum(1 for doc in documents if doc.metadata.get("level") == "summary_2"))
        self.assertEqual(0, sum(1 for doc in documents if doc.metadata.get("level") == "summary_3"))
        self.assertEqual(5, len(thesis.metadata["generation_source_node_ids"]))
        self.assertEqual(1, len(thesis.metadata["child_ids"]))
        self.assertEqual([2], pipeline._file_hierarchy_manifest["forced_rollup_levels"])

    def test_force_checkpoint_namespace_does_not_touch_normal_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("source", encoding="utf-8")
            config = PipelineConfig(checkpoint_dir="checkpoints", resume_within_file=True)
            pipeline = hierarchy_stub(config)
            pipeline.project_dir = root
            pipeline.active_context = {}
            pipeline.active_errata = None
            pipeline.record_diagnostic = lambda *args, **kwargs: None
            pipeline._active_checkpoint_namespace = "logical.force.run-1"
            document = leaf(0)

            pipeline.save_file_checkpoint(source, "logical", "hierarchy", [document])

            force_path = checkpoint_path(config, root, source, "logical.force.run-1", "hierarchy")
            normal_path = checkpoint_path(config, root, source, "logical", "hierarchy")
            self.assertTrue(force_path.is_file())
            self.assertFalse(normal_path.exists())

            with patch.object(runtime, "load_runtime_deps", lambda: None), patch.object(runtime, "Document", StubDocument, create=True):
                restored = pipeline.load_file_checkpoint(source, "logical", "hierarchy")
            self.assertEqual([document.metadata["node_id"]], [item.metadata["node_id"] for item in restored])


if __name__ == "__main__":
    unittest.main()
