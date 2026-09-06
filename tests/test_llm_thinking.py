import types
import unittest

from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.runtime import FakeChain


class BindableModel:
    def __init__(self):
        self.bind_calls = []
        self.bound_models = []

    def bind(self, **kwargs):
        self.bind_calls.append(kwargs)
        bound = object()
        self.bound_models.append(bound)
        return bound


class Prompt:
    def __or__(self, model):
        return model


class ThinkingControlTests(unittest.TestCase):
    def make_pipeline(self, *, fake_llm=False):
        pipeline = PodcastRagPipeline.__new__(PodcastRagPipeline)
        pipeline.config = types.SimpleNamespace(fake_llm=fake_llm)
        pipeline.llm = BindableModel()
        return pipeline

    def test_no_think_is_bound_per_request(self):
        pipeline = self.make_pipeline()

        result = pipeline.make_chain(Prompt(), enable_thinking=False)

        self.assertIs(result, pipeline.llm.bound_models[0])
        self.assertEqual(
            pipeline.llm.bind_calls,
            [{"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}],
        )

    def test_thinking_can_be_enabled_per_request(self):
        pipeline = self.make_pipeline()

        result = pipeline.make_chain(Prompt(), enable_thinking=True)

        self.assertIs(result, pipeline.llm.bound_models[0])
        self.assertEqual(
            pipeline.llm.bind_calls,
            [{"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}],
        )

    def test_none_uses_shared_model_without_binding(self):
        pipeline = self.make_pipeline()

        result = pipeline.make_chain(Prompt(), enable_thinking=None)

        self.assertIs(result, pipeline.llm)
        self.assertEqual(pipeline.llm.bind_calls, [])

    def test_fake_model_does_not_attempt_request_binding(self):
        pipeline = self.make_pipeline(fake_llm=True)

        result = pipeline.make_chain(Prompt(), enable_thinking=False)

        self.assertIsInstance(result, FakeChain)
        self.assertEqual(pipeline.llm.bind_calls, [])


if __name__ == "__main__":
    unittest.main()
