import unittest

from podcast_rag.ecosystem_delta import DeltaError, apply_delta, plan_delta, validate_delta


class DeltaTests(unittest.TestCase):
    def setUp(self):
        self.old = {"leaf": {"content": "before", "parent_id": "episode", "topic_id": "topic", "position_id": "position"}, "same": {"content": "same"}}

    def plan(self, new):
        return plan_delta(self.old, new, parent_corpus_id="corpus-1", correction_set_id="correction-1", processing_fingerprint="p1", representation_fingerprint="r1")

    def test_noop_is_empty_and_deterministic(self):
        first, second = self.plan(self.old), self.plan(self.old)
        self.assertEqual(first["delta_id"], second["delta_id"])
        self.assertEqual([], first["changed_document_ids"])

    def test_changed_leaf_closes_derived_effects(self):
        new = {**self.old, "leaf": {**self.old["leaf"], "content": "after"}}
        delta = self.plan(new)
        self.assertEqual(["leaf"], delta["changed_document_ids"])
        self.assertEqual(["episode"], delta["invalidated"]["ancestors"])
        self.assertEqual(["topic"], delta["invalidated"]["topics"])
        self.assertEqual(["position"], delta["invalidated"]["positions"])
        self.assertEqual("after", apply_delta(delta, self.old, new, approved_correction_set_id="correction-1")["leaf"]["content"])

    def test_removed_has_reason_and_is_advisory(self):
        delta = self.plan({"same": self.old["same"]})
        self.assertIn("leaf", delta["reasons"])
        self.assertTrue(delta["removals_advisory"])

    def test_apply_requires_approved_correction(self):
        delta = self.plan(self.old)
        with self.assertRaisesRegex(DeltaError, "approved"):
            apply_delta(delta, self.old, self.old, approved_correction_set_id="wrong")

    def test_identity_is_validated(self):
        delta = self.plan(self.old); delta["processing_fingerprint"] = "changed"
        with self.assertRaisesRegex(DeltaError, "identity"):
            validate_delta(delta)


if __name__ == "__main__": unittest.main()
