import json,tempfile,unittest
from pathlib import Path
from podcast_rag.evaluation.campaign_export import BaselineBindingError,export_campaign_run
class CampaignExportTests(unittest.TestCase):
 def test_identity_aligned_export_and_critical_mode(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); pack=root/"pack.json"; results=root/"results.json"; out=root/"out.json"
   pack.write_text(json.dumps({"pack_id":"p","dataset":{"corpus_fingerprint":"c","queries":[{"query_id":"q1","judgments":[{}]}]}}),encoding="utf-8")
   results.write_text(json.dumps({"strategy_id":"dense-default","manifest":{"corpus_fingerprint":"c"},"queries":[{"query_id":"q1","latency_ms":2,"results":[{"document_id":"d1","page_content":"text","metadata":{"speaker":"A","source_span_id":"s1"}}]}]}),encoding="utf-8")
   a=export_campaign_run(evaluation_pack_path=pack,retrieval_results_path=results,output_path=out,corpus_release_id="r1",corpus_fingerprint="c",release_critical_query_ids={"q1"}); b=export_campaign_run(evaluation_pack_path=pack,retrieval_results_path=results,output_path=out,corpus_release_id="r1",corpus_fingerprint="c",release_critical_query_ids={"q1"})
   self.assertEqual(a["run_id"],b["run_id"]); self.assertTrue(a["queries"][0]["release_critical"]); self.assertEqual("s1",a["queries"][0]["diagnostics"]["source_span_id"][0])
 def test_missing_judgments_and_incompatible_corpus_fail(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); p=root/"p"; r=root/"r"; p.write_text(json.dumps({"dataset":{"queries":[]}})); r.write_text(json.dumps({"queries":[]}))
   with self.assertRaisesRegex(BaselineBindingError,"judgments"): export_campaign_run(evaluation_pack_path=p,retrieval_results_path=r,output_path=root/"o",corpus_release_id="r",corpus_fingerprint="c")
if __name__=="__main__": unittest.main()
