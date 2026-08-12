import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.state import export_representation_corpus


class RepresentationCorpusTests(unittest.TestCase):
    def test_export_preserves_ids_and_reports_coverage(self):
        fixture=Path(__file__).parent/"fixtures/processed_cache_v2_1.json"
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"processed"; source.mkdir()
            (source/"fixture.processed_documents.json").write_bytes(fixture.read_bytes())
            output=root/"corpus.json"; result=export_representation_corpus(source,output)
            value=json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("representation-corpus-1.0",value["contract_version"])
            self.assertEqual(result["document_count"],value["coverage"]["included"])
            self.assertTrue(all(item["document_id"]==item["metadata"]["stable_document_id"] for item in value["documents"]))
            self.assertTrue(all(item["display_text"] and item["dense_text"] and item["lexical_text"] for item in value["documents"]))

    def test_query_fixtures_cover_required_slices(self):
        value=json.loads((Path(__file__).parent/"fixtures/m2_retrieval_queries.json").read_text(encoding="utf-8"))
        self.assertTrue({"exact_term","temporal","quotation","semantic","hierarchy"}.issubset({item["slice"] for item in value["queries"]}))


if __name__=="__main__": unittest.main()
