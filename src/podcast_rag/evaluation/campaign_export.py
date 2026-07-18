"""Identity-aligned page-content-v1 campaign run exports."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
from .runner import _load_retrieval_run

class BaselineBindingError(ValueError): pass
def _sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")

def export_campaign_run(*,evaluation_pack_path:Path,retrieval_results_path:Path,output_path:Path,
                        corpus_release_id:str,corpus_fingerprint:str,release_critical_query_ids:set[str]|None=None)->dict[str,Any]:
 pack=json.loads(evaluation_pack_path.read_text(encoding="utf-8")); pack_fingerprint=_sha(evaluation_pack_path)
 dataset=pack.get("dataset",{}); queries=dataset.get("queries",[])
 if not queries: raise BaselineBindingError("evaluation pack has no judgments")
 expected=dataset.get("corpus_fingerprint")
 if expected and expected!=corpus_fingerprint: raise BaselineBindingError("incompatible corpus fingerprint")
 run=_load_retrieval_run(retrieval_results_path); manifest=run.get("manifest",{})
 if manifest.get("corpus_fingerprint") and manifest["corpus_fingerprint"]!=corpus_fingerprint: raise BaselineBindingError("retrieval run corpus fingerprint is incompatible")
 critical=release_critical_query_ids or set(); output_queries=[]
 for item in run.get("queries",[]):
  qid=str(item.get("query_id") or ""); ranked=[]
  for rank,result in enumerate(item.get("results") or [],1):
   metadata=dict(result.get("metadata") or {}); doc_id=result.get("document_id") or metadata.get("stable_document_id")
   if not doc_id: raise BaselineBindingError(f"ranked result for {qid} lacks evidence identity")
   ranked.append({"rank":rank,"document_id":doc_id,"score":result.get("score"),"page_content":result.get("page_content",result.get("text","")),"metadata":metadata})
  output_queries.append({"query_id":qid,"release_critical":qid in critical,"ranked_results":ranked,"diagnostics":{"speaker":[r["metadata"].get("speaker") for r in ranked],"date":[r["metadata"].get("episode_date") for r in ranked],"node_type":[r["metadata"].get("node_type") for r in ranked],"source_span_id":[r["metadata"].get("source_span_id") for r in ranked],"latency_ms":item.get("latency_ms"),"failure":item.get("failure"),"exclusion":item.get("exclusion")}})
 evidence_identity=[{"query_id":q["query_id"],"documents":[r["document_id"] for r in q["ranked_results"]]} for q in output_queries]
 query_identity=hashlib.sha256(_canonical(sorted(str(q.get("query_id")) for q in queries))).hexdigest()
 identity_payload={"contract_version":"page-content-v1","pack_fingerprint":pack_fingerprint,"corpus_release_id":corpus_release_id,"corpus_fingerprint":corpus_fingerprint,"query_identity":query_identity,"strategy_id":run.get("strategy_id"),"evidence":evidence_identity}
 value={**identity_payload,"run_id":"run_"+hashlib.sha256(_canonical(identity_payload)).hexdigest(),"pack_id":pack.get("pack_id"),"queries":output_queries,"raw_results_sha256":_sha(retrieval_results_path)}
 output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8"); return value
