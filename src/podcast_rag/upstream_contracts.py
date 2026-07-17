"""Dependency-free parsers for upstream ecosystem artifacts."""
import hashlib,json
from pathlib import Path
from typing import Any
class UpstreamContractError(ValueError): pass
def _canonical(v:Any): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
def parse_correction_fixture(path:str|Path)->dict[str,Any]:
 value=json.loads(Path(path).read_text(encoding="utf-8")); manifest=value.get("manifest",value); transcript=value.get("transcript")
 if manifest.get("contract_version")!="correction-manifest-v1": raise UpstreamContractError("unsupported correction manifest")
 if transcript is not None:
  actual=hashlib.sha256(_canonical(transcript)).hexdigest()
  if actual!=manifest.get("source_transcript_hash"): raise UpstreamContractError("stale transcript hash")
  spans={str(x.get("source_span_id",x.get("id",""))):x for x in transcript.get("segments",[])}
  for c in manifest.get("accepted_corrections",[]):
   if spans.get(str(c.get("source_span_id")),{}).get(str(c.get("field")))!=c.get("before"): raise UpstreamContractError("before value mismatch")
 return manifest
