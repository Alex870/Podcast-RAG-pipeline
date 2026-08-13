# M5 Producer Prototype

`podcast_rag.advanced_retrieval` builds release-bound `evidence-graph-1.0` and `late-chunk-alignment-1.0` sidecars. Both require a complete `advanced-retrieval-entry-gate-1.0`; late chunking also requires a resolved immutable model revision.

Set `enable_advanced_retrieval_prototypes` to `true`, then use `--build-advanced-retrieval`, `--m5-entry-gate`, and `--corpus-release-id`. Outputs remain marked `prototype`, include exact token character offsets and incremental rebuild keys, and contain no model-generated vectors or implicit downloads.
