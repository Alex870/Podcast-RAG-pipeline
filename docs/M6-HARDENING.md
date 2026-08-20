# M6 Processing Hardening

`podcast-rag-hardening` reports cache integrity, checkpoint/state presence, concurrency, free-space estimates, and deterministic offline operations. It backs up configuration, state, snapshots, stop state, and file checkpoints using checksums and approval-gated atomic restore. It never downloads or loads a model during diagnostics.
