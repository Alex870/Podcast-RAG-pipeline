# processed-delta-v1

This additive contract records parent corpus/cache identities, approved correction
sets, processing and representation fingerprints, exact document effects,
derived-node invalidations, evidence mappings, stale judgments, validation,
timings, and failures. Removals remain advisory until Chroma reconciliation.
Empty successful deltas are valid and deterministic.

Notification-planned deltas add `affected_episode_ids` and
`affected_source_span_ids`. The planner requires the scope to resolve to at
least one corpus document and rejects changed, added, or removed documents
outside that scope. This fail-closed behavior keeps correction processing from
silently absorbing unrelated corpus drift.
