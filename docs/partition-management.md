# Partitioned processing

Podcast-RAG treats each podcast, meeting stream, or other context as an
independent processing partition.  A partition owns its derived caches,
checkpoints, topic state, errata, and releases.  The transcription producer
publishes handoff packages; Podcast-RAG reads them and never edits the
producer's files.

## Create a processing partition

Use the guided launcher menu, or run the equivalent commands from the project
directory:

```powershell
podcast-rag partitions create `
  --id podcast-history `
  --name "Podcast History" `
  --context-type podcast `
  --workflow-profile podcast

podcast-rag partitions create `
  --id work-meetings `
  --name "Work Meetings" `
  --context-type meeting `
  --workflow-profile anonymous_meeting

podcast-rag partitions use --id podcast-history
podcast-rag partitions list
```

The command creates the generated registry entry and the isolated directory
layout.  `partition_id` and `corpus_id` are immutable.  Changing a display
name does not change cache identity; archiving is reversible after another
partition is selected as active.

The guided menu also collects optional owner, tags, privacy, and retention
metadata.  Automation can supply repeated `--tag` options, plus
`--privacy key=value` and `--retention key=value` options.

## Publish and process a handoff

The upstream generator should stage a complete package and atomically promote
it into the selected partition's `handoff_inbox`.  The package must contain a
manifest and the exact transcript artifacts named by that manifest.

```powershell
podcast-rag handoff validate --manifest .\partitions\podcast-history\handoff_inbox\handoff-01\manifest.json
podcast-rag handoff scan --manifest .\partitions\podcast-history\handoff_inbox
podcast-rag process --partition podcast-history
podcast-rag process --manifest .\partitions\podcast-history\handoff_inbox\handoff-01\manifest.json --episode podcast-2026-01-03
```

Validation is dependency-free and checks package-relative paths, hashes,
contract versions, episode/source-span identity, stable source status, scope,
and correction guards before any model work.  The manifest-selected
transcript variant is authoritative; filenames are not used to choose a
managed variant.

## Inspect status and publish a release

```powershell
podcast-rag status --partition podcast-history
podcast-rag export --partition podcast-history --release release-2026-09-05-01
```

The release records partition/corpus identity, handoff IDs, episode UIDs,
cache schema, representation profile, embedding model, and cache fingerprints.
Downstream importers must reject releases containing mixed identities.

## Legacy input

Existing flat transcript directories remain readable through the compatibility
adapter, but are labeled with a synthetic `legacy:<root-fingerprint>` scope
and cannot silently join a managed release.  Register one explicitly when
needed:

```powershell
podcast-rag partitions adopt-legacy --id default --source C:\path\to\legacy\root
```

This registration does not rewrite or move the source directory.  A producer
handoff export or an explicit migration is required before it can become a
portable managed corpus.
