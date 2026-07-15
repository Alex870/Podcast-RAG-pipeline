# Retrieval Query Sets

Query sets use JSON Lines: one query object per line. The included `podcast-baseline-v1.jsonl` is an honest draft template, not fabricated ground truth. Replace its prompts with real corpus questions, add stable document IDs to `relevance`, and change `status` to `judged` after review.

Relevance grades:

- `3`: directly answers the question with attributable evidence
- `2`: strongly supports part of the answer
- `1`: useful background
- `0`: explicitly judged not relevant

The evaluation runner skips draft records and judged records without relevance labels. Unanswerable-query scoring belongs to the downstream answer evaluation layer because an empty relevance set cannot be scored with ordinary ranking metrics.
