# Real / sourced subset

`data/real/` is the **sourced real-text subset**, not the final training corpus.
It is built from NEREL and FactRuEval by default; Collection3 is optional after
licence review. This subset intentionally maps only labels that the original
annotation actually supports (for example, generic news `DATE` is not converted to
`BIRTH_DATE`).

The final model trains on `data/hybrid/`, which additionally contains
`redmadrobot-rnd/pii_train` (published mixed/pseudonymized PII corpus) and the focused
synthetic task supplement. See `COVERAGE.md` for the full class matrix.
