# Focused synthetic supplement

These JSONL files are locally generated, deterministic training data used only to
close gaps in ready Russian corpora. They contain non-issued synthetic values and
exact character offsets. Default sizes are 12,000 train / 1,500 dev / 1,500 test.

Regenerate exactly with:

```bash
python -m ru_pii.synthetic_supplement --output data/supplement --seed 20260922
```

The supplement contains supervision for every hackathon-required PII class, plus
coarse passport/driver-licence spans and public-person/public-address context signals.
It deliberately does **not** mark bonus RedMadRobot labels such as SNILS/OMS/URL/IP as
negative, because those classes are learned from the published source corpus instead.
