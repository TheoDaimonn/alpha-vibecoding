# Russian GLiNER PII model

This directory contains the trained checkpoint prepared for local inference.

```python
from ru_pii.inference import RussianPIIDetector

detector = RussianPIIDetector.from_pretrained(
    "models/gliner-ru-pii-small",
    device="cpu",
)
entities = detector.predict(
    "Клиент Иванов Иван Иванович, email: test@example.org",
    include_auxiliary=False,
)
```

The `thresholds.json` file contains thresholds calibrated on the development
split. The large `model.safetensors` file is stored with Git LFS.