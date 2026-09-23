"""Print the curated real-corpus source registry."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
items=json.loads((root/'configs/datasets.json').read_text(encoding='utf-8'))
for item in items:
    print(f"{item['id']:24} {item['role']:34} {item['text_provenance']}")
