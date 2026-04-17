import json
import pickle
from pathlib import Path


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def append_json_list(path, record):
    records = load_json(path, default=[])
    records.append(record)
    save_json(path, records)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
