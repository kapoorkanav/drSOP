def tokenize_comorbidities(text) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip().lower() for p in text.replace(";", ",").split(",")]
    return [p for p in parts if p]
