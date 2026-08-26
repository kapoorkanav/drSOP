def tokenize_comorbidities(text) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip().lower() for p in text.replace(";", ",").split(",")]
    return [p for p in parts if p]


def parse_locale_number(value) -> float:
    """Parses a number that may use ',' as the decimal separator (BRSET is Brazilian-locale
    data, e.g. diabetes_time_y values like "10,00"). Returns NaN for anything unparseable --
    missing, empty, or a data-entry typo (e.g. "1O" with a letter O) -- rather than raising, so
    such rows get caught by the complete-metadata filter instead of crashing the pipeline."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return float("nan")
