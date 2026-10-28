"""Look up public FDA drug-label information by medicine name."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://api.fda.gov/drug/label.json"
LABEL_FIELDS = (
    ("Active ingredient", "active_ingredient"),
    ("Purpose", "purpose"),
    ("Indications and usage", "indications_and_usage"),
    ("Warnings", "warnings"),
    ("Adverse reactions", "adverse_reactions"),
)


def _field_text(value: object, limit: int = 1200) -> str:
    if isinstance(value, list):
        text = "\n".join(str(item) for item in value if item)
    elif isinstance(value, str):
        text = value
    else:
        return ""
    return text.strip()[:limit]


def search_drug_labels(medicine_name: str, limit: int = 3) -> list[dict[str, str]]:
    """Fetch up to three public FDA labels matching a generic or brand name."""
    name = medicine_name.strip()
    if not name:
        raise ValueError("Enter a medicine name to search FDA labels.")
    if limit < 1 or limit > 10:
        raise ValueError("FDA label result limit must be between 1 and 10.")

    escaped_name = name.replace("\\", "\\\\").replace('"', '\\"')
    query = (
        f'openfda.generic_name:"{escaped_name}" OR '
        f'openfda.brand_name:"{escaped_name}"'
    )
    url = f"{API_URL}?{urlencode({'search': query, 'limit': limit})}"

    try:
        with urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code == 404:
            return []
        raise
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"FDA label service is unavailable: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("FDA label service returned invalid JSON.") from error

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise RuntimeError("FDA label service returned an unexpected response.")

    labels = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue

        openfda = result.get("openfda", {})
        if not isinstance(openfda, dict):
            openfda = {}
        brand = _field_text(openfda.get("brand_name"), 200)
        generic = _field_text(openfda.get("generic_name"), 200)
        title = " / ".join(part for part in (brand, generic) if part) or name
        sections = []
        for heading, field in LABEL_FIELDS:
            content = _field_text(result.get(field))
            if content:
                sections.append(f"{heading}: {content}")
        if sections:
            labels.append(
                {
                    "title": title,
                    "content": "\n".join(sections),
                    "url": url,
                }
            )
    return labels
