
def extract_domain(url):
    from urllib.parse import urlparse
    if not url or type(url) == float:
        url = "unknown"
    parsed = urlparse(url)
    domain = parsed.netloc
    
    if not domain:
        return "unknown"
    
    elif domain.startswith("www."):
        domain = domain[4:]
    
    return domain.lower()


def load_test_data(filepath):
    import json
    import pandas as pd

    # Load JSON file (list of dicts)
    with open(filepath, "r") as f:
        records = json.load(f)

    # Convert to DataFrame (BEST PRACTICE)
    df = pd.DataFrame(records)

    # Build payload for API
    payload = df[["news_url", "title"]].to_dict(orient="records")

    return payload, df