
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


def load_test_data(filepath="data/test_fake_news.json"):
    import json
    import pandas as pd

    # Load your converted JSON (from CSV)
    with open(filepath) as f:
        records = json.load(f)

    news_urls = records["news_url"].values()
    titles = records["title"].values()
    labels = records["label"].values()
    
    # payload for request
    payload = []
    for url, title in zip(news_urls, titles):
        payload.append({"news_url": url, "title": title})

    # dataframe to combine prediction with
    df = pd.DataFrame(data={"news_url": news_urls, "title": titles, "label": labels})
    
    return payload, df
