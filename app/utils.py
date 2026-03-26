
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

