import requests

url = "https://translate.googleapis.com/translate_a/single"
params = {
    "client": "gtx",
    "sl": "la",
    "tl": "en",
    "dt": "t",
    "q": "arma virumque cano",
}
r = requests.get(url, params=params, timeout=10)
print(r.status_code)
print(r.text[:200])
