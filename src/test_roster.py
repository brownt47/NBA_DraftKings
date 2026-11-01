import requests

url = "https://site.api.espn.com/apis/search/v2"

params = {
    "query": "Michael Ajayi",
    "limit": 10,
}

response = requests.get(url, params=params, timeout=30)

print("HTTP status:", response.status_code)
print("URL:", response.url)
print("Response:")
print(response.text[:3000])
