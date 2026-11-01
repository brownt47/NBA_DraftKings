import requests

game_id = "401909088"

url = (
    "https://site.api.espn.com/apis/site/v2/"
    f"sports/basketball/nba/summary?event={game_id}"
)

response = requests.get(url, timeout=20)
response.raise_for_status()

data = response.json()

print("Top-level keys:")
print(list(data.keys()))

print("\nBoxscore keys:")
boxscore = data.get("boxscore", {})
print(list(boxscore.keys()))

print("\nTeam statistics:")
for team in boxscore.get("teams", []):
    print("\nTeam:", team.get("team", {}).get("displayName"))

    for stat in team.get("statistics", []):
        print(
            stat.get("name"),
            "=",
            stat.get("displayValue")
        )


print("\nGame header:")
header = data.get("header", {})

for competition in header.get("competitions", []):
    print("Status:", competition.get("status"))

    for competitor in competition.get("competitors", []):
        print(
            competitor.get("team", {}).get("displayName"),
            "Score:", competitor.get("score"),
            "Home/Away:", competitor.get("homeAway")
        )
print("\nPeriod information:")

for competition in header.get("competitions", []):
    print("Status period:", competition.get("status", {}).get("period"))
    print("Competition keys:", list(competition.keys()))

print("Header season:", header.get("season"))

print("\nGame duration:")

for competition in header.get("competitions", []):
    print("Boxscore minutes:", competition.get("boxscoreMinutes"))
    print("Status:", competition.get("status"))

    print("\nPeriod scoring:")

for competition in header.get("competitions", []):
    for competitor in competition.get("competitors", []):
        print(
            competitor.get("team", {}).get("displayName"),
            competitor.get("linescores")
        )