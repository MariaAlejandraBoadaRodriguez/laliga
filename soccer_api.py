import requests
import json
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ.get("API_FOOTBALL_KEY")

headers = {
    "x-apisports-key": API_KEY
}

def get_laliga_fixtures(season, limit=380):
    url = "https://v3.football.api-sports.io/fixtures"
    params = {
        "league": 140,      # LaLiga
        "season": season,   # ej: 2022
        "status": "FT"      # solo partidos finalizados
    }

    res = requests.get(url, headers=headers, params=params)
    print("Status code:", res.status_code)

    data = res.json()
    if "errors" in data and data["errors"]:
        print("Errores de la API:", data["errors"])
        return []

    fixtures_raw = data.get("response", [])
    print(f"Total partidos devueltos por la API: {len(fixtures_raw)}")

    fixtures_clean = []
    for f in fixtures_raw[:limit]:
        fixtures_clean.append({
            "fixture_id": f["fixture"]["id"],
            "date": f["fixture"]["date"],
            "status": f["fixture"]["status"]["short"],
            "league": f["league"]["name"],
            "round": f["league"]["round"],
            "home_team": f["teams"]["home"]["name"],
            "away_team": f["teams"]["away"]["name"],
            "home_goals": f["goals"]["home"],
            "away_goals": f["goals"]["away"],
        })

    # Mostrar en consola
    print(json.dumps(fixtures_clean, indent=4, ensure_ascii=False))

    return fixtures_clean


if __name__ == "__main__":
    fixtures = get_laliga_fixtures(2022)

    # Guardar a CSV
    df = pd.DataFrame(fixtures)
    df.to_csv("soccer_api_2022.csv", index=False)

    print("\n✔ Archivo 'soccer_api_2022.csv' generado correctamente.")
