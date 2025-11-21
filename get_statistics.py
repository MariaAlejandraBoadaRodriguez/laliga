import requests
import json
import pandas as pd
from soccer_api import API_KEY 

headers = {
    "x-apisports-key": API_KEY
}

def get_match_statistics(fixture_id):
    """
    Extrae estadísticas por partido usando el endpoint:
    https://v3.football.api-sports.io/fixtures/statistics
    """
    url = "https://v3.football.api-sports.io/fixtures/statistics"
    params = {"fixture": fixture_id}

    res = requests.get(url, headers=headers, params=params)
    data = res.json()

    if "errors" in data and data["errors"]:
        print(f"[ERROR] Fixture {fixture_id}: {data['errors']}")
        return None

    if not data.get("response"):
        print(f"[WARNING] No stats for fixture {fixture_id}")
        return None

    # La respuesta trae 2 bloques: Home y Away
    try:
        home_stats = data["response"][0]
        away_stats = data["response"][1]
    except:
        print(f"[WARNING] Stats malformed for fixture {fixture_id}")
        return None

    # Convertir lista de stats en dict {stat_name: value}
    def parse_stats(team_stats):
        stats_dict = {}
        for s in team_stats["statistics"]:
            name = s["type"]
            value = s["value"]
            # Convertir valores tipo "35%" -> 35
            if isinstance(value, str) and "%" in value:
                value = float(value.replace("%", ""))
            stats_dict[name] = value
        return stats_dict

    home = parse_stats(home_stats)
    away = parse_stats(away_stats)

    # Unificamos con prefijo home_ y away_
    flat = {"fixture_id": fixture_id}
    for k, v in home.items():
        flat[f"home_{k}"] = v
    for k, v in away.items():
        flat[f"away_{k}"] = v

    return flat


def download_all_stats(fixture_ids, limit=None):
    """
    Descarga las estadísticas de TODOS los partidos (opcional limitar).
    """
    stats_rows = []
    total = len(fixture_ids) if not limit else min(limit, len(fixture_ids))

    print(f"Descargando estadísticas de {total} partidos...")

    for i, fid in enumerate(fixture_ids[:total]):
        print(f"[{i+1}/{total}] Fixture {fid}")
        row = get_match_statistics(fid)
        if row:
            stats_rows.append(row)

    df_stats = pd.DataFrame(stats_rows)
    print("\nMuestras procesadas:", len(df_stats))
    print(df_stats.head())

    return df_stats


if __name__ == "__main__":
    # Ejemplo de uso con tus fixtures descargados anteriormente
    from soccer_api import get_laliga_fixtures

    fixtures = get_laliga_fixtures(2022)
    fixture_ids = [f["fixture_id"] for f in fixtures]

    stats_df = download_all_stats(fixture_ids)  # <-- sube el límite si quieres todo
    stats_df.to_csv("stats_2022_sample.csv", index=False)
