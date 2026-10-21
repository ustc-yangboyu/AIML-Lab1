"""构建 extra world 数据集(基于 extra_world_data 的全球城市数据库)"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import re
import numpy as np
import pandas as pd
import os

from configs import Extra_Configs as configs

np.random.seed(configs.SEED)

SENTENCE_FMT = "What's the longitude and latitude of {}, {}?"
SENTENCE_PREFIX = "What's the longitude and latitude of "

COUNTRY_MAP = {
    'Bahamas': 'The Bahamas',
    'Belorussia': 'Belarus',
    'Bosna and Hercegovina': 'Bosnia and Herzegovina',
    'BosniaandHerzegovina': 'Bosnia and Herzegovina',
    'Burma': 'Myanmar',
    'Columbia': 'Colombia',
    'Czech': 'Czechia',
    'Czech Republic': 'Czechia',
    'Gambia': 'The Gambia',
    'Holland': 'Netherlands',
    'Ivory Coast': 'Côte d’Ivoire',
    'Kampuchea': 'Cambodia',
    'Korea': 'South Korea',
    'Lao Peoples Democratic Republic': 'Laos',
    'Macedonia': 'North Macedonia',
    'PuertoRico': 'Puerto Rico',
    'Salvador': 'El Salvador',
    'Slovia': 'Slovakia',
    'Swaziland': 'Eswatini',
    'The Democratic Republic of Congo': 'Democratic Republic of the Congo',
    'Democratic Republic of Congo': 'Democratic Republic of the Congo',
    'United States of America': 'United States',
    'Vaticano': 'Vatican City'
}

BAD_CITY_CHARS = re.compile(r"[0-9\\'\"?!()]")

def parse_sql(path):
    rows = []
    pat = re.compile(r"INSERT INTO `erdai_api_weather_static_heweather` VALUES \('(\d+)', (.*)\);$")
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = pat.match(line.rstrip('\r\n'))
            if m:
                rows.append(next(csv.reader([m.group(2)], delimiter=',', quotechar="'", skipinitialspace=True)))
    return rows

def is_ascii(s):
    return bool(s) and all(ord(c) < 128 for c in s)

def is_number(s):
    return bool(re.match(r'^-?\d+(\.\d+)?$', s))

def norm_name(s):
    return re.sub(r'\s+', ' ', s.replace('.', '').replace('\\', '').replace("'", '').lower()).strip()

def clean_country(s):
    return re.sub(r'\s+', ' ', s.replace('\\', '').replace("'", '').strip())

def extract_place(sentence, country):
    body = sentence[len(SENTENCE_PREFIX):-1]
    suffix = ', ' + country
    if body.endswith(suffix):
        return body[:-len(suffix)]
    return body

def check_sentence(sentence, country):
    start = sentence.rfind(country)
    if start < 0 or sentence[start + len(country):].strip() != '?':
        return False
    return True

if __name__ == "__main__":
    os.makedirs(configs.OUT_DIR, exist_ok = True)
    os.makedirs(configs.RESULTS_DIR, exist_ok = True)

    train_orig = pd.read_csv(os.path.join(configs.ORIG_DIR, 'train.csv'))
    valid_orig = pd.read_csv(os.path.join(configs.ORIG_DIR, 'valid.csv'))
    Orig_Countries = set(train_orig.country) | set(valid_orig.country)
    Orig_Places = set(norm_name(extract_place(s, c)) for s, c in zip(train_orig.sentence, train_orig.country))
    Orig_Places |= set(norm_name(extract_place(s, c)) for s, c in zip(valid_orig.sentence, valid_orig.country))

    rows = parse_sql(configs.SQL_PATH)
    print(f"SQL Parsed: {len(rows)}")

    Clean_Rows, Dropped_Countries = [], {}
    Seen_Pairs = set()
    for r in rows:
        city, country = r[0].strip(), clean_country(r[1])
        lat, lon = r[3].strip(), r[4].strip()
        if not (is_ascii(city) and is_number(lat) and is_number(lon)):
            continue
        if BAD_CITY_CHARS.search(city) or norm_name(city) == norm_name(country):
            continue
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            continue
        country = COUNTRY_MAP.get(country, country)
        if country not in Orig_Countries:
            Dropped_Countries[r[1]] = Dropped_Countries.get(r[1], 0) + 1
            continue
        pair = (norm_name(city), norm_name(country))
        if pair in Seen_Pairs:
            continue
        Seen_Pairs.add(pair)
        Clean_Rows.append({'city': city, 'country': country, 'lat': lat, 'lon': lon})
    print(f"Clean Rows: {len(Clean_Rows)} (dropped countries: {len(Dropped_Countries)})")

    By_Country = {}
    for r in Clean_Rows:
        By_Country.setdefault(r['country'], []).append(r)

    Extra_Rows = []
    for country, cities in sorted(By_Country.items()):
        cap = configs.MAX_US if country == 'United States' else configs.MAX_PER_COUNTRY
        quota = min(cap, len(cities))
        head_n = int(quota * configs.HEAD_RATIO)
        head = cities[:head_n]
        tail_pool = cities[head_n:]
        tail_n = quota - len(head)
        tail = list(np.random.permutation(tail_pool)[:tail_n]) if tail_n > 0 else []
        Extra_Rows.extend(head + tail)
    print(f"After Per-Country Sampling: {len(Extra_Rows)}")

    Kept_Rows = [r for r in Extra_Rows if norm_name(r['city']) not in Orig_Places]
    print(f"After Dedup Against Original: {len(Kept_Rows)} (removed {len(Extra_Rows) - len(Kept_Rows)})")
    if len(Kept_Rows) > configs.MAX_TOTAL:
        Kept_Rows = list(np.random.permutation(Kept_Rows)[:configs.MAX_TOTAL])
        print(f"Truncated To: {len(Kept_Rows)}")

    Records = []
    for r in Kept_Rows:
        sentence = SENTENCE_FMT.format(r['city'], r['country'])
        assert check_sentence(sentence, r['country']), sentence
        Records.append({'sentence': sentence, 'country': r['country'],
                        'latitude': f"{r['lat']:.5f}", 'longitude': f"{r['lon']:.5f}"})
    np.random.shuffle(Records)

    Extra_df = pd.DataFrame(Records)
    Out_Path = os.path.join(configs.OUT_DIR, 'extra_cities.csv')
    Extra_df.to_csv(Out_Path, index = False)
    print(f"Saved: {Out_Path} ({len(Extra_df)} rows)")

    Stats = {
        'n_sql_rows': len(rows),
        'n_clean_rows': len(Clean_Rows),
        'n_dropped_country_rows': sum(Dropped_Countries.values()),
        'n_dropped_dup_against_original': len(Extra_Rows) - len(Kept_Rows),
        'n_final': len(Records),
        'n_countries': int(Extra_df.country.nunique()),
        'top_countries': Extra_df.country.value_counts().head(10).to_dict(),
        'dropped_countries': dict(sorted(Dropped_Countries.items(), key = lambda kv: -kv[1])[:50])
    }
    with open(os.path.join(configs.RESULTS_DIR, 'extra_stats.json'), 'w') as f:
        json.dump(Stats, f, ensure_ascii = False, indent = 2)
    print(json.dumps({k: v for k, v in Stats.items() if k != 'dropped_countries'}, ensure_ascii = False, indent = 2))
