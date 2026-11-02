# app.py
import re
import io
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree

st.set_page_config(page_title="Сопоставление фин ап ↔ возможная АП", layout="wide")
st.title("Сопоставление фин ап клиента с возможной АП")

EPS_M = 15.0  # фиксированный порог гео, метров

# =========================================================
# Нормализация города
# =========================================================
CITY_SYNONYMS = {
    "питер": "санкт-петербург",
    "спб": "санкт-петербург",
    "санкт петербург": "санкт-петербург",
    "екатеренбург": "екатеринбург",
    "екб": "екатеринбург",
    "нижнийновгород": "нижний новгород",
    "ростовнадону": "ростов-на-дону",
    "ростов на дону": "ростов-на-дону",
}

def norm_city(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower().replace("ё", "е")
    s = re.sub(r"[.,;]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if s in CITY_SYNONYMS:
        return CITY_SYNONYMS[s]
    key = s.replace(" ", "")
    if key in CITY_SYNONYMS:
        return CITY_SYNONYMS[key]
    return s

# =========================================================
# Нормализация адреса
# =========================================================
TYPE_RE = re.compile(
    r"\b(пр-кт|пр-т|проспект|пр-д|проезд|просека|просек|переулок|пер|"
    r"бульвар|б-р|набережная|наб|площадь|пл|шоссе|ш|улица|ул|"
    r"аллея|мкр|микрорайон|тракт|тупик|туп|спуск|съезд|дорога|дор|"
    r"линия|квартал|кв-л|тер|проулок|протока)\b\.?",
    re.IGNORECASE,
)

DOM_RE = re.compile(r"(?:д\.?|дом)\s*([0-9]+[0-9a-zA-Zа-яА-Я/\-]*)", re.IGNORECASE)
HOUSE_AFTER_TYPE_RE = re.compile(r"^\s*([0-9]+[0-9a-zA-Zа-яА-Я/\-]*)")

GARBAGE_RE = re.compile(
    r"\b(пом|помещ|помещение|оф|офис|эт|этаж|кв|квартира|стр|строение|"
    r"корп|корпус|лит|литера)\b\.?\s*[\w\-/]*",
    re.IGNORECASE,
)

# удаляем область/республику/город — БЕЗ бага "г\s"
REGION_CITY_RE = re.compile(
    r"\b(респ\.?|республика|обл\.?|область|край|г\.|город)\s*",
    re.IGNORECASE,
)

def clean_street_name(name):
    if not name:
        return ""
    name = name.lower().replace("ё", "е")
    name = TYPE_RE.sub(" ", name)
    name = re.sub(r"[.,;:\(\)\"'«»]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def canon_house(h):
    if not h:
        return ""
    h = h.lower().replace("ё", "е")
    h = h.replace(" ", "").replace("-", "")
    return h

def parse_addr_fin(addr, city_hint):
    """Адрес из фин-апа:
    'Челябинская обл. г.Челябинск Скульптора Головницкого ул 20А пом.1'
    → ('головницкого', '20а')
    """
    if pd.isna(addr):
        return "", ""
    s = str(addr).lower().replace("ё", "е")

    # убрать область/город по маркерам (без "г\s")
    s = REGION_CITY_RE.sub(" ", s)
    if city_hint:
        s = s.replace(city_hint.lower(), " ")
    s = re.sub(r"\s+", " ", s).strip()

    # убрать пом/стр/корп и хвост
    s = GARBAGE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    m = TYPE_RE.search(s)
    if not m:
        return "", ""

    street = s[: m.start()].strip()
    words = street.split()
    # название улицы обычно 1–3 слова — берём последние 3
    if len(words) > 3:
        words = words[-3:]
    street = clean_street_name(" ".join(words))

    after = s[m.end():]
    hm = HOUSE_AFTER_TYPE_RE.match(after)
    if hm:
        house = hm.group(1)
    else:
        tail = re.search(r"([0-9]+[0-9a-zA-Zа-яА-Я/]*)\s*$", street)
        house = tail.group(1) if tail else ""

    return street, canon_house(house)

def parse_addr_ap(addr):
    """Адрес из файла 2: 'ул. Запарина; д.130' → ('запарина', '130')"""
    if pd.isna(addr):
        return "", ""
    s = str(addr).lower().replace("ё", "е")
    s = GARBAGE_RE.sub(" ", s)

    house = ""
    hm = DOM_RE.search(s)
    if hm:
        house = canon_house(hm.group(1))
        s = s[: hm.start()]
    else:
        # fallback: последнее число в строке — это дом
        tail = re.search(r"([0-9]+[0-9a-zA-Zа-яА-Я/]*)\s*$", s)
        if tail:
            house = canon_house(tail.group(1))
            s = s[: tail.start()]

    s = re.sub(r"[;]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    street = clean_street_name(s)
    return street, house

# =========================================================
# Геометрия
# =========================================================
def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.inf
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))

def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    s = re.sub(r"\s+", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan

# =========================================================
# Загрузка
# =========================================================
def load_fin(uploaded):
    df = pd.read_excel(uploaded)
    df.columns = [c.strip() for c in df.columns]
    df["Широта"] = df["Широта"].map(to_float)
    df["Долгота"] = df["Долгота"].map(to_float)
    df["_city_norm"] = df["город"].map(norm_city)
    parsed = df.apply(lambda r: parse_addr_fin(r["Адрес"], r["_city_norm"]), axis=1)
    df["_street"] = [p[0] for p in parsed]
    df["_house"] = [p[1] for p in parsed]
    return df

def load_ap(uploaded):
    df = pd.read_excel(uploaded)
    df.columns = [c.strip() for c in df.columns]
    df["Широта"] = df["Широта"].map(to_float)
    df["Долгота"] = df["Долгота"].map(to_float)
    df["_city_norm"] = df["город"].map(norm_city)
    parsed = df["адрес"].map(parse_addr_ap)
    df["_street"] = [p[0] for p in parsed]
    df["_house"] = [p[1] for p in parsed]
    return df

# =========================================================
# Сопоставление
# =========================================================
def match(fin, ap, eps_m=EPS_M):
    ap_valid = ap.dropna(subset=["Широта", "Долгота"]).reset_index(drop=True)

    # группировка по городу
    city_to_indices = {}
    for i in range(len(ap_valid)):
        city_to_indices.setdefault(ap_valid.at[i, "_city_norm"], []).append(i)

    # кеш KDTree на город
    city_trees = {}
    def get_city_tree(city):
        if city in city_trees:
            return city_trees[city]
        idxs = city_to_indices.get(city, [])
        if not idxs:
            city_trees[city] = (None, None)
            return None, None
        coords = ap_valid.loc[idxs, ["Широта", "Долгота"]].to_numpy()
        tree = cKDTree(coords)
        city_trees[city] = (tree, idxs)
        return city_trees[city]

    # результирующие колонки
    ap_out_cols = [
        "ID_Точки", "название", "адрес", "Широта", "Долгота",
        "Полигон", "город", "тип", "сеть", "база",
    ]
    addr_flag = []
    geo_flag = []
    matched_ap = []  # список dict с полями АП или None

    for _, r in fin.iterrows():
        city = r["_city_norm"]
        best_ap_idx = None

        # --- канал "адрес" ---
        addr_ap_idx = None
        if city:
            idxs = city_to_indices.get(city, [])
            if r["_street"] and r["_house"]:
                best_d = np.inf
                for gi in idxs:
                    a = ap_valid.loc[gi]
                    if a["_street"] == r["_street"] and a["_house"] == r["_house"]:
                        d = haversine_m(r["Широта"], r["Долгота"], a["Широта"], a["Долгота"])
                        if d < best_d:
                            best_d = d
                            addr_ap_idx = gi
        addr_ok = addr_ap_idx is not None

        # --- канал "гео" ---
        geo_ok = False
        geo_ap_idx = None
        if city:
            tree, idxs = get_city_tree(city)
            if tree is not None and not pd.isna(r["Широта"]) and not pd.isna(r["Долгота"]):
                lat0 = r["Широта"]
                cos_lat = cos(radians(lat0)) or 1.0
                # радиус в градусах с запасом (учитываем долготу)
                r_deg = (eps_m * 3.0) / 111320.0 / min(cos_lat, 1.0)
                local_idxs = tree.query_ball_point([r["Широта"], r["Долгота"]], r_deg)
                best_d = np.inf
                for li in local_idxs:
                    gi = idxs[li]
                    a = ap_valid.loc[gi]
                    d = haversine_m(lat0, r["Долгота"], a["Широта"], a["Долгота"])
                    if d <= eps_m and d < best_d:
                        best_d = d
                        geo_ap_idx = gi
                geo_ok = geo_ap_idx is not None

        # приоритет вывода: адрес → гео
        if addr_ok:
            best_ap_idx = addr_ap_idx
        elif geo_ok:
            best_ap_idx = geo_ap_idx

        addr_flag.append("ДА" if addr_ok else "НЕТ")
        geo_flag.append("ДА" if geo_ok else "НЕТ")
        matched_ap.append(ap_valid.loc[best_ap_idx].to_dict() if best_ap_idx is not None else None)

    fin = fin.copy()
    fin["Совпадение по адресу (ДА/НЕТ)"] = addr_flag
    fin["Совпадение по ГЕО (ДА/НЕТ)"] = geo_flag

    for c in ap_out_cols:
        fin[c + "_ap"] = [m.get(c) if m else None for m in matched_ap]

    return fin

# =========================================================
# UI
# =========================================================
col1, col2 = st.columns(2)
with col1:
    file_fin = st.file_uploader("Файл 1: фин ап клиента (.xlsx)", type=["xlsx"], key="fin")
with col2:
    file_ap = st.file_uploader("Файл 2: возможная АП (.xlsx)", type=["xlsx"], key="ap")

if file_fin and file_ap:
    with st.spinner("Читаем файлы..."):
        fin = load_fin(file_fin)
        ap = load_ap(file_ap)

    st.write(f"Фин ап: {len(fin)} строк. Возможная АП: {len(ap)} строк.")

    with st.spinner("Сопоставляем..."):
        result = match(fin, ap)

    # финальная сборка строго в заданном порядке
    result_out = result[[
        "ТЕРРИТОРИЯ", "КЛАСТЕР", "ИМЯ МАГАЗИНА", "SAP НОМЕР МАГАЗИНА",
        "АК наименование", "УА ПЛАНОГРАММЫ", "СТАТУС МАГАЗИНА", "Адрес",
        "город", "Широта", "Долгота",
        "Совпадение по адресу (ДА/НЕТ)", "Совпадение по ГЕО (ДА/НЕТ)",
        "ID_Точки_ap", "название_ap", "адрес_ap", "Широта_ap", "Долгота_ap",
        "Полигон_ap", "город_ap", "тип_ap", "сеть_ap", "база_ap",
    ]].rename(columns={
        "ID_Точки_ap": "ID_Точки",
        "название_ap": "название",
        "адрес_ap": "адрес",
        "Широта_ap": "Широта",
        "Долгота_ap": "Долгота",
        "Полигон_ap": "Полигон",
        "город_ap": "город",
        "тип_ap": "тип",
        "сеть_ap": "сеть",
        "база_ap": "база",
    })

    # выгрузка в Excel
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        result_out.to_excel(w, index=False, sheet_name="Результат")

        addr_da = int((result_out["Совпадение по адресу (ДА/НЕТ)"] == "ДА").sum())
        geo_da = int((result_out["Совпадение по ГЕО (ДА/НЕТ)"] == "ДА").sum())
        any_da = int(
            (
                (result_out["Совпадение по адресу (ДА/НЕТ)"] == "ДА")
                | (result_out["Совпадение по ГЕО (ДА/НЕТ)"] == "ДА")
            ).sum()
        )
        metrics = pd.DataFrame({
            "Показатель": [
                "Всего строк",
                "Совпадение по адресу (ДА)",
                "Совпадение по ГЕО (ДА)",
                "Совпало хотя бы по одному каналу",
                "Не найдено",
            ],
            "Значение": [
                len(result_out),
                addr_da,
                geo_da,
                any_da,
                len(result_out) - any_da,
            ],
        })
        metrics.to_excel(w, index=False, sheet_name="Метрики")
    bio.seek(0)

    st.success("Готово.")
    st.dataframe(result_out.head(50), use_container_width=True)
    st.download_button(
        label="Скачать результат (.xlsx)",
        data=bio,
        file_name="result_сопоставление.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
