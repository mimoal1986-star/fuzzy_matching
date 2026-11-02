# app.py
import re
import io

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

def norm_city_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.lower().str.replace("ё", "е", regex=False)
    out = out.str.replace(r"[.,;]+", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.map(lambda x: CITY_SYNONYMS.get(x, CITY_SYNONYMS.get(x.replace(" ", ""), x)))
    out = out.where(out != "nan", "")
    out = out.where(out != "none", "")
    return out

# =========================================================
# Нормализация адреса — векторизованно
# =========================================================
TYPE_PATTERN = (
    r"пр-кт|пр-т|проспект|пр-д|проезд|просека|просек|переулок|пер|"
    r"бульвар|б-р|набережная|наб|площадь|пл|шоссе|ш|улица|ул|"
    r"аллея|мкр|микрорайон|тракт|тупик|туп|спуск|съезд|дорога|дор|"
    r"линия|квартал|кв-л|тер|проулок|протока"
)
TYPE_RE = re.compile(rf"\b({TYPE_PATTERN})\b\.?", re.IGNORECASE)

GARBAGE_RE = re.compile(
    r"\b(пом|помещ|помещение|оф|офис|эт|этаж|кв|квартира|стр|строение|"
    r"корп|корпус|лит|литера)\b\.?\s*[\w\-/]*",
    re.IGNORECASE,
)

REGION_CITY_RE = re.compile(
    r"\b(респ\.?|республика|обл\.?|область|край|г\.|город)\s*",
    re.IGNORECASE,
)

DOM_RE = re.compile(r"(?:д\.?|дом)\s*([0-9]+[0-9a-zA-Zа-яА-Я/\-]*)", re.IGNORECASE)

# для фин-апа: улица + маркер + дом одной regex.
# ВАЖНО: маркер отделён пробелами (\s+), чтобы не срабатывать на «улица» внутри названия улицы.
FIN_ADDR_RE = re.compile(
    rf"^(?P<street>.*?)\s+(?P<type>{TYPE_PATTERN})\b\.?\s+(?P<house>[0-9]+[0-9a-zA-Zа-яА-Я/\-]*)",
    re.IGNORECASE,
)

def _clean_street(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(TYPE_RE, " ", regex=True)
    s = s.str.replace(r"[.,;:\(\)\"'«»]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def _canon_house(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
    s = s.where(s != "nan", "")
    return s

def parse_fin_series(addr: pd.Series, city: pd.Series):
    """Векторный парсинг адресов фин-апа."""
    s = addr.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(REGION_CITY_RE, " ", regex=True)

    # --- убрать город-подсказку из строки адреса, без apply(axis=1) ---
    city_clean = city.fillna("").astype(str).str.lower().to_numpy()
    s_clean = s.fillna("").astype(str).to_numpy()
    s_clean = [
        s_i.replace(c_i, " ") if c_i and c_i in s_i else s_i
        for s_i, c_i in zip(s_clean, city_clean)
    ]
    s = pd.Series(s_clean, index=s.index)

    s = s.str.replace(GARBAGE_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    extracted = s.str.extract(FIN_ADDR_RE)
    street_raw = extracted["street"].fillna("")
    house_raw = extracted["house"].fillna("")

    # улица: последние 3 слова + убрать тип
    street = street_raw.str.split().str[-3:].str.join(" ")
    street = _clean_street(street)
    house = _canon_house(house_raw)

    return street.fillna(""), house.fillna("")

def parse_ap_series(addr: pd.Series):
    """Векторный парсинг адресов файла 2: 'ул. Запарина; д.130'."""
    s = addr.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(GARBAGE_RE, " ", regex=True)

    # дом по "д." / "дом"
    house_raw = s.str.extract(r"(?:д\.?|дом)\s*([0-9]+[0-9a-zA-Zа-яА-Я/\-]*)")[0].fillna("")
    has_dom = house_raw.ne("")

    # обрезаем строку по первому "д." / "дом" — через .str.split (regex=True поддерживается)
    parts = s.str.split(r"(?:д\.?|дом)\s*[0-9]", n=1, regex=True, expand=True)
    s_street = parts[0].where(has_dom, s)

    # fallback: если "д." нет — последнее число в строке = дом
    fallback = s_street.str.extract(r"([0-9]+[0-9a-zA-Zа-яА-Я/]*)\s*$")
    fallback_house = fallback[0].fillna("")
    need_fallback = (~has_dom) & fallback_house.ne("")
    house_raw = house_raw.where(has_dom, fallback_house)
    if need_fallback.any():
        s_street.loc[need_fallback] = s_street.loc[need_fallback].str.replace(
            r"[0-9]+[0-9a-zA-Zа-яА-Я/]*\s*$", "", regex=True
        )

    s_street = s_street.str.replace(r"[;]+", " ", regex=True)
    s_street = s_street.str.replace(r"\s+", " ", regex=True).str.strip()
    street = _clean_street(s_street)
    house = _canon_house(house_raw)
    return street.fillna(""), house.fillna("")

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(r"\s+", "", regex=True),
        errors="coerce",
    )

# =========================================================
# Haversine — numpy
# =========================================================
R_EARTH = 6371000.0

def haversine_np(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(a))

# =========================================================
# Загрузка
# =========================================================
@st.cache_data(show_spinner=False)
def load_fin(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    df["Широта"] = to_float_series(df["Широта"])
    df["Долгота"] = to_float_series(df["Долгота"])
    df["_city_norm"] = norm_city_series(df["город"])
    street, house = parse_fin_series(df["Адрес"], df["_city_norm"])
    df["_street"] = street
    df["_house"] = house
    return df

@st.cache_data(show_spinner=False)
def load_ap(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    df["Широта"] = to_float_series(df["Широта"])
    df["Долгота"] = to_float_series(df["Долгота"])
    df["_city_norm"] = norm_city_series(df["город"])
    street, house = parse_ap_series(df["адрес"])
    df["_street"] = street
    df["_house"] = house
    return df

# =========================================================
# Сопоставление — векторизованное
# ВАЖНО: без `_fin`/`_ap` — чтобы Streamlit кешировал по содержимому DataFrame
# и сбрасывал кеш при загрузке нового файла.
# =========================================================
@st.cache_data(show_spinner=False)
def match(fin: pd.DataFrame, ap: pd.DataFrame, eps_m: float = EPS_M) -> pd.DataFrame:
    fin = fin.copy()
    ap = ap.dropna(subset=["Широта", "Долгота"]).reset_index(drop=True)

    # ---------- Адресный канал через merge ----------
    fin_key = fin.reset_index()[["index", "_city_norm", "_street", "_house"]].copy()
    fin_key = fin_key[
        (fin_key["_street"] != "") & (fin_key["_house"] != "") & (fin_key["_city_norm"] != "")
    ]

    ap_key = ap[["_city_norm", "_street", "_house"]].copy()
    ap_key["_ap_idx"] = np.arange(len(ap))
    ap_key = ap_key[
        (ap_key["_street"] != "") & (ap_key["_house"] != "") & (ap_key["_city_norm"] != "")
    ]

    cand = fin_key.merge(ap_key, on=["_city_norm", "_street", "_house"], how="inner")

    addr_ok = pd.Series(False, index=fin.index)
    addr_ap_idx = pd.Series(np.nan, index=fin.index)

    if len(cand) > 0:
        fin_lat = fin.loc[cand["index"], "Широта"].to_numpy()
        fin_lon = fin.loc[cand["index"], "Долгота"].to_numpy()
        ap_lat = ap.loc[cand["_ap_idx"], "Широта"].to_numpy()
        ap_lon = ap.loc[cand["_ap_idx"], "Долгота"].to_numpy()
        cand = cand.copy()
        cand["_dist"] = haversine_np(fin_lat, fin_lon, ap_lat, ap_lon)
        cand_best = cand.sort_values("_dist").drop_duplicates("index", keep="first")
        addr_ok.loc[cand_best["index"].values] = True
        addr_ap_idx.loc[cand_best["index"].values] = cand_best["_ap_idx"].values

    # ---------- Гео-канал через KDTree батчем ----------
    geo_ok = pd.Series(False, index=fin.index)
    geo_ap_idx = pd.Series(np.nan, index=fin.index)

    fin_valid_mask = (
        fin["Широта"].notna() & fin["Долгота"].notna() & (fin["_city_norm"] != "")
    )

    for city, fin_idx_city in fin[fin_valid_mask].groupby("_city_norm").groups.items():
        ap_idx_city = ap.index[ap["_city_norm"] == city]
        if len(ap_idx_city) == 0:
            continue

        ap_coords = ap.loc[ap_idx_city, ["Широта", "Долгота"]].to_numpy()
        tree = cKDTree(ap_coords)

        fin_idx_city = np.array(list(fin_idx_city))
        fin_coords = fin.loc[fin_idx_city, ["Широта", "Долгота"]].to_numpy()

        # радиус в градусах с учётом cos(lat); query_ball_point требует скаляр — берём максимум
        cos_lat = np.clip(np.cos(np.radians(fin_coords[:, 0])), 0.1, 1.0)
        r_deg = (eps_m * 3.0) / 111320.0 / cos_lat
        r_max = float(np.max(r_deg))

        neighbours = tree.query_ball_point(fin_coords, r_max)

        fin_local = []
        ap_local = []
        for i, lst in enumerate(neighbours):
            if lst:
                fin_local.extend([i] * len(lst))
                ap_local.extend(lst)
        if not fin_local:
            continue

        fin_local = np.array(fin_local)
        ap_local = np.array(ap_local)

        lat1 = fin_coords[fin_local, 0]
        lon1 = fin_coords[fin_local, 1]
        lat2 = ap_coords[ap_local, 0]
        lon2 = ap_coords[ap_local, 1]

        dist = haversine_np(lat1, lon1, lat2, lon2)
        mask = dist <= eps_m
        if not mask.any():
            continue

        fin_local = fin_local[mask]
        ap_local = ap_local[mask]
        dist = dist[mask]

        fin_global = fin_idx_city[fin_local]
        ap_global = ap_idx_city[ap_local]

        tmp = pd.DataFrame({"fin": fin_global, "ap": ap_global, "d": dist})
        tmp = tmp.sort_values("d").drop_duplicates("fin", keep="first")

        geo_ok.loc[tmp["fin"].values] = True
        geo_ap_idx.loc[tmp["fin"].values] = tmp["ap"].values

    # ---------- Сборка ----------
    fin["Совпадение по адресу (ДА/НЕТ)"] = np.where(addr_ok, "ДА", "НЕТ")
    fin["Совпадение по ГЕО (ДА/НЕТ)"] = np.where(geo_ok, "ДА", "НЕТ")

    chosen_ap_idx = addr_ap_idx.where(addr_ok, geo_ap_idx)

    ap_out_cols = [
        "ID_Точки", "название", "адрес", "Широта", "Долгота",
        "Полигон", "город", "тип", "сеть", "база",
    ]
    chosen_valid = chosen_ap_idx.dropna()
    if len(chosen_valid) > 0:
        ap_lookup = ap.loc[chosen_valid.astype(int).values, ap_out_cols].copy()
        ap_lookup.index = chosen_valid.index
    else:
        ap_lookup = pd.DataFrame(columns=ap_out_cols, index=fin.index)

    for c in ap_out_cols:
        fin[c + "_ap"] = ap_lookup[c].reindex(fin.index)

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
    fin_bytes = file_fin.getvalue()
    ap_bytes = file_ap.getvalue()

    with st.spinner("Читаем файлы..."):
        fin = load_fin(fin_bytes)
        ap = load_ap(ap_bytes)

    st.write(f"Фин ап: {len(fin)} строк. Возможная АП: {len(ap)} строк.")

    with st.spinner("Сопоставляем..."):
        result = match(fin, ap, EPS_M)

    result_out = result[[
        "ТЕРРИТОРИЯ", "КЛАСТЕР", "ИМЯ МАГАЗИНА", "SAP НОМЕР МАГАЗИНА",
        "АК наименование", "УА ПЛАНОГРАММЫ", "СТАТУС МАГАЗИНА", "Адрес",
        "город", "Широта", "Долгота",
        "Совпадение по адресу (ДА/НЕТ)", "Совпадение по ГЕО (ДА/НЕТ)",
        "ID_Точки_ap", "название_ap", "адрес_ap",
        "Широта_ap", "Долгота_ap",
        "Полигон_ap", "город_ap", "тип_ap", "сеть_ap", "база_ap",
    ]].rename(columns={
        "ID_Точки_ap": "ID_Точки",
        "название_ap": "название",
        "адрес_ap": "адрес",
        "Широта_ap": "Широта_АП",
        "Долгота_ap": "Долгота_АП",
        "Полигон_ap": "Полигон",
        "город_ap": "город_АП",
        "тип_ap": "тип",
        "сеть_ap": "сеть",
        "база_ap": "база",
    })

    assert result_out.columns.is_unique, f"Duplicate columns: {list(result_out.columns)}"


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
                len(result_out), addr_da, geo_da, any_da, len(result_out) - any_da,
            ],
        })
        metrics.to_excel(w, index=False, sheet_name="Метрики")
    bio.seek(0)

    st.success("Готово.")
    st.dataframe(result_out.head(50), width="stretch")
    st.download_button(
        label="Скачать результат (.xlsx)",
        data=bio,
        file_name="result_сопоставление.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
