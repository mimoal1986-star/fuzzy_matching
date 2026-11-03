# app.py
import re
import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree

st.set_page_config(
    page_title="Подбор ближайшего магазина фин-ап к возможной АП",
    layout="wide",
)
st.title("Подбор ближайшего магазина фин-ап к возможной АП")

# =========================================================
# Жёсткий порог для Приоритета 2 (координаты)
# =========================================================
EPS_M = 15.0  # FIX п.3: жёстко 15 м для П2

# =========================================================
# Сайдбар
# =========================================================
with st.sidebar:
    st.header("Настройки")
    n_target = st.number_input(
        "Кол-во подборов (X)",
        min_value=1, max_value=100_000, value=2350, step=50,
        help="Набрали X подборов — калькулятор останавливается.",
    )
    max_prio3_km = st.slider(
        "Макс. расстояние для Приоритета 3, км",
        min_value=0, max_value=500, value=100, step=10,
        help="0 = без ограничения. Защита от мусора при ошибках геокодирования.",
    )

st.caption(
    f"Приоритет 1 — совпадение адреса. "
    f"Приоритет 2 — координаты ≤ {EPS_M:.0f} м (жёстко). "
    f"Приоритет 3 — ближайшие пары, добиваем до {n_target}."
    + (f" Порог П3: ≤ {max_prio3_km} км." if max_prio3_km > 0 else " Порог П3: без ограничения.")
)

# =========================================================
# Нормализация города
# =========================================================
CITY_SYNONYMS = {
    "питер": "санкт-петербург",
    "спб": "санкт-петербург",
    "санкт петербург": "санкт-петербург",
    "екатеренбург": "екатеринбург",
    "екб": "екатеринбург",
    "екат": "екатеринбург",
    "члб": "челябинск",
    "челяба": "челябинск",
    "мск": "москва",
    "н новгород": "нижний новгород",
    "н.новгород": "нижний новгород",
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
# Общие регулярки
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
    r"\b(респ\.?|республика|обл\.?|область|край|г\.|город|р-н|район)\s*",
    re.IGNORECASE,
)

REGION_NAMES_RE = re.compile(
    r"\b("
    r"башкортостан[а]?|татарстан[а]?|"
    r"свердловская|свердловской|челябинская|челябинской|"
    r"нижегородская|нижегородской|воронежская|воронежской|"
    r"ростовская|ростовской|волгоградская|волгоградской|"
    r"самарская|самарской|иркутская|иркутской|"
    r"калужская|калужской|омская|омской|"
    r"новосибирская|новосибирской|"
    r"красноярский|красноярского|пермский|пермского|"
    r"московская|московской|ленинградская|ленинградской|"
    r"краснодарский|краснодарского|приморский|приморского|"
    r"хабаровский|хабаровского|забайкальский|забайкальского"
    r")\b\.?",
    re.IGNORECASE,
)

SETTLEMENT_RE = re.compile(
    r"\b(?:п\.|пос\.|поселок|рп\.|рабочий поселок|с\.|село|"
    r"д\.|деревня|кп\.|коттеджный поселок|сп\.|сп\.?|станица|ст-ца|"
    r"х\.|хутор|мкр\.?|микрорайон)\s*(?![0-9])[а-яёa-z\-]+\s*",
    re.IGNORECASE,
)

FIN_ADDR_HEAD_RE = re.compile(
    rf"^(?P<type>{TYPE_PATTERN})\b\.?\s*(?P<street>.+?)\s+"
    rf"(?P<house>[0-9]+[0-9a-zA-Zа-яА-Я/\-]*)"
    rf"(?:\s+(?P<letter>[а-яa-z]))?"
    rf"(?:\s*(?:корп\.?|к\.|стр\.?|с\.|лит\.?)\s*(?P<korpus>[0-9a-zA-Zа-яА-Я]+))?",
    re.IGNORECASE,
)
FIN_ADDR_MID_RE = re.compile(
    rf"^(?P<street>.+?)\s+(?P<type>{TYPE_PATTERN})\b\.?\s+"
    rf"(?P<house>[0-9]+[0-9a-zA-Zа-яА-Я/\-]*)"
    rf"(?:\s+(?P<letter>[а-яa-z]))?"
    rf"(?:\s*(?:корп\.?|к\.|стр\.?|с\.|лит\.?)\s*(?P<korpus>[0-9a-zA-Zа-яА-Я]+))?",
    re.IGNORECASE,
)
FIN_ADDR_FALLBACK_RE = re.compile(
    r"^(?P<street>.+?)\s+(?P<house>[0-9]+[0-9a-zA-Zа-яА-Я/\-]*)"
    r"(?:\s+(?P<letter>[а-яa-z]))?"
    r"(?:\s*(?:корп\.?|к\.|стр\.?|с\.|лит\.?)\s*(?P<korpus>[0-9a-zA-Zа-яА-Я]+))?"
    r"\s*$",
    re.IGNORECASE,
)

HOUSE_WITH_KORPUS_RE = re.compile(
    r"(?:д\.?|дом)\s*([0-9]+[0-9a-zA-Zа-яА-Я/\-]*)"
    r"(?:\s*(?:корп\.?|к\.|стр\.?|с\.|лит\.?)\s*([0-9a-zA-Zа-яА-Я]+))?",
    re.IGNORECASE,
)
HOUSE_KORPUS_LOOSE_RE = re.compile(
    r"([0-9]+[0-9a-zA-Zа-яА-Я/\-]*)"
    r"(?:\s*(?:корп\.?|к\.|стр\.?|с\.|лит\.?)\s*([0-9a-zA-Zа-яА-Я]+))?",
    re.IGNORECASE,
)

NO_FALLBACK_MARKER_RE = re.compile(
    r"\b(п\.|пом\.?|помещ\.?|оф\.?|офис\.?|эт\.?|этаж\.?|кв\.?|квартира)\b",
    re.IGNORECASE,
)

# =========================================================
# Канонизация дома
# =========================================================
def _canon_house_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(r"\s+", "", regex=True).str.replace("-", "", regex=False)
    s = s.str.replace(r"^0+(?=\d)", "", regex=True)
    s = s.where(s != "nan", "")
    s = s.where(s != "none", "")
    return s

def _house_base(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    base = s.str.extract(r"^([0-9]+(?:/[0-9]+)?)")[0].fillna("")
    return base

def _house_with_korpus(house_raw: pd.Series,
                       letter_raw: pd.Series,
                       korpus_raw: pd.Series) -> pd.Series:
    h = _canon_house_series(house_raw)
    l = _canon_house_series(letter_raw)
    k = _canon_house_series(korpus_raw)
    out = h.copy()
    has_l = l.ne("") & h.ne("")
    out.loc[has_l] = h.loc[has_l] + l.loc[has_l]
    has_k = k.ne("") & h.ne("")
    out.loc[has_k] = h.loc[has_k] + k.loc[has_k]
    out.loc[h.eq("")] = ""
    return out

def _clean_street_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(REGION_NAMES_RE, " ", regex=True)
    s = s.str.replace(TYPE_RE, " ", regex=True)
    s = s.str.replace(r"[.,;:\(\)\"'«»]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

INITIALS_RE = re.compile(r"\b[а-яa-z]\.(?:\s*[а-яa-z]\.)*\b", re.IGNORECASE)

def _strip_initials(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(INITIALS_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def _street_first_n(series: pd.Series, n: int = 3) -> pd.Series:
    s = _strip_initials(series)
    return s.str.split().str[:n].str.join(" ")

# =========================================================
# Парсинг фин-апа
# =========================================================
def _extract_fin_addr(s: pd.Series):
    head = s.str.extract(FIN_ADDR_HEAD_RE)
    mid = s.str.extract(FIN_ADDR_MID_RE)
    fb = s.str.extract(FIN_ADDR_FALLBACK_RE)

    street_h = head["street"].fillna("")
    house_h = head["house"].fillna("")
    letter_h = head["letter"].fillna("")
    korpus_h = head["korpus"].fillna("")

    street_m = mid["street"].fillna("")
    house_m = mid["house"].fillna("")
    letter_m = mid["letter"].fillna("")
    korpus_m = mid["korpus"].fillna("")

    street_f = fb["street"].fillna("")
    house_f = fb["house"].fillna("")
    letter_f = fb["letter"].fillna("")
    korpus_f = fb["korpus"].fillna("")

    use_head = house_h.ne("") & street_h.ne("")
    use_mid = (~use_head) & house_m.ne("") & street_m.ne("")
    use_fb = (~use_head) & (~use_mid) & house_f.ne("") & street_f.ne("")

    street = np.where(use_head, street_h, np.where(use_mid, street_m, street_f))
    house = np.where(use_head, house_h, np.where(use_mid, house_m, house_f))
    letter = np.where(use_head, letter_h, np.where(use_mid, letter_m, letter_f))
    korpus = np.where(use_head, korpus_h, np.where(use_mid, korpus_m, korpus_f))

    street = pd.Series(street, index=s.index)
    house = pd.Series(house, index=s.index)
    letter = pd.Series(letter, index=s.index)
    korpus = pd.Series(korpus, index=s.index)
    return street, house, letter, korpus

def parse_fin_series(addr: pd.Series, city: pd.Series):
    s = addr.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(REGION_CITY_RE, " ", regex=True)
    s = s.str.replace(SETTLEMENT_RE, " ", regex=True)

    city_clean = (
        city.fillna("").astype(str).str.lower().str.replace("ё", "е", regex=False)
    ).to_numpy()
    s_clean = s.fillna("").astype(str).to_numpy()

    new_s = []
    for s_i, c_i in zip(s_clean, city_clean):
        if c_i and c_i in s_i:
            tail = s_i.split(c_i, 1)[-1].strip()
            if tail:
                new_s.append(tail)
            else:
                new_s.append(re.sub(rf"\b{re.escape(c_i)}\b", " ", s_i).strip())
        else:
            new_s.append(s_i)
    s = pd.Series(new_s, index=s.index)

    s = s.str.replace(GARBAGE_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    street_raw, house_raw, letter_raw, korpus_raw = _extract_fin_addr(s)

    street = _clean_street_series(street_raw)
    house = _house_with_korpus(house_raw, letter_raw, korpus_raw)
    return street.fillna(""), house.fillna("")

# =========================================================
# Парсинг АП
# =========================================================
def parse_ap_series(addr: pd.Series):
    s = addr.astype(str).str.lower().str.replace("ё", "е", regex=False)

    ext = s.str.extract(HOUSE_WITH_KORPUS_RE)
    house_raw = ext[0].fillna("")
    korpus_raw = ext[1].fillna("")

    need_loose = house_raw.eq("")
    if need_loose.any():
        ext2 = s.loc[need_loose].str.extract(HOUSE_KORPUS_LOOSE_RE)
        house_raw.loc[need_loose] = ext2[0].fillna("")
        korpus_raw.loc[need_loose] = ext2[1].fillna("")

    letter_raw = pd.Series("", index=s.index)
    house = _house_with_korpus(house_raw, letter_raw, korpus_raw)
    has_dom = house.ne("")

    s = s.str.replace(GARBAGE_RE, " ", regex=True)
    parts = s.str.split(r"(?:д\.?|дом)\s*[0-9]", n=1, regex=True, expand=True)
    s_street = parts[0].where(has_dom, s)

    fallback = s_street.str.extract(r"([0-9]+[0-9a-zA-Zа-яА-Я/]*)\s*$")
    fallback_house = fallback[0].fillna("")
    has_no_marker = ~s.str.contains(NO_FALLBACK_MARKER_RE)
    need_fallback = (~has_dom) & fallback_house.ne("") & has_no_marker
    house = house.where(has_dom, fallback_house)
    house = _canon_house_series(house)
    if need_fallback.any():
        s_street.loc[need_fallback] = s_street.loc[need_fallback].str.replace(
            r"[0-9]+[0-9a-zA-Zа-яА-Я/]*\s*$", "", regex=True
        )

    s_street = s_street.str.replace(r"[;]+", " ", regex=True)
    s_street = s_street.str.replace(r"\s+", " ", regex=True).str.strip()
    street = _clean_street_series(s_street)
    return street.fillna(""), house.fillna("")

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(r"\s+", "", regex=True),
        errors="coerce",
    )

# =========================================================
# Haversine
# =========================================================
R_EARTH = 6371000.0

def haversine_np(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(a))

# =========================================================
# Загрузка
# =========================================================
FIN_REQUIRED = ["Адрес", "город", "Широта", "Долгота"]
AP_REQUIRED = ["ID_Точки", "название", "адрес", "Широта", "Долгота", "город"]

def _check_cols(df: pd.DataFrame, required: list, name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле «{name}» нет обязательных колонок: {missing}. "
            f"Есть: {list(df.columns)}"
        )

@st.cache_data(show_spinner=False)
def load_fin(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    _check_cols(df, FIN_REQUIRED, "фин ап")
    df = df.reset_index(drop=True)
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
    _check_cols(df, AP_REQUIRED, "возможная АП")
    df = df.reset_index(drop=True)
    df["Широта"] = to_float_series(df["Широта"])
    df["Долгота"] = to_float_series(df["Долгота"])
    df["_city_norm"] = norm_city_series(df["город"])
    street, house = parse_ap_series(df["адрес"])
    df["_street"] = street
    df["_house"] = house
    return df

# =========================================================
# Подготовка ключей
# =========================================================
def _prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
    key = df.reset_index(drop=True).copy()
    key["_row"] = np.arange(len(key))
    key = key[["_row", "_city_norm", "_street", "_house"]].copy()
    key["_house_base"] = _house_base(key["_house"])
    key["_street3"] = _street_first_n(key["_street"], 3)
    key = key[
        (key["_street"] != "") & (key["_house"] != "") & (key["_city_norm"] != "")
    ].copy()
    return key

# =========================================================
# Основной подбор
#   П1 = адрес
#   П2 = координаты ≤ EPS_M (15 м)
#   П3 = ближайший оставшийся (с опц. порогом max_prio3_km)
#   Лимит X: П1 → П2 → П3 (П3 — ровно до остатка X)
# =========================================================
@st.cache_data(show_spinner=False)
def match(fin: pd.DataFrame, ap: pd.DataFrame,
          n_target: int, max_prio3_km: float) -> pd.DataFrame:
    fin = fin.reset_index(drop=True).copy()
    ap = ap.reset_index(drop=True).copy()

    ap_valid = ap["Широта"].notna() & ap["Долгота"].notna()
    ap_work = ap[ap_valid].reset_index(drop=True).copy()
    n_ap = len(ap_work)
    if n_ap == 0:
        return pd.DataFrame()

    fin_has_coords = (fin["Широта"].notna() & fin["Долгота"].notna()).to_numpy()
    fin_lat_all = fin["Широта"].to_numpy()
    fin_lon_all = fin["Долгота"].to_numpy()
    ap_lat_all = ap_work["Широта"].to_numpy()
    ap_lon_all = ap_work["Долгота"].to_numpy()

    fin_key = _prepare_keys(fin)
    ap_key = _prepare_keys(ap_work)

    # результаты по индексу ap_work (0..n_ap-1)
    addr_ok = np.zeros(n_ap, dtype=bool)
    addr_fin = np.full(n_ap, -1, dtype=int)
    addr_dist = np.full(n_ap, np.nan)

    geo_ok = np.zeros(n_ap, dtype=bool)
    geo_fin = np.full(n_ap, -1, dtype=int)
    geo_dist = np.full(n_ap, np.nan)

    # ---------- Приоритет 1: адресные каналы ----------
    def _apply_channel(cand: pd.DataFrame):
        if len(cand) == 0:
            return
        cand = cand[~addr_ok[cand["_ap_idx"].to_numpy()]]
        if len(cand) == 0:
            return
        f_rows = cand["_row"].to_numpy()
        a_rows = cand["_ap_idx"].to_numpy()

        both = fin_has_coords[f_rows]
        d = np.full(len(cand), np.nan)
        if both.any():
            d[both] = haversine_np(
                fin_lat_all[f_rows[both]], fin_lon_all[f_rows[both]],
                ap_lat_all[a_rows[both]], ap_lon_all[a_rows[both]],
            )
        cand = cand.assign(_dist=d)
        best = (
            cand.sort_values("_dist", na_position="last")
                .drop_duplicates("_ap_idx", keep="first")
        )
        b_a = best["_ap_idx"].to_numpy()
        addr_ok[b_a] = True
        addr_fin[b_a] = best["_row"].to_numpy()
        addr_dist[b_a] = best["_dist"].to_numpy()

    # канал 1: (город, улица, дом)
    _apply_channel(fin_key.merge(ap_key, on=["_city_norm", "_street", "_house"], how="inner"))
    # канал 2: (город, улица, дом_база)
    _apply_channel(fin_key.merge(ap_key, on=["_city_norm", "_street", "_house_base"], how="inner"))
    # канал 3: (город, улица_3слова, дом_база)
    _apply_channel(fin_key.merge(ap_key, on=["_city_norm", "_street3", "_house_base"], how="inner"))
    # канал 4: (город, дом_база) при уникальной улице в обеих таблицах
    ap_uniq = (ap_key.groupby(["_city_norm", "_house_base"])["_street"]
               .nunique().reset_index().rename(columns={"_street": "_ap_nuniq"}))
    fin_uniq = (fin_key.groupby(["_city_norm", "_house_base"])["_street"]
                .nunique().reset_index().rename(columns={"_street": "_fin_nuniq"}))
    uniq = ap_uniq.merge(fin_uniq, on=["_city_norm", "_house_base"], how="inner")
    uniq = uniq[(uniq["_ap_nuniq"] == 1) & (uniq["_fin_nuniq"] == 1)]
    if len(uniq) > 0:
        ap_key_uniq = ap_key.merge(uniq[["_city_norm", "_house_base"]],
                                   on=["_city_norm", "_house_base"], how="inner")
        fin_key_uniq = fin_key.merge(uniq[["_city_norm", "_house_base"]],
                                     on=["_city_norm", "_house_base"], how="inner")
        cand4 = fin_key_uniq.merge(ap_key_uniq, on=["_city_norm", "_house_base"],
                                   how="inner", suffixes=("_f", "_a"))
        cand4 = cand4[cand4["_house_f"] == cand4["_house_a"]]
        if len(cand4) > 0:
            _apply_channel(cand4[["_row", "_ap_idx"]])

    # ---------- Приоритет 2: координаты ≤ EPS_M ----------
    # FIX п.11: итерируем только по непокрытым П1 АП (экономия KDTree/запросов)
    ap_need_geo = ap_work[~addr_ok]
    if len(ap_need_geo) > 0:
        for city, ap_idx_city in ap_need_geo.groupby("_city_norm").groups.items():
            ap_idx_city = np.asarray(ap_idx_city)  # позиции в ap_work (RangeIndex)
            fin_idx_city = np.flatnonzero(
                fin_has_coords & (fin["_city_norm"].to_numpy() == city)
            )
            if len(fin_idx_city) == 0:
                continue
            fin_coords = np.column_stack([fin_lat_all[fin_idx_city],
                                           fin_lon_all[fin_idx_city]])
            tree = cKDTree(fin_coords)
            ap_coords = np.column_stack([ap_lat_all[ap_idx_city],
                                          ap_lon_all[ap_idx_city]])

            cos_lat = np.clip(np.cos(np.radians(ap_coords[:, 0])), 0.1, 1.0)
            r_deg = (EPS_M * 3.0) / 111320.0 / cos_lat
            r_max = float(np.max(r_deg))

            neighbours = tree.query_ball_point(ap_coords, r_max)
            a_local, f_local = [], []
            for i, lst in enumerate(neighbours):
                if lst:
                    a_local.extend([i] * len(lst))
                    f_local.extend(lst)
            if not a_local:
                continue
            a_local = np.array(a_local); f_local = np.array(f_local)
            dist = haversine_np(ap_coords[a_local, 0], ap_coords[a_local, 1],
                                fin_coords[f_local, 0], fin_coords[f_local, 1])
            mask = dist <= EPS_M
            if not mask.any():
                continue
            tmp = pd.DataFrame({
                "ap": ap_idx_city[a_local[mask]],
                "fin": fin_idx_city[f_local[mask]],
                "d": dist[mask],
            }).sort_values("d").drop_duplicates("ap", keep="first")
            tmp = tmp[~addr_ok[tmp["ap"].to_numpy()]]
            if len(tmp) == 0:
                continue
            a = tmp["ap"].to_numpy()
            geo_ok[a] = True
            geo_fin[a] = tmp["fin"].to_numpy()
            geo_dist[a] = tmp["d"].to_numpy()

    # ---------- Приоритет 3: ближайший оставшийся ----------
    prio3_ok = np.zeros(n_ap, dtype=bool)
    prio3_fin = np.full(n_ap, -1, dtype=int)
    prio3_dist = np.full(n_ap, np.nan)

    need_p3 = ~(addr_ok | geo_ok)
    max_prio3_m = (max_prio3_km * 1000.0) if max_prio3_km and max_prio3_km > 0 else None

    if need_p3.any():
        for city, ap_idx_city in ap_work[need_p3].groupby("_city_norm").groups.items():
            ap_idx_city = np.asarray(ap_idx_city)
            fin_idx_city = np.flatnonzero(
                fin_has_coords & (fin["_city_norm"].to_numpy() == city)
            )
            if len(fin_idx_city) == 0:
                continue
            fin_coords = np.column_stack([fin_lat_all[fin_idx_city],
                                           fin_lon_all[fin_idx_city]])
            tree = cKDTree(fin_coords)
            ap_coords = np.column_stack([ap_lat_all[ap_idx_city],
                                          ap_lon_all[ap_idx_city]])
            d, idx = tree.query(ap_coords, k=1)
            if max_prio3_m is not None:
                within = d <= max_prio3_m
                if not within.any():
                    continue
                a = ap_idx_city[within]
                prio3_ok[a] = True
                prio3_fin[a] = fin_idx_city[idx[within]]
                prio3_dist[a] = d[within]
            else:
                a = ap_idx_city
                prio3_ok[a] = True
                prio3_fin[a] = fin_idx_city[idx]
                prio3_dist[a] = d

    # ---------- Лимит X: П1 → П2 → П3 ----------
    taken = np.zeros(n_ap, dtype=bool)

    # FIX п.3/п.12: NaN в П1 гарантированно уходит в конец (inf-замена), stable sort
    p1_idx = np.flatnonzero(addr_ok)
    if len(p1_idx) > 0:
        d1 = addr_dist[p1_idx]
        d1_sort = np.where(np.isnan(d1), np.inf, d1)
        p1_order = p1_idx[np.argsort(d1_sort, kind="stable")]
        if len(p1_order) > n_target:
            p1_order = p1_order[:n_target]
        taken[p1_order] = True

    remaining = n_target - int(taken.sum())
    if remaining > 0:
        p2_idx = np.flatnonzero(geo_ok & ~taken)
        if len(p2_idx) > 0:
            p2_order = p2_idx[np.argsort(geo_dist[p2_idx], kind="stable")][:remaining]
            taken[p2_order] = True

    remaining = n_target - int(taken.sum())
    if remaining > 0:
        p3_idx = np.flatnonzero(prio3_ok & ~taken)
        if len(p3_idx) > 0:
            p3_order = p3_idx[np.argsort(prio3_dist[p3_idx], kind="stable")][:remaining]
            taken[p3_order] = True

    # ---------- Сборка результата ----------
    ap_cols_out = ["ID_Точки", "название", "адрес",
                   "Широта", "Долгота", "Полигон", "город", "тип", "сеть", "база"]
    ap_cols_present = [c for c in ap_cols_out if c in ap_work.columns]
    out = ap_work[ap_cols_present].copy()
    out = out.rename(columns={
        "Широта": "Широта_АП",
        "Долгота": "Долгота_АП",
        "город": "город_АП",
    })

    out["Совпадение по адресу (ДА/НЕТ)"] = np.where(addr_ok, "ДА", "НЕТ")
    out["Совпадение по ГЕО (ДА/НЕТ)"] = np.where(geo_ok, "ДА", "НЕТ")
    out["Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)"] = np.where(addr_ok | geo_ok, "ДА", "НЕТ")
    out["Приоритет 1 (ДА/НЕТ)"] = np.where(addr_ok, "ДА", "НЕТ")
    out["Приоритет 2 (ДА/НЕТ)"] = np.where(~addr_ok & geo_ok, "ДА", "НЕТ")
    out["Приоритет 3 (ДА/НЕТ)"] = np.where(~addr_ok & ~geo_ok & prio3_ok, "ДА", "НЕТ")

    chosen_fin = np.where(addr_ok, addr_fin,
                  np.where(geo_ok, geo_fin, prio3_fin))
    chosen_dist = np.where(addr_ok, addr_dist,
                   np.where(geo_ok, geo_dist, prio3_dist))
    out["Расстояние по гео"] = chosen_dist

    out["_chosen_fin"] = chosen_fin
    out["_taken"] = taken
    out["_prio_num"] = np.where(addr_ok, 1, np.where(geo_ok, 2,
                       np.where(prio3_ok, 3, 4)))

    # FIX п.16: сначала отсекаем по taken, потом мержим фин-поля — меньше работы
    out = out[out["_taken"]].copy()

    fin_out_cols = ["ТЕРРИТОРИЯ", "КЛАСТЕР", "ИМЯ МАГАЗИНА", "SAP НОМЕР МАГАЗИНА",
                    "АК наименование", "УА ПЛАНОГРАММЫ", "СТАТУС МАГАЗИНА",
                    "Адрес", "город", "Широта", "Долгота"]
    fin_cols_present = [c for c in fin_out_cols if c in fin.columns]

    if len(out) > 0:
        chosen_vals = out["_chosen_fin"].astype(int).to_numpy()
        fin_lookup = fin.loc[chosen_vals, fin_cols_present].copy()
        fin_lookup.index = out.index
        for c in fin_cols_present:
            out[c] = fin_lookup[c]
    for c in fin_out_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out.sort_values(["_prio_num", "Расстояние по гео"], na_position="last")
    return out

# =========================================================
# UI
# =========================================================
col1, col2 = st.columns(2)
with col1:
    file_fin = st.file_uploader("Файл 1: фин ап клиента (.xlsx)", type=["xlsx"], key="fin")
with col2:
    file_ap = st.file_uploader("Файл 2: возможная АП (.xlsx)", type=["xlsx"], key="ap")

if file_fin and file_ap:
    try:
        fin = load_fin(file_fin.getvalue())
        ap = load_ap(file_ap.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.write(f"Фин ап: {len(fin)} строк. Возможная АП: {len(ap)} строк.")

    with st.spinner("Подбираем..."):
        result = match(fin, ap, int(n_target), float(max_prio3_km))

    if result.empty:
        st.warning("Не удалось подобрать ни одной АП (нет координат / городов / магазинов).")
        st.stop()

    final = pd.DataFrame({
        "ID_Точки": result["ID_Точки"],
        "название": result["название"],
        "адрес": result["адрес"],
        "Широта_АП": result["Широта_АП"],
        "Долгота_АП": result["Долгота_АП"],
        "Полигон": result.get("Полигон", np.nan),
        "город_АП": result["город_АП"],
        "тип": result.get("тип", np.nan),
        "сеть": result.get("сеть", np.nan),
        "база": result.get("база", np.nan),
        "Совпадение по адресу (ДА/НЕТ)": result["Совпадение по адресу (ДА/НЕТ)"],
        "Совпадение по ГЕО (ДА/НЕТ)": result["Совпадение по ГЕО (ДА/НЕТ)"],
        "Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)": result["Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)"],
        "Приоритет 1 (ДА/НЕТ)": result["Приоритет 1 (ДА/НЕТ)"],
        "Приоритет 2 (ДА/НЕТ)": result["Приоритет 2 (ДА/НЕТ)"],
        "Приоритет 3 (ДА/НЕТ)": result["Приоритет 3 (ДА/НЕТ)"],
        "Расстояние по гео": result["Расстояние по гео"],
        "ТЕРРИТОРИЯ": result["ТЕРРИТОРИЯ"],
        "КЛАСТЕР": result["КЛАСТЕР"],
        "ИМЯ МАГАЗИНА": result["ИМЯ МАГАЗИНА"],
        "SAP НОМЕР МАГАЗИНА": result["SAP НОМЕР МАГАЗИНА"],
        "АК наименование": result["АК наименование"],
        "УА ПЛАНОГРАММЫ": result["УА ПЛАНОГРАММЫ"],
        "СТАТУС МАГАЗИНА": result["СТАТУС МАГАЗИНА"],
        "Адрес": result["Адрес"],
        "город": result["город"],
        "Широта": result["Широта"],
        "Долгота": result["Долгота"],
    })

    st.success(f"Подобрано {len(final)} из {n_target} (цель). Достигли X — стоп.")
    st.dataframe(final.head(50), use_container_width=True)

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        final.to_excel(w, index=False, sheet_name="Результат")
        p1 = int((final["Приоритет 1 (ДА/НЕТ)"] == "ДА").sum())
        p2 = int((final["Приоритет 2 (ДА/НЕТ)"] == "ДА").sum())
        p3 = int((final["Приоритет 3 (ДА/НЕТ)"] == "ДА").sum())
        metrics = pd.DataFrame({
            "Показатель": [
                "Цель X",
                "Подобрано всего",
                "Приоритет 1 (адрес)",
                f"Приоритет 2 (координаты ≤ {EPS_M:.0f} м)",
                "Приоритет 3 (ближайшие, добивка)",
                "Осталось до X (не хватило АП)",
            ],
            "Значение": [
                int(n_target),
                len(final),
                p1, p2, p3,
                max(0, int(n_target) - len(final)),
            ],
        })
        metrics.to_excel(w, index=False, sheet_name="Метрики")
    bio.seek(0)

    st.download_button(
        label="Скачать результат (.xlsx)",
        data=bio,
        file_name="result_подбор_АП.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
