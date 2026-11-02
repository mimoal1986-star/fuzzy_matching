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

# п.4 + п.22: регионы — вырезаем явно, включая родительный падеж
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

# п.5 (новый): населённые пункты — вырезаем, если после них НЕТ цифры
#   п.Луч 77  -> " 77" (улица пустая, дом 77) — но это НЕ улица, оставляем как есть
#   п.Луч, ул.Ленина 5 -> " , ул.ленина 5"
SETTLEMENT_RE = re.compile(
    r"\b(?:п\.|пос\.|поселок|рп\.|рабочий поселок|с\.|село|"
    r"д\.|деревня|кп\.|коттеджный поселок|сп\.|сп\.?|станица|ст-ца|"
    r"х\.|хутор|мкр\.?|микрорайон)\s*(?![0-9])[а-яёa-z\-]+\s*",
    re.IGNORECASE,
)

# =========================================================
# п.1/п.29/п.10: парсинг адреса фин-апа
#   Поддерживает формы:
#     "Чичерина ул. 35"          (маркер в середине)
#     "ул. Чичерина 35"          (маркер в начале)
#     "ул.Чичерина 35"           (маркер в начале без пробела)
#     "Чичерина ул. 35 В"        (буква через пробел -> 35в)
#     "Чичерина ул. 35 к.1"      (корпус -> 35к1)
#     "Чичерина ул. 35 стр.2"    (строение -> 35с2)
#   + fallback "число в конце" (п.8)
# =========================================================
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
# п.8: fallback — число в конце, без маркера типа
FIN_ADDR_FALLBACK_RE = re.compile(
    r"^(?P<street>.+?)\s+(?P<house>[0-9]+[0-9a-zA-Zа-яА-Я/\-]*)"
    r"(?:\s+(?P<letter>[а-яa-z]))?"
    r"(?:\s*(?:корп\.?|к\.|стр\.?|с\.|лит\.?)\s*(?P<korpus>[0-9a-zA-Zа-яА-Я]+))?"
    r"\s*$",
    re.IGNORECASE,
)

# п.2: дом + опциональный корпус/строение (для АП)
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

# п.5: маркеры, при которых fallback «последнее число = дом» запрещён
NO_FALLBACK_MARKER_RE = re.compile(
    r"\b(п\.|пом\.?|помещ\.?|оф\.?|офис\.?|эт\.?|этаж\.?|кв\.?|квартира)\b",
    re.IGNORECASE,
)

# =========================================================
# п.6 + п.2: канонизация дома
# =========================================================
def _canon_house_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(r"\s+", "", regex=True).str.replace("-", "", regex=False)
    # п.6: ведущие нули — 05 -> 5, 005 -> 5, но 0 -> 0
    s = s.str.replace(r"^0+(?=\d)", "", regex=True)
    s = s.where(s != "nan", "")
    s = s.where(s != "none", "")
    return s

# п.23: _house_base ДО _canon_house — 35-2 -> 35
def _house_base(series: pd.Series) -> pd.Series:
    """Дом без корпуса:
    35б -> 35, 12с2 -> 12, 10к1 -> 10,
    35/2 -> 35/2, 35-2 -> 35, 35В -> 35.
    """
    s = series.astype(str)
    base = s.str.extract(r"^([0-9]+(?:/[0-9]+)?)")[0].fillna("")
    return base

# =========================================================
# п.2: собрать house + letter + корпус
# =========================================================
def _house_with_korpus(house_raw: pd.Series,
                       letter_raw: pd.Series,
                       korpus_raw: pd.Series) -> pd.Series:
    h = _canon_house_series(house_raw)
    l = _canon_house_series(letter_raw)
    k = _canon_house_series(korpus_raw)
    out = h.copy()
    # п.2: буква через пробел -> 35 В -> 35в
    has_l = l.ne("") & h.ne("")
    out.loc[has_l] = h.loc[has_l] + l.loc[has_l]
    # п.2: корпус/строение -> 35к1, 35с2
    has_k = k.ne("") & h.ne("")
    out.loc[has_k] = h.loc[has_k] + k.loc[has_k]
    out.loc[h.eq("")] = ""
    return out

# =========================================================
# п.7: единый нормализатор улицы
# =========================================================
def _clean_street_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(REGION_NAMES_RE, " ", regex=True)   # п.4 + п.22
    s = s.str.replace(TYPE_RE, " ", regex=True)
    s = s.str.replace(r"[.,;:\(\)\"'«»]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

# =========================================================
# п.6: убрать инициалы из street3
# =========================================================
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
# п.1 + п.7 + п.9 + п.10 + п.24 + п.29 + п.8: парсинг фин-апа
# =========================================================
def _extract_fin_addr(s: pd.Series):
    """(street_raw, house_raw) с fallback head→mid→fallback-число."""
    head = s.str.extract(FIN_ADDR_HEAD_RE)
    mid = s.str.extract(FIN_ADDR_MID_RE)
    fb = s.str.extract(FIN_ADDR_FALLBACK_RE)

    def _combine(d, prefix):
        return (d[f"{prefix}_street"].fillna(""),
                d[f"{prefix}_house"].fillna(""),
                d[f"{prefix}_letter"].fillna(""),
                d[f"{prefix}_korpus"].fillna(""))

    street_h, house_h, letter_h, korpus_h = _combine(head, "type_first")  # unused, для совместимости
    # head/mid/fallback возвращают одинаковые имена групп, extractor создаст
    # колонки по именам групп, без префикса — берём напрямую
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

    # приоритет: head -> mid -> fallback
    use_head = house_h.ne("") & street_h.ne("")
    use_mid = (~use_head) & house_m.ne("") & street_m.ne("")
    use_fb = (~use_head) & (~use_mid) & house_f.ne("") & street_f.ne("")

    street = np.where(use_head, street_h,
                      np.where(use_mid, street_m, street_f))
    house = np.where(use_head, house_h,
                     np.where(use_mid, house_m, house_f))
    letter = np.where(use_head, letter_h,
                      np.where(use_mid, letter_m, letter_f))
    korpus = np.where(use_head, korpus_h,
                      np.where(use_mid, korpus_m, korpus_f))

    # для невыбранных — пустые
    street = pd.Series(street, index=s.index)
    house = pd.Series(house, index=s.index)
    letter = pd.Series(letter, index=s.index)
    korpus = pd.Series(korpus, index=s.index)
    return street, house, letter, korpus

def parse_fin_series(addr: pd.Series, city: pd.Series):
    s = addr.astype(str).str.lower().str.replace("ё", "е", regex=False)
    s = s.str.replace(REGION_CITY_RE, " ", regex=True)

    # п.5 (новый): вырезаем населённые пункты, если после них НЕТ цифры
    s = s.str.replace(SETTLEMENT_RE, " ", regex=True)

    # п.10: city нормализуем так же, как s
    city_clean = (
        city.fillna("").astype(str).str.lower().str.replace("ё", "е", regex=False)
    ).to_numpy()
    s_clean = s.fillna("").astype(str).to_numpy()

    # п.9 + п.24: режем префикс до города; если хвост пустой —
    # вырезаем город как слово; REGION_NAMES_RE потом добивает регион
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

    # п.1: НЕ режем [-3:]
    street = _clean_street_series(street_raw)
    house = _house_with_korpus(house_raw, letter_raw, korpus_raw)
    return street.fillna(""), house.fillna("")

# =========================================================
# п.2 + п.5 + п.29: парсинг адреса АП
# =========================================================
def parse_ap_series(addr: pd.Series):
    s = addr.astype(str).str.lower().str.replace("ё", "е", regex=False)

    # 1) сначала пробуем "д.35 корп.Б"
    ext = s.str.extract(HOUSE_WITH_KORPUS_RE)
    house_raw = ext[0].fillna("")
    korpus_raw = ext[1].fillna("")

    # 2) где не нашли "д." — loose-вариант (дом + корпус без "д.")
    need_loose = house_raw.eq("")
    if need_loose.any():
        ext2 = s.loc[need_loose].str.extract(HOUSE_KORPUS_LOOSE_RE)
        house_raw.loc[need_loose] = ext2[0].fillna("")
        korpus_raw.loc[need_loose] = ext2[1].fillna("")

    letter_raw = pd.Series("", index=s.index)
    house = _house_with_korpus(house_raw, letter_raw, korpus_raw)
    has_dom = house.ne("")

    # теперь чистим мусор в остальной строке
    s = s.str.replace(GARBAGE_RE, " ", regex=True)

    # улица — всё до "д." / "дом"
    parts = s.str.split(r"(?:д\.?|дом)\s*[0-9]", n=1, regex=True, expand=True)
    s_street = parts[0].where(has_dom, s)

    # п.5: fallback "последнее число = дом" — ТОЛЬКО если нет п./пом./оф./эт./кв.
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
# Haversine — numpy
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
# п.3: подготовка ключей
# =========================================================
def _prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
    key = df.reset_index(drop=True).copy()
    key["_row"] = np.arange(len(key))
    key = key[["_row", "_city_norm", "_street", "_house"]].copy()
    # п.23: _house_base ДО _canon_house
    key["_house_base"] = _house_base(key["_house"])
    key["_street3"] = _street_first_n(key["_street"], 3)
    key = key[
        (key["_street"] != "") & (key["_house"] != "") & (key["_city_norm"] != "")
    ].copy()
    return key

@st.cache_data(show_spinner=False)
def match(fin: pd.DataFrame, ap: pd.DataFrame, eps_m: float = EPS_M) -> pd.DataFrame:
    fin = fin.copy()
    ap = ap.dropna(subset=["Широта", "Долгота"]).reset_index(drop=True)

    fin_key = _prepare_keys(fin)
    ap_key = _prepare_keys(ap).rename(columns={"_row": "_ap_idx"})

    n_fin = len(fin)
    addr_ok_arr = np.zeros(n_fin, dtype=bool)
    addr_ap_arr = np.full(n_fin, np.nan)

    fin_lat_all = fin["Широта"].to_numpy()
    fin_lon_all = fin["Долгота"].to_numpy()
    ap_lat_all = ap["Широта"].to_numpy()
    ap_lon_all = ap["Долгота"].to_numpy()

    # п.26: строки без координат НЕ участвуют в адресном канале
    fin_has_coords = fin["Широта"].notna().to_numpy() & fin["Долгота"].notna().to_numpy()

    def _apply_channel(cand: pd.DataFrame):
        if len(cand) == 0:
            return
        rows = cand["_row"].to_numpy()
        valid = (rows >= 0) & (rows < n_fin) & fin_has_coords[rows]
        if not valid.all():
            cand = cand.loc[valid].copy()
            rows = cand["_row"].to_numpy()
        if len(cand) == 0:
            return
        cand = cand[~addr_ok_arr[rows]]
        if len(cand) == 0:
            return
        rows = cand["_row"].to_numpy()
        ap_idx = cand["_ap_idx"].to_numpy()
        cand = cand.copy()
        cand["_dist"] = haversine_np(
            fin_lat_all[rows], fin_lon_all[rows],
            ap_lat_all[ap_idx], ap_lon_all[ap_idx],
        )
        cand_best = cand.sort_values("_dist").drop_duplicates("_row", keep="first")
        b_rows = cand_best["_row"].to_numpy()
        addr_ok_arr[b_rows] = True
        addr_ap_arr[b_rows] = cand_best["_ap_idx"].to_numpy()

    # --- канал 1: точный (city, street, house) ---
    _apply_channel(fin_key.merge(ap_key, on=["_city_norm", "_street", "_house"], how="inner"))

    # --- канал 2: (city, street, house_base) ---
    _apply_channel(fin_key.merge(ap_key, on=["_city_norm", "_street", "_house_base"], how="inner"))

    # --- канал 3: (city, street3, house_base) ---
    _apply_channel(fin_key.merge(ap_key, on=["_city_norm", "_street3", "_house_base"], how="inner"))

    # --- канал 4: (city, house_base) при nunique(street)==1 И в АП, И в фин-апе
    #     п.30: полные _house должны совпадать (35 == 35, 35б == 35б, 35 != 35б)
    ap_uniq = (
        ap_key.groupby(["_city_norm", "_house_base"])["_street"]
        .nunique().reset_index().rename(columns={"_street": "_ap_nuniq"})
    )
    fin_uniq = (
        fin_key.groupby(["_city_norm", "_house_base"])["_street"]
        .nunique().reset_index().rename(columns={"_street": "_fin_nuniq"})
    )
    uniq = ap_uniq.merge(fin_uniq, on=["_city_norm", "_house_base"], how="inner")
    uniq = uniq[(uniq["_ap_nuniq"] == 1) & (uniq["_fin_nuniq"] == 1)]
    if len(uniq) > 0:
        ap_key_uniq = ap_key.merge(
            uniq[["_city_norm", "_house_base"]], on=["_city_norm", "_house_base"], how="inner"
        )
        fin_key_uniq = fin_key.merge(
            uniq[["_city_norm", "_house_base"]], on=["_city_norm", "_house_base"], how="inner"
        )
        # п.27: явные суффиксы + явный rename
        cand4 = fin_key_uniq.merge(
            ap_key_uniq,
            on=["_city_norm", "_house_base"],
            how="inner",
            suffixes=("_f", "_a"),
        )
        # колонки _house_f / _house_a гарантированы (обе таблицы содержат _house)
        cand4 = cand4[cand4["_house_f"] == cand4["_house_a"]]
        if len(cand4) > 0:
            _apply_channel(cand4[["_row", "_ap_idx"]])

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

        dist = haversine_np(
            fin_coords[fin_local, 0], fin_coords[fin_local, 1],
            ap_coords[ap_local, 0], ap_coords[ap_local, 1],
        )
        mask = dist <= eps_m
        if not mask.any():
            continue

        fin_global = fin_idx_city[fin_local[mask]]
        ap_global = ap_idx_city[ap_local[mask]]

        tmp = pd.DataFrame({"fin": fin_global, "ap": ap_global, "d": dist[mask]})
        tmp = tmp.sort_values("d").drop_duplicates("fin", keep="first")

        geo_ok.loc[tmp["fin"].values] = True
        geo_ap_idx.loc[tmp["fin"].values] = tmp["ap"].values

    # ---------- Сборка ----------
    addr_ok = pd.Series(addr_ok_arr, index=fin.index)
    addr_ap_idx = pd.Series(addr_ap_arr, index=fin.index)

    fin["Совпадение по адресу (ДА/НЕТ)"] = np.where(addr_ok, "ДА", "НЕТ")
    fin["Совпадение по ГЕО (ДА/НЕТ)"] = np.where(geo_ok, "ДА", "НЕТ")
    fin["Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)"] = np.where(addr_ok | geo_ok, "ДА", "НЕТ")

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
        "Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)",
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

    # п.8: диагностические листы
    diag_fin = fin[["Адрес", "город", "_city_norm", "_street", "_house"]].copy()
    diag_fin["_house_base"] = _house_base(diag_fin["_house"])
    diag_fin["_street3"] = _street_first_n(diag_fin["_street"], 3)
    diag_fin = diag_fin.rename(columns={
        "_city_norm": "город_norm", "_street": "улица_norm",
        "_house": "дом_norm", "_house_base": "дом_база",
        "_street3": "улица_3слова",
    })

    diag_ap = ap[["ID_Точки", "адрес", "город", "_city_norm", "_street", "_house"]].copy()
    diag_ap["_house_base"] = _house_base(diag_ap["_house"])
    diag_ap["_street3"] = _street_first_n(diag_ap["_street"], 3)
    diag_ap = diag_ap.rename(columns={
        "_city_norm": "город_norm", "_street": "улица_norm",
        "_house": "дом_norm", "_house_base": "дом_база",
        "_street3": "улица_3слова",
    })

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        result_out.to_excel(w, index=False, sheet_name="Результат")
        diag_fin.to_excel(w, index=False, sheet_name="Разбор_фин")
        diag_ap.to_excel(w, index=False, sheet_name="Разбор_АП")

        addr_da = int((result_out["Совпадение по адресу (ДА/НЕТ)"] == "ДА").sum())
        geo_da = int((result_out["Совпадение по ГЕО (ДА/НЕТ)"] == "ДА").sum())
        any_da = int((result_out["Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)"] == "ДА").sum())
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
    st.dataframe(result_out.head(50), use_container_width=True)
    st.download_button(
        label="Скачать результат (.xlsx)",
        data=bio,
        file_name="result_сопоставление.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
