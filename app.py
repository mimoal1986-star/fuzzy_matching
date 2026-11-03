# app.py
import re
import io
from difflib import SequenceMatcher

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
# Пороги
# =========================================================
EPS_M = 15.0
FUZZY_THRESHOLD = 0.88

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

st.caption(
    f"Приоритет 1 — совпадение адреса (5 каналов: точный → fuzzy). "
    f"Приоритет 2 — координаты ≤ {EPS_M:.0f} м (жёстко). "
    f"Приоритет 3 — топ ближайших пар, добиваем до {n_target}. "
    f"Достигли {n_target} — стоп."
)

# =========================================================
# Нормализация города
# =========================================================
CITY_SYNONYMS = {
    "питер": "санкт-петербург",
    "спб": "санкт-петербург",
    "санкт петербург": "санкт-петербург",
    "санктпетербург": "санкт-петербург",
    "екатеренбург": "екатеринбург",
    "екб": "екатеринбург",
    "екат": "екатеринбург",
    "члб": "челябинск",
    "челяба": "челябинск",
    "мск": "москва",
    "н новгород": "нижний новгород",
    "н.новгород": "нижний новгород",
    "нижнийновгород": "нижний новгород",
    "ростов": "ростов-на-дону",
    "ростов на дону": "ростов-на-дону",
    "ростовнадону": "ростов-на-дону",
}

# стемы прилагательных регионов — ловят все падежи
REGION_ADJ_STEMS = {
    "московск", "ленинградск", "свердловск", "челябинск",
    "нижегородск", "воронежск", "ростовск", "волгоградск",
    "самарск", "иркутск", "омск", "новосибирск", "красноярск",
    "пермск", "краснодарск", "приморск", "хабаровск",
    "башкортостан", "татарстан", "пермск", "краснодарск",
    "приморск", "хабаровск", "красноярск",
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
# Стоп-слова
# =========================================================
STREET_TYPE_TOKENS = {
    "ул", "улица", "пр", "прт", "проспект", "просп", "пер", "переулок",
    "б", "бур", "бульвар", "наб", "набережная", "пл", "площадь",
    "ш", "шоссе", "проезд", "прд", "туп", "тупик", "аллея",
    "мкр", "микрорайон", "кв", "квартал", "квл", "линия", "тракт",
    "дор", "дорога", "спуск", "съезд", "просек", "просека",
    "проулок", "протока", "тер", "территория",
}
BUILDING_MARKERS = {
    "д", "дом", "корп", "корпус", "к", "стр", "строение", "с",
    "лит", "литера", "пом", "помещ", "помещение", "оф", "офис",
    "эт", "этаж", "кв", "квартира", "нп", "вл", "владение",
    "соор", "сооружение",
}
REGION_MARKERS = {
    "г", "город", "обл", "область", "край", "респ", "республика",
    "р", "н", "район", "рн", "п", "пос", "поселок", "с", "село",
    "д", "деревня", "рп", "х", "хутор", "ст", "ца", "станица",
}
# стоп-набор ТОЛЬКО для улицы (после _split_house)
STREET_STOP_TOKENS = STREET_TYPE_TOKENS | REGION_MARKERS

# маркеры корпуса (для типа, обрабатываются в _split_house)
KORPUS_MARKERS = {"корп", "корпус", "к", "стр", "строение", "с", "лит", "литера"}
HOUSE_MARKERS = {"д", "дом"}

# =========================================================
# Токенизация
# =========================================================
def _tokenize(addr: str) -> list:
    s = str(addr).lower().replace("ё", "е")
    return [t for t in re.split(r"[^0-9a-zа-я]+", s) if t]

def _city_stems(city_norm: str) -> set:
    """Стемы для вырезания города/региона из адреса."""
    stems = set()
    if not city_norm:
        return stems
    stems.add(city_norm)
    if len(city_norm) >= 4:
        stems.add(city_norm[:4])
    if len(city_norm) >= 5:
        stems.add(city_norm[:5])
    # составные города (санкт-петербург, ростов-на-дону) — все части, без порога
    parts = re.split(r"[- ]+", city_norm)
    if len(parts) > 1:
        for part in parts:
            if len(part) >= 2:
                stems.add(part)
                if len(part) >= 4:
                    stems.add(part[:4])
    return stems

def _strip_tokens(tokens, city_norm):
    """Выбрасывает стоп-слова, город и его прилагательные формы — для УЛИЦЫ."""
    stems = _city_stems(city_norm)
    out = []
    for t in tokens:
        if t in STREET_STOP_TOKENS:
            continue
        if any(t.startswith(s) for s in REGION_ADJ_STEMS):
            continue
        if any(t == s or t.startswith(s) for s in stems):
            continue
        out.append(t)
    return out

# =========================================================
# Разбор на улицу / дом / корпус — ДО выброса стоп-слов
# =========================================================
def _split_house(tokens):
    """
    Возвращает (street_tokens_raw, house, korpus, korpus_type).
    Маркеры (д/дом/корп/стр/с/к/лит) обрабатываются здесь и удаляются.
    """
    street, house, korpus, korpus_type = [], "", "", ""
    seen_letter = False
    pending = None
    pending_type = ""

    for t in tokens:
        if t in HOUSE_MARKERS:
            pending = "house"; pending_type = ""
            continue
        if t in KORPUS_MARKERS:
            pending = "korpus"; pending_type = t
            continue
        if t.isdigit():
            if pending == "house":
                if not house:
                    house = t
                pending = None; pending_type = ""
            elif pending == "korpus":
                if not korpus:
                    korpus = t
                    korpus_type = pending_type
                pending = None; pending_type = ""
            else:
                if not seen_letter and not house and not korpus:
                    street.append(t)          # 250-летия
                elif not house:
                    house = t
                elif not korpus:
                    korpus = t
            continue
        # буква
        seen_letter = True
        pending = None; pending_type = ""
        street.append(t)

    return street, house, korpus, korpus_type

# =========================================================
# Полная нормализация адреса
# =========================================================
def normalize_address(addr: str, city_norm: str) -> dict:
    tokens = _tokenize(addr)
    street_raw_tokens, house, korpus, _ktype = _split_house(tokens)
    street_tokens = _strip_tokens(street_raw_tokens, city_norm)
    street_sort = " ".join(sorted(street_tokens))
    first_token = street_tokens[0] if street_tokens else ""
    return {
        "_street_set": "|".join(sorted(street_tokens)),
        "_street_sort": street_sort,
        "_street_raw": " ".join(street_tokens),
        "_street_first": first_token,
        "_house": house,
        "_korpus": korpus,
        "_house_base": house,
    }

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

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(r"\s+", "", regex=True),
        errors="coerce",
    )

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

def _apply_norm(df: pd.DataFrame, addr_col: str) -> pd.DataFrame:
    rows = [
        normalize_address(a, c)
        for a, c in zip(df[addr_col].astype(str).tolist(),
                        df["_city_norm"].astype(str).tolist())
    ]
    norm = pd.DataFrame(rows, index=df.index)
    for col in norm.columns:
        df[col] = norm[col]
    return df

@st.cache_data(show_spinner=False)
def load_fin(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    _check_cols(df, FIN_REQUIRED, "фин ап")
    df = df.reset_index(drop=True)
    df["Широта"] = to_float_series(df["Широта"])
    df["Долгота"] = to_float_series(df["Долгота"])
    df["_city_norm"] = norm_city_series(df["город"])
    df = _apply_norm(df, "Адрес")
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
    df = _apply_norm(df, "адрес")
    return df

# =========================================================
# Подготовка ключей
# =========================================================
def _prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
    key = df.reset_index(drop=True).copy()
    key["_row"] = np.arange(len(key))
    key = key[["_row", "_city_norm", "_street_set", "_street_sort",
               "_street_raw", "_street_first", "_house", "_korpus",
               "_house_base"]].copy()
    key = key[
        (key["_city_norm"] != "")
        & (key["_house"] != "")
        & (key["_street_set"] != "")   # пустая улица — не участвует
    ].copy()
    return key

# =========================================================
# Основной подбор
# =========================================================
@st.cache_data(show_spinner=False)
def match(fin: pd.DataFrame, ap: pd.DataFrame, n_target: int):
    fin = fin.reset_index(drop=True).copy()
    ap = ap.reset_index(drop=True).copy()

    ap_valid = ap["Широта"].notna() & ap["Долгота"].notna()
    ap_work = ap[ap_valid].reset_index(drop=True).copy()
    n_ap = len(ap_work)
    if n_ap == 0:
        return pd.DataFrame(), np.zeros(0, dtype=int)

    fin_has_coords = (fin["Широта"].notna() & fin["Долгота"].notna()).to_numpy()
    fin_lat_all = fin["Широта"].to_numpy()
    fin_lon_all = fin["Долгота"].to_numpy()
    ap_lat_all = ap_work["Широта"].to_numpy()
    ap_lon_all = ap_work["Долгота"].to_numpy()

    fin_key = _prepare_keys(fin)
    ap_key = _prepare_keys(ap_work).rename(columns={"_row": "_ap_idx"})

    addr_ok = np.zeros(n_ap, dtype=bool)
    addr_fin = np.full(n_ap, -1, dtype=int)
    addr_dist = np.full(n_ap, np.nan)
    addr_channel = np.zeros(n_ap, dtype=int)

    geo_ok = np.zeros(n_ap, dtype=bool)
    geo_fin = np.full(n_ap, -1, dtype=int)
    geo_dist = np.full(n_ap, np.nan)

    def _apply_channel(cand: pd.DataFrame, channel_no: int, rank_by: str = "_dist"):
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

        if rank_by == "_dist":
            best = (
                cand.sort_values("_dist", na_position="last")
                    .drop_duplicates("_ap_idx", keep="first")
            )
        else:
            # для fuzzy: лучший по ratio, при равенстве — по расстоянию
            best = (
                cand.sort_values([rank_by, "_dist"],
                                 ascending=[False, True], na_position="last")
                    .drop_duplicates("_ap_idx", keep="first")
            )

        b_a = best["_ap_idx"].to_numpy()
        addr_ok[b_a] = True
        addr_fin[b_a] = best["_row"].to_numpy()
        addr_dist[b_a] = best["_dist"].to_numpy()
        addr_channel[b_a] = channel_no

    # --- Канал 1: (город, street_set, house, korpus) ---
    _apply_channel(
        fin_key.merge(ap_key, on=["_city_norm", "_street_set", "_house", "_korpus"],
                      how="inner"),
        1,
    )

    # --- Канал 2: (город, street_set, house) — корпус игнорируется ---
    _apply_channel(
        fin_key.merge(ap_key, on=["_city_norm", "_street_set", "_house"], how="inner"),
        2,
    )

    # --- Канал 3: fuzzy по street_sort + (город, house, street_first) ---
    fin_fuzzy = fin_key[["_row", "_city_norm", "_house",
                         "_street_first", "_street_sort"]].rename(
        columns={"_row": "_row_f"}
    )
    ap_fuzzy = ap_key[["_ap_idx", "_city_norm", "_house",
                       "_street_first", "_street_sort"]].rename(
        columns={"_street_sort": "_street_sort_a"}
    )
    cand3 = ap_fuzzy.merge(
        fin_fuzzy,
        on=["_city_norm", "_house", "_street_first"],
        how="inner",
        suffixes=("_a", "_f"),
    )
    if len(cand3) > 0:
        cand3 = cand3[~addr_ok[cand3["_ap_idx"].to_numpy()]]
    if len(cand3) > 0:
        ratios = np.fromiter(
            (SequenceMatcher(None, a, f).ratio()
             for a, f in zip(cand3["_street_sort_a"], cand3["_street_sort_f"])),
            dtype=float, count=len(cand3),
        )
        cand3 = cand3.assign(_ratio=ratios)
        cand3 = cand3[cand3["_ratio"] >= FUZZY_THRESHOLD]
        if len(cand3) > 0:
            _apply_channel(
                cand3.rename(columns={"_row_f": "_row"})[["_row", "_ap_idx", "_ratio"]],
                3,
                rank_by="_ratio",
            )

    # --- Канал 4: (город, house_base) при уникальной улице + полный дом ---
    ap_uniq = (ap_key.groupby(["_city_norm", "_house_base"])["_street_set"]
               .nunique().reset_index().rename(columns={"_street_set": "_ap_nuniq"}))
    fin_uniq = (fin_key.groupby(["_city_norm", "_house_base"])["_street_set"]
                .nunique().reset_index().rename(columns={"_street_set": "_fin_nuniq"}))
    uniq = ap_uniq.merge(fin_uniq, on=["_city_norm", "_house_base"], how="inner")
    uniq = uniq[(uniq["_ap_nuniq"] == 1) & (uniq["_fin_nuniq"] == 1)]
    if len(uniq) > 0:
        ap_key_uniq = ap_key.merge(uniq[["_city_norm", "_house_base"]],
                                   on=["_city_norm", "_house_base"], how="inner")
        fin_key_uniq = fin_key.merge(uniq[["_city_norm", "_house_base"]],
                                     on=["_city_norm", "_house_base"], how="inner")
        cand4 = fin_key_uniq.merge(ap_key_uniq, on=["_city_norm", "_house_base"],
                                   how="inner", suffixes=("_f", "_a"))
        # сверка полного дома (на случай, если _house_base != _house)
        if len(cand4) > 0:
            cand4 = cand4[cand4["_house_f"] == cand4["_house_a"]]
        if len(cand4) > 0:
            _apply_channel(cand4[["_row", "_ap_idx"]], 4)

    # ---------- П2: координаты ≤ EPS_M ----------
    ap_need_geo = ap_work[~addr_ok]
    if len(ap_need_geo) > 0:
        for city, ap_idx_city in ap_need_geo.groupby("_city_norm").groups.items():
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

    # ---------- П3: ближайший без порога ----------
    prio3_ok = np.zeros(n_ap, dtype=bool)
    prio3_fin = np.full(n_ap, -1, dtype=int)
    prio3_dist = np.full(n_ap, np.nan)

    need_p3 = ~(addr_ok | geo_ok)
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
            _d_deg, idx = tree.query(ap_coords, k=1)
            d_m = haversine_np(
                ap_coords[:, 0], ap_coords[:, 1],
                fin_coords[idx, 0], fin_coords[idx, 1],
            )
            a = ap_idx_city
            prio3_ok[a] = True
            prio3_fin[a] = fin_idx_city[idx]
            prio3_dist[a] = d_m

    # ---------- Лимит X ----------
    taken = np.zeros(n_ap, dtype=bool)

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

    # ---------- Сборка ----------
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

    out["Канал адреса"] = np.where(addr_ok, addr_channel, 0).astype(int)

    out["Улица_norm_АП"] = ap_work["_street_raw"].to_numpy()
    out["Дом_norm_АП"] = ap_work["_house"].to_numpy()

    chosen_fin = np.where(addr_ok, addr_fin,
                  np.where(geo_ok, geo_fin, prio3_fin))
    chosen_dist = np.where(addr_ok, addr_dist,
                   np.where(geo_ok, geo_dist, prio3_dist))
    out["Расстояние по гео,м"] = chosen_dist

    out["_chosen_fin"] = chosen_fin
    out["_taken"] = taken
    out["_prio_num"] = np.where(addr_ok, 1, np.where(geo_ok, 2,
                       np.where(prio3_ok, 3, 4)))

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
        fin_diag = fin.loc[chosen_vals, ["_street_raw", "_house"]].copy()
        fin_diag.index = out.index
        out["Улица_norm_фин"] = fin_diag["_street_raw"]
        out["Дом_norm_фин"] = fin_diag["_house"]
    else:
        out["Улица_norm_фин"] = np.nan
        out["Дом_norm_фин"] = np.nan

    for c in fin_out_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out.sort_values(["_prio_num", "Расстояние по гео,м"], na_position="last")
    return out, addr_channel

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
        result, _channel = match(fin, ap, int(n_target))

    if result.empty:
        st.warning("Не удалось подобрать ни одной АП (нет координат / городов / магазинов).")
        st.stop()

    final = pd.DataFrame({
        "ID_Точки": result["ID_Точки"],
        "название": result["название"],
        "адрес": result["адрес"],
        "Широта_АП": result["Широта_АП"],
        "Долгота_АП": result["Долгота_АП"],
        "город_АП": result["город_АП"],
        "Совпадение по адресу (ДА/НЕТ)": result["Совпадение по адресу (ДА/НЕТ)"],
        "Совпадение по ГЕО (ДА/НЕТ)": result["Совпадение по ГЕО (ДА/НЕТ)"],
        "Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)": result["Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)"],
        "Приоритет 1 (ДА/НЕТ)": result["Приоритет 1 (ДА/НЕТ)"],
        "Приоритет 2 (ДА/НЕТ)": result["Приоритет 2 (ДА/НЕТ)"],
        "Приоритет 3 (ДА/НЕТ)": result["Приоритет 3 (ДА/НЕТ)"],
        "Расстояние по гео,м": result["Расстояние по гео,м"],
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

    diag_fin = fin[["Адрес", "город", "_city_norm", "_street_raw",
                    "_street_set", "_house", "_korpus", "_house_base"]].copy()
    diag_fin = diag_fin.rename(columns={
        "_city_norm": "город_norm",
        "_street_raw": "улица",
        "_street_set": "улица_множество",
        "_house": "дом",
        "_korpus": "корпус",
        "_house_base": "дом_база",
    })

    diag_ap = ap[["ID_Точки", "адрес", "город", "_city_norm", "_street_raw",
                  "_street_set", "_house", "_korpus", "_house_base"]].copy()
    diag_ap = diag_ap.rename(columns={
        "_city_norm": "город_norm",
        "_street_raw": "улица",
        "_street_set": "улица_множество",
        "_house": "дом",
        "_korpus": "корпус",
        "_house_base": "дом_база",
    })

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        final.to_excel(w, index=False, sheet_name="Результат")
        diag_fin.to_excel(w, index=False, sheet_name="Разбор_фин")
        diag_ap.to_excel(w, index=False, sheet_name="Разбор_АП")

        p1 = int((final["Приоритет 1 (ДА/НЕТ)"] == "ДА").sum())
        p2 = int((final["Приоритет 2 (ДА/НЕТ)"] == "ДА").sum())
        p3 = int((final["Приоритет 3 (ДА/НЕТ)"] == "ДА").sum())

        ch_counts = result.loc[result["Приоритет 1 (ДА/НЕТ)"] == "ДА", "Канал адреса"]
        ch1 = int((ch_counts == 1).sum())
        ch2 = int((ch_counts == 2).sum())
        ch3 = int((ch_counts == 3).sum())
        ch4 = int((ch_counts == 4).sum())

        metrics = pd.DataFrame({
            "Показатель": [
                "Цель X",
                "Подобрано всего",
                "Приоритет 1 (адрес), всего",
                "  канал 1: (город, улица, дом, корпус)",
                "  канал 2: (город, улица, дом) — корпус игнор.",
                f"  канал 3: fuzzy улицы ≥ {FUZZY_THRESHOLD} + (город, дом)",
                "  канал 4: (город, дом_база, уник. улица)",
                f"Приоритет 2 (координаты ≤ {EPS_M:.0f} м)",
                "Приоритет 3 (ближайшие, добивка)",
                "Осталось до X (не хватило АП)",
            ],
            "Значение": [
                int(n_target),
                len(final),
                p1, ch1, ch2, ch3, ch4,
                p2, p3,
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
