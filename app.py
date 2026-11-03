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
    f"Приоритет 1 — совпадение адреса (3 канала: точный → fuzzy → уникальный дом). "
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

REGION_ADJ_STEMS = {
    "московск", "ленинградск", "свердловск", "челябинск",
    "нижегородск", "воронежск", "ростовск", "волгоградск",
    "самарск", "иркутск", "омск", "новосибирск", "красноярск",
    "пермск", "краснодарск", "приморск", "хабаровск",
    "башкортостан", "татарстан",
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
# Стоп-слова (единый список — выбрасываем всё)
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
STOP_TOKENS = STREET_TYPE_TOKENS | BUILDING_MARKERS | REGION_MARKERS

# =========================================================
# Токенизация и стемы
# =========================================================
def _tokenize(addr: str) -> list:
    s = str(addr).lower().replace("ё", "е")
    return [t for t in re.split(r"[^0-9a-zа-я]+", s) if t]

def _city_stems(city_norm: str) -> set:
    stems = set()
    if not city_norm:
        return stems
    stems.add(city_norm)
    if len(city_norm) >= 4:
        stems.add(city_norm[:4])
    if len(city_norm) >= 5:
        stems.add(city_norm[:5])
    parts = re.split(r"[- ]+", city_norm)
    if len(parts) > 1:
        for part in parts:
            if len(part) >= 2:
                stems.add(part)
                if len(part) >= 4:
                    stems.add(part[:4])
    return stems

def _strip_tokens(tokens, city_norm):
    stems = _city_stems(city_norm)
    out = []
    for t in tokens:
        if t in STOP_TOKENS:
            continue
        if any(t.startswith(s) for s in REGION_ADJ_STEMS):
            continue
        if any(t == s or t.startswith(s) for s in stems):
            continue
        out.append(t)
    return out

# =========================================================
# Нормализация адреса — «мешок токенов»
# =========================================================
def normalize_address(addr: str, city_norm: str) -> dict:
    tokens = _tokenize(addr)
    tokens = _strip_tokens(tokens, city_norm)
    token_raw = " ".join(tokens)
    sorted_tokens = sorted(tokens)
    token_set = "|".join(sorted_tokens)
    token_sort = " ".join(sorted_tokens)
    first_token = sorted_tokens[0] if sorted_tokens else ""
    # _house_base — последний числовой токен (для канала 3)
    house_base = ""
    for t in reversed(tokens):
        if re.match(r"^\d", t):
            house_base = t
            break
    return {
        "_token_set": token_set,
        "_token_sort": token_sort,
        "_token_raw": token_raw,
        "_first_token": first_token,
        "_house_base": house_base,
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
    key = key[["_row", "_city_norm", "_token_set", "_token_sort",
               "_token_raw", "_first_token", "_house_base"]].copy()
    key = key[(key["_city_norm"] != "") & (key["_token_set"] != "")].copy()
    return key

# =========================================================
# Основной подбор
#   П1 = адрес (3 канала: точный → fuzzy → уникальный дом)
#   П2 = координаты ≤ EPS_M
#   П3 = топ ближайших
#   Лимит X: П1 → П2 → П3
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

    # --- Канал 1: (город, token_set) ---
    _apply_channel(
        fin_key.merge(ap_key, on=["_city_norm", "_token_set"], how="inner"),
        1,
    )

    # --- Канал 2: fuzzy по token_sort + (город, first_token) ---
    fin_fuzzy = fin_key[["_row", "_city_norm", "_first_token",
                         "_token_sort"]].rename(
        columns={"_row": "_row_f", "_token_sort": "_token_sort_f"}
    )
    ap_fuzzy = ap_key[["_ap_idx", "_city_norm", "_first_token",
                       "_token_sort"]].rename(
        columns={"_token_sort": "_token_sort_a"}
    )
    cand2 = ap_fuzzy.merge(
        fin_fuzzy, on=["_city_norm", "_first_token"], how="inner",
    )
    if len(cand2) > 0:
        cand2 = cand2[~addr_ok[cand2["_ap_idx"].to_numpy()]]
    if len(cand2) > 0:
        ratios = np.fromiter(
            (SequenceMatcher(None, a, f).ratio()
             for a, f in zip(cand2["_token_sort_a"], cand2["_token_sort_f"])),
            dtype=float, count=len(cand2),
        )
        cand2 = cand2.assign(_ratio=ratios)
        cand2 = cand2[cand2["_ratio"] >= FUZZY_THRESHOLD]
        if len(cand2) > 0:
            _apply_channel(
                cand2[["_row_f", "_ap_idx", "_ratio"]].rename(columns={"_row_f": "_row"}),
                2,
                rank_by="_ratio",
            )

    # --- Канал 3: (город, house_base) при уникальной token_set ---
    ap_uniq = (ap_key.groupby(["_city_norm", "_house_base"])["_token_set"]
               .nunique().reset_index().rename(columns={"_token_set": "_ap_nuniq"}))
    fin_uniq = (fin_key.groupby(["_city_norm", "_house_base"])["_token_set"]
                .nunique().reset_index().rename(columns={"_token_set": "_fin_nuniq"}))
    uniq = ap_uniq.merge(fin_uniq, on=["_city_norm", "_house_base"], how="inner")
    uniq = uniq[(uniq["_ap_nuniq"] == 1) & (uniq["_fin_nuniq"] == 1)]
    if len(uniq) > 0:
        ap_key_uniq = ap_key.merge(uniq[["_city_norm", "_house_base"]],
                                   on=["_city_norm", "_house_base"], how="inner")
        fin_key_uniq = fin_key.merge(uniq[["_city_norm", "_house_base"]],
                                     on=["_city_norm", "_house_base"], how="inner")
        cand3 = fin_key_uniq.merge(ap_key_uniq, on=["_city_norm", "_house_base"],
                                   how="inner")
        if len(cand3) > 0:
            _apply_channel(cand3[["_row", "_ap_idx"]], 3)

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

    out["Токены_АП"] = ap_work["_token_raw"].to_numpy()

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
        fin_diag = fin.loc[chosen_vals, ["_token_raw"]].copy()
        fin_diag.index = out.index
        out["Токены_фин"] = fin_diag["_token_raw"]
    else:
        out["Токены_фин"] = np.nan

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

    diag_fin = fin[["Адрес", "город", "_city_norm", "_token_raw",
                    "_token_set", "_house_base"]].copy()
    diag_fin = diag_fin.rename(columns={
        "_city_norm": "город_norm",
        "_token_raw": "токены",
        "_token_set": "множество",
        "_house_base": "дом_база",
    })

    diag_ap = ap[["ID_Точки", "адрес", "город", "_city_norm", "_token_raw",
                  "_token_set", "_house_base"]].copy()
    diag_ap = diag_ap.rename(columns={
        "_city_norm": "город_norm",
        "_token_raw": "токены",
        "_token_set": "множество",
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

        metrics = pd.DataFrame({
            "Показатель": [
                "Цель X",
                "Подобрано всего",
                "Приоритет 1 (адрес), всего",
                "  канал 1: точное множество токенов",
                f"  канал 2: fuzzy токенов ≥ {FUZZY_THRESHOLD}",
                "  канал 3: уникальный дом в городе",
                f"Приоритет 2 (координаты ≤ {EPS_M:.0f} м)",
                "Приоритет 3 (ближайшие, добивка)",
                "Осталось до X (не хватило АП)",
            ],
            "Значение": [
                int(n_target),
                len(final),
                p1, ch1, ch2, ch3,
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
