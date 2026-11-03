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
K_NEAREST = 10

# =========================================================
# Дефолтные квоты по городам
# =========================================================
DEFAULT_QUOTAS = {
    "волгоград": 92,
    "воронеж": 126,
    "екатеринбург": 173,
    "иркутск": 50,
    "казань": 125,
    "краснодар": 76,
    "красноярск": 50,
    "москва": 578,
    "нижний новгород": 134,
    "новосибирск": 121,
    "омск": 83,
    "пермь": 86,
    "ростов-на-дону": 131,
    "самара": 167,
    "санкт-петербург": 244,
    "уфа": 87,
    "челябинск": 74,
}

# =========================================================
# Сайдбар — квоты по городам
# =========================================================
with st.sidebar:
    st.header("Квоты по городам")
    quota_df = pd.DataFrame({
        "Город": list(DEFAULT_QUOTAS.keys()),
        "Квота": list(DEFAULT_QUOTAS.values()),
    })
    edited = st.data_editor(
        quota_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Город": st.column_config.TextColumn(disabled=True),
            "Квота": st.column_config.NumberColumn(min_value=0, step=1),
        },
        key="quota_editor",
    )
    city_quotas = {
        str(row["Город"]).strip().lower(): int(row["Квота"])
        for _, row in edited.iterrows()
        if int(row["Квота"]) > 0
    }
    total_quota = sum(city_quotas.values())
    st.markdown(f"**Итого подборов (сумма квот): {total_quota}**")

st.caption(
    f"Приоритет 1 — совпадение адреса (3 канала). "
    f"Приоритет 2 — координаты ≤ {EPS_M:.0f} м. "
    f"Приоритет 3 — ближайший свободный магазин. "
    f"Каждый магазин фин-ап используется не более 1 раза. "
    f"Квоты по городам соблюдаются на каждом шаге."
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
STOP_TOKENS = STREET_TYPE_TOKENS | BUILDING_MARKERS | REGION_MARKERS

# =========================================================
# Токенизация и нормализация адреса
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

def normalize_address(addr: str, city_norm: str) -> dict:
    tokens = _tokenize(addr)
    tokens = _strip_tokens(tokens, city_norm)
    token_raw = " ".join(tokens)
    sorted_tokens = sorted(tokens)
    token_set = "|".join(sorted_tokens)
    token_sort = " ".join(sorted_tokens)
    first_token = sorted_tokens[0] if sorted_tokens else ""
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
# =========================================================
@st.cache_data(show_spinner=False)
def match(fin: pd.DataFrame, ap: pd.DataFrame, city_quotas_tuple: tuple):
    city_quotas = dict(city_quotas_tuple)

    fin = fin.reset_index(drop=True).copy()
    ap = ap.reset_index(drop=True).copy()

    ap_valid = ap["Широта"].notna() & ap["Долгота"].notna()
    ap_work = ap[ap_valid].reset_index(drop=True).copy()
    n_ap = len(ap_work)
    n_fin = len(fin)
    if n_ap == 0 or n_fin == 0:
        return pd.DataFrame(), np.zeros(0, dtype=bool), {}

    fin_has_coords = (fin["Широта"].notna() & fin["Долгота"].notna()).to_numpy()
    fin_lat_all = fin["Широта"].to_numpy()
    fin_lon_all = fin["Долгота"].to_numpy()
    ap_lat_all = ap_work["Широта"].to_numpy()
    ap_lon_all = ap_work["Долгота"].to_numpy()

    fin_key = _prepare_keys(fin)
    ap_key = _prepare_keys(ap_work).rename(columns={"_row": "_ap_idx"})

    fin_used = np.zeros(n_fin, dtype=bool)
    ap_taken = np.zeros(n_ap, dtype=bool)
    city_taken = {c: 0 for c in city_quotas}
    ap_city = ap_work["_city_norm"].to_numpy()

    result_pairs = []

    def _city_has_room(city: str) -> bool:
        return city_taken.get(city, 0) < city_quotas.get(city, 0)

    def _try_take_pair(a: int, f: int, dist: float, priority: int, channel: int):
        if ap_taken[a] or fin_used[f]:
            return False
        city = ap_city[a]
        if city not in city_quotas:
            return False
        if not _city_has_room(city):
            return False
        ap_taken[a] = True
        fin_used[f] = True
        city_taken[city] += 1
        result_pairs.append({
            "ap_idx": int(a),
            "fin_idx": int(f),
            "priority": priority,
            "channel": channel,
            "dist": float(dist) if (dist is not None and not np.isnan(dist)) else np.nan,
        })
        return True

    def _apply_channel_greedy(cand: pd.DataFrame, channel_no: int, priority_no: int,
                              rank_by: str = "_dist"):
        """
        Жадный проход по cand (содержит _row, _ap_idx).
        Если нет _dist — вычисляет haversine по координатам.
        """
        if len(cand) == 0:
            return
        cand = cand[
            ~fin_used[cand["_row"].to_numpy()]
            & ~ap_taken[cand["_ap_idx"].to_numpy()]
        ]
        if len(cand) == 0:
            return

        # FIX п.1: досчитываем _dist, если его нет
        if "_dist" not in cand.columns:
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
            cand = cand.sort_values("_dist", na_position="last")
        else:
            cand = cand.sort_values([rank_by, "_dist"],
                                    ascending=[False, True], na_position="last")

        for _, row in cand.iterrows():
            _try_take_pair(
                int(row["_ap_idx"]), int(row["_row"]),
                float(row["_dist"]) if not np.isnan(row["_dist"]) else np.nan,
                priority_no, channel_no,
            )

    # ---------- П1: адресные каналы ----------
    # канал 1: точное множество токенов
    _apply_channel_greedy(
        fin_key.merge(ap_key, on=["_city_norm", "_token_set"], how="inner"),
        1, 1,
    )

    # канал 2: fuzzy по token_sort + (город, first_token)
    fin_fuzzy = fin_key[["_row", "_city_norm", "_first_token",
                         "_token_sort"]].rename(
        columns={"_row": "_row_f", "_token_sort": "_token_sort_f"}
    )
    ap_fuzzy = ap_key[["_ap_idx", "_city_norm", "_first_token",
                       "_token_sort"]].rename(
        columns={"_token_sort": "_token_sort_a"}
    )
    cand2 = ap_fuzzy.merge(fin_fuzzy, on=["_city_norm", "_first_token"], how="inner")
    if len(cand2) > 0:
        cand2 = cand2[
            ~fin_used[cand2["_row_f"].to_numpy()]
            & ~ap_taken[cand2["_ap_idx"].to_numpy()]
        ]
    if len(cand2) > 0:
        ratios = np.fromiter(
            (SequenceMatcher(None, a, f).ratio()
             for a, f in zip(cand2["_token_sort_a"], cand2["_token_sort_f"])),
            dtype=float, count=len(cand2),
        )
        cand2 = cand2.assign(_ratio=ratios)
        cand2 = cand2[cand2["_ratio"] >= FUZZY_THRESHOLD]
        if len(cand2) > 0:
            cand2 = cand2[["_row_f", "_ap_idx", "_ratio"]].rename(
                columns={"_row_f": "_row"}
            )
            # _dist досчитается внутри _apply_channel_greedy
            _apply_channel_greedy(cand2, 2, 1, rank_by="_ratio")

    # канал 3: уникальный дом в городе
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
        # _dist досчитается внутри
        _apply_channel_greedy(cand3[["_row", "_ap_idx"]], 3, 1)

    # ---------- П2: координаты ≤ EPS_M (жадно по расстоянию) ----------
    ap_need_geo = ap_work[~ap_taken]
    geo_candidates = []
    if len(ap_need_geo) > 0:
        for city, ap_idx_city in ap_need_geo.groupby("_city_norm").groups.items():
            if city not in city_quotas or not _city_has_room(city):
                continue
            ap_idx_city = np.asarray(ap_idx_city)
            fin_idx_city = np.flatnonzero(
                fin_has_coords
                & (fin["_city_norm"].to_numpy() == city)
                & ~fin_used
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
            for ai, fi, di in zip(
                ap_idx_city[a_local[mask]],
                fin_idx_city[f_local[mask]],
                dist[mask],
            ):
                geo_candidates.append((int(ai), int(fi), float(di)))

    if geo_candidates:
        geo_df = pd.DataFrame(geo_candidates, columns=["_ap_idx", "_row", "_dist"])
        geo_df = geo_df.sort_values("_dist")
        for _, row in geo_df.iterrows():
            _try_take_pair(int(row["_ap_idx"]), int(row["_row"]),
                           float(row["_dist"]), 2, 0)

    # ---------- П3: k ближайших свободных (жадно) ----------
    ap_need_p3 = ap_work[~ap_taken]
    p3_candidates = []
    if len(ap_need_p3) > 0:
        for city, ap_idx_city in ap_need_p3.groupby("_city_norm").groups.items():
            if city not in city_quotas or not _city_has_room(city):
                continue
            ap_idx_city = np.asarray(ap_idx_city)
            fin_idx_city = np.flatnonzero(
                fin_has_coords
                & (fin["_city_norm"].to_numpy() == city)
                & ~fin_used
            )
            if len(fin_idx_city) == 0:
                continue
            fin_coords = np.column_stack([fin_lat_all[fin_idx_city],
                                           fin_lon_all[fin_idx_city]])
            tree = cKDTree(fin_coords)
            ap_coords = np.column_stack([ap_lat_all[ap_idx_city],
                                          ap_lon_all[ap_idx_city]])
            k = min(K_NEAREST, len(fin_coords))
            d_mat, idx_mat = tree.query(ap_coords, k=k)
            if k == 1:
                d_mat = d_mat[:, None]
                idx_mat = idx_mat[:, None]
            # векторный пересчёт в метры
            for i in range(len(ap_idx_city)):
                valid = np.isfinite(d_mat[i])
                if not valid.any():
                    continue
                j_valid = np.flatnonzero(valid)
                f_glob = fin_idx_city[idx_mat[i, j_valid]]
                d_m = haversine_np(
                    ap_coords[i, 0], ap_coords[i, 1],
                    fin_coords[idx_mat[i, j_valid], 0],
                    fin_coords[idx_mat[i, j_valid], 1],
                )
                for f_g, d_g in zip(f_glob, d_m):
                    p3_candidates.append((int(ap_idx_city[i]), int(f_g), float(d_g)))

    if p3_candidates:
        p3_df = pd.DataFrame(p3_candidates, columns=["_ap_idx", "_row", "_dist"])
        p3_df = p3_df.sort_values("_dist")
        for _, row in p3_df.iterrows():
            _try_take_pair(int(row["_ap_idx"]), int(row["_row"]),
                           float(row["_dist"]), 3, 0)

    # ---------- Сборка ----------
    if not result_pairs:
        return pd.DataFrame(), fin_used, city_taken

    pairs_df = pd.DataFrame(result_pairs)

    ap_cols_out = ["ID_Точки", "название", "адрес",
                   "Широта", "Долгота", "Полигон", "город", "тип", "сеть", "база"]
    ap_cols_present = [c for c in ap_cols_out if c in ap_work.columns]
    ap_part = ap_work.loc[pairs_df["ap_idx"].to_numpy(), ap_cols_present].reset_index(drop=True)
    ap_part = ap_part.rename(columns={
        "Широта": "Широта_АП",
        "Долгота": "Долгота_АП",
        "город": "город_АП",
    })

    fin_cols_out = ["ТЕРРИТОРИЯ", "КЛАСТЕР", "ИМЯ МАГАЗИНА", "SAP НОМЕР МАГАЗИНА",
                    "АК наименование", "УА ПЛАНОГРАММЫ", "СТАТУС МАГАЗИНА",
                    "Адрес", "город", "Широта", "Долгота"]
    fin_cols_present = [c for c in fin_cols_out if c in fin.columns]
    fin_part = fin.loc[pairs_df["fin_idx"].to_numpy(), fin_cols_present].reset_index(drop=True)

    out = pd.concat([ap_part, fin_part], axis=1)

    # FIX п.35: добавляем _city_norm_АП для корректных метрик по городам
    out["_city_norm_АП"] = ap_work["_city_norm"].to_numpy()[pairs_df["ap_idx"].to_numpy()]

    pr = pairs_df["priority"].to_numpy()
    ch = pairs_df["channel"].to_numpy()
    dist = pairs_df["dist"].to_numpy()

    out["Совпадение по адресу (ДА/НЕТ)"] = np.where(pr == 1, "ДА", "НЕТ")
    out["Совпадение по ГЕО (ДА/НЕТ)"] = np.where(pr == 2, "ДА", "НЕТ")
    out["Совпадение по Адресу ИЛИ ГЕО (ДА/НЕТ)"] = np.where(pr <= 2, "ДА", "НЕТ")
    out["Приоритет 1 (ДА/НЕТ)"] = np.where(pr == 1, "ДА", "НЕТ")
    out["Приоритет 2 (ДА/НЕТ)"] = np.where(pr == 2, "ДА", "НЕТ")
    out["Приоритет 3 (ДА/НЕТ)"] = np.where(pr == 3, "ДА", "НЕТ")
    out["Канал адреса"] = np.where(pr == 1, ch, 0).astype(int)
    out["Расстояние по гео,м"] = dist

    out["_prio_num"] = pr

    for c in fin_cols_out:
        if c not in out.columns:
            out[c] = np.nan

    out = out.sort_values(["_prio_num", "Расстояние по гео,м"], na_position="last")
    return out, fin_used, city_taken

# =========================================================
# UI
# =========================================================
col1, col2 = st.columns(2)
with col1:
    file_fin = st.file_uploader("Файл 1: фин ап клиента (.xlsx)", type=["xlsx"], key="fin")
with col2:
    file_ap = st.file_uploader("Файл 2: возможная АП (.xlsx)", type=["xlsx"], key="ap")

if file_fin and file_ap and total_quota > 0:
    try:
        fin = load_fin(file_fin.getvalue())
        ap = load_ap(file_ap.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.write(f"Фин ап: {len(fin)} строк. Возможная АП: {len(ap)} строк. "
             f"Суммарная квота: {total_quota}.")

    with st.spinner("Подбираем..."):
        result, fin_used, city_taken = match(
            fin, ap, tuple(sorted(city_quotas.items()))
        )

    if result.empty:
        st.warning("Не удалось подобрать ни одной пары (нет координат / городов / магазинов).")
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

    st.success(f"Подобрано {len(final)} (суммарная квота {total_quota}).")
    st.dataframe(final.head(50), use_container_width=True)

    # диагностика
    diag_fin = fin[["Адрес", "город", "_city_norm", "_token_raw", "_token_set",
                    "_house_base"]].copy()
    # FIX п.6: fin_used корректной длины и типа
    if len(fin_used) == len(diag_fin):
        diag_fin["использован"] = np.where(fin_used, "ДА", "НЕТ")
    else:
        diag_fin["использован"] = "?"
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

    # FIX п.35: метрики по городам — через _city_norm_АП
    quota_rows = []
    res_city = result["_city_norm_АП"].to_numpy()
    for city, q in city_quotas.items():
        mask_c = res_city == city
        p1 = int((mask_c & (result["Приоритет 1 (ДА/НЕТ)"] == "ДА").to_numpy()).sum())
        p2 = int((mask_c & (result["Приоритет 2 (ДА/НЕТ)"] == "ДА").to_numpy()).sum())
        p3 = int((mask_c & (result["Приоритет 3 (ДА/НЕТ)"] == "ДА").to_numpy()).sum())
        taken = city_taken.get(city, 0)
        quota_rows.append({
            "Город": city,
            "Квота": q,
            "Подобрано": taken,
            "П1": p1, "П2": p2, "П3": p3,
            "Остаток": max(0, q - taken),
        })
    quota_metrics = pd.DataFrame(quota_rows).sort_values("Город")

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        final.to_excel(w, index=False, sheet_name="Результат")
        diag_fin.to_excel(w, index=False, sheet_name="Разбор_фин")
        diag_ap.to_excel(w, index=False, sheet_name="Разбор_АП")
        quota_metrics.to_excel(w, index=False, sheet_name="Квоты_по_городам")

        p1 = int((final["Приоритет 1 (ДА/НЕТ)"] == "ДА").sum())
        p2 = int((final["Приоритет 2 (ДА/НЕТ)"] == "ДА").sum())
        p3 = int((final["Приоритет 3 (ДА/НЕТ)"] == "ДА").sum())
        ch_counts = result.loc[result["Приоритет 1 (ДА/НЕТ)"] == "ДА", "Канал адреса"]
        ch1 = int((ch_counts == 1).sum())
        ch2 = int((ch_counts == 2).sum())
        ch3 = int((ch_counts == 3).sum())

        ap_cities = set(ap["_city_norm"].unique())
        out_of_quota_cities = ap_cities - set(city_quotas.keys())
        n_ap_out = int(ap["_city_norm"].isin(out_of_quota_cities).sum())

        metrics = pd.DataFrame({
            "Показатель": [
                "Суммарная квота",
                "Подобрано всего",
                "Приоритет 1 (адрес)",
                "  канал 1: точное множество токенов",
                f"  канал 2: fuzzy ≥ {FUZZY_THRESHOLD}",
                "  канал 3: уникальный дом",
                f"Приоритет 2 (координаты ≤ {EPS_M:.0f} м)",
                "Приоритет 3 (ближайший свободный)",
                "Магазинов фин использовано",
                "Магазинов фин всего",
                "АП вне квот (пропущены)",
            ],
            "Значение": [
                int(total_quota),
                len(final),
                p1, ch1, ch2, ch3,
                p2, p3,
                int(fin_used.sum()) if len(fin_used) == len(fin) else 0,
                int(len(fin)),
                n_ap_out,
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
