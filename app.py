from __future__ import annotations

from datetime import datetime
import re
from collections import defaultdict

import pandas as pd
import streamlit as st
import requests
import trafilatura

import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# Config / Frames
# -----------------------------
FRAMES = [
    "Meritocracy",
    "Global inclusion",
    "Individual achievement",
    "American modernity",
    "Commercialization",
    "Exchange / youth programs",
    "Geopolitics",
    "Backlash / limits",
]

SENTIMENTS = ["Positive", "Neutral", "Negative"]

FRAME_KEYWORDS = {
    "Meritocracy": [
        "hard work", "earned", "grind", "opportunity", "from nothing", "work ethic", "self-made", "discipline"
    ],
    "Global inclusion": [
        "global", "worldwide", "international", "diverse", "inclusion", "community", "fans around the world", "bridge"
    ],
    "Individual achievement": [
        "legacy", "mvp", "greatest", "record", "all-time", "career", "superstar", "achievement"
    ],
    "American modernity": [
        "innovation", "modern", "cutting-edge", "technology", "progress", "future", "leadership", "american"
    ],
    "Commercialization": [
        "sponsor", "brand", "merch", "broadcast rights", "deal", "revenue", "market", "partnership", "advertising"
    ],
    "Exchange / youth programs": [
        "academy", "youth", "jr nba", "clinic", "camp", "development", "grassroots", "training", "school"
    ],
    "Geopolitics": [
        "government", "policy", "sanctions", "diplomacy", "censorship", "national", "state", "security", "relations"
    ],
    "Backlash / limits": [
        "criticism", "boycott", "controversy", "backlash", "protest", "resentment", "limits", "decline"
    ],
}

SHEET_COLUMNS = [
    "id",
    "created_at",
    "country",
    "source",
    "doc_date",
    "title",
    "url",
    "full_text",
    "auto_frame",
    "auto_frame_score",
    "auto_summary",
    "auto_method",
    "auto_keywords",
    "frames_human",
    "sentiment_human",
    "soft_power_score_human",
    "coder",
    "coded_at",
    "memo",
    "status",
    "error",
]


# -----------------------------
# Google Sheets backend
# -----------------------------
def gs_client():
    creds_info = dict(st.secrets["gcp_service_account"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(creds)


def gs_worksheet():
    gc = gs_client()
    sh = gc.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    ws = sh.worksheet(st.secrets["sheets"]["worksheet_name"])
    return ws


@st.cache_data(ttl=30)
def gs_load_df() -> pd.DataFrame:
    ws = gs_worksheet()
    records = ws.get_all_records()  # row 1 = header
    df = pd.DataFrame(records)

    if df.empty:
        # Ensure expected columns exist even when empty
        return pd.DataFrame(columns=SHEET_COLUMNS)

    for c in SHEET_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    return df


def gs_next_id(df: pd.DataFrame) -> int:
    if df.empty or "id" not in df.columns:
        return 1
    ids = pd.to_numeric(df["id"], errors="coerce").dropna()
    return int(ids.max()) + 1 if not ids.empty else 1


def gs_append_row(row: dict) -> int:
    df = gs_load_df()
    new_id = gs_next_id(df)

    clean = {c: "" for c in SHEET_COLUMNS}
    clean.update(row)
    clean["id"] = new_id
    clean["created_at"] = clean.get("created_at") or datetime.utcnow().isoformat()

    ws = gs_worksheet()
    ws.append_row([clean[c] for c in SHEET_COLUMNS], value_input_option="RAW")

    st.cache_data.clear()
    return new_id


def gs_find_row_num(doc_id: int) -> int | None:
    ws = gs_worksheet()
    col = ws.col_values(1)  # column A
    target = str(doc_id).strip()
    for idx, v in enumerate(col[1:], start=2):  # start at row 2
        if str(v).strip() == target:
            return idx
    return None


def gs_update(doc_id: int, updates: dict) -> bool:
    ws = gs_worksheet()
    row_num = gs_find_row_num(doc_id)
    if not row_num:
        return False

    header = ws.row_values(1)
    col_index = {name: i + 1 for i, name in enumerate(header)}

    cells = []
    for k, v in updates.items():
        if k in col_index:
            cells.append(gspread.Cell(row_num, col_index[k], v))

    if not cells:
        return False

    ws.update_cells(cells, value_input_option="RAW")
    st.cache_data.clear()
    return True


# -----------------------------
# Scrape / Extract
# -----------------------------
def fetch_article(url: str, timeout: int = 20) -> tuple[str | None, str | None, str | None, str]:
    """
    Returns (final_url, title, text, error_message).
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        meta = trafilatura.metadata.extract_metadata(html)
        title = meta.title if meta and meta.title else None

        text = extracted.strip() if isinstance(extracted, str) and extracted.strip() else None
        return resp.url, title, text, ""
    except Exception as e:
        return None, None, None, str(e)


# -----------------------------
# Framing + Keywords + Summary
# -----------------------------
def normalize_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def keyword_hits(text: str) -> dict[str, list[str]]:
    """
    Returns {frame: [keywords_hit]} for transparency/auditability.
    """
    t = normalize_text(text)
    hits: dict[str, list[str]] = {}
    for frame, kws in FRAME_KEYWORDS.items():
        frame_hits = []
        for kw in kws:
            kw_n = normalize_text(kw)
            if " " in kw_n:
                if kw_n in t:
                    frame_hits.append(kw)
            else:
                if kw_n in t:
                    frame_hits.append(kw)
        if frame_hits:
            hits[frame] = sorted(set(frame_hits))
    return hits


def pick_best_frame(text: str) -> tuple[str | None, float, dict[str, float]]:
    """
    Returns (best_frame, confidence_0to1, raw_scores).
    One best frame only.
    """
    t = normalize_text(text)
    scores = defaultdict(float)

    for frame, kws in FRAME_KEYWORDS.items():
        for kw in kws:
            kw_n = normalize_text(kw)
            if " " in kw_n:
                if kw_n in t:
                    scores[frame] += 2.0
            else:
                scores[frame] += t.count(kw_n) * 1.0

    scores = dict(scores)
    if not scores:
        return None, 0.0, {}

    best = max(scores, key=scores.get)
    best_score = scores.get(best, 0.0)
    if best_score <= 0:
        return None, 0.0, scores

    conf = min(1.0, best_score / 10.0)
    return best, conf, scores


def short_summary(text: str, max_sentences: int = 3) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", t)
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:max_sentences])[:900]


# -----------------------------
# Analytics helpers
# -----------------------------
def frame_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=["frame", "count"])
    s = df[col].fillna("").astype(str).str.split(",").explode().str.strip()
    s = s[s != ""]
    if s.empty:
        return pd.DataFrame(columns=["frame", "count"])
    return s.value_counts().rename_axis("frame").reset_index(name="count")


def sentiment_distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count", "share"])
    counts = df[col].fillna("Uncoded").value_counts().rename_axis("sentiment").reset_index(name="count")
    total = counts["count"].sum()
    counts["share"] = (counts["count"] / total).round(3)
    return counts


# -----------------------------
# Wix-friendly UI polish
# -----------------------------
st.set_page_config(page_title="NBA Soft Power Tracker", layout="wide")

# Hide Streamlit chrome for a cleaner embedded look
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🏀 NBA Soft Power Tracker")
st.caption("Scrape documents → auto-frame by keywords → human code → compare cases.")

tab_add, tab_ingest, tab_batch, tab_code, tab_explore, tab_compare = st.tabs(
    ["➕ Add (manual)", "🌐 Import URL", "🧠 Batch scrape", "🏷️ Code", "🔎 Explore", "📊 Compare"]
)


# -----------------------------
# Add manual
# -----------------------------
with tab_add:
    st.subheader("Add a document manually")
    with st.form("add_doc_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            country = st.text_input("Country / market *", placeholder="China, France, etc.")
        with c2:
            source = st.text_input("Source *", placeholder="NBA.com, ESPN, CCTV, Le Monde...")
        with c3:
            doc_date = st.date_input("Document date *")

        title = st.text_input("Title *")
        url = st.text_input("URL (optional)")
        full_text = st.text_area("Full text (optional)", height=180)

        submitted = st.form_submit_button("Save")
        if submitted:
            if not country.strip() or not source.strip() or not title.strip():
                st.error("Country, Source, and Title are required.")
            else:
                best_frame, conf, _ = pick_best_frame(full_text or "")
                summ = short_summary(full_text or "")
                hits = keyword_hits(full_text or "")

                gs_append_row(
                    {
                        "country": country.strip(),
                        "source": source.strip(),
                        "doc_date": str(doc_date),
                        "title": title.strip(),
                        "url": url.strip() if url.strip() else "",
                        "full_text": full_text.strip() if full_text.strip() else "",
                        "auto_frame": best_frame or "",
                        "auto_frame_score": float(conf) if conf is not None else "",
                        "auto_summary": summ,
                        "auto_method": "rule_based_keywords_v1",
                        "auto_keywords": str(hits),
                        "status": "OK",
                        "error": "",
                    }
                )
                st.success("Saved to Google Sheets.")


# -----------------------------
# Import single URL
# -----------------------------
with tab_ingest:
    st.subheader("Import from URL (scrape → extract → auto-frame + summary)")
    url_in = st.text_input("URL", placeholder="https://...")

    if st.button("Fetch & Preview"):
        if not url_in.strip():
            st.error("Paste a URL first.")
        else:
            final_url, title, text, err = fetch_article(url_in.strip())
            if not text:
                st.error(f"Extraction failed. Site may block scraping.\n\nError: {err}")
            else:
                st.session_state["ingest_url"] = final_url or url_in.strip()
                st.session_state["ingest_title"] = title or ""
                st.session_state["ingest_text"] = text

    if st.session_state.get("ingest_text"):
        with st.form("save_ingest"):
            c1, c2, c3 = st.columns(3)
            with c1:
                country = st.text_input("Country / market *")
            with c2:
                source = st.text_input("Source *")
            with c3:
                doc_date = st.date_input("Document date *")

            title = st.text_input("Title *", value=st.session_state.get("ingest_title", ""))
            final_url = st.text_input("URL", value=st.session_state.get("ingest_url", ""))
            full_text = st.text_area(
                "Extracted full text (edit if needed)",
                height=220,
                value=st.session_state.get("ingest_text", ""),
            )

            if st.form_submit_button("Save imported document"):
                if not country.strip() or not source.strip() or not title.strip():
                    st.error("Country, Source, and Title are required.")
                else:
                    best_frame, conf, _ = pick_best_frame(full_text)
                    summ = short_summary(full_text)
                    hits = keyword_hits(full_text)

                    gs_append_row(
                        {
                            "country": country.strip(),
                            "source": source.strip(),
                            "doc_date": str(doc_date),
                            "title": title.strip(),
                            "url": final_url.strip() if final_url.strip() else "",
                            "full_text": full_text.strip(),
                            "auto_frame": best_frame or "",
                            "auto_frame_score": float(conf) if conf is not None else "",
                            "auto_summary": summ,
                            "auto_method": "rule_based_keywords_v1",
                            "auto_keywords": str(hits),
                            "status": "OK",
                            "error": "",
                        }
                    )

                    st.success("Saved to Google Sheets.")
                    st.session_state.pop("ingest_url", None)
                    st.session_state.pop("ingest_title", None)
                    st.session_state.pop("ingest_text", None)


# -----------------------------
# Batch scrape
# -----------------------------
with tab_batch:
    st.subheader("Batch scrape → auto-frame + keyword hits + summary")
    urls_text = st.text_area("URLs (one per line)", height=160, placeholder="https://...\nhttps://...\n...")

    c1, c2, c3 = st.columns(3)
    with c1:
        default_country = st.text_input("Default country/market", value="")
    with c2:
        default_source = st.text_input("Default source label", value="")
    with c3:
        batch_date = st.date_input("Document date for batch")

    if st.button("Run batch scrape"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.error("Paste at least one URL.")
        else:
            rows = []
            for u in urls:
                final_url, title, text, err = fetch_article(u)

                if not text:
                    rows.append(
                        {
                            "url": u,
                            "status": "FAILED",
                            "title": title or "",
                            "auto_frame": "",
                            "confidence": 0.0,
                            "summary": "",
                            "keyword_hits": "",
                            "full_text": "",
                            "error": err,
                        }
                    )
                    continue

                best_frame, conf, _ = pick_best_frame(text)
                summ = short_summary(text)
                hits = keyword_hits(text)

                rows.append(
                    {
                        "url": final_url or u,
                        "status": "OK",
                        "title": title or "(no title detected)",
                        "auto_frame": best_frame or "",
                        "confidence": conf,
                        "summary": summ,
                        "keyword_hits": str(hits),
                        "full_text": text,
                        "error": "",
                    }
                )

            out = pd.DataFrame(rows)
            st.session_state["batch_out"] = out

    out = st.session_state.get("batch_out")
    if isinstance(out, pd.DataFrame) and not out.empty:
        st.dataframe(out.drop(columns=["full_text"], errors="ignore"), use_container_width=True, hide_index=True)

        if st.button("Save all OK rows to Google Sheets"):
            ok = out[out["status"] == "OK"].copy()
            if ok.empty:
                st.warning("No OK rows to save.")
            else:
                for _, r in ok.iterrows():
                    gs_append_row(
                        {
                            "country": (default_country.strip() or "Unknown"),
                            "source": (default_source.strip() or "Unknown"),
                            "doc_date": str(batch_date),
                            "title": str(r["title"]),
                            "url": str(r["url"]),
                            "full_text": str(r["full_text"]),
                            "auto_frame": str(r["auto_frame"]),
                            "auto_frame_score": float(r["confidence"]) if pd.notna(r["confidence"]) else "",
                            "auto_summary": str(r["summary"]),
                            "auto_method": "rule_based_keywords_v1",
                            "auto_keywords": str(r.get("keyword_hits", "")),
                            "status": "OK",
                            "error": "",
                        }
                    )
                st.success(f"Saved {len(ok)} documents to Google Sheets.")


# -----------------------------
# Code (human)
# -----------------------------
with tab_code:
    st.subheader("Human coding (frames/sentiment/score)")
    df = gs_load_df()

    if df.empty:
        st.info("No documents yet.")
    else:
        df2 = df.copy()
        # Ensure id is numeric for sorting/selection
        df2["id_num"] = pd.to_numeric(df2["id"], errors="coerce")
        df2 = df2.sort_values("id_num", ascending=False)

        df2["label"] = df2.apply(
            lambda r: f"#{r['id']} | {r['country']} | {r['source']} | {r['doc_date']} | {r['title']} | auto:{r.get('auto_frame','')}",
            axis=1,
        )

        selection = st.selectbox("Select a document", df2["label"].tolist())
        doc_id = int(selection.split("|")[0].replace("#", "").strip())

        row = df2[df2["id"].astype(str) == str(doc_id)].iloc[0]

        st.write(f"**Auto frame:** {row.get('auto_frame','')}  |  **Auto score:** {row.get('auto_frame_score','')}")
        st.write(f"**Auto summary:** {row.get('auto_summary','')}")
        if str(row.get("url", "")).strip():
            st.write(f"**URL:** {row['url']}")

        if str(row.get("full_text", "")).strip():
            with st.expander("Full text"):
                st.write(row["full_text"])

        with st.form("code_form"):
            frames = st.multiselect("Frames (human) *", FRAMES, default=[])
            sentiment = st.selectbox("Sentiment *", SENTIMENTS, index=1)
            soft_power_score = st.slider("Soft power signal strength (1–5) *", 1, 5, 3)
            coder = st.text_input("Coder *", value="Devin")
            memo = st.text_area("Memo (optional)", height=120)

            if st.form_submit_button("Save coding"):
                if not frames or not coder.strip():
                    st.error("Frames and Coder are required.")
                else:
                    ok = gs_update(
                        doc_id,
                        {
                            "frames_human": ",".join(frames),
                            "sentiment_human": sentiment,
                            "soft_power_score_human": int(soft_power_score),
                            "coder": coder.strip(),
                            "coded_at": datetime.utcnow().isoformat(),
                            "memo": memo.strip() if memo.strip() else "",
                        },
                    )
                    if ok:
                        st.success("Saved to Google Sheets.")
                    else:
                        st.error("Could not update row (ID not found).")


# -----------------------------
# Explore + sort
# -----------------------------
with tab_explore:
    st.subheader("Explore + sort")
    df = gs_load_df()

    if df.empty:
        st.info("No data yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            countries = sorted([c for c in df["country"].dropna().unique().tolist() if str(c).strip()])
            country_filter = st.multiselect("Country", countries, default=countries)
        with c2:
            auto_frames = sorted([x for x in df["auto_frame"].dropna().unique().tolist() if str(x).strip()])
            auto_frame_filter = st.multiselect("Auto frame", auto_frames, default=auto_frames)
        with c3:
            keyword_search = st.text_input("Keyword search (full text / hits)", placeholder="e.g., sponsor, diplomacy, boycott")
        with c4:
            sort_by = st.selectbox("Sort by", ["doc_date (newest)", "auto_frame_score (high→low)", "country", "source", "id (newest)"])

        filtered = df.copy()
        if country_filter:
            filtered = filtered[filtered["country"].isin(country_filter)]
        if auto_frame_filter:
            filtered = filtered[filtered["auto_frame"].fillna("").isin(auto_frame_filter)]

        if keyword_search.strip():
            ks = keyword_search.strip().lower()
            filtered = filtered[
                filtered["full_text"].fillna("").astype(str).str.lower().str.contains(ks)
                | filtered["auto_keywords"].fillna("").astype(str).str.lower().str.contains(ks)
                | filtered["title"].fillna("").astype(str).str.lower().str.contains(ks)
            ]

        if sort_by == "doc_date (newest)":
            filtered = filtered.sort_values("doc_date", ascending=False)
        elif sort_by == "auto_frame_score (high→low)":
            filtered["auto_frame_score_num"] = pd.to_numeric(filtered["auto_frame_score"], errors="coerce")
            filtered = filtered.sort_values("auto_frame_score_num", ascending=False)
        elif sort_by == "country":
            filtered = filtered.sort_values("country", ascending=True)
        elif sort_by == "source":
            filtered = filtered.sort_values("source", ascending=True)
        elif sort_by == "id (newest)":
            filtered["id_num"] = pd.to_numeric(filtered["id"], errors="coerce")
            filtered = filtered.sort_values("id_num", ascending=False)

        show_cols = [
            "id", "country", "source", "doc_date", "title", "url",
            "auto_frame", "auto_frame_score", "auto_summary",
            "frames_human", "sentiment_human", "soft_power_score_human",
            "coder", "coded_at",
        ]
        existing = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[existing], use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="nba_soft_power_export.csv",
            mime="text/csv",
        )


# -----------------------------
# Compare cases (human-coded)
# -----------------------------
with tab_compare:
    st.subheader("Compare cases (human-coded)")
    df = gs_load_df()

    if df.empty:
        st.info("No data yet.")
    else:
        coded = df.copy()
        coded_mask = (
            coded["frames_human"].fillna("").astype(str).str.len().gt(0)
            | coded["sentiment_human"].fillna("").astype(str).str.len().gt(0)
            | pd.to_numeric(coded["soft_power_score_human"], errors="coerce").notna()
        )
        coded = coded[coded_mask]

        if coded.empty:
            st.warning("No human-coded items yet.")
        else:
            countries = sorted([c for c in coded["country"].dropna().unique().tolist() if str(c).strip()])
            if len(countries) < 2:
                st.warning("Add at least two countries.")
            else:
                a = st.selectbox("Case A", countries, index=0)
                b = st.selectbox("Case B", countries, index=1)

                df_a = coded[coded["country"] == a].copy()
                df_b = coded[coded["country"] == b].copy()

                s1, s2, s3, s4 = st.columns(4)
                s1.metric(f"{a}: coded docs", int(df_a["id"].nunique()))
                s2.metric(f"{b}: coded docs", int(df_b["id"].nunique()))

                a_score = pd.to_numeric(df_a["soft_power_score_human"], errors="coerce").dropna()
                b_score = pd.to_numeric(df_b["soft_power_score_human"], errors="coerce").dropna()
                s3.metric(f"{a}: avg score", round(float(a_score.mean()), 2) if not a_score.empty else "—")
                s4.metric(f"{b}: avg score", round(float(b_score.mean()), 2) if not b_score.empty else "—")

                left, right = st.columns(2)
                with left:
                    st.write(f"**{a}: sentiment**")
                    st.dataframe(sentiment_distribution(df_a, col="sentiment_human"), use_container_width=True, hide_index=True)
                    st.write(f"**{a}: frames**")
                    st.dataframe(frame_counts(df_a, col="frames_human"), use_container_width=True, hide_index=True)

                with right:
                    st.write(f"**{b}: sentiment**")
                    st.dataframe(sentiment_distribution(df_b, col="sentiment_human"), use_container_width=True, hide_index=True)
                    st.write(f"**{b}: frames**")
                    st.dataframe(frame_counts(df_b, col="frames_human"), use_container_width=True, hide_index=True)