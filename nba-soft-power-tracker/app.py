import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st

DB_PATH = "tracker.db"

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


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            doc_date TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            excerpt TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            frames TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            soft_power_score INTEGER NOT NULL,
            coder TEXT NOT NULL,
            coded_at TEXT NOT NULL,
            memo TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


def insert_document(country, source, doc_date, title, url, excerpt):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (country, source, doc_date, title, url, excerpt, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (country, source, doc_date, title, url, excerpt, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def insert_code(document_id, frames, sentiment, soft_power_score, coder, memo):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO codes (document_id, frames, sentiment, soft_power_score, coder, coded_at, memo)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            document_id,
            ",".join(frames),
            sentiment,
            int(soft_power_score),
            coder,
            datetime.utcnow().isoformat(),
            memo,
        ),
    )
    conn.commit()
    conn.close()


def load_data():
    conn = get_conn()
    docs = pd.read_sql_query("SELECT * FROM documents;", conn)
    codes = pd.read_sql_query("SELECT * FROM codes;", conn)
    conn.close()

    if docs.empty:
        return pd.DataFrame()

    # left-join latest code per document (MVP: assumes 0 or 1 coding pass; if multiple, take most recent)
    if codes.empty:
        docs["frames"] = ""
        docs["sentiment"] = ""
        docs["soft_power_score"] = None
        docs["coder"] = ""
        docs["coded_at"] = ""
        docs["memo"] = ""
        return docs

    codes_sorted = codes.sort_values("coded_at").groupby("document_id").tail(1)
    merged = docs.merge(codes_sorted, left_on="id", right_on="document_id", how="left", suffixes=("", "_code"))

    # tidy
    keep_cols = [
        "id",
        "country",
        "source",
        "doc_date",
        "title",
        "url",
        "excerpt",
        "frames",
        "sentiment",
        "soft_power_score",
        "coder",
        "coded_at",
        "memo",
        "created_at",
    ]
    return merged[keep_cols].sort_values(["doc_date", "created_at"], ascending=[False, False])


def delete_document(doc_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM codes WHERE document_id = ?;", (doc_id,))
    cur.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
    conn.commit()
    conn.close()


# ---------- UI ----------

st.set_page_config(page_title="NBA Soft Power Tracker (MVP)", layout="wide")
init_db()

st.title("🏀 NBA Soft Power Tracker (Beginner MVP)")
st.caption("Log documents by country/source, code narrative frames, filter, and export.")


tab_add, tab_code, tab_explore = st.tabs(["➕ Add document", "🏷️ Code document", "🔎 Explore & export"])

with tab_add:
    st.subheader("Add a document")
    with st.form("add_doc_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            country = st.text_input("Country / market *", placeholder="China, France, etc.")
        with col2:
            source = st.text_input("Source *", placeholder="NBA.com, ESPN, CCTV, Le Monde...")
        with col3:
            doc_date = st.date_input("Document date *")

        title = st.text_input("Title *")
        url = st.text_input("URL (optional)")
        excerpt = st.text_area("Excerpt / notes (optional)", height=140, placeholder="Paste a key paragraph you’re coding...")

        submitted = st.form_submit_button("Save document")
        if submitted:
            if not country.strip() or not source.strip() or not title.strip():
                st.error("Country, Source, and Title are required.")
            else:
                insert_document(
                    country=country.strip(),
                    source=source.strip(),
                    doc_date=str(doc_date),
                    title=title.strip(),
                    url=url.strip() if url.strip() else None,
                    excerpt=excerpt.strip() if excerpt.strip() else None,
                )
                st.success("Saved.")

with tab_code:
    st.subheader("Code a document")
    df = load_data()

    if df.empty:
        st.info("No documents yet. Add one in the first tab.")
    else:
        # Pick a document
        options = df[["id", "country", "source", "doc_date", "title"]].copy()
        options["label"] = options.apply(
            lambda r: f"#{r['id']} | {r['country']} | {r['source']} | {r['doc_date']} | {r['title']}",
            axis=1,
        )
        selection = st.selectbox("Select a document", options["label"].tolist())
        doc_id = int(selection.split("|")[0].replace("#", "").strip())

        doc_row = df[df["id"] == doc_id].iloc[0]

        with st.expander("View document details", expanded=True):
            st.write(f"**Country:** {doc_row['country']}")
            st.write(f"**Source:** {doc_row['source']}")
            st.write(f"**Date:** {doc_row['doc_date']}")
            st.write(f"**Title:** {doc_row['title']}")
            if pd.notna(doc_row.get("url")) and str(doc_row.get("url")).strip():
                st.write(f"**URL:** {doc_row['url']}")
            if pd.notna(doc_row.get("excerpt")) and str(doc_row.get("excerpt")).strip():
                st.write("**Excerpt/Notes:**")
                st.write(doc_row["excerpt"])

        st.markdown("### Coding fields")
        with st.form("code_form"):
            frames = st.multiselect("Frames (multi-select) *", FRAMES, default=None)
            sentiment = st.selectbox("Sentiment *", SENTIMENTS, index=1)
            soft_power_score = st.slider("Soft power signal strength (1–5) *", 1, 5, 3)
            coder = st.text_input("Coder name/initials *", value="Devin")
            memo = st.text_area("Coding memo (optional)", height=120)

            code_submitted = st.form_submit_button("Save coding")
            if code_submitted:
                if not frames or not coder.strip():
                    st.error("Frames and Coder are required.")
                else:
                    insert_code(
                        document_id=doc_id,
                        frames=frames,
                        sentiment=sentiment,
                        soft_power_score=soft_power_score,
                        coder=coder.strip(),
                        memo=memo.strip() if memo.strip() else None,
                    )
                    st.success("Coding saved. Go to Explore to see updates.")

with tab_explore:
    st.subheader("Explore & export")
    df = load_data()

    if df.empty:
        st.info("No data yet.")
    else:
        # Filters
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            countries = sorted([c for c in df["country"].dropna().unique().tolist()])
            country_filter = st.multiselect("Filter: Country", countries, default=countries)
        with c2:
            sources = sorted([s for s in df["source"].dropna().unique().tolist()])
            source_filter = st.multiselect("Filter: Source", sources, default=sources)
        with c3:
            sentiment_vals = sorted([s for s in df["sentiment"].dropna().unique().tolist()])
            sentiment_filter = st.multiselect("Filter: Sentiment", sentiment_vals, default=sentiment_vals)
        with c4:
            frame_text = st.text_input("Filter: frame contains", placeholder="e.g., Meritocracy")

        filtered = df.copy()
        filtered = filtered[filtered["country"].isin(country_filter)]
        filtered = filtered[filtered["source"].isin(source_filter)]
        if sentiment_filter:
            filtered = filtered[filtered["sentiment"].fillna("").isin(sentiment_filter)]
        if frame_text.strip():
            filtered = filtered[filtered["frames"].fillna("").str.contains(frame_text.strip(), case=False, na=False)]

        st.markdown("### Quick summary")
        left, right = st.columns(2)

        with left:
            st.write("**Documents by country**")
            st.dataframe(
                filtered.groupby("country")["id"].count().rename("count").sort_values(ascending=False).reset_index(),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.write("**Frame frequency (rough count)**")
            # explode frames
            frames_series = (
                filtered["frames"]
                .fillna("")
                .astype(str)
                .str.split(",")
                .explode()
                .str.strip()
            )
            frames_series = frames_series[frames_series != ""]
            if frames_series.empty:
                st.write("No frames coded in this filtered view.")
            else:
                st.dataframe(
                    frames_series.value_counts().rename_axis("frame").reset_index(name="count"),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Records")
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="nba_soft_power_tracker_export.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### Danger zone")
        del_id = st.number_input("Delete document by ID", min_value=1, step=1)
        if st.button("Delete (permanent)"):
            delete_document(int(del_id))
            st.warning(f"Deleted document #{int(del_id)}. Refresh the page.")