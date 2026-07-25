"""LeetCoach AI — Streamlit app.

"""
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Load pre-trained artifacts 
# --------------------------------------------------------------------------
ART_DIR = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_artifacts():
    df = pd.read_json(ART_DIR / "problems.json")
    df["Slug"] = df["Link"].str.rstrip("/").str.split("/").str[-1]
    slug_to_id = dict(zip(df["Slug"], df["ID"]))
    with open(ART_DIR / "solve_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(ART_DIR / "config.pkl", "rb") as f:
        cfg = pickle.load(f)
    return df, slug_to_id, clf, cfg


df, SLUG_TO_ID, clf, CFG = load_artifacts()
PRIOR, SCALE = CFG["PRIOR"], CFG["SCALE"]
TARGET_LOW, TARGET_HIGH = CFG["TARGET_LOW"], CFG["TARGET_HIGH"]


def p_solve(theta, beta):
    return 1 / (1 + np.exp(-(theta - beta) / SCALE))


# --------------------------------------------------------------------------
# Live LeetCode profile fetch (unofficial GraphQL endpoint, no login needed)
# --------------------------------------------------------------------------
LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"

PROFILE_QUERY = """
query userProfile($username: String!) {
  matchedUser(username: $username) {
    username
    profile { ranking reputation }
    submitStatsGlobal { acSubmissionNum { difficulty count } }
    tagProblemCounts {
      advanced     { tagName tagSlug problemsSolved }
      intermediate { tagName tagSlug problemsSolved }
      fundamental  { tagName tagSlug problemsSolved }
    }
  }
  recentAcSubmissionList(username: $username, limit: 20) { title titleSlug timestamp }
  userContestRanking(username: $username) { rating globalRanking attendedContestsCount }
}
"""


def fetch_leetcode_profile(username: str, timeout: int = 10) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Referer": f"https://leetcode.com/{username}/",
        "User-Agent": "Mozilla/5.0 (compatible; LeetCoachAI/0.1)",
    }
    resp = requests.post(
        LEETCODE_GRAPHQL_URL,
        json={"query": PROFILE_QUERY, "variables": {"username": username}},
        headers=headers, timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise ValueError("LeetCode API error — the username may be private or the API schema may have changed.")
    if not data["data"].get("matchedUser"):
        raise ValueError(f"No public LeetCode user found for '{username}'.")
    return data["data"]


def profile_to_theta(profile: dict, gain_per_solve: float = 60.0) -> dict:
    theta = {}
    tpc = profile["matchedUser"]["tagProblemCounts"]
    for tier in ("fundamental", "intermediate", "advanced"):
        for tag in tpc.get(tier, []):
            theta[tag["tagName"]] = PRIOR + gain_per_solve * math.log1p(tag["problemsSolved"])
    return theta


def get_recent_solved_ids(profile: dict) -> list:
    slugs = [s["titleSlug"] for s in profile.get("recentAcSubmissionList", [])]
    return [SLUG_TO_ID[s] for s in slugs if s in SLUG_TO_ID]


# --------------------------------------------------------------------------
# Recommendation engine 
# --------------------------------------------------------------------------
def recommend(theta_by_topic: dict, solved_ids: list, k: int = 10, allow_premium: bool = False) -> pd.DataFrame:
    cand = df[df["InDSAScope"] & ~df["ID"].isin(solved_ids)].copy()
    if not allow_premium:
        cand = cand[~cand["Premium Only"]]

    def theta_for(topics):
        vals = [theta_by_topic.get(t, PRIOR) for t in topics] or [PRIOR]
        return np.mean(vals), np.min(vals)

    tm, tmin = zip(*cand["TopicList"].apply(theta_for))
    cand["theta_mean"], cand["theta_min"] = tm, tmin
    cand["gap"] = cand["theta_mean"] - cand["Beta"]
    cand["n_topics"] = cand["TopicList"].apply(len)

    X_cand = cand[["theta_mean", "theta_min", "Beta", "gap", "n_topics"]]
    X_cand.columns = ["theta_mean", "theta_min", "beta", "gap", "n_topics"]
    cand["p_solve"] = clf.predict_proba(X_cand)[:, 1]

    is_new_topic = cand["TopicList"].apply(
        lambda ts: all(t not in theta_by_topic for t in ts) if ts else False
    )
    in_zone = cand["p_solve"].between(TARGET_LOW, TARGET_HIGH)
    cand["score"] = (
        in_zone.astype(float) * 2.0
        - (cand["p_solve"] - 0.675).abs()
        + is_new_topic.astype(float) * 0.5
        + cand["LikeRatio"] * 0.2
    )
    cand["reason"] = np.select(
        [is_new_topic & in_zone, is_new_topic, in_zone],
        ["New topic, good difficulty", "New topic for you", "Good challenge level"],
        default="Solid next step",
    )
    return cand.sort_values("score", ascending=False).head(k)[
        ["ID", "Title", "Difficulty", "TopicList", "p_solve", "Link", "reason"]
    ]


def format_recommendations_html(rec_df: pd.DataFrame) -> str:
    if rec_df.empty:
        return "<p>No eligible unsolved problems matched — try allowing premium problems.</p>"
    rows = []
    for _, r in rec_df.iterrows():
        topics = ", ".join(r["TopicList"][:3]) if r["TopicList"] else "—"
        rows.append(
            f"<tr><td><a href='{r['Link']}' target='_blank'>{r['Title']}</a></td>"
            f"<td>{r['Difficulty']}</td><td>{topics}</td>"
            f"<td>{r['p_solve']*100:.0f}%</td><td>{r['reason']}</td></tr>"
        )
    return (
        "<table style='width:100%; border-collapse:collapse;'>"
        "<tr><th align='left'>Problem</th><th align='left'>Difficulty</th>"
        "<th align='left'>Topics</th><th align='left'>Est. solve chance</th>"
        "<th align='left'>Why</th></tr>" + "".join(rows) + "</table>"
    )


def plot_profile_charts(ac_counts: dict, theta_by_topic: dict, recommended_topics: set, top_n: int = 12):
    """Two-panel chart: solved-by-difficulty (left) + top topics by estimated
    mastery (right), with bars for topics currently being recommended
    highlighted so the chart visually connects to the suggestion list below."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    diffs = ["Easy", "Medium", "Hard"]
    counts = [ac_counts.get(d, 0) for d in diffs]
    axes[0].bar(diffs, counts, color=["#4CAF50", "#FF9800", "#F44336"])
    axes[0].set_title("Your solved problems by difficulty")
    axes[0].set_ylabel("Solved")
    for i, c in enumerate(counts):
        axes[0].text(i, c, str(c), ha="center", va="bottom")

    combined = dict(theta_by_topic)
    for rt in recommended_topics:
        combined.setdefault(rt, PRIOR)  # show targeted-but-untouched topics too, at baseline

    if combined:
        rec_items = sorted([(t, v) for t, v in combined.items() if t in recommended_topics],
                            key=lambda x: x[1], reverse=True)[:top_n // 2]
        other_items = sorted([(t, v) for t, v in combined.items() if t not in recommended_topics],
                              key=lambda x: x[1], reverse=True)[:top_n - len(rec_items)]
        items = sorted(rec_items + other_items, key=lambda x: x[1], reverse=True)
        names = [t[0] for t in items][::-1]
        vals = [t[1] for t in items][::-1]
        colors = ["#FF5252" if n in recommended_topics else "#3F51B5" for n in names]
        axes[1].barh(names, vals, color=colors)
        axes[1].axvline(PRIOR, color="gray", linestyle="--", linewidth=1)
        axes[1].set_title("Your topic mastery (red = targeted by suggestions)")
        axes[1].set_xlabel("Estimated skill (θ)")
    else:
        axes[1].text(0.5, 0.5, "Not enough topic data yet", ha="center", va="center",
                      transform=axes[1].transAxes)
        axes[1].set_title("Your topic mastery")

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="LeetCoach AI", page_icon="🧭", layout="centered")
st.title("🧭 LeetCoach AI")
st.write(
    "Enter a public LeetCode username. LeetCoach pulls your real solve history, "
    "estimates your per-topic mastery, and recommends what to solve next — "
    "problems that stretch you without being discouraging, prioritizing topics "
    "you haven't practiced yet."
)

col1, col2 = st.columns([2, 1])
with col1:
    username = st.text_input("LeetCode username", value="your-username")
with col2:
    k = st.slider("Recommendations", 5, 20, 10)
allow_premium = st.checkbox("Include premium-only problems", value=False)

if st.button("Get recommendations", type="primary"):
    username = (username or "").strip()
    if not username:
        st.warning("Enter a LeetCode username to get started.")
    else:
        with st.spinner("Pulling your profile and scoring problems..."):
            try:
                profile = fetch_leetcode_profile(username)
            except requests.exceptions.RequestException:
                st.error("Couldn't reach LeetCode right now — please try again in a moment.")
                st.stop()
            except ValueError as e:
                st.error(str(e))
                st.stop()

            mu = profile["matchedUser"]
            ac = {d["difficulty"]: d["count"] for d in mu["submitStatsGlobal"]["acSubmissionNum"]}
            rating = profile.get("userContestRanking")
            summary = (
                f"**{mu['username']}** — {ac.get('All', 0)} problems solved "
                f"({ac.get('Easy', 0)} Easy / {ac.get('Medium', 0)} Medium / {ac.get('Hard', 0)} Hard)"
            )
            if rating:
                summary += f"  \nContest rating: {rating['rating']:.0f} (global rank {rating['globalRanking']:,})"
            st.markdown(summary)

            theta = profile_to_theta(profile)
            solved_ids = get_recent_solved_ids(profile)
            recs = recommend(theta, solved_ids, k=int(k), allow_premium=allow_premium)

            recommended_topics = {t for ts in recs["TopicList"] for t in ts}
            fig = plot_profile_charts(ac, theta, recommended_topics)
            st.pyplot(fig)
            if theta:
                st.caption("Red bars = topics your recommendations below are targeting. Dashed line = baseline (untouched topic).")

            st.subheader("Recommended next problems")
            st.markdown(format_recommendations_html(recs), unsafe_allow_html=True)

st.caption(
    "Built by [Shovon](https://www.linkedin.com/in/shoovoon/) · "
    "[GitHub](https://github.com/al-shovon) ·"
)
