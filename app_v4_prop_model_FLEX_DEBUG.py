# app_v4_prop_model_FLEX_DEBUG.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFL Player Prop Model (Flex + Debug)", layout="centered")

# ====== GOOGLE SHEETS (EXPORT LINKS) ======
SHEET_TOTAL_OFFENSE = "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv"
SHEET_TOTAL_PASS_OFF = "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv"
SHEET_TOTAL_RUSH_OFF = "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv"
SHEET_TOTAL_SCORE_OFF = "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv"
SHEET_PLAYER_REC = "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv"
SHEET_PLAYER_RUSH = "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv"
SHEET_PLAYER_PASS = "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv"
SHEET_DEF_RB = "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv"
SHEET_DEF_QB = "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv"
SHEET_DEF_WR = "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv"
SHEET_DEF_TE = "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv"

def load_sheet(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def find_player_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        low = col.lower()
        if "player" in low or "name" in low:
            return col
    if len(df.columns) >= 2:
        return df.columns[1]
    return df.columns[0]

# ====== LOAD SHEETS ======
try:
    total_off = load_sheet(SHEET_TOTAL_OFFENSE)
    total_pass = load_sheet(SHEET_TOTAL_PASS_OFF)
    total_rush = load_sheet(SHEET_TOTAL_RUSH_OFF)
    total_score = load_sheet(SHEET_TOTAL_SCORE_OFF)
    p_rec = load_sheet(SHEET_PLAYER_REC)
    p_rush = load_sheet(SHEET_PLAYER_RUSH)
    p_pass = load_sheet(SHEET_PLAYER_PASS)
    d_rb = load_sheet(SHEET_DEF_RB)
    d_qb = load_sheet(SHEET_DEF_QB)
    d_wr = load_sheet(SHEET_DEF_WR)
    d_te = load_sheet(SHEET_DEF_TE)
except Exception as e:
    st.error(f"❌ Error loading one or more Google Sheets: {e}")
    st.stop()

# detect player columns for debug
rec_player_col = find_player_column(p_rec)
rush_player_col = find_player_column(p_rush)
pass_player_col = find_player_column(p_pass)

with st.sidebar:
    st.header("🔎 Debug info")
    st.write("Receiving sheet player column:", rec_player_col)
    st.write("Rushing sheet player column:", rush_player_col)
    st.write("Passing sheet player column:", pass_player_col)
    st.write("If these don't look right, rename your column in Google Sheets or move player names to column 2.")

st.title("🏈 NFL Player Prop Model (Flex + Debug)")
st.write("This version auto-detects the player-name column and falls back to the 2nd column if needed.")

# ====== UI ======
player_name = st.text_input("Player name (as in your Google Sheet):")
opponent_team = st.text_input("Opponent team name (must match 'Team' in defense sheets):")

prop_options = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "targets",
    "carries",
    "anytime_td",
]
selected_props = st.multiselect("Select props to evaluate", prop_options, default=["rushing_yards"])

lines = {}
for prop in selected_props:
    if prop == "anytime_td":
        continue
    lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

if not player_name or not opponent_team or not selected_props:
    st.stop()

# ====== FIND PLAYER IN SHEETS (FLEX) ======
player_df = None
player_pos = None

if player_name.lower() in p_rec[rec_player_col].astype(str).str.lower().values:
    player_df = p_rec[p_rec[rec_player_col].astype(str).str.lower() == player_name.lower()].copy()
    player_pos = player_df.iloc[0].get("Position", "WR")
elif player_name.lower() in p_rush[rush_player_col].astype(str).str.lower().values:
    player_df = p_rush[p_rush[rush_player_col].astype(str).str.lower() == player_name.lower()].copy()
    player_pos = player_df.iloc[0].get("Position", "RB")
elif player_name.lower() in p_pass[pass_player_col].astype(str).str.lower().values:
    player_df = p_pass[p_pass[pass_player_col].astype(str).str.lower() == player_name.lower()].copy()
    player_pos = "QB"
else:
    st.error("❌ Player not found in receiving, rushing, or passing sheets. Check spelling or the column order.")
    st.stop()

# ====== HELPER: yardage/receptions model ======
def run_yardage_model(player_df: pd.DataFrame, player_pos: str, prop_type: str, opponent_team: str):
    if prop_type == "passing_yards":
        defense_df = d_qb
    elif prop_type in ["rushing_yards", "carries"]:
        defense_df = d_rb if player_pos != "QB" else d_qb
    elif prop_type in ["receiving_yards", "receptions", "targets"]:
        if player_pos == "TE":
            defense_df = d_te
        elif player_pos == "RB":
            defense_df = d_rb
        else:
            defense_df = d_wr
    else:
        return None

    stat_col = None
    if prop_type == "passing_yards":
        for cand in ["Passing_Yards", "Passing_Yds", "Pass_Yds", "Pass_Yards", "PassYds"]:
            if cand in player_df.columns:
                stat_col = cand
                break
    elif prop_type == "rushing_yards":
        for cand in ["Rushing_Yards", "Rush_Yds", "Rush_Yards"]:
            if cand in player_df.columns:
                stat_col = cand
                break
    elif prop_type == "receiving_yards":
        for cand in ["Receiving_Yards", "Rec_Yds", "Rec_Yards"]:
            if cand in player_df.columns:
                stat_col = cand
                break
    elif prop_type == "receptions":
        for cand in ["Receptions", "Rec", "Catches"]:
            if cand in player_df.columns:
                stat_col = cand
                break
    elif prop_type == "targets":
        for cand in ["Targets", "Tgt", "Tgts"]:
            if cand in player_df.columns:
                stat_col = cand
                break
    elif prop_type == "carries":
        for cand in ["Carries", "Rush_Att", "Rushing_Att"]:
            if cand in player_df.columns:
                stat_col = cand
                break

    if stat_col is None:
        return None

    merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    merged["rolling_avg_3"] = merged[stat_col].rolling(3, 1).mean()
    season_avg = merged[stat_col].mean()

    def_cols = [c for c in defense_df.columns if "Allowed" in c or "allowed" in c]
    def_col = def_cols[0] if def_cols else None

    X = merged[["rolling_avg_3"] + ([def_col] if def_col else [])].fillna(0)
    y = merged[stat_col].fillna(0)

    if len(X) < 2:
        return None

    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds)) if len(y) > 1 else np.std(y)

    opp_row = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    if opp_row.empty:
        return None

    rolling_next = merged[stat_col].tail(3).mean()
    feat_next = [rolling_next]
    if def_col:
        feat_next.append(float(opp_row.iloc[0][def_col]))
    pred_next = model.predict([feat_next])[0]

    return dict(
        stat_col=stat_col,
        season_avg=season_avg,
        last3=rolling_next,
        pred_next=pred_next,
        rmse=rmse,
        merged=merged,
        defense_df=defense_df,
    )

# ====== HELPER: anytime TD ======
def estimate_anytime_td(player_df: pd.DataFrame, defense_df: pd.DataFrame, opponent_team: str, player_pos: str):
    td_cands = ["Receiving_TDs", "Rec_TD", "Rush_TD", "Rushing_TDs", "Passing_TDs", "Pass_TD", "TD"]
    td_col = None
    for c in td_cands:
        if c in player_df.columns:
            td_col = c
            break
    if td_col is None:
        return None

    td_series = player_df[td_col].fillna(0)
    y = (td_series > 0).astype(int)
    merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    merged["td_rolling_3"] = td_series.rolling(3, 1).mean()
    X = merged[["td_rolling_3"]].fillna(0)

    if y.sum() == 0 or len(X) < 2:
        return float(y.mean())

    lg = LogisticRegression()
    lg.fit(X, y)

    opp_row = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    feat = [td_series.tail(3).mean()]
    prob = lg.predict_proba([feat])[0][1]
    return prob

# ====== RENDER RESULTS ======
st.header("📊 Results")

for prop in selected_props:
    if prop == "anytime_td":
        continue
    res = run_yardage_model(player_df, player_pos, prop, opponent_team)
    if res is None:
        st.warning(f"Could not model {prop} for this player. Check column names in your sheet.")
        continue

    line_val = lines.get(prop, None)
    if line_val is None:
        continue

    pred_next = res["pred_next"]
    rmse = res["rmse"]
    z = (line_val - pred_next) / rmse if rmse and rmse > 0 else 1
    prob_over = 1 - norm.cdf(z)
    prob_under = 1 - prob_over

    st.subheader(f"Prop: {prop}")
    st.markdown(
        "**Line:** {}  
**Predicted:** {:.2f}  
**Season avg:** {:.2f}  
**Last 3 games avg:** {:.2f}  
**Probability of over:** {:.1f}%  
**Probability of under:** {:.1f}%".format(
            line_val, pred_next, res["season_avg"], res["last3"], prob_over * 100, prob_under * 100
        )
    )

    fig1, ax1 = plt.subplots()
    ax1.bar(["Predicted", "Line"], [pred_next, line_val], color=["skyblue", "salmon"])
    ax1.set_ylabel(res["stat_col"])
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    m = res["merged"]
    ax2.plot(m.index + 1, m[res["stat_col"]], marker="o")
    ax2.set_title(f"{player_name} {res['stat_col']} by game")
    ax2.set_xlabel("Game #")
    ax2.set_ylabel(res["stat_col"])
    ax2.grid(True)
    st.pyplot(fig2)

# Anytime TD
if "anytime_td" in selected_props:
    if player_pos == "QB":
        def_df = d_qb
    elif player_pos == "RB":
        def_df = d_rb
    elif player_pos == "TE":
        def_df = d_te
    else:
        def_df = d_wr

    td_prob = estimate_anytime_td(player_df, def_df, opponent_team, player_pos)
    st.subheader("🔥 Anytime TD")
    if td_prob is not None:
        st.write("**Anytime TD probability:** {:.1f}%".format(td_prob * 100))
    else:
        st.write("Not enough TD data to estimate.")
