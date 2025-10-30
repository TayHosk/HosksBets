# app_v4_prop_model.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFL Player Prop Model (v4)", layout="centered")

# ====== UPDATED GOOGLE SHEET IDS ======
SHEET_TOTAL_OFFENSE = "1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48"
SHEET_TOTAL_PASS_OFF = "1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng"
SHEET_TOTAL_RUSH_OFF = "14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90"
SHEET_TOTAL_SCORE_OFF = "1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw"
SHEET_PLAYER_REC = "a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM"
SHEET_PLAYER_RUSH = "1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k"
SHEET_PLAYER_PASS = "1I9YNSQMylW_waJs910q4S6SM8CZE"
SHEET_DEF_RB = "1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY"
SHEET_DEF_QB = "1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660"
SHEET_DEF_WR = "14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo"
SHEET_DEF_TE = "1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4"

def load_sheet(sheet_id: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    return pd.read_csv(url)

# ====== LOAD ALL SHEETS ======
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
except Exception:
    st.error("❌ Error loading one or more Google Sheets. Make sure all 11 are shared as 'Anyone with the link'.")
    st.stop()

st.title("🏈 NFL Player Prop Model (v4)")
st.write("Single player → multiple props → yards + over/under + anytime TD, using your live Google Sheets.")

# ====== USER INPUTS ======
player_name = st.text_input("Player name (exact as in your player sheets)")
opponent_team = st.text_input("Opponent team name (must match 'Team' in defense sheets)")

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

# ====== FIND PLAYER IN SHEETS ======
player_df = None
player_pos = None

if player_name.lower() in p_rec["Player"].str.lower().values:
    player_df = p_rec[p_rec["Player"].str.lower() == player_name.lower()].copy()
    player_pos = player_df.iloc[0].get("Position", "WR")
elif player_name.lower() in p_rush["Player"].str.lower().values:
    player_df = p_rush[p_rush["Player"].str.lower() == player_name.lower()].copy()
    player_pos = player_df.iloc[0].get("Position", "RB")
elif player_name.lower() in p_pass["Player"].str.lower().values:
    player_df = p_pass[p_pass["Player"].str.lower() == player_name.lower()].copy()
    player_pos = "QB"
else:
    st.error("❌ Player not found in receiving, rushing, or passing sheets.")
    st.stop()

# ====== HELPER FUNCTIONS ======
def run_yardage_model(player_df, player_pos, prop_type, opponent_team):
    defense_df = None
    td_col_player = None
    td_col_def = None

    if prop_type == "passing_yards":
        stat_col = "Passing_Yds" if "Passing_Yds" in player_df.columns else "Passing_Yards"
        defense_df = d_qb
        td_col_player = "Passing_TDs" if "Passing_TDs" in player_df.columns else None
        td_col_def = "Pass_TDs_Allowed"
    elif prop_type == "rushing_yards":
        stat_col = "Rush_Yds" if "Rush_Yds" in player_df.columns else "Rushing_Yards"
        defense_df = d_rb if player_pos != "QB" else d_qb
        td_col_player = "Rushing_TDs" if "Rushing_TDs" in player_df.columns else None
        td_col_def = "Rush_TDs_Allowed"
    elif prop_type == "receiving_yards":
        stat_col = "Rec_Yds" if "Rec_Yds" in player_df.columns else "Receiving_Yards"
        defense_df = d_wr if player_pos not in ["RB", "TE"] else (d_rb if player_pos == "RB" else d_te)
        td_col_player = "Receiving_TDs" if "Receiving_TDs" in player_df.columns else None
        td_col_def = "Rec_TDs_Allowed"
    elif prop_type in ["receptions", "targets"]:
        stat_col = "Rec" if prop_type == "receptions" else "Tgt"
        defense_df = d_wr
    elif prop_type == "carries":
        stat_col = "Carries" if "Carries" in player_df.columns else "Rush_Att"
        defense_df = d_rb
    else:
        return None

    merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    merged["rolling_avg_3"] = merged[stat_col].rolling(3, 1).mean()
    season_avg = merged[stat_col].mean()

    allowed_cols = [c for c in defense_df.columns if "Allowed" in c]
    def_col = allowed_cols[0] if allowed_cols else None
    X = merged[["rolling_avg_3"] + ([def_col] if def_col else [])].fillna(0)
    y = merged[stat_col].fillna(0)
    if len(X) < 2:
        return None

    model = LinearRegression().fit(X, y)
    opp_row = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    rolling_next = merged[stat_col].tail(3).mean()
    feat = [rolling_next]
    if def_col:
        feat.append(float(opp_row.iloc[0][def_col]))
    pred = model.predict([feat])[0]
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    return dict(pred=pred, rmse=rmse, season_avg=season_avg, last3=rolling_next, merged=merged, stat_col=stat_col)

def estimate_anytime_td(player_df, defense_df, opponent_team, td_col_player, td_col_def):
    if td_col_player not in player_df.columns:
        return None
    td_series = player_df[td_col_player].fillna(0)
    y = (td_series > 0).astype(int)
    merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    merged["td_rolling_3"] = td_series.rolling(3, 1).mean()
    X = merged[["td_rolling_3"]].fillna(0)
    if td_col_def in defense_df.columns:
        X[td_col_def] = merged[td_col_def].fillna(0)
    if y.sum() == 0 or len(X) < 2:
        return y.mean()
    model = LogisticRegression()
    model.fit(X, y)
    opp = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    next_feat = [td_series.tail(3).mean()]
    if td_col_def in opp.columns:
        next_feat.append(float(opp.iloc[0][td_col_def]))
    return model.predict_proba([next_feat])[0][1]

# ====== RUN AND DISPLAY ======
st.header("📊 Results")

for prop in selected_props:
    if prop == "anytime_td":
        continue
    res = run_yardage_model(player_df, player_pos, prop, opponent_team)
    if not res:
        st.warning(f"Couldn't model {prop}.")
        continue
    line = lines[prop]
    z = (line - res["pred"]) / res["rmse"] if res["rmse"] > 0 else 0
    p_over = 1 - norm.cdf(z)
    p_under = 1 - p_over
    st.subheader(f"{prop.title().replace('_', ' ')}")
    st.write(f"**Predicted:** {res['pred']:.1f} | **Line:** {line} | **Over:** {p_over*100:.1f}% | **Under:** {p_under*100:.1f}%")
    fig, ax = plt.subplots()
    ax.bar(["Predicted", "Line"], [res["pred"], line], color=["skyblue", "salmon"])
    st.pyplot(fig)

if "anytime_td" in selected_props:
    def_df = d_qb if player_pos == "QB" else d_rb if player_pos == "RB" else d_wr if player_pos == "WR" else d_te
    td_col_def = "Pass_TDs_Allowed" if player_pos == "QB" else "Rush_TDs_Allowed" if player_pos == "RB" else "Rec_TDs_Allowed"
    td_col_player = "Passing_TDs" if player_pos == "QB" else "Rushing_TDs" if player_pos == "RB" else "Receiving_TDs"
    td_prob = estimate_anytime_td(player_df, def_df, opponent_team, td_col_player, td_col_def)
    st.subheader("🔥 Anytime TD Probability")
    st.write(f"**Chance to score a TD:** {td_prob*100:.1f}%")
