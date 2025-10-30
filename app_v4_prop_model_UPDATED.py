# app_v4_prop_model_UPDATED.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFL Player Prop Model (v4) - Updated", layout="centered")

# ====== UPDATED GOOGLE SHEET EXPORT LINKS ======
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
except Exception as e:
    st.error(f"❌ Error loading one or more Google Sheets. Details: {e}")
    st.stop()

st.title("🏈 NFL Player Prop Model (v4)")
st.write("Now running with verified Google Sheets export URLs. You can select multiple props, calculate over/under probabilities, and see anytime TD odds.")

# ====== USER INPUTS ======
player_name = st.text_input("Player name (exact as in your player sheets)")
opponent_team = st.text_input("Opponent team name (must match 'Team' in defense sheets)")

prop_options = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "carries", "anytime_td"]
selected_props = st.multiselect("Select props to evaluate", prop_options, default=["rushing_yards"])

lines = {}
for prop in selected_props:
    if prop == "anytime_td":
        continue
    lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

if not player_name or not opponent_team or not selected_props:
    st.stop()

# ====== DETERMINE PLAYER SOURCE ======
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
    if prop_type == "passing_yards":
        defense_df = d_qb
        stat_col = "Passing_Yards" if "Passing_Yards" in player_df.columns else "Passing_Yds"
    elif prop_type == "rushing_yards":
        defense_df = d_rb if player_pos != "QB" else d_qb
        stat_col = "Rushing_Yards" if "Rushing_Yards" in player_df.columns else "Rush_Yds"
    elif prop_type == "receiving_yards":
        defense_df = d_wr if player_pos not in ["RB", "TE"] else (d_rb if player_pos == "RB" else d_te)
        stat_col = "Receiving_Yards" if "Receiving_Yards" in player_df.columns else "Rec_Yds"
    elif prop_type in ["receptions", "targets"]:
        defense_df = d_wr
        stat_col = "Receptions" if prop_type == "receptions" else "Targets"
    elif prop_type == "carries":
        defense_df = d_rb
        stat_col = "Carries" if "Carries" in player_df.columns else "Rush_Att"
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

def estimate_anytime_td(player_df, defense_df, opponent_team):
    td_col = "TDs" if "TDs" in player_df.columns else None
    if not td_col:
        return None
    td_series = player_df[td_col].fillna(0)
    y = (td_series > 0).astype(int)
    merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    merged["td_rolling_3"] = td_series.rolling(3, 1).mean()
    X = merged[["td_rolling_3"]].fillna(0)
    if y.sum() == 0 or len(X) < 2:
        return y.mean()
    model = LogisticRegression()
    model.fit(X, y)
    opp = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    next_feat = [td_series.tail(3).mean()]
    return model.predict_proba([next_feat])[0][1]

# ====== RUN ======
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
    td_prob = estimate_anytime_td(player_df, def_df, opponent_team)
    st.subheader("🔥 Anytime TD Probability")
    st.write(f"**Chance to score a TD:** {td_prob*100:.1f}%")
