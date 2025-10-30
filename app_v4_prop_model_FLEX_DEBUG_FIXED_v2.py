# app_v4_prop_model_FLEX_DEBUG_FIXED_v2.py
# Clean and syntax-safe version ready for Streamlit Cloud deployment.
# Features: debug info, 11 Google Sheets, player column detection with 2nd column fallback.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFL Player Prop Model (Flex + Debug Fixed)", layout="centered")

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

# ====== LOAD HELPERS ======
def load_sheet(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def find_player_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "player" in col.lower() or "name" in col.lower():
            return col
    if len(df.columns) >= 2:
        return df.columns[1]
    return df.columns[0]

# ====== LOAD SHEETS ======
try:
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

rec_player_col = find_player_column(p_rec)
rush_player_col = find_player_column(p_rush)
pass_player_col = find_player_column(p_pass)

with st.sidebar:
    st.header("🔎 Debug info")
    st.write("Receiving player column:", rec_player_col)
    st.write("Rushing player column:", rush_player_col)
    st.write("Passing player column:", pass_player_col)
    st.write("If any look wrong, rename the column or move player names to column 2.")

# ====== MAIN APP ======
st.title("🏈 NFL Player Prop Model (Flex + Debug Fixed)")
st.write("Auto-detects player name column and supports Over/Under & Anytime TD analysis.")

player_name = st.text_input("Enter player name:")
opponent_team = st.text_input("Enter opponent team:")
prop_options = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "carries", "anytime_td"]
selected_props = st.multiselect("Select props", prop_options, default=["rushing_yards"])

lines = {}
for prop in selected_props:
    if prop != "anytime_td":
        lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

if not player_name or not opponent_team or not selected_props:
    st.stop()

# ====== PLAYER LOOKUP ======
player_df = None
player_pos = None

if player_name.lower() in p_rec[rec_player_col].astype(str).str.lower().values:
    player_df = p_rec[p_rec[rec_player_col].astype(str).str.lower() == player_name.lower()]
    player_pos = "WR"
elif player_name.lower() in p_rush[rush_player_col].astype(str).str.lower().values:
    player_df = p_rush[p_rush[rush_player_col].astype(str).str.lower() == player_name.lower()]
    player_pos = "RB"
elif player_name.lower() in p_pass[pass_player_col].astype(str).str.lower().values:
    player_df = p_pass[p_pass[pass_player_col].astype(str).str.lower() == player_name.lower()]
    player_pos = "QB"
else:
    st.error("Player not found in any data sheets.")
    st.stop()

# ====== MODEL ======
def run_yardage_model(player_df, prop_type, opponent_team):
    defense_df = d_qb if prop_type == "passing_yards" else d_rb if prop_type in ["rushing_yards", "carries"] else d_wr
    stat_col = None
    for c in player_df.columns:
        if prop_type.split("_")[0] in c.lower() and ("yard" in c.lower() or "rec" in c.lower() or "rush" in c.lower()):
            stat_col = c
            break
    if not stat_col:
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
    opp_row = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    if opp_row.empty:
        return None

    rolling_next = merged[stat_col].tail(3).mean()
    feat_next = [rolling_next]
    if def_col:
        feat_next.append(float(opp_row.iloc[0][def_col]))
    pred_next = model.predict([feat_next])[0]
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    return dict(pred=pred_next, rmse=rmse, season_avg=season_avg, last3=rolling_next, stat_col=stat_col)

# ====== RESULTS ======
st.header("📊 Results")

for prop in selected_props:
    if prop == "anytime_td":
        continue
    res = run_yardage_model(player_df, prop, opponent_team)
    if not res:
        st.warning(f"Could not model {prop}.")
        continue
    line_val = lines[prop]
    z = (line_val - res["pred"]) / res["rmse"] if res["rmse"] > 0 else 0
    p_over = 1 - norm.cdf(z)
    p_under = norm.cdf(z)

    st.subheader(f"{prop.title().replace('_', ' ')}")
    st.write(f"**Predicted:** {res['pred']:.1f} | **Line:** {line_val} | **Over:** {p_over*100:.1f}% | **Under:** {p_under*100:.1f}%")

    fig, ax = plt.subplots()
    ax.bar(["Predicted", "Line"], [res["pred"], line_val], color=["skyblue", "salmon"])
    ax.set_title(prop.title())
    st.pyplot(fig)

# Anytime TD
if "anytime_td" in selected_props:
    defense_df = d_qb if player_pos == "QB" else d_rb if player_pos == "RB" else d_te if player_pos == "TE" else d_wr
    td_col = next((c for c in player_df.columns if "TD" in c), None)
    if td_col:
        td_series = player_df[td_col].fillna(0)
        y = (td_series > 0).astype(int)
        if y.sum() > 0:
            lg = LogisticRegression()
            X = np.arange(len(y)).reshape(-1, 1)
            lg.fit(X, y)
            td_prob = lg.predict_proba([[len(y)+1]])[0][1]
            st.subheader("🔥 Anytime TD")
            st.write(f"**Anytime TD probability:** {td_prob*100:.1f}%")
        else:
            st.warning("Not enough TD data for this player.")
    else:
        st.warning("No TD column found for this player.")
