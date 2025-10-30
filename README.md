# 🏈 NFL Player Prop Model (v4)

This Streamlit web app predicts NFL player prop outcomes using live Google Sheets data. 
It allows you to analyze single players across multiple prop types — such as passing, rushing, and receiving yards — 
and calculates probabilities for over/under and anytime touchdown bets.

## ⚙️ Features
- Connects directly to 10 Google Sheets for live data updates
- Predicts player stat performance (yards, receptions, carries, etc.)
- Calculates probability of hitting the sportsbook line (Over/Under)
- Estimates Anytime TD probability using logistic regression
- Displays visual charts for performance trends and matchup data

## 🚀 How to Run Locally
1. Clone this repo  
   ```bash
   git clone https://github.com/<your-username>/nfl-prop-model.git
   cd nfl-prop-model
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app  
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment on Streamlit Cloud
- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Connect your GitHub repo
- Select `app.py` as your main file
- Click **Deploy**

Your app will automatically sync with updates to your Google Sheets.
