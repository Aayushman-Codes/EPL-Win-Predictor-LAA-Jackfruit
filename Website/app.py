#pip install streamlit xgboost pandas numpy scikit-learn
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np

# --- 1. Bulletproof HTML Renderer ---
# This stops VS Code's auto-formatter from breaking Streamlit
def render_safe_html(raw_string):
    # This removes all leading spaces from every line so Streamlit NEVER treats it as a code block
    clean_string = "\n".join([line.strip() for line in raw_string.split("\n")])
    st.markdown(clean_string, unsafe_allow_html=True)

# --- 2. Page Configuration ---
st.set_page_config(page_title="EPL Match Predictor", layout="centered")

# --- 3. Inject Custom CSS ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif !important;
        background-color: #0d1117 !important;
        color: #f4f1eb !important; 
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .custom-header {
        background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 50%, #0a1628 100%);
        border-bottom: 1px solid #30363d;
        padding: 2rem 2rem 1.5rem;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .header-badge {
        display: inline-block;
        background: rgba(240,180,41,0.12);
        border: 1px solid rgba(240,180,41,0.3);
        color: #7c12de; 
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
    }
    .custom-header h1 {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        letter-spacing: 3px;
        color: #f4f1eb;
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    .custom-header h1 span { color: #7c12de; }
    .custom-header p {
        color: #8b949e;
        font-size: 14px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .result-section {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.75rem;
        margin-top: 2rem;
    }
    .card-title {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title::after { content: ''; flex: 1; height: 1px; background: #30363d; }

    .teams-display {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .team-display-name { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 1.5px; line-height: 1; }
    .team-display-name.home { color: #4a9eff; text-align: left; }
    .team-display-name.away { color: #ff6b6b; text-align: right; }
    .result-vs { font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; color: #30363d; letter-spacing: 3px; text-align: center; }

    .win-bar-container { background: #0d1117; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
    .win-bar-labels { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .win-bar-label { font-size: 13px; font-weight: 500; }
    .win-bar-label.home { color: #4a9eff; }
    .win-bar-label.away { color: #ff6b6b; }
    .win-bar-track { height: 8px; background: #1c2129; border-radius: 4px; overflow: hidden; display: flex; }

    .win-bar-home { height: 100%; background: #4a9eff; transition: width 0.8s; }
    .win-bar-draw { height: 100%; background: #8b949e; transition: width 0.8s; }
    .win-bar-away { height: 100%; background: #ff6b6b; transition: width 0.8s; }

    .win-bar-pcts { display: flex; justify-content: space-between; margin-top: 6px; }
    .win-bar-pct { font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 1px; }
    .win-bar-pct.home { color: #4a9eff; }
    .win-bar-pct.draw { color: #8b949e; text-align: center; flex: 1; }
    .win-bar-pct.away { color: #ff6b6b; }

    .verdict-box {
        background: rgba(240,180,41,0.08);
        border: 1px solid rgba(240,180,41,0.2);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        text-align: center;
        justify-content: center;
    }
    .verdict-text { font-size: 15px; color: #e8e4dc; line-height: 1.5; }
    .verdict-text strong { color: #7c12de; font-weight: 600; }
    .confidence-pill {
        display: inline-block; border-radius: 20px; font-size: 11px; font-weight: 500;
        padding: 2px 10px; letter-spacing: 0.5px; margin-left: 6px; vertical-align: middle;
    }
    .conf-high   { background: rgba(74,180,94,0.15); color: #4ab45e; border: 1px solid rgba(74,180,94,0.3); }
    .conf-medium { background: rgba(240,180,41,0.15); color: #c8952a; border: 1px solid rgba(240,180,41,0.3); }
    .conf-low    { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); }

    .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 1rem; }
    .stat-cell { background: #0d1117; border-radius: 8px; padding: 10px 12px; text-align: center; }
    .stat-cell-label { font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: #8b949e; margin-bottom: 4px; }
    .stat-cell-value { font-size: 18px; font-weight: 600; color: #f4f1eb; }
</style>
"""
render_safe_html(custom_css)

# --- 4. Load the Trained Model ---
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('epl_xgboost_model.json')
    return model

model = load_model()

# --- 5. Database of Team Stats ---
TEAM_STATS = {
  "Arsenal":{"htp":62,"atp":62,"htFormPts":9,"atFormPts":9,"htgd":28,"atgd":25,"htgs":45,"atgs":40,"htgc":28,"atgc":32,"hwStreak3":1,"awStreak3":1,"hlStreak3":0,"alStreak3":0,"hwRate":0.57,"awRate":0.42},
  "Aston Villa":{"htp":50,"atp":50,"htFormPts":7,"atFormPts":7,"htgd":5,"atgd":2,"htgs":38,"atgs":35,"htgc":38,"atgc":40,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.44,"awRate":0.38},
  "Birmingham":{"htp":40,"atp":40,"htFormPts":5,"atFormPts":5,"htgd":-5,"atgd":-8,"htgs":30,"atgs":28,"htgc":38,"atgc":42,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "Blackburn":{"htp":44,"atp":44,"htFormPts":6,"atFormPts":6,"htgd":2,"atgd":-2,"htgs":34,"atgs":30,"htgc":33,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.40,"awRate":0.33},
  "Blackpool":{"htp":36,"atp":36,"htFormPts":5,"atFormPts":5,"htgd":-10,"atgd":-12,"htgs":28,"atgs":25,"htgc":40,"atgc":44,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.35,"awRate":0.27},
  "Bolton":{"htp":42,"atp":42,"htFormPts":6,"atFormPts":6,"htgd":-2,"atgd":-5,"htgs":32,"atgs":28,"htgc":35,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.39,"awRate":0.31},
  "Bournemouth":{"htp":42,"atp":42,"htFormPts":6,"atFormPts":6,"htgd":0,"atgd":-3,"htgs":33,"atgs":30,"htgc":34,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.40,"awRate":0.32},
  "Bradford":{"htp":32,"atp":32,"htFormPts":4,"atFormPts":4,"htgd":-15,"atgd":-18,"htgs":24,"atgs":22,"htgc":42,"atgc":46,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.30,"awRate":0.22},
  "Brighton":{"htp":40,"atp":40,"htFormPts":5,"atFormPts":5,"htgd":-3,"atgd":-6,"htgs":30,"atgs":27,"htgc":34,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "Burnley":{"htp":42,"atp":42,"htFormPts":6,"atFormPts":6,"htgd":-1,"atgd":-4,"htgs":31,"atgs":28,"htgc":33,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.41,"awRate":0.32},
  "Cardiff":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-8,"atgd":-10,"htgs":28,"atgs":25,"htgc":38,"atgc":42,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.36,"awRate":0.28},
  "Charlton":{"htp":42,"atp":42,"htFormPts":6,"atFormPts":6,"htgd":-1,"atgd":-3,"htgs":32,"atgs":29,"htgc":34,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.40,"awRate":0.32},
  "Chelsea":{"htp":70,"atp":70,"htFormPts":11,"atFormPts":11,"htgd":38,"atgd":35,"htgs":55,"atgs":50,"htgc":24,"atgc":28,"hwStreak3":1,"awStreak3":1,"hlStreak3":0,"alStreak3":0,"hwRate":0.63,"awRate":0.50},
  "Coventry":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-6,"atgd":-8,"htgs":28,"atgs":25,"htgc":36,"atgc":40,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.36,"awRate":0.28},
  "Crystal Palace":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "Derby":{"htp":36,"atp":36,"htFormPts":5,"atFormPts":5,"htgd":-10,"atgd":-12,"htgs":26,"atgs":23,"htgc":38,"atgc":42,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.34,"awRate":0.26},
  "Everton":{"htp":54,"atp":54,"htFormPts":8,"atFormPts":8,"htgd":10,"atgd":8,"htgs":42,"atgs":38,"htgc":36,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.48,"awRate":0.38},
  "Fulham":{"htp":44,"atp":44,"htFormPts":6,"atFormPts":6,"htgd":0,"atgd":-3,"htgs":34,"atgs":30,"htgc":36,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.40,"awRate":0.32},
  "Huddersfield":{"htp":36,"atp":36,"htFormPts":5,"atFormPts":5,"htgd":-12,"atgd":-14,"htgs":25,"atgs":22,"htgc":38,"atgc":42,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.33,"awRate":0.25},
  "Hull":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-7,"atgd":-9,"htgs":28,"atgs":25,"htgc":37,"atgc":40,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.36,"awRate":0.28},
  "Ipswich":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "Leeds":{"htp":46,"atp":46,"htFormPts":7,"atFormPts":7,"htgd":5,"atgd":2,"htgs":36,"atgs":32,"htgc":33,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.43,"awRate":0.35},
  "Leicester":{"htp":50,"atp":50,"htFormPts":8,"atFormPts":8,"htgd":8,"atgd":6,"htgs":40,"atgs":36,"htgc":34,"atgc":36,"hwStreak3":0,"awStreak3":1,"hlStreak3":0,"alStreak3":0,"hwRate":0.46,"awRate":0.38},
  "Liverpool":{"htp":68,"atp":68,"htFormPts":11,"atFormPts":11,"htgd":36,"atgd":32,"htgs":54,"atgs":48,"htgc":24,"atgc":28,"hwStreak3":1,"awStreak3":1,"hlStreak3":0,"alStreak3":0,"hwRate":0.62,"awRate":0.50},
  "Man City":{"htp":72,"atp":72,"htFormPts":12,"atFormPts":12,"htgd":42,"atgd":38,"htgs":60,"atgs":54,"htgc":22,"atgc":26,"hwStreak3":1,"awStreak3":1,"hlStreak3":0,"alStreak3":0,"hwRate":0.65,"awRate":0.52},
  "Man United":{"htp":74,"atp":74,"htFormPts":12,"atFormPts":12,"htgd":44,"atgd":40,"htgs":60,"atgs":54,"htgc":22,"atgc":26,"hwStreak3":1,"awStreak3":1,"hlStreak3":0,"alStreak3":0,"hwRate":0.66,"awRate":0.54},
  "Middlesboro":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-5,"atgd":-8,"htgs":28,"atgs":25,"htgc":35,"atgc":39,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.37,"awRate":0.29},
  "Middlesbrough":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.39,"awRate":0.31},
  "Newcastle":{"htp":52,"atp":52,"htFormPts":8,"atFormPts":8,"htgd":10,"atgd":7,"htgs":42,"atgs":38,"htgc":36,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.47,"awRate":0.37},
  "Norwich":{"htp":36,"atp":36,"htFormPts":5,"atFormPts":5,"htgd":-10,"atgd":-12,"htgs":27,"atgs":24,"htgc":38,"atgc":42,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.34,"awRate":0.26},
  "Portsmouth":{"htp":44,"atp":44,"htFormPts":6,"atFormPts":6,"htgd":0,"atgd":-3,"htgs":34,"atgs":30,"htgc":36,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.40,"awRate":0.32},
  "QPR":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-6,"atgd":-8,"htgs":28,"atgs":25,"htgc":36,"atgc":40,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.36,"awRate":0.28},
  "Reading":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "Sheffield United":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-5,"atgd":-8,"htgs":28,"atgs":25,"htgc":35,"atgc":39,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.37,"awRate":0.29},
  "Southampton":{"htp":44,"atp":44,"htFormPts":7,"atFormPts":7,"htgd":2,"atgd":0,"htgs":34,"atgs":31,"htgc":34,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.41,"awRate":0.33},
  "Stoke":{"htp":44,"atp":44,"htFormPts":6,"atFormPts":6,"htgd":0,"atgd":-3,"htgs":32,"atgs":28,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.40,"awRate":0.31},
  "Sunderland":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "Swansea":{"htp":42,"atp":42,"htFormPts":6,"atFormPts":6,"htgd":-1,"atgd":-3,"htgs":32,"atgs":29,"htgc":34,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.39,"awRate":0.31},
  "Tottenham":{"htp":60,"atp":60,"htFormPts":10,"atFormPts":10,"htgd":22,"atgd":19,"htgs":50,"atgs":45,"htgc":30,"atgc":33,"hwStreak3":1,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.55,"awRate":0.44},
  "Watford":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30},
  "West Brom":{"htp":42,"atp":42,"htFormPts":6,"atFormPts":6,"htgd":-1,"atgd":-3,"htgs":32,"atgs":29,"htgc":34,"atgc":36,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.39,"awRate":0.31},
  "West Ham":{"htp":48,"atp":48,"htFormPts":7,"atFormPts":7,"htgd":5,"atgd":3,"htgs":38,"atgs":34,"htgc":36,"atgc":38,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":0,"hwRate":0.44,"awRate":0.35},
  "Wigan":{"htp":38,"atp":38,"htFormPts":5,"atFormPts":5,"htgd":-6,"atgd":-8,"htgs":28,"atgs":25,"htgc":36,"atgc":40,"hwStreak3":0,"awStreak3":0,"hlStreak3":1,"alStreak3":1,"hwRate":0.36,"awRate":0.28},
  "Wolves":{"htp":40,"atp":40,"htFormPts":6,"atFormPts":6,"htgd":-3,"atgd":-5,"htgs":30,"atgs":27,"htgc":34,"atgc":37,"hwStreak3":0,"awStreak3":0,"hlStreak3":0,"alStreak3":1,"hwRate":0.38,"awRate":0.30}
}

# --- 6. Custom Header ---
header_html = """
<div class="custom-header">
  <div class="header-badge">XGBoost Prediction Model</div>
  <h1>EPL Match <span>Predictor</span></h1>
  <p>Tree-based model trained on 6,840 Premier League matches (2000–2019)</p>
</div>
"""
render_safe_html(header_html)

# --- 7. User Inputs ---
col1, col_vs, col2 = st.columns([2, 0.5, 2])

with col1:
    team_a = st.selectbox("Team A", list(TEAM_STATS.keys()), index=0)
with col_vs:
    st.markdown('<div style="text-align: center; margin-top: 30px; font-family: \'Bebas Neue\', sans-serif; font-size: 2rem; color: #8b949e;">VS</div>', unsafe_allow_html=True)
with col2:
    team_b = st.selectbox("Team B", list(TEAM_STATS.keys()), index=12)

venue = st.radio(
    "Venue",
    ["Team A Home Ground", "Neutral Ground", "Team B Home Ground"],
    horizontal=True,
    index=0
)

# --- 8. Prediction Function ---
def get_model_prediction(home_team, away_team):
    home = TEAM_STATS[home_team]
    away = TEAM_STATS[away_team]
    
    points_diff = home['htp'] - away['atp']
    form_diff = home['htFormPts'] - away['atFormPts']
    gd_diff = home['htgd'] - away['atgd']
    
    input_features = pd.DataFrame([{
        'HTP': home['htp'], 
        'HTFormPts': home['htFormPts'], 
        'HTGD': home['htgd'], 
        'HTWinStreak3': home['hwStreak3'], 
        'HTWinStreak5': 0, 
        'HTLossStreak3': home['hlStreak3'],
        'HTGS': home['htgs'], 
        'HTGC': home['htgc'], 
        'ATP': away['atp'], 
        'ATFormPts': away['atFormPts'], 
        'ATGD': away['atgd'], 
        'ATWinStreak3': away['awStreak3'], 
        'ATWinStreak5': 0, 
        'ATLossStreak3': away['alStreak3'],
        'ATGS': away['atgs'], 
        'ATGC': away['atgc'],
        'DiffPts': points_diff, 
        'DiffFormPts': form_diff,
        'Points_Diff': points_diff, 
        'Form_Diff': form_diff, 
        'GD_Diff': gd_diff
    }])
    probs = model.predict_proba(input_features)[0]
    return probs[1], probs[0] 

# --- 9. Execute Logic ---
if st.button("Predict Match Outcome", use_container_width=True):
    if team_a == team_b:
        st.error("Please select two different teams.")
    else:
        if venue == "Team A Home Ground":
            pA, pB = get_model_prediction(team_a, team_b)
        elif venue == "Team B Home Ground":
            pB, pA = get_model_prediction(team_b, team_a) 
        else: 
            pA1, pB1 = get_model_prediction(team_a, team_b)
            pB2, pA2 = get_model_prediction(team_b, team_a)
            pA = (pA1 + pA2) / 2
            pB = (pB1 + pB2) / 2
            
        evenness = 1 - abs(pA - 0.5) * 2
        pDraw_raw = 0.20 + 0.12 * evenness
        
        total = pA + pB + pDraw_raw
        pctA = round((pA / total) * 100)
        pctB = round((pB / total) * 100)
        pctDraw = 100 - pctA - pctB

        conf_diff = abs((pctA/100) - (pctB/100))
        if conf_diff > 0.2:
            conf_class = "conf-high"
            conf_label = "High confidence"
        elif conf_diff > 0.08:
            conf_class = "conf-medium"
            conf_label = "Medium confidence"
        else:
            conf_class = "conf-low"
            conf_label = "Tight match"

        if (pctA/100) > (pctB/100) + 0.04:
            verdict = f"<strong>{team_a}</strong> are predicted to win this match. <span class='confidence-pill {conf_class}'>{conf_label}</span>"
        elif (pctB/100) > (pctA/100) + 0.04:
            verdict = f"<strong>{team_b}</strong> are predicted to win this match. <span class='confidence-pill {conf_class}'>{conf_label}</span>"
        else:
            verdict = f"This match is predicted to be very evenly contested. A <strong>Draw</strong> is highly likely. <span class='confidence-pill {conf_class}'>{conf_label}</span>"

        # The HTML string is passed to our bulletproof renderer
        result_html = f"""
        <div class="result-section">
            <div class="card-title">Prediction Result</div>
            
            <div class="teams-display">
                <div class="team-display-name home">{team_a}</div>
                <div class="result-vs">VS</div>
                <div class="team-display-name away">{team_b}</div>
            </div>

            <div class="win-bar-container">
                <div class="win-bar-labels">
                    <span class="win-bar-label home">{team_a}</span>
                    <span class="win-bar-label" style="color: #8b949e; text-transform: uppercase; font-size: 11px; letter-spacing: 1px;">Draw</span>
                    <span class="win-bar-label away">{team_b}</span>
                </div>
                <div class="win-bar-track">
                    <div class="win-bar-home" style="width: {pctA}%;"></div>
                    <div class="win-bar-draw" style="width: {pctDraw}%;"></div>
                    <div class="win-bar-away" style="width: {pctB}%;"></div>
                </div>
                <div class="win-bar-pcts">
                    <span class="win-bar-pct home">{pctA}%</span>
                    <span class="win-bar-pct draw">{pctDraw}%</span>
                    <span class="win-bar-pct away">{pctB}%</span>
                </div>
            </div>

            <div class="verdict-box">
                <div class="verdict-text">{verdict}</div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-cell">
                    <div class="stat-cell-label">{team_a} Win %</div>
                    <div class="stat-cell-value" style="color: #4a9eff;">{pctA}%</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-cell-label">Draw %</div>
                    <div class="stat-cell-value" style="color: #8b949e;">{pctDraw}%</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-cell-label">{team_b} Win %</div>
                    <div class="stat-cell-value" style="color: #ff6b6b;">{pctB}%</div>
                </div>
            </div>
        </div>
        """
        
        # This function call ensures it renders correctly no matter how VS Code indents it!
        render_safe_html(result_html)