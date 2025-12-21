import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# 1. Setup Page
st.set_page_config(page_title="Phone Recommender", layout="wide")

# 2. Data Processing (Wrapped in cache for speed)
@st.cache_data
def load_and_preprocess_data():
    original_df = pd.read_csv('mobile.csv')
    df = original_df.copy()
    
    # Drop unwanted cols
    cols_to_drop = ['img','tag','fm','sim','memoryExternal']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
    
    # Fill nulls
    for col in ['processor', 'storage', 'battery', 'display', 'camera', 'version']:
        df[col] = df[col].fillna(f'unknown {col}')
    
    df.drop_duplicates(inplace=True)
    
    # String cleaning
    df['processor'] = df['processor'].str.replace(",","").str.replace("Processor","")
    df['storage'] = df['storage'].astype(str).str.lower().str.replace('inbuilt', '', regex=False).str.replace(',', '', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['battery'] = df['battery'].str.split('Battery').str[0]
    df['display'] = df['display'].str.split(",").str[0]
    df['camera'] = df['camera'].str.split('&').str[1].fillna('')
    df['camera'] = df['camera'].str.split('Front Camera').str[0]
    
    # Version cleaning
    df['version'] = df['version'].replace(['unknown version', 'v30'], 'unknown')
    df['version'] = df['version'].str.replace(r'\(.*\)', '', regex=True).str.replace(' v', ' ', regex=False).str.replace(r'\.\d+', '', regex=True).str.strip()
    
    for c in ['processor','storage','battery','camera','version']:
        df[c] = df[c].str.lower()

    # Feature extraction
    df['proc_value'] = df['processor'].str.extract(r'(\d+\.?\d*)\s*ghz').astype(float).fillna(0)
    df['ram_val'] = df['storage'].str.extract(r'(\d+)\s*gb\s*ram').astype(float).fillna(0)
    df['stor_val'] = df['storage'].str.extract(r'(\d+)\s*gb(?!\s*ram)').astype(float).fillna(0)
    df['batt_val'] = df['battery'].str.extract(r'(\d+)\s*mah').astype(float).fillna(0)
    df['cam_val'] = df['camera'].str.extract(r'(\d+)').astype(float).fillna(0)

    # Scaling
    cols_to_scale = ['proc_value', 'ram_val', 'stor_val', 'batt_val', 'cam_val']
    scaler = MinMaxScaler()
    df[[c + '_norm' for c in cols_to_scale]] = scaler.fit_transform(df[cols_to_scale])
    
    return df

df = load_and_preprocess_data()

# 3. Weights
weights = {
    'Gaming': {'proc':0.4,'rm':0.3,'strg':0.1,'bttr':0.1,'cmr':0.1},
    'Photography': {'proc':0.1,'rm':0.1,'strg':0.2,'bttr':0.1,'cmr':0.5},
    'Balanced':{'proc':0.2,'rm':0.2,'strg':0.2,'bttr':0.2,'cmr':0.2}
}

# 4. Recommendation Function
def get_recommendations(user_budget, user_intent, user_os):
    mask = (df['price'] <= user_budget) & (df['version'].str.contains(user_os, case=False, na=False))
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        st.error(f"No {user_os} phones found under ₹{user_budget:,}")
        return

    w = weights[user_intent]
    filtered_df['final_score'] = (
        filtered_df['proc_value_norm'] * w['proc'] +
        filtered_df['ram_val_norm'] * w['rm'] +
        filtered_df['stor_val_norm'] * w['strg'] +
        filtered_df['batt_val_norm'] * w['bttr'] +
        filtered_df['cam_val_norm'] * w['cmr']
    )
    
    top_5 = filtered_df.sort_values(by='final_score', ascending=False).head(5)
    
    st.markdown(f"### Top {user_os} Recommendations for {user_intent}")
    for idx, row in top_5.iterrows():
        with st.expander(f"{row['Name']} - ₹{row['price']:,}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**OS:** {row['version'].title()}")
                st.write(f"**Processor:** {row['processor']}")
                st.write(f"**Battery:** {row['battery']}")
            with col2:
                st.write(f"**RAM/Storage:** {row['storage']}")
                st.write(f"**Camera:** {row['camera']}")

# 5. UI Elements
st.title("📱 SmartPhone Finder")
st.sidebar.header("Filter Preferences")

user_budget = st.sidebar.slider("Your Budget (₹)", 5000, 150000, 50000, step=1000)
user_intent = st.sidebar.selectbox("Priority", list(weights.keys()))
user_os = st.sidebar.selectbox("OS Type", ["Android", "Ios"])

if st.sidebar.button("Recommend Me"):
    get_recommendations(user_budget, user_intent, user_os)
else:
    st.info("Adjust the filters in the sidebar and click 'Recommend Me' to see results.")

