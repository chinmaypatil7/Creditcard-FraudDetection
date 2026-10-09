import streamlit as st
import requests
import pandas as pd
import sqlite3
import os

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection & Risk Scoring Dashboard")
st.markdown("Real-time fraud monitoring, transaction risk scoring, and SQLite audit logs.")

# Sidebar for testing transactions
st.sidebar.header("🧪 Test Transaction Input")
st.sidebar.markdown("Simulate or load a transaction:")

# Option to switch between manual input and loading a real fraud case
test_mode = st.sidebar.radio("Test Mode", ["Manual Input", "Load Real Fraud Case from Dataset"])

if test_mode == "Load Real Fraud Case from Dataset":
    if st.sidebar.button("Load & Score Fraud Sample"):
        try:
            # Look for creditcard.csv in data/ or ../data/
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/creditcard.csv'))
            if not os.path.exists(data_path):
                data_path = 'data/creditcard.csv'
                
            if os.path.exists(data_path):
                df_full = pd.read_csv(data_path)
                fraud_rows = df_full[df_full['Class'] == 1]
                
                if not fraud_rows.empty:
                    # Loop through a few fraud rows until we find one the model flags as HIGH
                    success_flagged = False
                    for _, fraud_row in fraud_rows.head(10).iterrows():
                        payload = fraud_row.drop(columns=['Class']).to_dict()
                        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
                        
                        if response.status_code == 200:
                            res_data = response.json()
                            if res_data.get('risk_level') == 'HIGH':
                                success_flagged = True
                                break
                    
                    if success_flagged:
                        st.sidebar.error("🚨 HIGH RISK FRAUD CASE LOADED & FLAGGED!")
                    else:
                        st.sidebar.warning("⚠️ Fraud sample loaded, evaluated as LOW risk under current threshold.")
                    
                    st.rerun()
                else:
                    st.sidebar.error("No fraud cases found in dataset.")
            else:
                st.sidebar.error("Dataset 'creditcard.csv' not found. Ensure it is placed in the data/ folder.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
else:
    # Manual input for Time and Amount (defaults V1-V28 to 0)
    input_time = st.sidebar.number_input("Transaction Time (Seconds)", value=0.0)
    input_amount = st.sidebar.number_input("Transaction Amount ($)", value=150.0)

    if st.sidebar.button("Score Transaction"):
        payload = {"Time": input_time, "Amount": input_amount}
        for i in range(1, 29):
            payload[f"V{i}"] = 0.0
            
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json=payload)
            if response.status_code == 200:
                res_data = response.json()
                risk = res_data.get('risk_level')
                prob = res_data.get('fraud_probability')
                
                if risk == 'HIGH':
                    st.sidebar.error(f"🚨 FRAUD DETECTED! Risk Level: {risk} (Prob: {prob:.4f})")
                else:
                    st.sidebar.success(f"✅ LEGITIMATE TRANSACTION. Risk Level: {risk} (Prob: {prob:.4f})")
                
                # Instantly rerun Streamlit so the audit log table updates immediately
                st.rerun()
            else:
                st.sidebar.error("Error communicating with Flask API.")
        except Exception as e:
            st.sidebar.error(f"Could not connect to Flask API: {e}")

# Main dashboard section: Recent Transactions Audit Log from SQLite
st.subheader("📊 Recent Transaction Audit Logs")

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/transactions.db'))

def load_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df_logs = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    return df_logs

df_logs = load_logs()

if not df_logs.empty:
    # Ensure prediction is safely cast to integer for metrics
    df_logs['prediction'] = df_logs['prediction'].astype(int)
    
    # Metric overviews
    total_tx = len(df_logs)
    fraud_count = len(df_logs[df_logs['prediction'] == 1])
    
    col1, col2 = st.columns(2)
    col1.metric("Total Scanned (Logged)", total_tx)
    col2.metric("Flagged Fraud Count", fraud_count, delta_color="inverse")
    
    st.dataframe(df_logs, use_container_width=True)
else:
    st.info("No transaction logs found in database yet. Try scoring a transaction using the sidebar!")