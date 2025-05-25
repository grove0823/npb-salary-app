import streamlit as st
import pandas as pd

# エンコーディングを指定して読み込み（← ここが重要）
batters = pd.read_csv("batters_with_theory.csv", encoding="cp932")
pitchers = pd.read_csv("pitchers_with_theory.csv", encoding="cp932")

player_type = st.selectbox("ポジションを選択", ["野手", "投手"])
df = batters if player_type == "野手" else pitchers

teams = sorted(df["team"].dropna().unique())
team = st.selectbox("チームを選択", teams)

filtered = df[df["team"] == team]
players = filtered["選手名"].dropna().unique()
player = st.selectbox("選手を選択", players)

data = filtered[filtered["選手名"] == player].iloc[0]

st.title(f"{player}（{team} / {player_type}）")
st.markdown(f"- ?? **実年俸**：{int(da
