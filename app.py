import streamlit as st
import pandas as pd

# Windowsで保存された日本語CSVファイルは cp932 で読む
batters = pd.read_csv("batters_with_salary.csv", encoding="utf-8")
pitchers = pd.read_csv("pitchers_with_salary.csv", encoding="utf-8")

player_type = st.selectbox("ポジションを選択", ["野手", "投手"])
df = batters if player_type == "野手" else pitchers

teams = sorted(df["team"].dropna().unique())
team = st.selectbox("チームを選択", teams)

filtered = df[df["team"] == team]
players = filtered["選手名"].dropna().unique()
player = st.selectbox("選手を選択", players)

data = filtered[filtered["選手名"] == player].iloc[0]

st.title(f"{player}（{team} / {player_type}）")
st.markdown(f"- ?? **実年俸**：{int(data['salary']):,}万円")
st.markdown(f"- ?? **理論年俸**：{int(data['theoretical_salary']):,}万円")

if player_type == "野手":
    st.subheader("打撃成績")
    st.markdown(f"- 打数：{data['打数']}")
    st.markdown(f"- 安打：{data['安打']}")
    st.markdown(f"- 本塁打：{data['本塁打']}")
    st.markdown(f"- OPS：{data['OPS']}")
else:
    st.subheader("投手成績")
    st.markdown(f"- 投球回：{data['投球回']}")
    st.markdown(f"- 奪三振：{data['奪三振']}")
    st.markdown(f"- 防御率：{data['防御率']}")
    st.markdown(f"- WHIP：{data['WHIP']}")
