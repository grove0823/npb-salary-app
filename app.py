import streamlit as st
import pandas as pd

# CSVファイルを読み込み（エンコーディングを修正）
batters = pd.read_csv("batters_with_salary.csv", encoding="utf-8")
pitchers = pd.read_csv("pitchers_with_salary.csv", encoding="utf-8")

# ポジション選択
player_type = st.selectbox("ポジションを選択", ["野手", "投手"])

# データフレームの選択
df = batters if player_type == "野手" else pitchers

# チーム選択
teams = sorted(df["team"].dropna().unique())
team = st.selectbox("チームを選択", teams)

# 選択されたチームでフィルタリング
filtered = df[df["team"] == team]

# 選手選択
players = filtered["選手名"].dropna().unique()
player = st.selectbox("選手を選択", players)

# 選手データを取得
data = filtered[filtered["選手名"] == player].iloc[0]

# 選手情報表示
st.title(f"{player}（{team} / {player_type}）")

# 年俸情報（KeyErrorを修正）
try:
    # 実年俸（salaryまたは年俸の列名を確認）
    if 'salary' in data.index:
        st.markdown(f"- 💰 **実年俸**：{int(data['salary']):,}万円")
    elif '年俸' in data.index:
        st.markdown(f"- 💰 **実年俸**：{int(data['年俸']):,}万円")
    elif '実年俸' in data.index:
        st.markdown(f"- 💰 **実年俸**：{int(data['実年俸']):,}万円")
    
    # 理論年俸（theoretical_salaryまたは理論年俸の列名を確認）
    if 'theoretical_salary' in data.index:
        st.markdown(f"- 📊 **理論年俸**：{int(data['theoretical_salary']):,}万円")
    elif '理論年俸' in data.index:
        st.markdown(f"- 📊 **理論年俸**：{int(data['理論年俸']):,}万円")
    elif '予想年俸' in data.index:
        st.markdown(f"- 📊 **理論年俸**：{int(data['予想年俸']):,}万円")
        
except KeyError as e:
    st.error(f"列が見つかりません: {e}")
    st.write("利用可能な列名:")
    st.write(list(data.index))

# 成績表示
if player_type == "野手":
    st.subheader("⚾ 打撃成績")
    try:
        if '打数' in data.index:
            st.markdown(f"- 打数：{data['打数']}")
        if '安打' in data.index:
            st.markdown(f"- 安打：{data['安打']}")
        if '本塁打' in data.index:
            st.markdown(f"- 本塁打：{data['本塁打']}")
        if 'OPS' in data.index:
            st.markdown(f"- OPS：{data['OPS']}")
    except KeyError as e:
        st.error(f"打撃成績の列が見つかりません: {e}")
        
else:
    st.subheader("🥎 投手成績")
    try:
        if '投球回' in data.index:
            st.markdown(f"- 投球回：{data['投球回']}")
        if '奪三振' in data.index:
            st.markdown(f"- 奪三振：{data['奪三振']}")
        if '防御率' in data.index:
            st.markdown(f"- 防御率：{data['防御率']}")
        if 'WHIP' in data.index:
            st.markdown(f"- WHIP：{data['WHIP']}")
    except KeyError as e:
        st.error(f"投手成績の列が見つかりません: {e}")

# デバッグ用（問題解決後は削除可能）
with st.expander("デバッグ情報（問題解決後は削除してください）"):
    st.write("利用可能な列名:")
    st.write(list(data.index))
    st.write("データの最初の5行:")
    st.write(df.head())
