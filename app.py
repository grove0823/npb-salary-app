import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import numpy as np

# 年度スイッチ
selected_year = st.sidebar.selectbox("年度を選択", [2024])  # 拡張時に2023など追加可

# CSV読み込み
batter_file = f"batters_with_salary_{selected_year}.csv"
pitcher_file = f"pitchers_with_salary_{selected_year}.csv"

batters = pd.read_csv(batter_file, encoding="utf-8")
pitchers = pd.read_csv(pitcher_file, encoding="utf-8")

# 年度でフィルタ
batters = batters[batters['年度'] == selected_year] if '年度' in batters.columns else batters
pitchers = pitchers[pitchers['年度'] == selected_year] if '年度' in pitchers.columns else pitchers

# ポジション別補正係数（例：DHは打撃重視、捕手は守備含む）
position_weights = {
    "DH": 1.2,
    "捕手": 1.1,
    "外野手": 1.0,
    "内野手": 1.0,
    "投手": 1.0
}

# 類似選手クラスタリング
def apply_clustering(df, features, n_clusters=5):
    df_clean = df.dropna(subset=features)
    if len(df_clean) < n_clusters:
        return df
    X = df_clean[features].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df.loc[df_clean.index, "cluster"] = kmeans.fit_predict(X_scaled)
    return df

# 特徴量例（適宜修正）
batter_features = ['打率', '本塁打', 'OPS']
pitcher_features = ['防御率', '奪三振', 'WHIP']

batters = apply_clustering(batters, batter_features)
pitchers = apply_clustering(pitchers, pitcher_features)

# 理論年俸補正処理
def apply_theoretical_adjustment(df, pos_col="position"):
    if "理論年俸" in df.columns and pos_col in df.columns:
        df["補正理論年俸"] = df.apply(
            lambda row: row["理論年俸"] * position_weights.get(str(row[pos_col]), 1.0), axis=1
        )
    return df

batters = apply_theoretical_adjustment(batters, pos_col="守備") if "守備" in batters.columns else batters
pitchers = apply_theoretical_adjustment(pitchers, pos_col="守備") if "守備" in pitchers.columns else pitchers

# 類似選手表示用関数
def show_similar_players(df, selected_player_name):
    if "cluster" not in df.columns or "選手名" not in df.columns:
        st.write("クラスタ情報が不足しています")
        return
    if selected_player_name not in df["選手名"].values:
        st.write("選手が見つかりません")
        return
    cluster_id = df[df["選手名"] == selected_player_name]["cluster"].values[0]
    st.subheader("🔍 類似選手")
    similar_players = df[df["cluster"] == cluster_id]["選手名"].tolist()
    for name in similar_players:
        if name != selected_player_name:
            st.write(f"- {name}")

# Streamlit UI 部分
st.title("NPB 理論年俸アプリ")
player_type = st.selectbox("ポジションを選択", ["野手", "投手"])
df = batters if player_type == "野手" else pitchers

teams = sorted(df["team"].dropna().unique())
team = st.selectbox("チームを選択", teams)
filtered = df[df["team"] == team]

players = filtered["選手名"].dropna().unique()
player = st.selectbox("選手を選択", players)

if player in filtered["選手名"].values:
    data = filtered[filtered["選手名"] == player].iloc[0]
    st.header(f"{player}（{team} / {player_type}）")

    col1, col2, col3 = st.columns(3)
    with col1:
        if "年俸" in data:
            st.metric("実年俸", f"{int(data['年俸']):,}万円")
    with col2:
        if "理論年俸" in data:
            st.metric("理論年俸", f"{int(data['理論年俸']):,}万円")
    with col3:
        if "補正理論年俸" in data:
            st.metric("補正理論年俸", f"{int(data['補正理論年俸']):,}万円")

    show_similar_players(df, player)
