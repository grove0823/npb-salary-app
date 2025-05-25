import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import numpy as np

# サイドバー設定
st.sidebar.title("⚙️ 設定")

# 年度スイッチ
selected_year = st.sidebar.selectbox("年度を選択", [2024])  # 拡張時に2023など追加可

# CSV読み込み（エラーハンドリング付き）
try:
    batter_file = f"batters_with_salary_{selected_year}.csv"
    pitcher_file = f"pitchers_with_salary_{selected_year}.csv"
    
    # ファイルが存在しない場合の代替
    try:
        batters = pd.read_csv(batter_file, encoding="utf-8")
    except FileNotFoundError:
        batters = pd.read_csv("batters_with_salary.csv", encoding="utf-8")
        st.sidebar.warning(f"⚠️ {batter_file} が見つからないため、汎用ファイルを使用")
    
    try:
        pitchers = pd.read_csv(pitcher_file, encoding="utf-8")
    except FileNotFoundError:
        pitchers = pd.read_csv("pitchers_with_salary.csv", encoding="utf-8")
        st.sidebar.warning(f"⚠️ {pitcher_file} が見つからないため、汎用ファイルを使用")
        
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()

# 年度でフィルタ
if '年度' in batters.columns:
    batters = batters[batters['年度'] == selected_year]
if '年度' in pitchers.columns:
    pitchers = pitchers[pitchers['年度'] == selected_year]

# ポジション別補正係数
position_weights = {
    "DH": 1.2,
    "指名打者": 1.2,
    "捕手": 1.15,
    "外野手": 1.0,
    "内野手": 1.05,
    "投手": 1.0,
    "一塁手": 1.0,
    "二塁手": 1.05,
    "三塁手": 1.05,
    "遊撃手": 1.1,
    "左翼手": 1.0,
    "中堅手": 1.05,
    "右翼手": 1.0
}

# セイバーメトリクス特徴量の定義
def get_available_features(df, is_batter=True):
    """利用可能な特徴量を取得"""
    if is_batter:
        # 野手の特徴量（優先順位順）
        priority_features = ['WAR', 'OPS', 'wRC+', 'OPS+', 'wOBA']
        basic_features = ['打率', '本塁打', '打点', '安打', '出塁率', '長打率']
        advanced_features = ['UZR', 'DRS', 'BABIP', 'ISO', 'BB%', 'K%']
    else:
        # 投手の特徴量
        priority_features = ['WAR', 'FIP', 'xFIP', 'ERA+', 'K%']
        basic_features = ['防御率', '奪三振', 'WHIP', '勝利', '投球回']
        advanced_features = ['SIERA', 'BB%', 'HR/9', 'LOB%', 'BABIP']
    
    all_features = priority_features + basic_features + advanced_features
    available = [f for f in all_features if f in df.columns]
    
    # 最低3つの特徴量は確保
    if len(available) < 3:
        fallback = ['打率', '本塁打', 'OPS'] if is_batter else ['防御率', '奪三振', 'WHIP']
        available = [f for f in fallback if f in df.columns]
    
    return available[:10]  # 最大10個

# 類似選手クラスタリング
def apply_clustering(df, features, n_clusters=5):
    """k-meansクラスタリングで類似選手をグループ化"""
    df = df.copy()
    
    # 利用可能な特徴量のみ使用
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) == 0:
        st.warning("クラスタリングに使用できる特徴量がありません")
        return df
    
    df_clean = df.dropna(subset=available_features)
    
    if len(df_clean) < n_clusters:
        n_clusters = max(2, len(df_clean) // 2)
    
    if len(df_clean) < 2:
        df['cluster'] = 0
        return df
    
    try:
        X = df_clean[available_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df.loc[df_clean.index, 'cluster'] = clusters
        df['cluster'] = df['cluster'].fillna(0).astype(int)
        
    except Exception as e:
        st.warning(f"クラスタリングエラー: {e}")
        df['cluster'] = 0
    
    return df

# 理論年俸計算関数
def calculate_theoretical_salary(df, is_batter=True):
    """セイバーメトリクスを使用した理論年俸計算"""
    df_clean = df.dropna()
    
    if len(df_clean) < 5:
        return df
    
    # 特徴量取得
    features = get_available_features(df_clean, is_batter)
    
    if len(features) == 0:
        return df
    
    try:
        # 年俸列を特定
        salary_col = None
        for col in ['年俸', 'salary', '実年俸']:
            if col in df_clean.columns:
                salary_col = col
                break
        
        if salary_col is None:
            return df
        
        # データ準備
        X = df_clean[features].values
        y = df_clean[salary_col].values
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 特徴量選択
        k_features = min(len(features), max(3, len(features) // 2))
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_scaled, y)
        
        # リッジ回帰
        model = Ridge(alpha=1.0)
        model.fit(X_selected, y)
        
        # 予測
        y_pred = model.predict(X_selected)
        
        # 結果をデータフレームに追加
        df_result = df.copy()
        df_result.loc[df_clean.index, '理論年俸'] = y_pred
        df_result['理論年俸'] = df_result['理論年俸'].fillna(df_result[salary_col] if salary_col in df_result.columns else 0)
        
        return df_result
        
    except Exception as e:
        st.error(f"理論年俸計算エラー: {e}")
        return df

# 特徴量取得とクラスタリング適用
batter_features = get_available_features(batters, is_batter=True)
pitcher_features = get_available_features(pitchers, is_batter=False)

# 理論年俸計算
batters = calculate_theoretical_salary(batters, is_batter=True)
pitchers = calculate_theoretical_salary(pitchers, is_batter=False)

# クラスタリング適用
batters = apply_clustering(batters, batter_features)
pitchers = apply_clustering(pitchers, pitcher_features)

# 理論年俸補正処理
def apply_theoretical_adjustment(df, pos_col="守備"):
    """ポジション別補正を適用"""
    if "理論年俸" in df.columns and pos_col in df.columns:
        df["補正理論年俸"] = df.apply(
            lambda row: row["理論年俸"] * position_weights.get(str(row[pos_col]), 1.0) 
            if pd.notna(row["理論年俸"]) else row["理論年俸"], 
            axis=1
        )
    else:
        # 補正列がない場合は理論年俸をコピー
        if "理論年俸" in df.columns:
            df["補正理論年俸"] = df["理論年俸"]
    return df

batters = apply_theoretical_adjustment(batters)
pitchers = apply_theoretical_adjustment(pitchers)

# 類似選手表示用関数
def show_similar_players(df, selected_player_name):
    """類似選手を表示"""
    if "cluster" not in df.columns or "選手名" not in df.columns:
        st.write("🔍 類似選手分析のためのデータが不足しています")
        return
    
    if selected_player_name not in df["選手名"].values:
        st.write("選手データが見つかりません")
        return
    
    try:
        cluster_id = df[df["選手名"] == selected_player_name]["cluster"].iloc[0]
        similar_players = df[df["cluster"] == cluster_id]
        
        st.subheader("🔍 類似選手（同クラスター）")
        
        if len(similar_players) <= 1:
            st.write("類似選手が見つかりませんでした")
            return
        
        # 年俸順でソート
        salary_col = None
        for col in ['年俸', 'salary', '実年俸']:
            if col in similar_players.columns:
                salary_col = col
                break
        
        if salary_col:
            similar_players = similar_players.sort_values(salary_col, ascending=False)
        
        count = 0
        for _, player_data in similar_players.iterrows():
            name = player_data["選手名"]
            if name != selected_player_name and count < 5:  # 最大5名まで表示
                team = player_data.get("team", "不明")
                if salary_col and pd.notna(player_data[salary_col]):
                    salary = int(player_data[salary_col])
                    st.write(f"- **{name}** ({team}) - {salary:,}万円")
                else:
                    st.write(f"- **{name}** ({team})")
                count += 1
        
        if count == 0:
            st.write("他に類似選手が見つかりませんでした")
            
    except Exception as e:
        st.error(f"類似選手表示エラー: {e}")

# メインUI
st.title(f"⚾ NPB理論年俸分析アプリ ({selected_year}年)")

# データ統計表示
col1, col2 = st.columns(2)
with col1:
    st.info(f"📊 野手データ: {len(batters)}名")
with col2:
    st.info(f"🥎 投手データ: {len(pitchers)}名")

# プレイヤー選択
player_type = st.selectbox("ポジションを選択", ["野手", "投手"])
df = batters if player_type == "野手" else pitchers

if len(df) == 0:
    st.error("選択されたカテゴリにデータがありません")
    st.stop()

# チーム選択
teams = sorted(df["team"].dropna().unique())
team = st.selectbox("チームを選択", teams)

filtered = df[df["team"] == team]

if len(filtered) == 0:
    st.warning("選択されたチームに選手データがありません")
    st.stop()

# 選手選択
players = sorted(filtered["選手名"].dropna().unique())
player = st.selectbox("選手を選択", players)

# 選手情報表示
if player in filtered["選手名"].values:
    data = filtered[filtered["選手名"] == player].iloc[0]
    
    st.header(f"🌟 {player}（{team} / {player_type}）")
    
    # 年俸情報表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        salary_col = None
        for col in ['年俸', 'salary', '実年俸']:
            if col in data.index and pd.notna(data[col]):
                salary_col = col
                break
        
        if salary_col:
            actual_salary = int(data[salary_col])
            st.metric("💰 実年俸", f"{actual_salary:,}万円")
        else:
            st.metric("💰 実年俸", "データなし")
    
    with col2:
        if "理論年俸" in data.index and pd.notna(data["理論年俸"]):
            theoretical_salary = int(data["理論年俸"])
            st.metric("📊 理論年俸", f"{theoretical_salary:,}万円")
        else:
            st.metric("📊 理論年俸", "計算不可")
    
    with col3:
        if "補正理論年俸" in data.index and pd.notna(data["補正理論年俸"]):
            adjusted_salary = int(data["補正理論年俸"])
            st.metric("⚖️ 補正理論年俸", f"{adjusted_salary:,}万円")
        else:
            st.metric("⚖️ 補正理論年俸", "計算不可")
    
    # 年俸比較
    if salary_col and "補正理論年俸" in data.index:
        if pd.notna(data[salary_col]) and pd.notna(data["補正理論年俸"]):
            difference = int(data[salary_col]) - int(data["補正理論年俸"])
            if difference > 0:
                st.success(f"💹 実年俸が補正理論年俸より **{difference:,}万円** 高い")
            elif difference < 0:
                st.info(f"📉 実年俸が補正理論年俸より **{abs(difference):,}万円** 低い")
            else:
                st.info("⚖️ 実年俸と補正理論年俸がほぼ同じ")
    
    # 成績表示
    if player_type == "野手":
        st.subheader("⚾ 打撃成績")
        stats_cols = st.columns(4)
        batter_stats = ['打率', '本塁打', '打点', 'OPS', '安打', '出塁率', '長打率', 'WAR']
    else:
        st.subheader("🥎 投手成績")
        stats_cols = st.columns(4)
        batter_stats = ['防御率', '奪三振', 'WHIP', 'WAR', '勝利', '投球回', 'FIP', 'K%']
    
    for i, stat in enumerate(batter_stats):
        if stat in data.index and pd.notna(data[stat]):
            with stats_cols[i % 4]:
                st.metric(stat, f"{data[stat]}")
    
    # 類似選手表示
    show_similar_players(df, player)
    
    # 詳細情報
    with st.expander("📈 詳細分析情報"):
        st.write("**使用された特徴量:**")
        features = batter_features if player_type == "野手" else pitcher_features
        available_features = [f for f in features if f in data.index and pd.notna(data[f])]
        
        for feature in available_features:
            st.write(f"- {feature}: {data[feature]}")
        
        if "cluster" in data.index:
            st.write(f"**クラスター番号:** {int(data['cluster'])}")
        
        if "守備" in data.index:
            position = data["守備"]
            weight = position_weights.get(str(position), 1.0)
            st.write(f"**ポジション:** {position} (補正係数: {weight})")

else:
    st.error("選手データが見つかりません")
