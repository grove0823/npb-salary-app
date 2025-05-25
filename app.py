import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# CSVファイルを読み込み
batters = pd.read_csv("batters_with_salary.csv", encoding="utf-8")
pitchers = pd.read_csv("pitchers_with_salary.csv", encoding="utf-8")

# 回帰分析による理論年俸計算関数
def calculate_theoretical_salary(df, is_batter=True):
    """セイバーメトリクスを含む成績データから理論年俸を計算"""
    # 欠損値を除去
    df_clean = df.dropna()
    
    if len(df_clean) < 10:  # データが少なすぎる場合
        return None
    
    try:
        if is_batter:
            # 野手の場合の特徴量（セイバーメトリクス重視）
            features = []
            feature_names = []
            
            # 基本的な打撃成績
            basic_stats = ['打数', '安打', '本塁打', '打点', '得点', '盗塁']
            for stat in basic_stats:
                if stat in df_clean.columns:
                    features.append(df_clean[stat].values)
                    feature_names.append(stat)
            
            # 打撃率系指標
            rate_stats = ['打率', '出塁率', '長打率']
            for stat in rate_stats:
                if stat in df_clean.columns:
                    features.append(df_clean[stat].values)
                    feature_names.append(stat)
            
            # セイバーメトリクス指標（重要度高）
            sabermetrics = ['OPS', 'OPS+', 'wOBA', 'wRC+', 'WAR', 'UZR', 'DRS', 'BABIP', 'ISO', 'BB%', 'K%']
            for stat in sabermetrics:
                if stat in df_clean.columns:
                    features.append(df_clean[stat].values)
                    feature_names.append(stat)
            
            # 追加のセイバーメトリクス
            advanced_stats = ['wRAA', 'wRSB', 'Clutch', 'RE24', 'REW', 'Off', 'Def', 'BsR']
            for stat in advanced_stats:
                if stat in df_clean.columns:
                    features.append(df_clean[stat].values)
                    feature_names.append(stat)
                
        else:
            # 投手の場合の特徴量（セイバーメトリクス重視）
            features = []
            feature_names = []
            
            # 基本的な投手成績
            basic_stats = ['投球回', '奪三振', '与四球', '勝利', '敗北', 'セーブ', 'ホールド']
            for stat in basic_stats:
                if stat in df_clean.columns:
                    features.append(df_clean[stat].values)
                    feature_names.append(stat)
            
            # 投手率系指標（低い方が良い指標は逆数化）
            if '防御率' in df_clean.columns:
                era_values = df_clean['防御率'].values
                era_values = np.where(era_values > 0, 1/era_values, 0)
                features.append(era_values)
                feature_names.append('防御率(逆数)')
                
            if 'WHIP' in df_clean.columns:
                whip_values = df_clean['WHIP'].values
                whip_values = np.where(whip_values > 0, 1/whip_values, 0)
                features.append(whip_values)
                feature_names.append('WHIP(逆数)')
            
            # セイバーメトリクス指標（重要度高）
            sabermetrics = ['WAR', 'FIP', 'xFIP', 'SIERA', 'tERA', 'ERA+', 'FIP-', 'K%', 'BB%', 'K-BB%', 'BABIP', 'LOB%', 'HR/FB']
            for stat in sabermetrics:
                if stat in df_clean.columns:
                    if stat in ['FIP', 'xFIP', 'SIERA', 'tERA']:  # 低い方が良い指標
                        values = df_clean[stat].values
                        values = np.where(values > 0, 1/values, 0)
                        features.append(values)
                        feature_names.append(f'{stat}(逆数)')
                    else:
                        features.append(df_clean[stat].values)
                        feature_names.append(stat)
            
            # 追加のセイバーメトリクス
            advanced_stats = ['WPA', 'pLI', 'inLI', 'gmLI', 'exLI', 'Clutch', 'FB%', 'GB%', 'LD%', 'IFFB%', 'Soft%', 'Med%', 'Hard%']
            for stat in advanced_stats:
                if stat in df_clean.columns:
                    features.append(df_clean[stat].values)
                    feature_names.append(stat)
        # 特徴量の重要度設定（セイバーメトリクス重視）
        if len(features) == 0:
            return None
            
        # 特徴量行列を作成
        X = np.column_stack(features)
        
        # 年俸データを取得
        salary_col = None
        for col in ['salary', '年俸', '実年俸']:
            if col in df_clean.columns:
                salary_col = col
                break
        
        if salary_col is None:
            return None
            
        y = df_clean[salary_col].values
        
        # 標準化（セイバーメトリクスは異なるスケールのため重要）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 特徴量選択（重要度の低い特徴量を除去）
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # 利用可能な特徴量数に応じてk値を調整
        k_features = min(10, len(features))  # 最大10個の特徴量を選択
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_scaled, y)
        
        # 選択された特徴量の名前を取得
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        
        # リッジ回帰を使用（過学習を防ぐため）
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_selected, y)
        
        # 理論年俸を予測
        y_pred = model.predict(X_selected)
        
        # 元のデータフレームに理論年俸を追加
        df_result = df_clean.copy()
        df_result['理論年俸'] = y_pred
        
        return df_result, model, scaler, selected_features, selector
        
    except Exception as e:
        st.error(f"回帰分析でエラーが発生しました: {e}")
        return None

# 理論年俸を計算
st.sidebar.info("📊 理論年俸を計算中...")

batter_result = calculate_theoretical_salary(batters, is_batter=True)
pitcher_result = calculate_theoretical_salary(pitchers, is_batter=False)

if batter_result:
    batters_with_theoretical = batter_result[0]
    st.sidebar.success("✅ 野手の理論年俸計算完了")
else:
    batters_with_theoretical = batters
    st.sidebar.warning("⚠️ 野手の理論年俸計算に失敗")

if pitcher_result:
    pitchers_with_theoretical = pitcher_result[0]
    st.sidebar.success("✅ 投手の理論年俸計算完了")
else:
    pitchers_with_theoretical = pitchers
    st.sidebar.warning("⚠️ 投手の理論年俸計算に失敗")

# メインアプリ
st.title("⚾ NPB選手年俸検索アプリ")

# ポジション選択
player_type = st.selectbox("ポジションを選択", ["野手", "投手"])

# データフレームの選択
if player_type == "野手":
    df = batters_with_theoretical
else:
    df = pitchers_with_theoretical

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
st.header(f"{player}（{team} / {player_type}）")

# 年俸情報
col1, col2 = st.columns(2)

with col1:
    # 実年俸
    salary_col = None
    for col in ['salary', '年俸', '実年俸']:
        if col in data.index:
            salary_col = col
            break
    
    if salary_col:
        actual_salary = int(data[salary_col])
        st.metric("💰 実年俸", f"{actual_salary:,}万円")
    else:
        st.error("実年俸データが見つかりません")

with col2:
    # 理論年俸
    if '理論年俸' in data.index:
        theoretical_salary = int(data['理論年俸'])
        st.metric("📊 理論年俸", f"{theoretical_salary:,}万円")
        
        # 差額を計算
        if salary_col:
            difference = actual_salary - theoretical_salary
            if difference > 0:
                st.success(f"💹 実年俸が理論年俸より {difference:,}万円 高い")
            elif difference < 0:
                st.info(f"📉 実年俸が理論年俸より {abs(difference):,}万円 低い")
            else:
                st.info("⚖️ 実年俸と理論年俸が同じ")
    else:
        st.warning("理論年俸を計算できませんでした")

# 成績表示
if player_type == "野手":
    st.subheader("⚾ 打撃成績")
    cols = st.columns(4)
    
    stats = ['打数', '安打', '本塁打', 'OPS', '打率', '打点']
    for i, stat in enumerate(stats):
        if stat in data.index:
            with cols[i % 4]:
                st.metric(stat, data[stat])
else:
    st.subheader("🥎 投手成績")
    cols = st.columns(4)
    
    stats = ['投球回', '奪三振', '防御率', 'WHIP', '勝利', '敗北']
    for i, stat in enumerate(stats):
        if stat in data.index:
            with cols[i % 4]:
                st.metric(stat, data[stat])

# 回帰分析の詳細情報
with st.expander("📈 セイバーメトリクス回帰分析の詳細"):
    if player_type == "野手" and batter_result:
        _, model, scaler, selected_features, selector = batter_result
        st.write("**野手の理論年俸計算に使用したセイバーメトリクス特徴量:**")
        st.write("*（特徴量選択により重要度の高い指標のみ使用）*")
        for i, feature in enumerate(selected_features):
            importance = abs(model.coef_[i])
            st.write(f"- **{feature}**: 係数 = {model.coef_[i]:.4f} (重要度: {importance:.4f})")
        st.write(f"- **切片**: {model.intercept_:.2f}")
        
        # モデル評価
        from sklearn.metrics import r2_score, mean_absolute_error
        y_true = batters_with_theoretical[[col for col in ['salary', '年俸', '実年俸'] if col in batters_with_theoretical.columns][0]].values
        y_pred = batters_with_theoretical['理論年俸'].values
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("決定係数 (R²)", f"{r2:.3f}")
        with col2:
            st.metric("平均絶対誤差", f"{mae:.0f}万円")
        with col3:
            st.metric("使用特徴量数", len(selected_features))
    
    elif player_type == "投手" and pitcher_result:
        _, model, scaler, selected_features, selector = pitcher_result
        st.write("**投手の理論年俸計算に使用したセイバーメトリクス特徴量:**")
        st.write("*（特徴量選択により重要度の高い指標のみ使用）*")
        for i, feature in enumerate(selected_features):
            importance = abs(model.coef_[i])
            st.write(f"- **{feature}**: 係数 = {model.coef_[i]:.4f} (重要度: {importance:.4f})")
        st.write(f"- **切片**: {model.intercept_:.2f}")
        
        # モデル評価
        from sklearn.metrics import r2_score, mean_absolute_error
        y_true = pitchers_with_theoretical[[col for col in ['salary', '年俸', '実年俸'] if col in pitchers_with_theoretical.columns][0]].values
        y_pred = pitchers_with_theoretical['理論年俸'].values
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("決定係数 (R²)", f"{r2:.3f}")
        with col2:
            st.metric("平均絶対誤差", f"{mae:.0f}万円")
        with col3:
            st.metric("使用特徴量数", len(selected_features))
    
    st.info("💡 **セイバーメトリクス活用のポイント**: WAR、OPS+、wRC+、FIP、xFIPなどの高度な指標を重視し、特徴量選択により最も年俸に影響する指標のみを使用しています。")

# デバッグ情報（開発時のみ）
with st.expander("🔧 デバッグ情報"):
    st.write("**利用可能な列名:**")
    st.write(list(data.index))
    st.write("**データサンプル:**")
    st.write(df.head())
