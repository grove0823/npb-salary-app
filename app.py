import streamlit as st
import pandas as pd

# �G���R�[�f�B���O���w�肵�ēǂݍ��݁i�� �������d�v�j
batters = pd.read_csv("batters_with_theory.csv", encoding="cp932")
pitchers = pd.read_csv("pitchers_with_theory.csv", encoding="cp932")

player_type = st.selectbox("�|�W�V������I��", ["���", "����"])
df = batters if player_type == "���" else pitchers

teams = sorted(df["team"].dropna().unique())
team = st.selectbox("�`�[����I��", teams)

filtered = df[df["team"] == team]
players = filtered["�I�薼"].dropna().unique()
player = st.selectbox("�I���I��", players)

data = filtered[filtered["�I�薼"] == player].iloc[0]

st.title(f"{player}�i{team} / {player_type}�j")
st.markdown(f"- ?? **���N��**�F{int(da
