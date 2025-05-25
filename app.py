import streamlit as st
import pandas as pd

# Windows�ŕۑ����ꂽ���{��CSV�t�@�C���� cp932 �œǂ�
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
st.markdown(f"- ?? **���N��**�F{int(data['salary']):,}���~")
st.markdown(f"- ?? **���_�N��**�F{int(data['theoretical_salary']):,}���~")

if player_type == "���":
    st.subheader("�Ō�����")
    st.markdown(f"- �Ő��F{data['�Ő�']}")
    st.markdown(f"- ���ŁF{data['����']}")
    st.markdown(f"- �{�ۑŁF{data['�{�ۑ�']}")
    st.markdown(f"- OPS�F{data['OPS']}")
else:
    st.subheader("���萬��")
    st.markdown(f"- ������F{data['������']}")
    st.markdown(f"- �D�O�U�F{data['�D�O�U']}")
    st.markdown(f"- �h�䗦�F{data['�h�䗦']}")
    st.markdown(f"- WHIP�F{data['WHIP']}")
