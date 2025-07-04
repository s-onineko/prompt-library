import streamlit as st
import pandas as pd
from googletrans import Translator
from textblob import TextBlob

st.set_page_config(layout="wide")
st.title("#C3002F:PROMPT LIBRARY")
st.write(" #### S&A GenAI Workshop- ORIGINAL SYSTEM / フレーズ抽出・辞書化システム")
st.write("***")
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください（user, type, prompt列）", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("アップロードしたデータ", df.head())

    df.columns = [c.lower() for c in df.columns]

    # ---- TextBlobで名詞句抽出 ----
    phrase_rows = []
    for i, row in df.iterrows():
        prompt = str(row["prompt"])
        blob = TextBlob(prompt)
        # 2語以上の名詞句のみ抽出
        for phrase in blob.noun_phrases:
            if len(phrase.split()) >= 2:
                phrase_rows.append({
                    "user": row["user"], "type": row["type"], "phrase": phrase
                })

    phrase_df = pd.DataFrame(phrase_rows).drop_duplicates().reset_index(drop=True)
    st.write("抽出フレーズ一覧", phrase_df)

    # ---- Google翻訳 ----
    if st.button("日本語訳を付与（処理を開始）"):
        translator = Translator()
        phrase_ja = []
        for phrase in phrase_df["phrase"]:
            try:
                ja = translator.translate(phrase, src="en", dest="ja").text
            except Exception as e:
                ja = "(翻訳エラー)"
            phrase_ja.append(ja)
        phrase_df["phrase_ja"] = phrase_ja
        st.session_state["phrase_df"] = phrase_df

    if "phrase_df" in st.session_state:
        phrase_df = st.session_state["phrase_df"]
        st.write("日本語訳付きフレーズ", phrase_df)

        # user/typeフィルター
        user_options = ["All"] + sorted(phrase_df["user"].unique())
        type_options = ["All"] + sorted(phrase_df["type"].unique())

        selected_user = st.selectbox("ユーザーで絞り込む", user_options)
        selected_type = st.selectbox("タイプで絞り込む", type_options)

        filtered_df = phrase_df.copy()
        if selected_user != "All":
            filtered_df = filtered_df[filtered_df["user"] == selected_user]
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df["type"] == selected_type]

        choices = filtered_df["phrase_ja"].tolist()
        selected = st.multiselect("使いたいフレーズ（日本語）を選択してください", choices)
        selected_phrases = [filtered_df[filtered_df["phrase_ja"] == s]["phrase"].values[0] for s in selected]

        prompt_text = ", ".join(selected_phrases)
        st.text_area("画像生成用テキスト（編集可）", prompt_text, height=100, key="prompt_text_area")
