import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
from textblob import TextBlob
import nltk

nltk_packages = [
    "brown",
    "punkt",
    "averaged_perceptron_tagger",
    "conll2000",
    "wordnet"
]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

st.set_page_config(layout="wide")
st.markdown(
    "<div style='font-size:48px; font-weight:bold; color:#C3002F;'>PROMPT LIBRARY</div>",
    unsafe_allow_html=True
)
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

    # ---- deep-translatorでGoogle翻訳 ----
    if st.button("日本語訳を付与（処理を開始）"):
        phrase_ja = []
        for phrase in phrase_df["phrase"]:
            try:
                ja = GoogleTranslator(source='auto', target='ja').translate(phrase)
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

        # セッション状態で選択ユーザー・タイプを保持
        selected_user = st.selectbox("ユーザーで絞り込む", user_options, key="selected_user")
        selected_type = st.selectbox("タイプで絞り込む", type_options, key="selected_type")

        # フィルタ適用
        filtered_df = phrase_df.copy()
        if selected_user != "All":
            filtered_df = filtered_df[filtered_df["user"] == selected_user]
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df["type"] == selected_type]

        choices = filtered_df["phrase_ja"].tolist()

        # 過去の選択値を復元・維持
        if "selected_phrases_ja" not in st.session_state:
            st.session_state.selected_phrases_ja = []

        # 絞り込みにより消えた選択肢を除外
        prev_selected = [v for v in st.session_state.selected_phrases_ja if v in choices]

        # multiselectをセッション状態で管理
        selected = st.multiselect(
            "使いたいフレーズ（日本語）を選択してください",
            choices,
            default=prev_selected,
            key="selected_phrases_ja"
        )

        # phrase英語バージョンに変換
        selected_phrases = [
            filtered_df[filtered_df["phrase_ja"] == s]["phrase"].values[0]
            for s in selected
        ]
        prompt_text = ", ".join(selected_phrases) + " --ar 16:9"

        # テキストエリアもセッション管理
        if "prompt_text" not in st.session_state:
            st.session_state.prompt_text = prompt_text
        # フレーズ選択が変わった時だけ初期値を更新
        elif st.session_state.selected_phrases_ja != prev_selected:
            st.session_state.prompt_text = prompt_text

        # テキストエリア本体
        def update_prompt_text():
            st.session_state.prompt_text = st.session_state.prompt_text_area

        st.text_area(
            "画像生成用テキスト（編集可）",
            value=st.session_state.prompt_text,
            height=100,
            key="prompt_text_area",
            on_change=update_prompt_text
        )
