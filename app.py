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

        selected_user = st.selectbox("ユーザーで絞り込む", user_options, key="selected_user")
        selected_type = st.selectbox("タイプで絞り込む", type_options, key="selected_type")

        # フィルタ適用
        filtered_df = phrase_df.copy()
        if selected_user != "All":
            filtered_df = filtered_df[filtered_df["user"] == selected_user]
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df["type"] == selected_type]

        choices = filtered_df["phrase_ja"].tolist()

        # --- グローバル選択管理（全phrase_jaをキーに持つ） ---
        if "global_selected_phrases_ja" not in st.session_state:
            st.session_state.global_selected_phrases_ja = []

        # 直前の選択内容（choicesに残ってる分だけ）
        current_selected = [v for v in st.session_state.global_selected_phrases_ja if v in choices]

        # multiselect（表示はフィルター済み、状態はglobal）
        selected = st.multiselect(
            "使いたいフレーズ（日本語）を選択してください",
            choices,
            default=current_selected,
        )

        # -- global管理を更新 --
        # フィルター外のものも含めて記憶する
        # まず今表示分(current_selected)を全部外し、選択された分(selected)を追加する
        # それ以外の（フィルター外の）選択肢はそのまま残す
        base_selected = set(st.session_state.global_selected_phrases_ja) - set(current_selected)
        st.session_state.global_selected_phrases_ja = list(base_selected | set(selected))

        # -- 編集テキストもglobal管理 --
        # 英語phraseのリスト
        all_selected_phrases = [
            phrase_df[phrase_df["phrase_ja"] == s]["phrase"].values[0]
            for s in st.session_state.global_selected_phrases_ja
            if s in phrase_df["phrase_ja"].values
        ]
        prompt_text = ", ".join(all_selected_phrases) + " --ar 16:9"

        if "global_prompt_text" not in st.session_state:
            st.session_state.global_prompt_text = prompt_text

        # multiselectの選択が変化した時のみ自動更新（テキストエリア編集優先）
        # choices(=表示)が変わった場合、textも自動同期したい場合は以下判定を有効に
        if set(all_selected_phrases) != set(st.session_state.global_prompt_text.replace(" --ar 16:9", "").split(", ")):
            st.session_state.global_prompt_text = prompt_text

        def update_text():
            st.session_state.global_prompt_text = st.session_state.prompt_text_area

        st.text_area(
            "画像生成用テキスト（編集可）",
            value=st.session_state.global_prompt_text,
            height=100,
            key="prompt_text_area",
            on_change=update_text
        )
