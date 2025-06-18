import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel

st.set_page_config(page_title="営業KPI分析ツール", layout="wide")
st.title("📊 営業KPI・定性データ分析アプリ")

st.markdown("""
このアプリは、CSVでアップロードされた営業データ（KPIおよびコメントなど）をもとに、
可視化と簡易的な因果関係の推定（ベイジアンネットワーク）を行います。
""")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード（例: sample_kpi.csv）", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ データを読み込みました")

    with st.expander("🔍 データプレビュー"):
        st.dataframe(df)

    # 基本統計
    with st.expander("📈 基本統計と可視化"):
        if {'アポ数', '成約数', '顧客満足度'}.issubset(df.columns):
            fig, ax = plt.subplots()
            df[['アポ数', '成約数', '顧客満足度']].plot(ax=ax)
            plt.title("KPI推移")
            st.pyplot(fig)

            st.bar_chart(df[['アポ数', '成約数']])

    # ベイジアンネットワーク推定
    with st.expander("🧠 ベイジアンネットワーク（因果関係の推定）"):
        numeric_df = df.select_dtypes(include=['number']).dropna()
        if len(numeric_df.columns) >= 2:
            try:
                hc = HillClimbSearch(numeric_df)
                best_model = hc.estimate(scoring_method=BicScore(numeric_df))
                model = BayesianModel(best_model.edges())

                st.write("推定されたネットワークのエッジ:", model.edges())
                st.graphviz_chart(model.to_daft())
            except Exception as e:
                st.error(f"ネットワーク推定に失敗しました: {e}")
        else:
            st.warning("数値列が2つ以上必要です")
else:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")
