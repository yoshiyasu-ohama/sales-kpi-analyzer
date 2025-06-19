import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

st.set_page_config(page_title="営業KPI分析ツール", layout="wide")
st.title("📊 営業KPI・定性データ分析アプリ")

st.markdown("""
このアプリは、CSVでアップロードされた営業データ（KPIおよびコメントなど）をもとに、
可視化と統計分析を行います。
""")

# サンプルデータの表示
with st.expander("📝 サンプルデータ形式"):
    sample_data = {
        '日付': ['2025-06-14', '2025-06-15', '2025-06-16'],
        'アポ数': [10, 12, 8],
        '成約数': [3, 5, 2],
        '顧客満足度': [4.2, 4.8, 3.9],
        'コメント': ['顧客は価格に敏感だった', '提案内容に興味を持っていた', '競合製品との比較が必要だった']
    }
    st.dataframe(pd.DataFrame(sample_data))

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ データを読み込みました")
        
        # データの前処理
        if '日付' in df.columns:
            try:
                df['日付'] = pd.to_datetime(df['日付'])
            except:
                st.warning("日付列の変換に失敗しました。文字列として処理します。")

        with st.expander("🔍 データプレビュー"):
            st.dataframe(df)
            st.write(f"**データ行数**: {len(df)}")
            st.write(f"**列数**: {len(df.columns)}")

        # 基本統計
        with st.expander("📊 基本統計情報"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**数値データの基本統計**")
                st.dataframe(df[numeric_cols].describe())
            else:
                st.warning("数値データが見つかりません。")

        # 可視化セクション
        with st.expander("📈 データ可視化"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 1:
                # KPI推移グラフ
                if {'アポ数', '成約数', '顧客満足度'}.issubset(df.columns):
                    st.subheader("KPI推移")
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # アポ数の推移
                    axes[0,0].plot(df.index, df['アポ数'], marker='o')
                    axes[0,0].set_title('アポ数推移')
                    axes[0,0].set_ylabel('件数')
                    
                    # 成約数の推移
                    axes[0,1].plot(df.index, df['成約数'], marker='s', color='green')
                    axes[0,1].set_title('成約数推移')
                    axes[0,1].set_ylabel('件数')
                    
                    # 顧客満足度の推移
                    axes[1,0].plot(df.index, df['顧客満足度'], marker='^', color='orange')
                    axes[1,0].set_title('顧客満足度推移')
                    axes[1,0].set_ylabel('評価')
                    
                    # 成約率の計算と表示
                    df['成約率'] = (df['成約数'] / df['アポ数']) * 100
                    axes[1,1].plot(df.index, df['成約率'], marker='d', color='red')
                    axes[1,1].set_title('成約率推移')
                    axes[1,1].set_ylabel('成約率(%)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                # 相関分析
                if len(numeric_cols) >= 2:
                    st.subheader("相関関係")
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    plt.title('データ間の相関関係')
                    st.pyplot(fig)
                    
                    # 強い相関関係のハイライト
                    st.write("**強い相関関係（|r| > 0.7）**")
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                strong_corr.append({
                                    '変数1': corr_matrix.columns[i],
                                    '変数2': corr_matrix.columns[j],
                                    '相関係数': round(corr_val, 3)
                                })
                    
                    if strong_corr:
                        st.dataframe(pd.DataFrame(strong_corr))
                    else:
                        st.info("強い相関関係は見つかりませんでした。")

                # 散布図
                if len(numeric_cols) >= 2:
                    st.subheader("散布図分析")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_var = st.selectbox("X軸の変数", numeric_cols, key="x_var")
                    with col2:
                        y_var = st.selectbox("Y軸の変数", numeric_cols, key="y_var", index=1 if len(numeric_cols) > 1 else 0)
                    
                    if x_var != y_var:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(df[x_var], df[y_var], alpha=0.7)
                        ax.set_xlabel(x_var)
                        ax.set_ylabel(y_var)
                        ax.set_title(f'{x_var} vs {y_var}')
                        
                        # 回帰線の追加
                        z = np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1)
                        p = np.poly1d(z)
                        ax.plot(df[x_var], p(df[x_var]), "r--", alpha=0.8)
                        
                        st.pyplot(fig)

        # テキスト分析（コメント列がある場合）
        if 'コメント' in df.columns:
            with st.expander("💬 コメント分析"):
                st.subheader("コメント一覧")
                comments_df = df[['日付', 'コメント']].dropna() if '日付' in df.columns else df[['コメント']].dropna()
                st.dataframe(comments_df)
                
                # 簡単なキーワード分析
                st.subheader("キーワード頻度分析")
                all_comments = ' '.join(df['コメント'].dropna().astype(str))
                
                # よく使われる単語の抽出（簡易版）
                words = all_comments.split()
                word_freq = {}
                for word in words:
                    if len(word) > 1:  # 1文字の単語は除外
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                if word_freq:
                    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    words_df = pd.DataFrame(sorted_words, columns=['単語', '頻度'])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(words_df['単語'], words_df['頻度'])
                    ax.set_xlabel('頻度')
                    ax.set_title('よく使われる単語 (Top 10)')
                    plt.tight_layout()
                    st.pyplot(fig)

        # パフォーマンス分析
        if {'アポ数', '成約数'}.issubset(df.columns):
            with st.expander("📊 パフォーマンス分析"):
                st.subheader("営業パフォーマンス指標")
                
                total_appointments = df['アポ数'].sum()
                total_contracts = df['成約数'].sum()
                avg_conversion_rate = (total_contracts / total_appointments * 100) if total_appointments > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("総アポ数", f"{total_appointments:,}件")
                
                with col2:
                    st.metric("総成約数", f"{total_contracts:,}件")
                
                with col3:
                    st.metric("平均成約率", f"{avg_conversion_rate:.1f}%")
                
                # 日別パフォーマンス
                if len(df) > 1:
                    st.subheader("日別パフォーマンス")
                    performance_df = df.copy()
                    performance_df['成約率'] = (performance_df['成約数'] / performance_df['アポ数'] * 100).fillna(0)
                    
                    # パフォーマンスランキング
                    performance_ranking = performance_df.nlargest(5, '成約率')[['日付', 'アポ数', '成約数', '成約率']]
                    st.write("**成約率上位5日**")
                    st.dataframe(performance_ranking)

    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {str(e)}")
        st.info("CSVファイルの形式を確認してください。")

else:
    st.info("CSVファイルをアップロードしてください。")
    
    # 使い方の説明
    with st.expander("📋 使い方"):
        st.markdown("""
        1. **CSVファイルを準備**: 営業データを含むCSVファイルを用意してください
        2. **ファイルをアップロード**: 上のファイルアップローダーからCSVファイルを選択
        3. **データを確認**: アップロード後、データプレビューで内容を確認
        4. **分析結果を確認**: 各セクションを展開して分析結果を確認
        
        **推奨するCSV形式**:
        - 日付列: YYYY-MM-DD形式
        - 数値列: アポ数、成約数、顧客満足度など
        - コメント列: 定性的な情報やメモ
        """)
