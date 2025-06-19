import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定（複数のオプションを試行）
def set_japanese_font():
    import matplotlib.font_manager as fm
    
    # 日本語フォントの候補リスト
    japanese_fonts = [
        'Hiragino Sans',
        'Yu Gothic',
        'Meiryo',
        'MS Gothic',
        'Takao PGothic',
        'IPAexGothic',
        'IPAPGothic',
        'VL PGothic',
        'Noto Sans CJK JP',
        'DejaVu Sans'
    ]
    
    # 利用可能なフォントを探す
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in japanese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            break
    else:
        # フォールバック設定
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = japanese_fonts + ['Arial', 'Liberation Sans']
    
    # 負の値表示のためのマイナス記号設定
    plt.rcParams['axes.unicode_minus'] = False

# フォント設定を適用
set_japanese_font()

# ページ設定
st.set_page_config(
    page_title="営業KPI分析ツール", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stExpander > div > div > div > div {
        padding: 1rem;
    }
    
    .plot-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
    
    /* レスポンシブ対応 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# メインタイトル
st.markdown('<h1 class="main-header">📊 営業KPI・定性データ分析アプリ</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
このアプリは、CSVでアップロードされた営業データ（KPIおよびコメントなど）をもとに、<br>
可視化と統計分析を行います。
</div>
""", unsafe_allow_html=True)

# サイドバーの設定
with st.sidebar:
    st.header("🔧 設定")
    
    # グラフ設定
    st.subheader("グラフ設定")
    figure_size = st.selectbox(
        "グラフサイズ",
        ["小 (8x6)", "中 (10x8)", "大 (12x10)", "特大 (14x12)"],
        index=1
    )
    
    size_map = {
        "小 (8x6)": (8, 6),
        "中 (10x8)": (10, 8),
        "大 (12x10)": (12, 10),
        "特大 (14x12)": (14, 12)
    }
    
    fig_size = size_map[figure_size]
    
    # 色テーマ
    color_theme = st.selectbox(
        "色テーマ",
        ["デフォルト", "ダーク", "パステル", "ビビッド"],
        index=0
    )
    
    color_palettes = {
        "デフォルト": "Set2",
        "ダーク": "dark",
        "パステル": "pastel",
        "ビビッド": "bright"
    }

# サンプルデータの表示
with st.expander("📝 サンプルデータ形式"):
    sample_data = {
        '日付': ['2025-06-14', '2025-06-15', '2025-06-16'],
        'アポ数': [10, 12, 8],
        '成約数': [3, 5, 2],
        '顧客満足度': [4.2, 4.8, 3.9],
        'コメント': ['顧客は価格に敏感だった', '提案内容に興味を持っていた', '競合製品との比較が必要だった']
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

# ファイルアップロード
uploaded_file = st.file_uploader(
    "CSVファイルをアップロード", 
    type=["csv"],
    help="営業データを含むCSVファイルをドラッグ&ドロップまたは選択してください"
)

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

        # データプレビュー
        with st.expander("🔍 データプレビュー", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>📊 データ情報</h4>
                    <p><strong>行数:</strong> {len(df):,}</p>
                    <p><strong>列数:</strong> {len(df.columns)}</p>
                    <p><strong>期間:</strong> {len(df)}日分</p>
                </div>
                """, unsafe_allow_html=True)

        # 基本統計
        with st.expander("📊 基本統計情報"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.subheader("数値データの基本統計")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                # 欠損値情報
                missing_data = df[numeric_cols].isnull().sum()
                if missing_data.sum() > 0:
                    st.subheader("欠損値情報")
                    missing_df = pd.DataFrame({
                        '列名': missing_data.index,
                        '欠損値数': missing_data.values,
                        '欠損率(%)': (missing_data.values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df[missing_df['欠損値数'] > 0], use_container_width=True)
            else:
                st.warning("数値データが見つかりません。")

        # 可視化セクション
        with st.expander("📈 データ可視化", expanded=True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 1:
                # カラーパレット設定
                sns.set_palette(color_palettes[color_theme])
                
                # KPI推移グラフ
                if {'アポ数', '成約数', '顧客満足度'}.issubset(df.columns):
                    st.subheader("📈 KPI推移")
                    
                    # レスポンシブなグラフレイアウト
                    if st.checkbox("グラフを個別表示", value=False):
                        # 個別表示モード
                        metrics = ['アポ数', '成約数', '顧客満足度']
                        for metric in metrics:
                            if metric in df.columns:
                                fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]//2))
                                ax.plot(df.index, df[metric], marker='o', linewidth=2, markersize=8)
                                ax.set_title(f'{metric}の推移', fontsize=16, pad=20)
                                ax.set_ylabel(metric, fontsize=12)
                                ax.set_xlabel('日数', fontsize=12)
                                ax.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                                plt.close()
                    else:
                        # 統合表示モード
                        fig, axes = plt.subplots(2, 2, figsize=fig_size)
                        fig.suptitle('営業KPI総合ダッシュボード', fontsize=18, y=0.98)
                        
                        # アポ数の推移
                        axes[0,0].plot(df.index, df['アポ数'], marker='o', color='#1f77b4', linewidth=2)
                        axes[0,0].set_title('アポ数推移', fontsize=14)
                        axes[0,0].set_ylabel('件数', fontsize=12)
                        axes[0,0].grid(True, alpha=0.3)
                        
                        # 成約数の推移
                        axes[0,1].plot(df.index, df['成約数'], marker='s', color='#2ca02c', linewidth=2)
                        axes[0,1].set_title('成約数推移', fontsize=14)
                        axes[0,1].set_ylabel('件数', fontsize=12)
                        axes[0,1].grid(True, alpha=0.3)
                        
                        # 顧客満足度の推移
                        axes[1,0].plot(df.index, df['顧客満足度'], marker='^', color='#ff7f0e', linewidth=2)
                        axes[1,0].set_title('顧客満足度推移', fontsize=14)
                        axes[1,0].set_ylabel('評価', fontsize=12)
                        axes[1,0].grid(True, alpha=0.3)
                        
                        # 成約率の計算と表示
                        df['成約率'] = (df['成約数'] / df['アポ数']) * 100
                        axes[1,1].plot(df.index, df['成約率'], marker='d', color='#d62728', linewidth=2)
                        axes[1,1].set_title('成約率推移', fontsize=14)
                        axes[1,1].set_ylabel('成約率(%)', fontsize=12)
                        axes[1,1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                # 相関分析
                if len(numeric_cols) >= 2:
                    st.subheader("🔗 相関関係分析")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        corr_matrix = df[numeric_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(fig_size[0]*0.8, fig_size[1]*0.8))
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        sns.heatmap(
                            corr_matrix, 
                            annot=True, 
                            cmap='RdBu_r', 
                            center=0, 
                            ax=ax,
                            mask=mask,
                            square=True,
                            cbar_kws={"shrink": .8}
                        )
                        ax.set_title('データ間の相関関係', fontsize=16, pad=20)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    
                    with col2:
                        # 強い相関関係のハイライト
                        st.markdown("**強い相関関係 (|r| > 0.7)**")
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
                            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
                        else:
                            st.info("強い相関関係は見つかりませんでした。")

                # 散布図
                if len(numeric_cols) >= 2:
                    st.subheader("📊 散布図分析")
                    
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        x_var = st.selectbox("X軸の変数", numeric_cols, key="x_var")
                    with col2:
                        y_var = st.selectbox("Y軸の変数", numeric_cols, key="y_var", index=1 if len(numeric_cols) > 1 else 0)
                    
                    with col3:
                        show_regression = st.checkbox("回帰線を表示", value=True)
                        show_correlation = st.checkbox("相関係数を表示", value=True)
                    
                    if x_var != y_var:
                        fig, ax = plt.subplots(figsize=fig_size)
                        
                        # 散布図
                        ax.scatter(df[x_var], df[y_var], alpha=0.7, s=100)
                        ax.set_xlabel(x_var, fontsize=12)
                        ax.set_ylabel(y_var, fontsize=12)
                        ax.set_title(f'{x_var} vs {y_var}', fontsize=16, pad=20)
                        ax.grid(True, alpha=0.3)
                        
                        # 回帰線の追加
                        if show_regression:
                            valid_data = df[[x_var, y_var]].dropna()
                            if len(valid_data) > 1:
                                z = np.polyfit(valid_data[x_var], valid_data[y_var], 1)
                                p = np.poly1d(z)
                                ax.plot(valid_data[x_var], p(valid_data[x_var]), "r--", alpha=0.8, linewidth=2)
                        
                        # 相関係数の表示
                        if show_correlation:
                            corr_coef = df[x_var].corr(df[y_var])
                            ax.text(0.05, 0.95, f'相関係数: {corr_coef:.3f}', 
                                   transform=ax.transAxes, fontsize=12,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

        # テキスト分析（コメント列がある場合）
        if 'コメント' in df.columns:
            with st.expander("💬 コメント分析"):
                st.subheader("📝 コメント一覧")
                comments_df = df[['日付', 'コメント']].dropna() if '日付' in df.columns else df[['コメント']].dropna()
                st.dataframe(comments_df, use_container_width=True)
                
                # 簡単なキーワード分析
                st.subheader("🔤 キーワード頻度分析")
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
                    
                    fig, ax = plt.subplots(figsize=fig_size)
                    bars = ax.barh(words_df['単語'], words_df['頻度'])
                    ax.set_xlabel('頻度', fontsize=12)
                    ax.set_title('よく使われる単語 (Top 10)', fontsize=16, pad=20)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # バーの色を変更
                    for bar in bars:
                        bar.set_color(plt.cm.viridis(bar.get_width() / max(words_df['頻度'])))
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

        # パフォーマンス分析
        if {'アポ数', '成約数'}.issubset(df.columns):
            with st.expander("🎯 パフォーマンス分析"):
                st.subheader("📊 営業パフォーマンス指標")
                
                total_appointments = df['アポ数'].sum()
                total_contracts = df['成約数'].sum()
                avg_conversion_rate = (total_contracts / total_appointments * 100) if total_appointments > 0 else 0
                
                # メトリクス表示
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("総アポ数", f"{total_appointments:,}件")
                
                with col2:
                    st.metric("総成約数", f"{total_contracts:,}件")
                
                with col3:
                    st.metric("平均成約率", f"{avg_conversion_rate:.1f}%")
                
                with col4:
                    avg_satisfaction = df['顧客満足度'].mean() if '顧客満足度' in df.columns else 0
                    st.metric("平均満足度", f"{avg_satisfaction:.1f}点")
                
                # 日別パフォーマンス
                if len(df) > 1:
                    st.subheader("🏆 日別パフォーマンス")
                    performance_df = df.copy()
                    performance_df['成約率'] = (performance_df['成約数'] / performance_df['アポ数'] * 100).fillna(0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # パフォーマンスランキング
                        st.markdown("**成約率上位5日**")
                        top_performance = performance_df.nlargest(5, '成約率')
                        display_cols = ['日付', 'アポ数', '成約数', '成約率'] if '日付' in performance_df.columns else ['アポ数', '成約数', '成約率']
                        st.dataframe(top_performance[display_cols], use_container_width=True)
                    
                    with col2:
                        # 成約数上位5日
                        st.markdown("**成約数上位5日**")
                        top_contracts = performance_df.nlargest(5, '成約数')
                        st.dataframe(top_contracts[display_cols], use_container_width=True)

    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {str(e)}")
        st.info("CSVファイルの形式を確認してください。")

else:
    st.info("📁 CSVファイルをアップロードしてください。")
    
    # 使い方の説明
    with st.expander("📋 使い方ガイド", expanded=True):
        st.markdown("""
        ### 📖 アプリの使い方
        
        1. **CSVファイルを準備**: 営業データを含むCSVファイルを用意してください
        2. **ファイルをアップロード**: 上のファイルアップローダーからCSVファイルを選択
        3. **設定を調整**: 左サイドバーでグラフサイズや色テーマを選択
        4. **データを確認**: アップロード後、データプレビューで内容を確認
        5. **分析結果を確認**: 各セクションを展開して分析結果を確認
        
        ### 📊 推奨するCSV形式
        
        | 列名 | 説明 | 例 |
        |------|------|-----|
        | 日付 | YYYY-MM-DD形式 | 2025-06-14 |
        | アポ数 | 営業アポイントメント数 | 10 |
        | 成約数 | 成約した件数 | 3 |
        | 顧客満足度 | 1-5点の評価 | 4.2 |
        | コメント | 定性的な情報 | 顧客は価格に敏感だった |
        
        ### 🎨 機能紹介
        
        - **📈 KPI可視化**: アポ数、成約数、満足度の推移をグラフで表示
        - **🔗 相関分析**: データ間の関係性をヒートマップで可視化
        - **📊 散布図**: 2つの変数の関係を詳細に分析
        - **💬 コメント分析**: テキストデータのキーワード頻度を分析
        - **🎯 パフォーマンス分析**: 営業成果の総合的な評価
        """)
