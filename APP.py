import streamlit as st
import pandas as pd
import statsmodels.api as sm
import io
import plotly.express as px

# 設定網頁標題與寬度
st.set_page_config(page_title="穩定幣變數分析儀表板", layout="wide")

st.title("📊 穩定幣變數分析儀表板")
st.markdown("請在下方選擇應變數 (Y)，並透過不同的標籤頁 (Tabs) 進行全方位的數據分析。")

file_name = '0408穩定幣與變數數據(變動率).csv'

try:
    df = pd.read_csv(file_name)
    
    # 嘗試將日期欄位設為索引 (若有 Date 或 日期 欄位)，這會讓走勢圖的 X 軸更漂亮
    if 'Date' in df.columns:
        df = df.set_index('Date')
    elif '日期' in df.columns:
        df = df.set_index('日期')
        
    # 只抓取數值型欄位，避免字串干擾畫圖與計算
    numeric_df = df.select_dtypes(include='number')
    columns = numeric_df.columns.tolist()

    # 1. 讓同學動態選擇應變數 (Y)
    default_index = columns.index('USDT') if 'USDT' in columns else 0
    target_y = st.selectbox("🎯 請選擇你要預測/分析的「應變數」(Y)：", columns, index=default_index)

    # ==========================================
    # 建立三個標籤頁
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 1. 獨立影響力分析 (迴歸)", "🔥 2. 相關係數熱力圖", "📈 3. 歷史波動走勢圖"])

    # ---------- 分頁 1：獨立影響力分析 ----------
    with tab1:
        st.subheader(f"探討各變數對 {target_y} 的獨立影響")
        if st.button("🚀 開始執行成對迴歸分析"):
            independent_vars = [col for col in columns if col != target_y]
            results_list = []
            
            progress_bar = st.progress(0)
            
            for i, x_var in enumerate(independent_vars):
                progress_bar.progress((i + 1) / len(independent_vars))
                temp_data = numeric_df[[target_y, x_var]].dropna()
                sample_size = len(temp_data)
                
                if sample_size < 10: continue

                y = temp_data[target_y]
                X = sm.add_constant(temp_data[x_var])
                
                try:
                    model = sm.OLS(y, X).fit()
                    p_val = model.pvalues[x_var]
                    coef = model.params[x_var]
                    r_squared = model.rsquared

                    if p_val < 0.05:
                        significance = '⭐ 是'
                        explanation = f'正向影響：當 {x_var} 上漲時，{target_y} 傾向跟著上漲。' if coef > 0 else f'負向影響：當 {x_var} 上漲時，{target_y} 傾向下跌。'
                    else:
                        significance = '否'
                        explanation = f'無顯著影響：此變數與 {target_y} 無明顯統計關聯。'

                    results_list.append({
                        '變數名稱': x_var,
                        '樣本天數': sample_size,
                        '影響係數(Coef)': round(coef, 6),
                        'P值(P-value)': round(p_val, 4),
                        '解釋力(R-squared)': round(r_squared, 6),
                        '顯著性': significance,
                        '結果解釋': explanation
                    })
                except:
                    pass

            if results_list:
                summary_table = pd.DataFrame(results_list).sort_values(by='解釋力(R-squared)', ascending=False)
                st.dataframe(summary_table, use_container_width=True)

                csv_buffer = io.BytesIO()
                summary_table.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button("📥 下載完整分析結果 (CSV)", data=csv_buffer.getvalue(), file_name=f"{target_y}_迴歸分析結果.csv", mime="text/csv")
            else:
                st.warning("找不到足夠的有效資料。")

    # ---------- 分頁 2：相關係數熱力圖 ----------
    with tab2:
        st.subheader("變數之間的全局相關性 (Correlation)")
        st.markdown("💡 **怎麼看這張圖？** 顏色越**紅**代表正相關越強 (一起漲跌)；顏色越**藍**代表負相關越強 (一邊漲一邊跌)。")
        
        # 計算相關係數矩陣
        corr_matrix = numeric_df.corr().round(2)
        
        # 繪製互動式熱力圖
        fig_heat = px.imshow(
            corr_matrix, 
            text_auto=True, # 自動在格子裡顯示數字
            aspect="auto", 
            color_continuous_scale='RdBu_r', # 紅藍配色
            zmin=-1, zmax=1 # 鎖定範圍 -1 到 1
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---------- 分頁 3：歷史波動走勢圖 ----------
    with tab3:
        st.subheader(f"觀測 {target_y} 與其他變數的波動軌跡")
        st.markdown("請在下方選擇你想與之比較的變數 (可複選)，觀察它們在同一段時間內的走勢關聯。")
        
        # 讓同學多選要畫在圖上的 X 變數
        options = [col for col in columns if col != target_y]
        
        # 預設選取前兩個變數當作範例 (避免一開始畫面空空的)
        default_selections = options[:2] if len(options) >= 2 else []
        selected_xs = st.multiselect("📊 請選擇要疊加對比的變數：", options, default=default_selections)
        
        if selected_xs or target_y:
            # 將 Y 與選定的 X 一起畫成折線圖
            vars_to_plot = [target_y] + selected_xs
            fig_line = px.line(df, y=vars_to_plot, title="數值波動走勢對比")
            st.plotly_chart(fig_line, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ 找不到資料檔案：{file_name}")
