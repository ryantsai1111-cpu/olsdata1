import streamlit as st
import pandas as pd
import statsmodels.api as sm
import io
import plotly.express as px

# 設定網頁標題與寬度
st.set_page_config(page_title="穩定幣變數分析儀表板", layout="wide")

st.title("📊 穩定幣變數分析儀表板")
st.markdown("請在下方選擇應變數 (Y)，並透過不同的標籤頁 (Tabs) 進行全方位的數據分析與歷史背景回顧。")

file_name = '0408穩定幣與變數數據(變動率).csv'

try:
    df = pd.read_csv(file_name)
    
    if 'Date' in df.columns:
        df = df.set_index('Date')
    elif '日期' in df.columns:
        df = df.set_index('日期')
        
    numeric_df = df.select_dtypes(include='number')
    columns = numeric_df.columns.tolist()

    default_index = columns.index('USDT') if 'USDT' in columns else 0
    target_y = st.selectbox("🎯 請選擇你要預測/分析的「應變數」(Y)：", columns, index=default_index)

    # ==========================================
    # 🌟 建立四個標籤頁 (新增了 tab4)
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. 獨立影響力分析", 
        "🔥 2. 相關係數熱力圖", 
        "📈 3. 歷史波動走勢圖", 
        "⏳ 4. 重大歷史事件時間軸"
    ])

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
        st.markdown("💡 **怎麼看這張圖？** 顏色越**紅**代表正相關越強；顏色越**藍**代表負相關越強。")
        corr_matrix = numeric_df.corr().round(2)
        fig_heat = px.imshow(
            corr_matrix, text_auto=True, aspect="auto", 
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---------- 分頁 3：歷史波動走勢圖 ----------
    with tab3:
        st.subheader(f"觀測 {target_y} 與其他變數的波動軌跡")
        st.markdown("請在下方選擇你想與之比較的變數 (可複選)，觀察它們在同一段時間內的走勢關聯。")
        options = [col for col in columns if col != target_y]
        default_selections = options[:2] if len(options) >= 2 else []
        selected_xs = st.multiselect("📊 請選擇要疊加對比的變數：", options, default=default_selections)
        
        if selected_xs or target_y:
            vars_to_plot = [target_y] + selected_xs
            fig_line = px.line(df, y=vars_to_plot, title="數值波動走勢對比")
            st.plotly_chart(fig_line, use_container_width=True)

    # ---------- 🌟 分頁 4：重大歷史事件時間軸 ----------
    with tab4:
        st.subheader("⏳ 穩定幣重大歷史事件時間軸")
        st.markdown("透過回顧歷史上發生過的重大脫鉤或監管事件，可以幫助你了解極端波動背後的時空背景。")
        
        # 內建整理好的網路歷史重大事件資料
        events_data = [
            {
                "日期": "2017-11-19", 
                "事件名稱": "Tether 熱錢包遭駭", 
                "受影響幣種": "USDT", 
                "說明": "Tether 熱錢包被駭客盜取約 3000 萬顆 USDT。雖然官方表示未影響用戶備付資金，但引發了早期加密市場對穩定幣安全性的恐慌。"
            },
            {
                "日期": "2020-03-12", 
                "事件名稱": "COVID-19「312大崩盤」", 
                "受影響幣種": "全體加密資產", 
                "說明": "新冠疫情爆發導致全球金融市場流動性瞬間枯竭，美股與比特幣同步大暴跌。投資人恐慌性拋售所有資產換取美元，引發穩定幣的極端震盪。"
            },
            {
                "日期": "2022-05-09", 
                "事件名稱": "Terra (UST) 死亡螺旋", 
                "受影響幣種": "UST, USDT", 
                "說明": "演算法穩定幣 UST 嚴重脫鉤並最終歸零，LUNA 生態系崩盤。此事件引發幣圈史詩級恐慌，連帶拖累 USDT 遭遇擠兌拋售，一度短暫跌至約 $0.95 進行折價交易。"
            },
            {
                "日期": "2023-03-11", 
                "事件名稱": "美國矽谷銀行 (SVB) 倒閉", 
                "受影響幣種": "USDC, USDT, DAI", 
                "說明": "USDC 發行商 Circle 宣布約有 33 億美元儲備金卡在破產的 SVB 中，引發市場強烈恐慌，USDC 價格重挫至 $0.87。大量避險資金從 USDC 出逃並湧入 USDT，反而讓 USDT 產生短暫溢價（最高漲至 $1.06）。"
            },
            {
                "日期": "2025-07-18", 
                "事件名稱": "美國《GENIUS 法案》簽署", 
                "受影響幣種": "全體穩定幣", 
                "說明": "美國簽署首部全面的聯邦穩定幣法案。法案強制要求發行商必須維持 1:1 的高流動性資產（如美國國庫券）儲備，大幅改變了穩定幣的監管要求，降低了未來的脫鉤風險。"
            }
        ]
        
        df_events = pd.DataFrame(events_data)
        
        # 繪製視覺化時間軸 (散佈圖)
        fig_timeline = px.scatter(
            df_events, 
            x="日期", 
            y="事件名稱", 
            color="受影響幣種",
            hover_data=["說明"],
            title="歷史重大脫鉤與市場事件分布圖"
        )
        # 調整圖表點的大小與樣式
        fig_timeline.update_traces(marker=dict(size=16, symbol="diamond"))
        fig_timeline.update_layout(yaxis_title="", xaxis_title="發生日期", showlegend=True)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # 顯示詳細文字說明區塊
        st.markdown("### 📝 事件詳細解析")
        for event in events_data:
            with st.expander(f"📌 {event['日期']} - {event['事件名稱']}"):
                st.markdown(f"**受波及標的：** `{event['受影響幣種']}`")
                st.write(event['說明'])

except FileNotFoundError:
    st.error(f"❌ 找不到資料檔案：{file_name}")
