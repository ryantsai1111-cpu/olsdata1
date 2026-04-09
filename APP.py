import streamlit as st
import pandas as pd
import statsmodels.api as sm
import io
import plotly.express as px

# 設定網頁標題與寬度
st.set_page_config(page_title="穩定幣變數分析儀表板", layout="wide")

st.title("📊 穩定幣變數分析儀表板")
st.markdown("請在下方選擇應變數 (Y)，並透過不同的標籤頁 (Tabs) 進行全方位的數據分析與歷史背景回顧。")

# --- 定義兩個資料檔的名稱 ---
file_rate = '0408穩定幣與變數數據(變動率).csv'
file_absolute = '0326穩定幣與變數數據(跑回歸用) - 工作表1.csv'  # <--- 新增的絕對值檔案

try:
    # 讀取變動率檔案 (供 Tab 1~3 使用)
    df_rate = pd.read_csv(file_rate)
    
    if 'Date' in df_rate.columns:
        df_rate = df_rate.set_index('Date')
    elif '日期' in df_rate.columns:
        df_rate = df_rate.set_index('日期')
        
    numeric_df_rate = df_rate.select_dtypes(include='number')
    columns_rate = numeric_df_rate.columns.tolist()

    default_index = columns_rate.index('USDT') if 'USDT' in columns_rate else 0
    target_y = st.selectbox("🎯 請選擇你要預測/分析的「應變數」(Y)：", columns_rate, index=default_index)

    # 建立 5 個標籤頁
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 1. 獨立影響力分析", 
        "🔥 2. 相關係數熱力圖", 
        "📈 3. 變動率走勢圖", 
        "⏳ 4. 歷史事件分析",
        "💰 5. 絕對價格走勢圖" # <--- 新增的 Tab
    ])

    # ---------- 分頁 1：獨立影響力分析 ----------
    with tab1:
        st.subheader(f"探討各變數對 {target_y} 的獨立影響")
        if st.button("🚀 開始執行成對迴歸分析"):
            independent_vars = [col for col in columns_rate if col != target_y]
            results_list = []
            progress_bar = st.progress(0)
            
            for i, x_var in enumerate(independent_vars):
                progress_bar.progress((i + 1) / len(independent_vars))
                temp_data = numeric_df_rate[[target_y, x_var]].dropna()
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
        corr_matrix = numeric_df_rate.corr().round(2)
        fig_heat = px.imshow(
            corr_matrix, text_auto=True, aspect="auto", 
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---------- 分頁 3：變動率走勢圖 ----------
    with tab3:
        st.subheader(f"觀測 {target_y} 與其他變數的「變動率」波動軌跡")
        st.markdown("此圖表顯示的是每日變動百分比，適合觀察短期波動性。")
        options = [col for col in columns_rate if col != target_y]
        default_selections = options[:2] if len(options) >= 2 else []
        selected_xs = st.multiselect("📊 請選擇要疊加對比的變數：", options, default=default_selections, key="ms_rate")
        
        if selected_xs or target_y:
            vars_to_plot = [target_y] + selected_xs
            fig_line = px.line(df_rate, y=vars_to_plot, title="變動率走勢對比")
            st.plotly_chart(fig_line, use_container_width=True)

    # ---------- 分頁 4：重大歷史事件與波動分析 ----------
    with tab4:
        st.subheader("⏳ 穩定幣重大歷史事件時間軸與結構分析")
        st.markdown("本頁面基於《2020至2026年全球穩定幣市場波動、總體經濟衝擊與結構演變深度分析報告》彙整，探討不同穩定幣機制在極端事件中的真實抗壓能力。")
        
        events_data = [
            {"日期": "2020-03-12", "事件名稱": "COVID-19「黑色星期四」流動性危機", "受影響幣種": "USDT, USDC", "波動方向": "溢價 (上漲)", "說明": "全球資本市場因疫情恐慌暴跌（比特幣單日跌 37.17%）。衍生品市場面臨連環強平，投資人瘋狂買入 USDT 與 USDC 避險。因傳統匯款延遲與區塊鏈擁堵，導致套利機制短暫失效，USDT 變動率異常高達 +5.48%。"},
            {"日期": "2022-05-09", "事件名稱": "Terra (UST) 演算法穩定幣死亡螺旋", "受影響幣種": "USDT, USDC", "波動方向": "USDT 恐慌拋售 / USDC 資金流入", "說明": "UST 崩潰引發對所有穩定幣儲備真實性的恐慌。USDT 因持有商業票據等資產，遭遇數十億美元擠兌贖回；同時間資金轉向被認為更安全（由現金與美債支持）的 USDC，造成板塊內資金輪動。"},
            {"日期": "2023-03-11", "事件名稱": "矽谷銀行 (SVB) 倒閉反噬", "受影響幣種": "USDC, DAI", "波動方向": "嚴重脫鉤 (下跌)", "說明": "Circle 宣布 33 億美元儲備金卡在破產的 SVB 中。因週末電匯關閉無法贖回，USDC 在二級市場暴跌至 $0.87。DAI 因儲備中包含大量 USDC，遭受「DeFi 嵌套風險」牽連而嚴重脫鉤。"},
            {"日期": "2024-09-01", "事件名稱": "MakerDAO 升級為 Sky Protocol (USDS)", "受影響幣種": "USDS", "波動方向": "市佔率擴張", "說明": "DAI 正式重塑為 USDS，引入跨鏈能力與原生收益（Sky Savings Rate）分配機制，將國庫券收益分給用戶，在聯準會高息環境下吸引了大量資本。"},
            {"日期": "2025-10-10", "事件名稱": "Ethena (USDe) 極端波動壓力測試", "受影響幣種": "USDe", "波動方向": "CEX 極端折價 / DEX 維持穩定", "說明": "宏觀地緣政治引發加密市場崩盤。依賴 Delta 中性對沖的 USDe，因中心化交易所 (Binance) 預言機與流動性枯竭短暫暴跌至 $0.65，但去中心化交易所 (DEX) 仍堅守 $0.99 附近，展現了流動性隔離效應。"},
            {"日期": "2025-11-15", "事件名稱": "PYUSD 於 DeFi 生態爆發性增長", "受影響幣種": "PYUSD", "波動方向": "供應量激增", "說明": "PayPal 發力去中心化金融領域，PYUSD 在以太坊上的供應量於短短一週內激增 6.25 億美元，迅速躍升為 DeFi 生態系中第六大穩定幣。"},
            {"日期": "2026-01-10", "事件名稱": "傳統股權市場與 Sky (USDS) 深度綁定", "受影響幣種": "USDS", "波動方向": "資本結構升級", "說明": "納斯達克上市公司 NovaBay 更名為 Stablecoin Development Corp (SDEV)，並持有高達 21.5 億枚 SKY 代幣（佔總供應量 9.15%），標誌著傳統金融與 USDS 協議的深度整合。"},
            {"日期": "2026-03-17", "事件名稱": "PayPal (PYUSD) 全球佈局擴展", "受影響幣種": "PYUSD", "波動方向": "全球採用率增長", "說明": "PYUSD 將使用權限擴展至全球 70 個市場。透過大幅優化跨國清算週期，將穩定幣的應用場景從「加密資產交易媒介」推進為「全球商業結算基礎設施」。"}
        ]
        
        df_events = pd.DataFrame(events_data)
        
        fig_timeline = px.scatter(
            df_events, x="日期", y="受影響幣種", color="波動方向",
            hover_name="事件名稱", title="歷史重大市場事件分佈與影響"
        )
        fig_timeline.update_traces(marker=dict(size=18, symbol="diamond"))
        fig_timeline.update_layout(yaxis_title="受影響幣種", xaxis_title="發生日期", showlegend=True, height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("### 🔍 穩定幣機制與波動原因深度解析")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**1. 避險擠兌與流動性錯配**")
            st.write("如 USDT 在 2022 年 UST 崩潰時遭遇的壓力。當市場極度恐慌時，若穩定幣的儲備資產（如商業票據）無法瞬間變現以應付龐大的贖回需求，就會在二級市場產生折價脫鉤。")
        with col2:
            st.warning("**2. 傳統金融的傳染風險**")
            st.write("如 USDC 在 2023 年 SVB 倒閉事件。法幣抵押型穩定幣高度依賴傳統銀行的託管安全。一旦託管銀行破產，即會引發恐慌拋售。")
        with col3:
            st.success("**3. DeFi 嵌套與預言機失效**")
            st.write("如 DAI 在 SVB 事件被牽連，以及 USDe 在 2025 年因交易所定價失常閃崩。依賴智能合約的穩定幣，風險常來自演算法連鎖反應或數據源錯誤。")

        st.markdown("---")
        for event in events_data:
            with st.expander(f"📌 {event['日期']} - {event['事件名稱']}"):
                st.markdown(f"**受波及標的：** `{event['受影響幣種']}` | **短期價格表現：** `{event['波動方向']}`")
                st.write(event['說明'])

    # ---------- 🌟 分頁 5：絕對價格走勢圖 (新增) ----------
    with tab5:
        st.subheader("💰 變數「絕對價格/數值」歷史走勢")
        st.markdown("此圖表讀取的是資產的原始價格（或指數原始點數），適合觀察長期趨勢或真實脫鉤幅度。")
        
        try:
            # 讀取絕對值檔案
            df_abs = pd.read_csv(file_absolute)
            if 'Date' in df_abs.columns:
                df_abs = df_abs.set_index('Date')
            elif '日期' in df_abs.columns:
                df_abs = df_abs.set_index('日期')
                
            columns_abs = df_abs.select_dtypes(include='number').columns.tolist()
            
            # 確保應變數 Y 在新檔案中也存在
            if target_y in columns_abs:
                options_abs = [col for col in columns_abs if col != target_y]
                
                # 因為是看價格，預設先不選其他對比項目，以免如比特幣(數萬)和USDT(1)放在同一張圖比例會跑掉
                selected_xs_abs = st.multiselect("📊 請選擇要疊加對比的變數：", options_abs, default=[], key="ms_abs")
                
                vars_to_plot_abs = [target_y] + selected_xs_abs
                fig_abs_line = px.line(df_abs, y=vars_to_plot_abs, title=f"原始數值走勢對比 (基準: {target_y})")
                st.plotly_chart(fig_abs_line, use_container_width=True)
                
                st.info("💡 **提示：** 絕對價格圖表如果將標的物（如 USDT 約 1 美元）與大數據（如 Bitcoin 約數萬美元）放在一起，會導致線條被壓縮。建議分開觀察，或僅對比數值相近的穩定幣。")
            else:
                st.warning(f"⚠️ 在絕對值檔案中找不到目前選擇的應變數：{target_y}")
                
        except FileNotFoundError:
            st.error(f"❌ 找不到絕對值資料檔案：{file_absolute}")
            st.info("請確認該檔案已上傳至 GitHub，且檔名與程式碼中設定的完全一致。")

except FileNotFoundError:
    st.error(f"❌ 找不到變動率資料檔案：{file_rate}")
