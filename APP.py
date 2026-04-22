import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io
import plotly.express as px

# 設定網頁標題與寬度
st.set_page_config(page_title="穩定幣全維度影響力分析儀表板", layout="wide")

st.title("📊 穩定幣全維度影響力分析儀表板")
st.markdown("本系統整合了自動化單變量/多變量迴歸、歷史事件對照與專業統計報表，旨在提供深度的穩定幣波動誘因剖析。")

# --- 定義資料檔名稱 ---
file_rate = '0408穩定幣與變數數據(變動率).csv'
file_absolute = '0326穩定幣與變數數據(跑回歸用) - 工作表1.csv'

try:
    # 讀取數據 (變動率檔案為核心分析源)
    df_rate = pd.read_csv(file_rate)
    
    if 'Date' in df_rate.columns:
        df_rate = df_rate.set_index('Date')
    elif '日期' in df_rate.columns:
        df_rate = df_rate.set_index('日期')
        
    numeric_df_rate = df_rate.select_dtypes(include='number')
    columns_rate = numeric_df_rate.columns.tolist()

    default_index = columns_rate.index('USDT') if 'USDT' in columns_rate else 0
    target_y = st.selectbox("🎯 選擇分析應變數 (Y)：", columns_rate, index=default_index)

    # ==========================================
    # 建立 7 個標籤頁
    # ==========================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 1. 獨立影響完整報告", 
        "🔥 2. 相關係數熱力圖", 
        "📈 3. 變動率走勢圖", 
        "⏳ 4. 歷史事件分析",
        "💰 5. 絕對價格走勢",
        "📄 6. 單一變數詳細報表",
        "🏆 7. 多變量複迴歸 (自動篩選)"
    ])

    # ---------- 分頁 1：獨立影響完整報告 (全面升級匯出功能) ----------
    with tab1:
        st.subheader(f"🚀 {target_y} 獨立影響因子完整量化報告 (含 Excel 完整指標)")
        st.markdown("自動遍歷所有變數進行迴歸，並將 ANOVA 與迴歸統計資訊整合至單一報表中。")
        
        if st.button("📈 執行全量獨立分析並生成完整報告"):
            independent_vars = [col for col in columns_rate if col != target_y]
            all_results = []
            progress_bar = st.progress(0)
            
            for i, x_var in enumerate(independent_vars):
                progress_bar.progress((i + 1) / len(independent_vars))
                temp_data = numeric_df_rate[[target_y, x_var]].dropna()
                
                if len(temp_data) < 10: continue

                y = temp_data[target_y]
                X = sm.add_constant(temp_data[x_var])
                
                try:
                    model = sm.OLS(y, X).fit()
                    conf = model.conf_int(alpha=0.05) 

                    # 🌟 整合 Excel 報表中出現的所有資訊
                    all_results.append({
                        '變數名稱': x_var,
                        '樣本天數(N)': int(model.nobs),
                        'Multiple R (相關係數)': np.sqrt(model.rsquared) if model.rsquared > 0 else 0,
                        'R Square (判定係數)': model.rsquared,
                        'Adjusted R Square': model.rsquared_adj,
                        '標準誤 (Standard Error)': np.sqrt(model.mse_resid),
                        '影響係數(Coef)': model.params[x_var],
                        '係數標準誤': model.bse[x_var],
                        't 統計量': model.tvalues[x_var],
                        'P 值 (P-value)': model.pvalues[x_var],
                        '下限 95%': conf.loc[x_var, 0],
                        '上限 95%': conf.loc[x_var, 1],
                        'F 統計量': model.fvalue,
                        'F 顯著性(Prob F)': model.f_pvalue,
                        '迴歸平方和 (SS Reg)': model.ess,
                        '殘差平方和 (SS Res)': model.ssr,
                        '總平方和 (SS Total)': model.centered_tss,
                        '迴歸均方 (MS Reg)': model.mse_model,
                        '殘差均方 (MS Res)': model.mse_resid,
                        '顯著性': '⭐ 是' if model.pvalues[x_var] < 0.05 else '否',
                        '性質': '正相關' if model.params[x_var] > 0 else '負相關'
                    })
                except:
                    pass

            if all_results:
                report_df = pd.DataFrame(all_results).sort_values(by='R Square (判定係數)', ascending=False)
                
                # 顯示表格 (僅顯示核心欄位以避免畫面過擠，但 CSV 會包含全部)
                display_cols = ['變數名稱', '樣本天數(N)', 'R Square (判定係數)', '影響係數(Coef)', 'P 值 (P-value)', '顯著性', '性質']
                st.dataframe(report_df[display_cols].style.format({
                    'R Square (判定係數)': '{:.4f}', '影響係數(Coef)': '{:.6f}', 'P 值 (P-value)': '{:.4e}'
                }), use_container_width=True)

                # 🌟 匯出包含所有 Excel 指標的 CSV
                csv_buffer = io.BytesIO()
                report_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下載完整分析報告 (包含 SS/MS/F 等全指標 CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"{target_y}_全維度詳細分析報告.csv",
                    mime="text/csv"
                )
                st.success("分析完成！下載的 CSV 檔案已包含 ANOVA 平方和、均方與所有檢定數據。")
            else:
                st.warning("資料筆數不足。")

    # ---------- 分頁 2：相關係數熱力圖 ----------
    with tab2:
        st.subheader("變數之間的全局相關性 (Correlation)")
        corr_matrix = numeric_df_rate.corr().round(2)
        fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---------- 分頁 3：變動率走勢圖 ----------
    with tab3:
        st.subheader(f"觀測 {target_y} 與其他變數的波動軌跡")
        options = [col for col in columns_rate if col != target_y]
        selected_xs = st.multiselect("📊 選擇對比變數：", options, default=options[:1], key="ms_rate")
        if selected_xs:
            fig_line = px.line(df_rate, y=[target_y] + selected_xs, title="變動率走勢對比")
            st.plotly_chart(fig_line, use_container_width=True)

    # ---------- 分頁 4：重大歷史事件分析 ----------
    with tab4:
        st.subheader("⏳ 穩定幣重大歷史事件時間軸")
        events_data = [
            {"日期": "2020-03-12", "事件名稱": "COVID-19 黑色星期四", "受影響幣種": "USDT, USDC", "波動方向": "溢價 (上漲)", "說明": "全球流動性枯竭引發加密資產恐慌拋售。交易員爭相換取穩定幣作為保證金與避風港。"},
            {"日期": "2022-05-09", "事件名稱": "Terra (UST) 死亡螺旋", "受影響幣種": "UST, USDT", "波動方向": "市場擠兌", "說明": "演算法穩定幣 UST 崩盤，引發市場對中心化穩定幣儲備的懷疑。"},
            {"日期": "2023-03-11", "事件名稱": "矽谷銀行 (SVB) 倒閉", "受影響幣種": "USDC, DAI", "波動方向": "嚴重脫鉤 (下跌)", "說明": "Circle 宣布 33 億美元儲備受困於 SVB。USDC 二級市場價格重挫至 $0.87，DAI 同步脫鉤。"},
            {"日期": "2024-09-01", "事件名稱": "Sky Protocol (USDS) 啟動", "受影響幣種": "USDS", "波動方向": "市佔擴張", "說明": "MakerDAO 品牌重塑，推出具備原生收益機制的 USDS。"},
            {"日期": "2025-10-10", "事件名稱": "Ethena (USDe) 壓力測試", "受影響幣種": "USDe", "波動方向": "交易所閃崩", "說明": "宏觀避險情緒引發加密市場崩跌，USDe 在 Binance 因預言機與流動性問題短暫跌至 $0.65。"},
            {"日期": "2025-11-15", "事件名稱": "PYUSD DeFi 增長", "受影響幣種": "PYUSD", "波動方向": "供應量激增", "說明": "PayPal 佈局 DeFi，PYUSD 在以太坊供應量一週激增 6.25 億美元。"},
            {"日期": "2026-01-10", "事件名稱": "NovaBay 企業併購 SDEV", "受影響幣種": "USDS", "波動方向": "結構升級", "說明": "上市公司持股 SKY 代幣，深化傳統金融與 Sky 協議的整合。"},
            {"日期": "2026-03-17", "事件名稱": "PayPal (PYUSD) 全球佈局", "受影響幣種": "PYUSD", "波動方向": "採用率提升", "說明": "PYUSD 擴展至全球 70 個市場，優化跨境支付商用結算基礎設施。"}
        ]
        df_events = pd.DataFrame(events_data)
        fig_timeline = px.scatter(df_events, x="日期", y="受影響幣種", color="波動方向", hover_name="事件名稱", title="重大市場事件")
        fig_timeline.update_traces(marker=dict(size=18, symbol="diamond"))
        st.plotly_chart(fig_timeline, use_container_width=True)
        for event in events_data:
            with st.expander(f"📌 {event['日期']} - {event['事件名稱']}"):
                st.write(event['說明'])

    # ---------- 分頁 5：絕對價格走勢 ----------
    with tab5:
        st.subheader("💰 絕對價格/數值歷史走勢")
        try:
            df_abs = pd.read_csv(file_absolute)
            if 'Date' in df_abs.columns: df_abs = df_abs.set_index('Date')
            elif '日期' in df_abs.columns: df_abs = df_abs.set_index('日期')
            cols_abs = df_abs.select_dtypes(include='number').columns.tolist()
            if target_y in cols_abs:
                selected_xs_abs = st.multiselect("📊 選擇對比變數：", [c for c in cols_abs if c != target_y], key="ms_abs")
                fig_abs = px.line(df_abs, y=[target_y] + selected_xs_abs, title="原始數值走勢對比")
                st.plotly_chart(fig_abs, use_container_width=True)
        except FileNotFoundError:
            st.error(f"❌ 找不到絕對值檔案。")

    # ---------- 分頁 6：單一變數詳細報表 (仿 Excel) ----------
    with tab6:
        st.subheader("📄 單一變數深度報表 (仿 Excel)")
        selected_x_detail = st.selectbox("🎯 選擇分析 X 變數：", [c for c in columns_rate if c != target_y], key="detail_x")
        if st.button("📊 產出 ANOVA 與詳細統計表"):
            temp = numeric_df_rate[[target_y, selected_x_detail]].dropna()
            if len(temp) > 10:
                y = temp[target_y]
                X = sm.add_constant(temp[selected_x_detail])
                model = sm.OLS(y, X).fit()
                st.markdown("#### 1. 迴歸統計")
                st.table(pd.DataFrame({"指標": ["Multiple R", "R Square", "Adjusted R Square", "Standard Error", "Observations"], "數值": [np.sqrt(model.rsquared), model.rsquared, model.rsquared_adj, np.sqrt(model.mse_resid), int(model.nobs)]}))
                st.markdown("#### 2. 變異數分析 (ANOVA)")
                anova = pd.DataFrame({"變異源": ["迴歸", "殘差", "總計"], "df": [int(model.df_model), int(model.df_resid), int(model.nobs-1)], "SS": [model.ess, model.ssr, model.centered_tss], "MS": [model.mse_model, model.mse_resid, np.nan], "F": [model.fvalue, np.nan, np.nan]})
                st.table(anova)
            else: st.warning("資料不足。")

    # ---------- 分頁 7：多變量複迴歸 (含 CSV 下載功能) ----------
    with tab7:
        st.subheader("🏆 多變量複迴歸分析 (確認真兇)")
        if st.button("🚀 執行多變量分析"):
            independent_vars = [col for col in columns_rate if col != target_y]
            significant_xs = []
            for x_var in independent_vars:
                temp_data = numeric_df_rate[[target_y, x_var]].dropna()
                if len(temp_data) < 10: continue
                try:
                    model_temp = sm.OLS(temp_data[target_y], sm.add_constant(temp_data[x_var])).fit()
                    if model_temp.pvalues[x_var] < 0.05: significant_xs.append(x_var)
                except: pass
                
            if significant_xs:
                multi_data = numeric_df_rate[[target_y] + significant_xs].dropna()
                model_multi = sm.OLS(multi_data[target_y], sm.add_constant(multi_data[significant_xs])).fit()
                st.markdown(f"### 📈 複迴歸整體模型表現 (N={int(model_multi.nobs)})")
                st.table(pd.DataFrame({"指標": ["R Square", "Adj. R Square", "Prob (F-statistic)"], "數值": [model_multi.rsquared, model_multi.rsquared_adj, model_multi.f_pvalue]}))
                
                multi_df = pd.DataFrame({"變數": significant_xs, "Coef": model_multi.params[significant_xs], "P-value": model_multi.pvalues[significant_xs]})
                st.dataframe(multi_df, use_container_width=True)
                csv_multi = io.BytesIO()
                multi_df.to_csv(csv_multi, index=False, encoding='utf-8-sig')
                st.download_button("📥 下載多變量分析 CSV", data=csv_multi.getvalue(), file_name=f"{target_y}_多變量分析.csv", mime="text/csv")
            else: st.warning("無顯著變數。")

except FileNotFoundError:
    st.error(f"❌ 找不到核心資料檔案。")
