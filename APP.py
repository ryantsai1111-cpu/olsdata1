import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io
import plotly.express as px

# 設定網頁標題與寬度
st.set_page_config(page_title="穩定幣全維度影響力分析儀表板", layout="wide")

st.title("📊 穩定幣全維度影響力分析儀表板")
st.markdown("本系統整合了自動化單變量/多變量迴歸、歷史事件對照與專業統計報表，提供深度的穩定幣波動誘因剖析。")

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

    # 建立標籤頁
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 1. 獨立影響完整報告", 
        "🔥 2. 相關係數熱力圖", 
        "📈 3. 變動率走勢圖", 
        "⏳ 4. 歷史事件分析",
        "💰 5. 絕對價格走勢",
        "📄 6. 單一變數詳細報表",
        "🏆 7. 多變量複迴歸 (自動篩選)"
    ])

    # ---------- 分頁 1：獨立影響完整報告 ----------
    with tab1:
        st.subheader(f"🚀 {target_y} 獨立影響因子完整量化報告 (含 Excel 完整指標)")
        if st.button("📈 執行全量獨立分析並生成報告"):
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
                    all_results.append({
                        '變數名稱': x_var, '樣本天數(N)': int(model.nobs),
                        'Multiple R': np.sqrt(model.rsquared), 'R Square': model.rsquared,
                        'Adj R Square': model.rsquared_adj, '標準誤': np.sqrt(model.mse_resid),
                        '影響係數(Coef)': model.params[x_var], 'P 值 (P-value)': model.pvalues[x_var],
                        '迴歸平方和 (SS Reg)': model.ess, '殘差平方和 (SS Res)': model.ssr,
                        'F 統計量': model.fvalue, '顯著性': '⭐ 是' if model.pvalues[x_var] < 0.05 else '否'
                    })
                except: pass
            if all_results:
                report_df = pd.DataFrame(all_results).sort_values(by='R Square', ascending=False)
                st.dataframe(report_df.style.format({'R Square': '{:.4f}', '影響係數(Coef)': '{:.6f}', 'P 值 (P-value)': '{:.4e}'}), use_container_width=True)
                csv_buffer = io.BytesIO()
                report_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button("📥 下載單變量詳細報告 (CSV)", data=csv_buffer.getvalue(), file_name=f"{target_y}_單變量完整報告.csv", mime="text/csv")

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
            {"日期": "2020-03-12", "事件名稱": "COVID-19 黑色星期四", "受影響幣種": "USDT, USDC", "波動方向": "溢價", "說明": "全球恐慌拋售引發避險需求。"},
            {"日期": "2022-05-09", "事件名稱": "Terra (UST) 死亡螺旋", "受影響幣種": "UST, USDT", "波動方向": "擠兌", "說明": "UST 崩盤引發儲備信任危機。"},
            {"日期": "2023-03-11", "事件名稱": "矽谷銀行 (SVB) 倒閉", "受影響幣種": "USDC, DAI", "波動方向": "脫鉤", "說明": "Circle 儲備受困 SVB 引發 USDC 暴跌。"},
            {"日期": "2024-09-01", "事件名稱": "Sky Protocol (USDS) 啟動", "受影響幣種": "USDS", "波動方向": "擴張", "說明": "MakerDAO 品牌重塑，推出 USDS。"},
            {"日期": "2025-10-10", "事件名稱": "Ethena (USDe) 壓力測試", "受影響幣種": "USDe", "波動方向": "閃崩", "說明": "交易所預言機失效導致 USDe 閃崩。"},
            {"日期": "2025-11-15", "事件名稱": "PYUSD DeFi 增長", "受影響幣種": "PYUSD", "波動方向": "激增", "說明": "PayPal 佈局 DeFi 推動供應量。"},
            {"日期": "2026-01-10", "事件名稱": "NovaBay 併購 SDEV", "受影響幣種": "USDS", "波動方向": "整合", "說明": "上市公司持股 SKY 強化傳統金融對接。"},
            {"日期": "2026-03-17", "事件名稱": "PayPal 全球佈局", "受影響幣種": "PYUSD", "波動方向": "採用", "說明": "PYUSD 擴展至 70 個全球市場。"}
        ]
        df_events = pd.DataFrame(events_data)
        fig_timeline = px.scatter(df_events, x="日期", y="受影響幣種", color="波動方向", hover_name="事件名稱", title="重大市場事件")
        fig_timeline.update_traces(marker=dict(size=18, symbol="diamond"))
        st.plotly_chart(fig_timeline, use_container_width=True)

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
            st.error("❌ 找不到絕對值檔案。")

    # ---------- 分頁 6：單一變數詳細報表 ----------
    with tab6:
        st.subheader("📄 單一變數詳細報表 (仿 Excel)")
        selected_x_detail = st.selectbox("🎯 選擇分析 X 變數：", [c for c in columns_rate if c != target_y], key="detail_x")
        if st.button("📊 產出詳細統計表"):
            temp = numeric_df_rate[[target_y, selected_x_detail]].dropna()
            if len(temp) > 10:
                y = temp[target_y]; X = sm.add_constant(temp[selected_x_detail])
                model = sm.OLS(y, X).fit()
                st.markdown("#### 1. 迴歸統計")
                st.table(pd.DataFrame({"指標": ["Multiple R", "R Square", "Adj R Square", "標準誤", "樣本數"], "數值": [np.sqrt(model.rsquared), model.rsquared, model.rsquared_adj, np.sqrt(model.mse_resid), int(model.nobs)]}))
                st.markdown("#### 2. 變異數分析 (ANOVA)")
                anova = pd.DataFrame({"變異源": ["迴歸", "殘差", "總計"], "df": [int(model.df_model), int(model.df_resid), int(model.nobs-1)], "SS": [model.ess, model.ssr, model.centered_tss], "MS": [model.mse_model, model.mse_resid, np.nan], "F": [model.fvalue, np.nan, np.nan]})
                st.table(anova)
            else: st.warning("資料不足。")

    # ---------- 🌟 分頁 7：多變量複迴歸 (自動篩選與完整輸出) ----------
    with tab7:
        st.subheader("🏆 多變量複迴歸分析 (自動篩選與 Excel 級完整報表)")
        st.markdown("**分析邏輯**：系統會自動篩選出單變量顯著 (**P < 0.05**) 的變數，並執行複迴歸。")
        
        if st.button("🚀 執行多變量複迴歸分析"):
            independent_vars = [col for col in columns_rate if col != target_y]
            significant_xs = []
            
            # 1. 篩選顯著變數
            for x_var in independent_vars:
                temp_data = numeric_df_rate[[target_y, x_var]].dropna()
                if len(temp_data) < 10: continue
                try:
                    model_temp = sm.OLS(temp_data[target_y], sm.add_constant(temp_data[x_var])).fit()
                    if model_temp.pvalues[x_var] < 0.05: significant_xs.append(x_var)
                except: pass
                
            if significant_xs:
                multi_data = numeric_df_rate[[target_y] + significant_xs].dropna()
                if len(multi_data) > 10:
                    y_multi = multi_data[target_y]
                    X_multi = sm.add_constant(multi_data[significant_xs])
                    model_multi = sm.OLS(y_multi, X_multi).fit()
                    
                    # --- 1. 迴歸統計 ---
                    st.markdown("### 📈 1. 迴歸統計 (Regression Statistics)")
                    reg_stats_multi = pd.DataFrame({
                        "指標": ["Multiple R", "R Square", "Adjusted R Square", "標準誤", "樣本數 (Observations)"],
                        "數值": [np.sqrt(model_multi.rsquared) if model_multi.rsquared > 0 else 0, 
                                model_multi.rsquared, model_multi.rsquared_adj, np.sqrt(model_multi.mse_resid), int(model_multi.nobs)]
                    })
                    st.table(reg_stats_multi.style.format({"數值": "{:.6f}"}))
                    
                    # --- 2. 變異數分析 (ANOVA) ---
                    st.markdown("### 📊 2. 變異數分析 (ANOVA)")
                    anova_multi = pd.DataFrame({
                        "變異源": ["迴歸 (Regression)", "殘差 (Residual)", "總和 (Total)"],
                        "df": [int(model_multi.df_model), int(model_multi.df_resid), int(model_multi.nobs - 1)],
                        "SS": [model_multi.ess, model_multi.ssr, model_multi.centered_tss],
                        "MS": [model_multi.mse_model, model_multi.mse_resid, np.nan],
                        "F": [model_multi.fvalue, np.nan, np.nan],
                        "Significance F": [model_multi.f_pvalue, np.nan, np.nan]
                    })
                    st.dataframe(anova_multi.style.format({"SS": "{:.6f}", "MS": "{:.6f}", "F": "{:.6f}", "Significance F": "{:.4e}"}, na_rep=""), use_container_width=True)
                    
                    # --- 3. 係數表 ---
                    st.markdown("### 🔍 3. 係數檢定表 (Coefficients)")
                    conf_int_multi = model_multi.conf_int(alpha=0.05)
                    coef_multi_df = pd.DataFrame({
                        "變數名稱": ["Intercept (截距)"] + significant_xs,
                        "影響係數 (Coef)": model_multi.params.values,
                        "標準誤 (Std Error)": model_multi.bse.values,
                        "t 統計量 (t Stat)": model_multi.tvalues.values,
                        "P 值 (P-value)": model_multi.pvalues.values,
                        "下限 95%": conf_int_multi[0].values,
                        "上限 95%": conf_int_multi[1].values,
                        "顯著性": ["⭐ 依然顯著" if p < 0.05 else "❌ 失去顯著性" for p in model_multi.pvalues.values]
                    })
                    st.dataframe(coef_multi_df.style.format({"影響係數 (Coef)": "{:.6f}", "標準誤 (Std Error)": "{:.6f}", "t 統計量 (t Stat)": "{:.4f}", "P 值 (P-value)": "{:.4e}", "下限 95%": "{:.6f}", "上限 95%": "{:.6f}"}), use_container_width=True)
                    
                    # --- 🌟 匯出完整 CSV 功能 ---
                    output_csv = io.StringIO()
                    output_csv.write("1. REGRESSION STATISTICS\n")
                    reg_stats_multi.to_csv(output_csv, index=False)
                    output_csv.write("\n2. ANOVA\n")
                    anova_multi.to_csv(output_csv, index=False)
                    output_csv.write("\n3. COEFFICIENTS\n")
                    coef_multi_df.to_csv(output_csv, index=False)
                    
                    st.download_button(
                        label="📥 下載多變量完整報告 (含 ANOVA 與 統計指標)",
                        data=output_csv.getvalue().encode('utf-8-sig'),
                        file_name=f"{target_y}_多變量完整分析報告.csv",
                        mime="text/csv"
                    )
                    st.success("✅ 多變量分析完成！報表已包含所有迴歸指標。")
                else:
                    st.error("⚠️ 合併後的有效資料天數不足。")
            else: st.warning("找不到具備顯著性的變數。")

except FileNotFoundError:
    st.error("❌ 找不到核心資料檔案。")
