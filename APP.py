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
        st.subheader(f"🚀 {target_y} 獨立影響因子完整量化報告 (單變量海選)")
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
                        '變數名稱': x_var,
                        '樣本天數': int(model.nobs),
                        '影響係數(Coef)': model.params[x_var],
                        '標準誤(Std Error)': model.bse[x_var],
                        't 統計量(t Stat)': model.tvalues[x_var],
                        'P 值(P-value)': model.pvalues[x_var],
                        '判定係數(R-squared)': model.rsquared,
                        '調整後 R-squared': model.rsquared_adj,
                        'F 統計量': model.fvalue,
                        'F 顯著性(Prob F)': model.f_pvalue,
                        '下限 95%': conf.loc[x_var, 0],
                        '上限 95%': conf.loc[x_var, 1],
                        '顯著性': '⭐ 是' if model.pvalues[x_var] < 0.05 else '否'
                    })
                except: pass

            if all_results:
                report_df = pd.DataFrame(all_results).sort_values(by='判定係數(R-squared)', ascending=False)
                st.dataframe(report_df.style.format({
                    '影響係數(Coef)': '{:.6f}', '標準誤(Std Error)': '{:.6f}', 't 統計量(t Stat)': '{:.4f}',
                    'P 值(P-value)': '{:.4e}', '判定係數(R-squared)': '{:.4f}', '調整後 R-squared': '{:.4f}',
                    'F 統計量': '{:.4f}', 'F 顯著性(Prob F)': '{:.4e}', '下限 95%': '{:.6f}', '上限 95%': '{:.6f}'
                }), use_container_width=True)
                csv_buffer = io.BytesIO()
                report_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button("📥 下載單變量分析報告 (CSV)", data=csv_buffer.getvalue(), file_name=f"{target_y}_單變量分析.csv", mime="text/csv")

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
            {"日期": "2020-03-12", "事件名稱": "COVID-19 黑色星期四", "受影響幣種": "USDT, USDC", "波動方向": "溢價 (上漲)", "說明": "全球流動性枯竭引發恐慌拋售。交易員爭相換取穩定幣作為保證金與避風港。"},
            {"日期": "2022-05-09", "事件名稱": "Terra (UST) 死亡螺旋", "受影響幣種": "UST, USDT", "波動方向": "市場擠兌", "說明": "演算法穩定幣 UST 崩盤，引發市場對穩定幣儲備的懷疑。"},
            {"日期": "2023-03-11", "事件名稱": "矽谷銀行 (SVB) 倒閉", "受影響幣種": "USDC, DAI", "波動方向": "嚴重脫鉤 (下跌)", "說明": "Circle 宣布儲備金受困於 SVB，USDC 價格重挫至 $0.87。"},
            {"日期": "2024-09-01", "事件名稱": "Sky Protocol (USDS) 啟動", "受影響幣種": "USDS", "波動方向": "市佔擴張", "說明": "MakerDAO 品牌重塑，推出具備原生收益機制的 USDS。"},
            {"日期": "2025-10-10", "事件名稱": "Ethena (USDe) 壓力測試", "受影響幣種": "USDe", "波動方向": "交易所閃崩", "說明": "地緣政治引發崩盤，USDe 在 Binance 因預言機與流動性問題跌至 $0.65。"},
            {"日期": "2025-11-15", "事件名稱": "PYUSD DeFi 增長", "受影響幣種": "PYUSD", "波動方向": "供應量激增", "說明": "PayPal 佈局 DeFi，PYUSD 供應量一週激增 6.25 億美元。"},
            {"日期": "2026-01-10", "事件名稱": "NovaBay 企業併購 SDEV", "受影響幣種": "USDS", "波動方向": "結構升級", "說明": "上市公司持股 SKY 代幣，深化傳統金融整合。"},
            {"日期": "2026-03-17", "事件名稱": "PayPal (PYUSD) 全球佈局", "受影響幣種": "PYUSD", "波動方向": "採用率提升", "說明": "PYUSD 擴展至全球 70 個市場，推動跨國商業結算基礎設施。"}
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
                if selected_xs_abs:
                    fig_abs = px.line(df_abs, y=[target_y] + selected_xs_abs, title="原始數值走勢對比")
                    st.plotly_chart(fig_abs, use_container_width=True)
        except FileNotFoundError:
            st.error("❌ 找不到絕對值檔案。")

    # ---------- 分頁 6：單一變數詳細報表 ----------
    with tab6:
        st.subheader("📄 單一變數深度報表 (仿 Excel)")
        selected_x_detail = st.selectbox("🎯 選擇分析 X 變數：", [c for c in columns_rate if c != target_y], key="detail_x")
        if st.button("📊 產出詳細統計表"):
            temp = numeric_df_rate[[target_y, selected_x_detail]].dropna()
            if len(temp) > 10:
                y = temp[target_y]
                X = sm.add_constant(temp[selected_x_detail])
                model = sm.OLS(y, X).fit()
                st.markdown("#### 1. 迴歸統計")
                reg_stats = pd.DataFrame({"指標": ["R Square", "Adjusted R Square", "Standard Error", "Observations"], "數值": [model.rsquared, model.rsquared_adj, np.sqrt(model.mse_resid), int(model.nobs)]})
                st.dataframe(reg_stats.style.format({"數值": "{:.6f}"}), use_container_width=True)
                st.markdown("#### 2. 係數檢定表")
                conf_int = model.conf_int(alpha=0.05)
                coef_df = pd.DataFrame({"變數": ["Intercept (截距)", selected_x_detail], "Coefficients": model.params.values, "P-value": model.pvalues.values, "Lower 95%": conf_int[0].values, "Upper 95%": conf_int[1].values})
                st.dataframe(coef_df.style.format({"Coefficients": "{:.6f}", "P-value": "{:.4e}", "Lower 95%": "{:.6f}", "Upper 95%": "{:.6f}"}), use_container_width=True)

    # ---------- 🌟 分頁 7：多變量複迴歸 (含 CSV 下載功能) ----------
    with tab7:
        st.subheader("🏆 多變量複迴歸分析 (確認真兇)")
        st.markdown("**分析邏輯**：系統會先自動篩選單變量中具備顯著性 (**P < 0.05**) 的變數，再執行複迴歸分析。")
        
        if st.button("🚀 執行多變量複迴歸並生成報表"):
            independent_vars = [col for col in columns_rate if col != target_y]
            significant_xs = []
            
            for x_var in independent_vars:
                temp_data = numeric_df_rate[[target_y, x_var]].dropna()
                if len(temp_data) < 10: continue
                y_temp = temp_data[target_y]
                X_temp = sm.add_constant(temp_data[x_var])
                try:
                    model_temp = sm.OLS(y_temp, X_temp).fit()
                    if model_temp.pvalues[x_var] < 0.05:
                        significant_xs.append(x_var)
                except: pass
                
            if not significant_xs:
                st.warning("⚠️ 找不到任何具備顯著性的變數。")
            else:
                multi_data = numeric_df_rate[[target_y] + significant_xs].dropna()
                if len(multi_data) > 10:
                    y_multi = multi_data[target_y]
                    X_multi = sm.add_constant(multi_data[significant_xs])
                    model_multi = sm.OLS(y_multi, X_multi).fit()
                    
                    st.markdown("---")
                    st.markdown(f"### 📈 複迴歸整體模型表現 (樣本數：{int(model_multi.nobs)} 天)")
                    model_stats = pd.DataFrame({
                        "指標": ["判定係數 (R-squared)", "調整後 R-squared", "模型整體顯著性 (Prob F)"],
                        "數值": [model_multi.rsquared, model_multi.rsquared_adj, model_multi.f_pvalue]
                    })
                    st.table(model_stats.style.format({"數值": "{:.4e}"}))
                    
                    st.markdown("### 🔍 各變數實質影響力對決 (係數表)")
                    multi_results = []
                    for var in significant_xs:
                        p_val = model_multi.pvalues[var]
                        multi_results.append({
                            "變數名稱": var,
                            "影響係數 (Coef)": model_multi.params[var],
                            "標準誤 (Std Error)": model_multi.bse[var],
                            "P 值 (P-value)": p_val,
                            "多變量顯著性": "⭐ 依然顯著" if p_val < 0.05 else "❌ 失去顯著性"
                        })
                    multi_df = pd.DataFrame(multi_results)
                    st.dataframe(multi_df.style.format({"影響係數 (Coef)": "{:.6f}", "標準誤 (Std Error)": "{:.6f}", "P 值 (P-value)": "{:.4e}"}), use_container_width=True)
                    
                    # 🌟 核心新增：CSV 下載按鈕
                    csv_multi_buffer = io.BytesIO()
                    multi_df.to_csv(csv_multi_buffer, index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下載多變量分析報告 (CSV)",
                        data=csv_multi_buffer.getvalue(),
                        file_name=f"{target_y}_多變量複迴歸分析.csv",
                        mime="text/csv"
                    )
                    st.success("✅ 多變量分析完成！你可以下載報告並與單變量結果進行對照。")
                else:
                    st.error("⚠️ 合併後的有效資料天數不足。")

except FileNotFoundError:
    st.error(f"❌ 找不到核心資料檔案。")
