import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    elif 'Time' in df_rate.columns:
        df_rate = df_rate.set_index('Time')
    numeric_df_rate = df_rate.select_dtypes(include='number')
    columns_rate = numeric_df_rate.columns.tolist()

    default_index = columns_rate.index('USDT') if 'USDT' in columns_rate else 0
    target_y = st.selectbox("🎯 選擇分析應變數 (Y)：", columns_rate, index=default_index)

    # 建立標籤頁
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 1. 獨立影響完整報告",
        "🔥 2. 相關係數熱力圖",
        "📈 3. 變動率走勢圖",
        "⏳ 4. 歷史事件分析",
        "💰 5. 絕對價格走勢",
        "📄 6. 單一變數詳細報表",
        "🏆 7. 多變量複迴歸 (自動篩選)",
        "🔬 8. 穩定幣穩定性研究"
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
            elif 'Time' in df_abs.columns: df_abs = df_abs.set_index('Time')
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

    # ---------- 🌟 分頁 7：多變量複迴歸 ----------
    with tab7:
        st.subheader("🏆 多變量複迴歸分析 (自動篩選與 Excel 級完整報表)")
        st.markdown("**分析邏輯**：系統會自動篩選出單變量顯著 (**P < 0.05**) 的變數，並執行複迴歸。")
        if st.button("🚀 執行多變量複迴歸分析"):
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
                if len(multi_data) > 10:
                    y_multi = multi_data[target_y]
                    X_multi = sm.add_constant(multi_data[significant_xs])
                    model_multi = sm.OLS(y_multi, X_multi).fit()
                    st.markdown("### 📈 1. 迴歸統計 (Regression Statistics)")
                    reg_stats_multi = pd.DataFrame({
                        "指標": ["Multiple R", "R Square", "Adjusted R Square", "標準誤", "樣本數 (Observations)"],
                        "數值": [np.sqrt(model_multi.rsquared) if model_multi.rsquared > 0 else 0,
                                 model_multi.rsquared, model_multi.rsquared_adj, np.sqrt(model_multi.mse_resid), int(model_multi.nobs)]
                    })
                    st.table(reg_stats_multi.style.format({"數值": "{:.6f}"}))
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

    # ============================================================
    # 🔬 分頁 8：自選變數多變量迴歸
    # ============================================================
    with tab8:
        st.subheader("🔬 自選變數多變量迴歸分析")
        try:
            all_numeric_cols = numeric_df_rate.columns.tolist()

            st.markdown("""
            <div style="background:rgba(41,121,255,0.08);border-left:3px solid #2979ff;
                        padding:12px 16px;border-radius:4px;margin-bottom:12px">
            <b>使用說明</b>：自由選擇應變數 (Y) 與自變數 (X)，系統會執行 OLS 多變量迴歸並輸出完整統計報表。<br>
            💡 建議每組只選一個代表變數，避免共線性問題（例如股市只選 S&amp;P500，匯率只選 DXY）。
            </div>
            """, unsafe_allow_html=True)

            # ── Y 選擇器 ──────────────────────────────────────────
            col_y_sel, col_hint = st.columns([1, 2])
            with col_y_sel:
                stable_defaults = [c for c in ["USDT", "USDC", "USDS"] if c in all_numeric_cols]
                y_default_idx = all_numeric_cols.index(stable_defaults[0]) if stable_defaults else 0
                y_selected = st.selectbox("🎯 選擇應變數 Y", all_numeric_cols, index=y_default_idx, key="tab8_y")
            with col_hint:
                st.markdown(" ")
                st.caption(f"目前選擇：**{y_selected}**　樣本數：{numeric_df_rate[y_selected].dropna().shape[0]:,} 筆")

            # ── 分組建議 ──────────────────────────────────────────
            GROUP_SUGGEST = {
                "📈 加密市場": ["bitcoin", "PAXG_Price"],
                "🏛️ 股市（擇一）": ["S&P500", "NASDAQ_Price", "道瓊平均工業指數"],
                "😰 市場情緒": ["VIX"],
                "💵 美元強弱（擇一）": ["DXY_Index", "USD/EUR", "USD/JPY", "USD/CNY"],
                "🏦 利率/總經": ["美國公債10年期殖利率", "T10YIE(類似通膨日資料)", "聯準會利率", "UNRATE"],
            }
            with st.expander("💡 分組建議參考（點擊展開）— 每組擇一可降低共線性"):
                for grp, vars_ in GROUP_SUGGEST.items():
                    available = [v for v in vars_ if v in all_numeric_cols and v != y_selected]
                    st.markdown(f"**{grp}**：{', '.join(available) if available else '—'}")

            # ── X 多選器 ──────────────────────────────────────────
            x_candidates = [c for c in all_numeric_cols if c != y_selected]
            x_defaults_suggest = [v for v in
                ["bitcoin", "S&P500", "VIX", "DXY_Index", "美國公債10年期殖利率", "T10YIE(類似通膨日資料)"]
                if v in x_candidates]
            x_selected = st.multiselect(
                "📊 選擇自變數 X（可複選，建議 3–8 個）",
                options=x_candidates,
                default=x_defaults_suggest,
                key="tab8_x"
            )

            # ── VIF 共線性警告 ─────────────────────────────────────
            if len(x_selected) >= 2:
                try:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    vif_data = numeric_df_rate[x_selected].dropna()
                    if len(vif_data) > len(x_selected) + 5:
                        X_vif = sm.add_constant(vif_data).values
                        vif_vals = {col: variance_inflation_factor(X_vif, j + 1) for j, col in enumerate(x_selected)}
                        high_vif = {k: v for k, v in vif_vals.items() if v > 10}
                        if high_vif:
                            st.warning(
                                "⚠️ 共線性警告：以下變數 VIF > 10，建議從中只保留一個：\n"
                                + "、".join([f"**{k}** (VIF={v:.1f})" for k, v in high_vif.items()])
                            )
                except Exception:
                    pass

            # ── 執行迴歸 ──────────────────────────────────────────
            if st.button("🚀 執行多變量迴歸", key="tab8_run"):
                if len(x_selected) == 0:
                    st.error("請至少選擇一個自變數 X。")
                else:
                    reg_data = numeric_df_rate[[y_selected] + x_selected].dropna()
                    if len(reg_data) < len(x_selected) + 10:
                        st.error(f"有效樣本數不足（{len(reg_data)} 筆），請減少 X 數量或換一個 Y。")
                    else:
                        y_vec = reg_data[y_selected]
                        X_mat = sm.add_constant(reg_data[x_selected])
                        model8 = sm.OLS(y_vec, X_mat).fit()
                        st.success(f"✅ 迴歸完成！Y = {y_selected}，X = {len(x_selected)} 個變數，樣本數 = {int(model8.nobs):,}")

                        # 1. 迴歸統計
                        st.markdown("#### 📈 1. 迴歸統計")
                        c1, c2, c3, c4, c5 = st.columns(5)
                        for col, (label, val) in zip([c1,c2,c3,c4,c5], [
                            ("Multiple R",  f"{np.sqrt(max(model8.rsquared,0)):.4f}"),
                            ("R²",          f"{model8.rsquared:.4f}"),
                            ("Adj R²",      f"{model8.rsquared_adj:.4f}"),
                            ("標準誤",       f"{np.sqrt(model8.mse_resid):.6f}"),
                            ("樣本數 N",     f"{int(model8.nobs):,}"),
                        ]):
                            col.metric(label, val)

                        # 2. ANOVA
                        st.markdown("#### 📊 2. 變異數分析（ANOVA）")
                        anova8 = pd.DataFrame({
                            "變異源":        ["迴歸 (Regression)", "殘差 (Residual)", "總和 (Total)"],
                            "df":            [int(model8.df_model), int(model8.df_resid), int(model8.nobs-1)],
                            "SS":            [model8.ess, model8.ssr, model8.centered_tss],
                            "MS":            [model8.mse_model, model8.mse_resid, np.nan],
                            "F":             [model8.fvalue, np.nan, np.nan],
                            "Significance F":[model8.f_pvalue, np.nan, np.nan],
                        })
                        st.dataframe(anova8.style.format(
                            {"SS":"{:.6f}","MS":"{:.6f}","F":"{:.4f}","Significance F":"{:.4e}"}, na_rep=""
                        ), use_container_width=True)

                        # 3. 係數表
                        st.markdown("#### 🔍 3. 係數檢定表")
                        conf8 = model8.conf_int(alpha=0.05)
                        coef8_df = pd.DataFrame({
                            "變數":        model8.params.index,
                            "係數 (Coef)": model8.params.values,
                            "標準誤":       model8.bse.values,
                            "t 值":        model8.tvalues.values,
                            "P 值":        model8.pvalues.values,
                            "下限 95%":    conf8[0].values,
                            "上限 95%":    conf8[1].values,
                            "顯著性":       ["✅ 顯著" if p < 0.05 else "❌ 不顯著" for p in model8.pvalues.values],
                        })
                        st.dataframe(coef8_df.style.format({
                            "係數 (Coef)":"{:.6f}","標準誤":"{:.6f}",
                            "t 值":"{:.4f}","P 值":"{:.4e}",
                            "下限 95%":"{:.6f}","上限 95%":"{:.6f}",
                        }), use_container_width=True)

                        # 4. 係數圖
                        st.markdown("#### 📊 4. 係數圖（含 95% 信賴區間）")
                        cp = coef8_df[coef8_df["變數"] != "const"].copy()
                        fig_coef8 = go.Figure()
                        fig_coef8.add_trace(go.Scatter(
                            x=cp["係數 (Coef)"], y=cp["變數"], mode="markers",
                            marker=dict(color=["#2979ff" if p < 0.05 else "#888888" for p in cp["P 值"]], size=10),
                            error_x=dict(
                                type="data", symmetric=False,
                                array=(cp["上限 95%"]-cp["係數 (Coef)"]).values,
                                arrayminus=(cp["係數 (Coef)"]-cp["下限 95%"]).values,
                                color="#aaaaaa", thickness=1.5, width=6
                            ),
                            text=cp["顯著性"],
                            hovertemplate="%{y}<br>係數: %{x:.6f}<br>%{text}<extra></extra>"
                        ))
                        fig_coef8.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                        fig_coef8.update_layout(
                            title=f"各 X 對 {y_selected} 的影響係數（藍色 = 顯著 p<0.05）",
                            xaxis_title="迴歸係數", yaxis_title="",
                            height=max(300, len(x_selected)*45+100),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                        )
                        st.plotly_chart(fig_coef8, use_container_width=True)

                        # 5. 殘差圖
                        st.markdown("#### 🔬 5. 殘差診斷")
                        st.caption("殘差應隨機散佈在 0 附近，若有明顯趨勢或擴張代表模型假設可能不成立。")
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            x=model8.fittedvalues, y=model8.resid, mode="markers",
                            marker=dict(color="#2979ff", size=3, opacity=0.4)
                        ))
                        fig_resid.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                        fig_resid.update_layout(
                            title="殘差 vs. 配適值", xaxis_title="配適值 (Fitted)", yaxis_title="殘差 (Residual)",
                            height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)

                        # 6. 下載
                        st.markdown("#### 📥 6. 下載完整報告")
                        out8 = io.StringIO()
                        out8.write(f"=== 多變量迴歸分析報告 ===\nY = {y_selected}\nX = {', '.join(x_selected)}\n\n")
                        out8.write("1. 迴歸統計\n")
                        pd.DataFrame({
                            "指標": ["Multiple R","R²","Adj R²","標準誤","樣本數 N"],
                            "數值": [np.sqrt(max(model8.rsquared,0)), model8.rsquared,
                                     model8.rsquared_adj, np.sqrt(model8.mse_resid), int(model8.nobs)]
                        }).to_csv(out8, index=False)
                        out8.write("\n2. ANOVA\n")
                        anova8.to_csv(out8, index=False)
                        out8.write("\n3. 係數表\n")
                        coef8_df.to_csv(out8, index=False)
                        st.download_button(
                            f"📥 下載 {y_selected} 多變量分析報告 (CSV)",
                            data=out8.getvalue().encode("utf-8-sig"),
                            file_name=f"{y_selected}_自選多變量分析報告.csv",
                            mime="text/csv"
                        )

        except FileNotFoundError:
            st.error("❌ 找不到資料檔案，請確認 CSV 檔案與 APP.py 在同一目錄。")

except FileNotFoundError:
    st.error("❌ 找不到核心資料檔案。")
