import streamlit as st
import pandas as pd
import statsmodels.api as sm
import io
import os

# 設定網頁標題與寬度
st.set_page_config(page_title="穩定幣變數影響力分析工具", layout="wide")

st.title("📊 穩定幣變數影響力分析工具")
st.markdown("請在下方選擇一個你想研究的「應變數 (Y)」，系統會自動將資料集中的其他欄位當作「自變數 (X)」，並逐一分析它們對 Y 的獨立影響力。")

# 1. 預設讀取綁定的 CSV 檔案 (確保檔名與上傳到 GitHub 的完全一致)
file_name = '0408穩定幣與變數數據(變動率).csv'

try:
    # 讀取相同資料夾下的檔案
    df = pd.read_csv(file_name)
    
    with st.expander("👀 點擊預覽資料集前 5 筆"):
        st.dataframe(df.head())

    # 2. 讓同學動態選擇應變數 (Y)
    columns = df.columns.tolist()
    
    # 預設選項可以設定為 USDT (如果有的話)
    default_index = columns.index('USDT') if 'USDT' in columns else 0
    target_y = st.selectbox("🎯 請選擇你要預測/分析的「應變數」(Y)：", columns, index=default_index)

    # 3. 按下按鈕後開始分析
    if st.button("🚀 開始執行成對迴歸分析"):
        
        # 自動將「非 Y」的所有欄位變成 X 變數清單
        independent_vars = [col for col in columns if col != target_y]
        
        results_list = []
        
        # 建立一個進度條讓同學知道進度
        progress_bar = st.progress(0)
        st.info(f"正在分析 {len(independent_vars)} 個變數，請稍候...")

        for i, x_var in enumerate(independent_vars):
            progress_bar.progress((i + 1) / len(independent_vars))
            
            # 確保資料是數值類型，避免字串欄位導致報錯
            if not pd.api.types.is_numeric_dtype(df[x_var]) or not pd.api.types.is_numeric_dtype(df[target_y]):
                continue
            
            temp_data = df[[target_y, x_var]].dropna()
            sample_size = len(temp_data)
            
            if sample_size < 10:
                continue

            # 執行簡單線性迴歸
            y = temp_data[target_y]
            X = sm.add_constant(temp_data[x_var])
            
            try:
                model = sm.OLS(y, X).fit()
                
                p_val = model.pvalues[x_var]
                coef = model.params[x_var]
                r_squared = model.rsquared

                # 產生白話文解釋
                if p_val < 0.05:
                    significance = '⭐ 是'
                    if coef > 0:
                        explanation = f'正向影響：當 {x_var} 上漲時，{target_y} 傾向跟著上漲。'
                    else:
                        explanation = f'負向影響：當 {x_var} 上漲時，{target_y} 傾向下跌。'
                else:
                    significance = '否'
                    explanation = f'無顯著影響：此變數與 {target_y} 的波動無統計上的明顯關聯。'

                results_list.append({
                    '變數名稱': x_var,
                    '樣本天數': sample_size,
                    '影響係數(Coef)': round(coef, 6),
                    'P值(P-value)': round(p_val, 4),
                    '解釋力(R-squared)': round(r_squared, 6),
                    '顯著性': significance,
                    '結果解釋': explanation
                })
            except Exception as e:
                pass

        # 4. 彙整結果並顯示
        if results_list:
            summary_table = pd.DataFrame(results_list)
            summary_table = summary_table.sort_values(by='解釋力(R-squared)', ascending=False)
            
            st.subheader("📈 分析結果彙整表 (依解釋力排序)")
            st.dataframe(summary_table, use_container_width=True)

            # 5. 提供 CSV 下載按鈕
            csv_buffer = io.BytesIO()
            summary_table.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_bytes = csv_buffer.getvalue()

            st.download_button(
                label="📥 下載完整分析結果 (CSV)",
                data=csv_bytes,
                file_name=f"{target_y}_影響力分析結果.csv",
                mime="text/csv"
            )
            st.balloons()
        else:
            st.warning("⚠️ 找不到足夠的有效數值資料來進行迴歸分析。")

except FileNotFoundError:
    st.error(f"❌ 找不到預設的資料檔案：{file_name}")
    st.info("請確認該 CSV 檔案已與 app.py 放在同一個 GitHub 資料夾中。")