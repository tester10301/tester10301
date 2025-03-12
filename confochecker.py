from openai import OpenAI
import traceback
import pandas as pd
import datetime
from pandas.tseries.offsets import BDay
import requests
import os
import json
import re
import base64
import io
import pdfplumber
import streamlit as st

def extract_text_from_pdf(pdf_content_bytes):
    """
    Extract text from PDF with improved settings to prevent word splitting.
    
    Args:
        pdf_content_bytes (bytes): PDF file content in bytes
        
    Returns:
        str: Extracted text with proper word spacing
    """
    text = ""
    pdf_bytes = io.BytesIO(pdf_content_bytes)
    
    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                if i > 1:  # Only process first two pages
                    break
                
                # Extract text with custom settings
                page_text = page.extract_text(
                    x_tolerance=3,  # Increase tolerance for horizontal text grouping
                    y_tolerance=3,  # Increase tolerance for vertical text grouping
                    layout=True,    # Maintain text layout
                    use_text_flow=True,  # Use text flow for better word grouping
                    extra_attrs=['size', 'fontname']  # Extract font information
                )
                
                if page_text:
                    text += page_text + "\n"
        
        # Basic cleanup without regex
        text = text.replace('\n\n', '\n').strip()
        
        return text
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def generate_prompt(text):
    prompt = f"""Do exactly as instructed. Extract data from the given trade confirmation.

    Take lookback period as number from Designated Maturity or Fixing Lookback.
    In Benchmark Lookback give format <Benchmark-Lookbeck "BD"> (example LIBOR AVG-1 80 or LIBOR OHP-3 BO).
    If Party A is Equity Amount Payer, then Direction is Short, else it is Long.

    Give data in json for all these fields:

        Settlement Currency: <text>
        Number of units: <number>
        Initial Price: <number>
        Notional Amount: <number>
        Lookback period: <number>
        Benchmark: <text from Floating Rate Option>
        Spread: <number>
        Early Termination Party A: <Yes or No>
        Early Termination Party B: <Yes or No>
        Benchmark Lookback: <text>
        Closing Break Costs: <whole number>
        Party A: <text>
        Direction: <text>
        Strike Date: <date>
        Trade Date: <date>
        Final Valuation Date: <date>
        Counterparty: <text>
        Swap Type: <Total Return Swap or Price Return Swap>
        Index: <text>
        Floating Amount Payment Date: <date> (Final Settlement Payment Date)
        Cash Settlement Payment Date: <date> (Final Settlement Payment Date)

    Input: {text}
    Output:
    """
    return prompt

def validate_match_conditions(matching, df_final):
    conditions = []
    validation_status = {}  # Dictionary to store column validation status
    
    # Create conditions only if columns exist in both dataframes
    if 'Strike Date' in matching.columns and 'Strike Date' in df_final.columns:
        strike_date_match = (matching['Strike Date'] == df_final['Strike Date'])
        conditions.append(('Strike Date', strike_date_match))
        validation_status['Strike Date'] = strike_date_match.all()
    
    if 'Expiry Date' in matching.columns and 'Expiry Date' in df_final.columns:
        expiry_date_match = (matching['Expiry Date'] == df_final['Expiry Date'])
        conditions.append(('Expiry Date', expiry_date_match))
        validation_status['Expiry Date'] = expiry_date_match.all()
    
    if 'Direction' in matching.columns and 'Direction' in df_final.columns:
        direction_match = (matching['Direction'].str.lower() == df_final['Direction'].str.lower())
        conditions.append(('Direction', direction_match))
        validation_status['Direction'] = direction_match.all()
    
    if 'Index' in matching.columns and 'Index' in df_final.columns:
        index_match = df_final['Index'].str.contains(matching['Index'].iloc[0], na=False)
        conditions.append(('Index', index_match))
        validation_status['Index'] = index_match.all()
    
    if 'Spread' in matching.columns and 'Spread' in df_final.columns:
        spread_match = (matching['Spread'] == df_final['Spread'])
        conditions.append(('Spread', spread_match))
        validation_status['Spread'] = spread_match.all()

    if 'Swap ccy' in matching.columns and 'Swap ccy' in df_final.columns:
        swap_ccy_match = (matching['Swap ccy'] == df_final['Swap ccy'])
        conditions.append(('Swap ccy', swap_ccy_match))
        validation_status['Swap ccy'] = swap_ccy_match.all()
    
    if 'Counterparty' in matching.columns and 'Counterparty' in df_final.columns:
        counterparty_match = df_final['Counterparty'].str.contains(matching['Counterparty'].iloc[0], na=False)
        conditions.append(('Counterparty', counterparty_match))
        validation_status['Counterparty'] = counterparty_match.all()
    
    if 'Market day lag' in matching.columns and 'Market day lag' in df_final.columns:
        market_day_lag_match = (matching['Market day lag'] == df_final['Market day lag'])
        conditions.append(('Market day lag', market_day_lag_match))
        validation_status['Market day lag'] = market_day_lag_match.all()
    
    if 'Payment day lag' in matching.columns and 'Payment day lag' in df_final.columns:
        payment_day_lag_match = (matching['Payment day lag'] == df_final['Payment day lag'])
        conditions.append(('Payment day lag', payment_day_lag_match))
        validation_status['Payment day lag'] = payment_day_lag_match.all()
    
    if 'Benchmark' in matching.columns and 'Benchmark' in df_final.columns:
        benchmark_match = df_final['Benchmark'].str.contains(matching['Benchmark'].iloc[0], na=False)
        conditions.append(('Benchmark', benchmark_match))
        validation_status['Benchmark'] = benchmark_match.all()
    
    if 'Units' in matching.columns and 'Units' in df_final.columns:
        units_match = (matching['Units'] == df_final['Units'])
        conditions.append(('Units', units_match))
        validation_status['Units'] = units_match.all()
    
    if 'BF' in matching.columns and 'BF' in df_final.columns:
        bf_match = (matching['BF'] == df_final['BF'])
        conditions.append(('BF', bf_match))
        validation_status['BF'] = bf_match.all()
    
    if 'Strike Price' in matching.columns and 'Strike Price' in df_final.columns:
        strike_price_match = (matching['Strike Price'] == df_final['Strike Price'])
        conditions.append(('Strike Price', strike_price_match))
        validation_status['Strike Price'] = strike_price_match.all()

    unmatched_values = []
    for col_name, condition in conditions:
        if not condition.all():
            unmatched_values.append(f"Unmatched Values in column: {col_name}")

    return unmatched_values, validation_status  # Return both unmatched values and validation status

def process_pdf(pdf_path, d1_suite_excel, filename):
    try:
        with st.spinner('Extracting text from PDF...'):
            text = extract_text_from_pdf(pdf_path)
            # st.info(f"📄 Extracted text from PDF ({len(text)} characters)")
        
        # Save extracted text to a file
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        response_path = os.path.join(downloads_path, f'response_text.txt')
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(text)

        prompt1 = generate_prompt(text)



        with st.spinner('Analyzing trade confirmation...'):
            completion = client.chat.completions.create(
                extra_body={},
                model="qwen/qwen2.5-vl-72b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt1
                            }
                        ]
                    }
                ]
            )
        
        if completion is None or completion.choices is None or len(completion.choices) == 0:
            st.error("❌ No data received from AI assistant")
            return None, None, None
            
        data = completion.choices[0].message.content
        # st.success("✅ Trade confirmation data extracted successfully")

        # Save response to a file
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        response_path = os.path.join(downloads_path, f'response_info_{filename}.txt')
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(data)

        try:
            # Find the JSON part of the response
            start_idx = data.find('{')
            end_idx = data.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                st.error("❌ No structured data found in response")
                st.code(data)
                return None, None, None
                
            json_string = data[start_idx:end_idx + 1]
            
            parsed_data = json.loads(json_string)
            
            df1 = pd.DataFrame([parsed_data])
            df = df1

            # Process dates
            date_columns = ['Final Valuation Date', 'Last Valuation Date', 
                          'Floating Amount Payment Date', 'Cash Settlement Payment Date',
                          'Strike Date', 'Trade Date']
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        st.warning(f"⚠️ Date conversion issue with {col}: {str(e)}")

            # Calculate day lags
            if all(col in df.columns for col in ['Final Valuation Date', 'Floating Amount Payment Date']):
                try:
                    df['Market day lag'] = df.apply(
                        lambda row: len(pd.bdate_range(start=pd.to_datetime(row['Final Valuation Date']), 
                                                    end=pd.to_datetime(row['Floating Amount Payment Date']))) - 1 
                        if pd.notnull(row['Final Valuation Date']) and pd.notnull(row['Floating Amount Payment Date']) 
                        else None, 
                        axis=1
                    )
                except Exception as e:
                    st.warning(f"⚠️ Market day lag calculation issue: {str(e)}")

            if all(col in df.columns for col in ['Final Valuation Date', 'Cash Settlement Payment Date']):
                try:
                    df['Payment day lag'] = df.apply(
                        lambda row: len(pd.bdate_range(start=pd.to_datetime(row['Final Valuation Date']), 
                                                    end=pd.to_datetime(row['Cash Settlement Payment Date']))) - 1
                        if pd.notnull(row['Final Valuation Date']) and pd.notnull(row['Cash Settlement Payment Date'])
                        else None,
                        axis=1
                    )
                except Exception as e:
                    st.warning(f"⚠️ Payment day lag calculation issue: {str(e)}")

            df_merged = df
            if 'Index' in df_merged.columns:
                df_merged['Index'] = df_merged['Index'].astype('string')

            if 'Number of units' not in df_merged.columns:
                st.error("❌ 'Number of units' column not found in extracted data")
                st.write(f"Available columns: {df_merged.columns.tolist()}")
                return None, None, None

            df_list = df_merged['Number of units'].tolist()
            dl_suite = d1_suite_excel
            
            if 'Units' not in dl_suite.columns:
                st.error("❌ 'Units' column not found in booking file")
                st.write(f"Available columns in excel: {dl_suite.columns.tolist()}")
                return None, None, None

            matching_index = dl_suite[dl_suite['Units'].isin(df_list)]

            df_final = df_merged.copy()

            # Rename columns safely
            column_mapping = {
                'Settlement Currency': 'Swap ccy',
                'Number of units': 'Units',
                'Benchmark_Lookback': 'Benchmark',
                'Initial Price': 'Strike Price',
                'Final Valuation Date': 'Expiry Date',
                'Closing Break Costs': 'BF'
            }
            
            df_final = df_final.rename(columns={k: v for k, v in column_mapping.items() 
                                              if k in df_final.columns})

            desired_columns = [
                'Strike Date', 'Trade Date', 'Direction', 'Swap Type', 'Index', 'Party A',
                'Swap ccy', 'Counterparty', 'Units', 'Strike Price', 'Expiry Date',
                'Notional Amount', 'Market day lag', 'Payment day lag', 'Early Termination Party A',
                'Benchmark', 'Spread', 'Early Termination Party B', 'BF'
            ]
            
            existing_columns = [col for col in desired_columns if col in df_final.columns]
            df_final = df_final[existing_columns]

            if matching_index.empty:
                st.warning("⚠️ No matching booking records found")
                return None, df_final, None

            matching = matching_index.copy()
            
            # Format dates in matching dataframe
            for date_col in ['Strike Date', 'Expiry Date']:
                if date_col in matching.columns:
                    matching[date_col] = pd.to_datetime(matching[date_col], errors='coerce').dt.strftime('%Y-%m-%d')

            unmatched_values, validation_status = validate_match_conditions(matching, df_final)

            return unmatched_values, df_final, matching

        except json.JSONDecodeError as e:
            st.error(f"❌ JSON parsing error: {str(e)}")
            st.code(json_string)
            return None, None, None
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.exception(e)
            return None, None, None

    except Exception as e:
        st.error(f"❌ Error in PDF processing: {str(e)}")
        st.exception(e)
        return None, None, None

def main():
    # Configure the page
    st.set_page_config(
        page_title="TRS Booking Validation Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    def style_dataframe(df, validation_status):
        # Create a copy to avoid modifying the original
        styled_df = df.copy()
        
        # Create a DataFrame of the same shape as styled_df filled with empty strings
        styles = pd.DataFrame('', index=styled_df.index, columns=styled_df.columns)
        
        # Apply styles based on validation status
        for col in styled_df.columns:
            if col in validation_status:
                if validation_status[col]:  # If values match
                    styles[col] = 'background-color: #D1FAE5;'  # Light green
                else:  # If values don't match
                    styles[col] = 'background-color: #FEE2E2;'  # Light red
        
        return styles

    # Add custom CSS
    st.markdown("""
    <style>
/* General layout */
.main .block-container {
    padding: 1rem 2rem;
}

/* Button */
.stButton>button {
    background: #4F46E5;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: background 0.3s;
}
.stButton>button:hover {
    background: #3730A3;
}

/* File uploader */
.file-uploader {
    border: 2px dashed #CBD5E1;
    border-radius: 8px;
    padding: 1rem;
    background: #F1F5F9;
}

/* Header */
.header-container {
    padding: 1rem 1.5rem;
    background: #4F46E5;
    border-radius: 8px;
    color: white;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-title {
    font-size: 1.75rem;
    margin: 0;
}
.subheader {
    margin: 0;
    font-size: 1rem;
    opacity: 0.9;
}

/* Cards */
.card {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    background: white;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Status boxes */
.success-container, .warning-container, .error-container {
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin: 1rem 0;
}
.success-container {
    background: #DCFCE7;
    border-left: 4px solid #16A34A;
}
.warning-container {
    background: #FEF9C3;
    border-left: 4px solid #D97706;
}
.error-container {
    background: #FEE2E2;
    border-left: 4px solid #DC2626;
}

/* Divider */
.divider {
    height: 2px;
    background: #E5E7EB;
    margin: 2rem 0;
    border-radius: 1px;
}

/* Dataframes */
.stDataFrame, .dataframe-container {
    border-radius: 8px;
    overflow: hidden;
    background: white;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border: 1px solid #E2E8F0;
}

/* Dataframe container title */
.dataframe-container .card-title {
    margin-bottom: 0.5rem;
}
</style>
    """, unsafe_allow_html=True)

    # Custom function to style cells based on validation status
    def highlight_cells(dataframe, column, validation_status):
        styles = []
        for col in dataframe.columns:
            if col == column and col in validation_status and not validation_status[col]:
                styles.append({
                    'selector': f'td.col{dataframe.columns.get_loc(col)}',
                    'props': 'background-color: #FEE2E2;'
                })
            elif col == column and col in validation_status and validation_status[col]:
                styles.append({
                    'selector': f'td.col{dataframe.columns.get_loc(col)}',
                    'props': 'background-color: #D1FAE5;'
                })
        return styles

    # Header with custom styling
    st.markdown("""
    <div class="header-container">
        <div class="header-icon">🔍</div>
        <div>
            <h1 class="header-title">TRS Booking Validation Tool</h1>
            <p class="subheader">Verify trade confirmations against booking records</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section with cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">
                <span class="card-icon">📄</span> Upload Trade Confirmations
            </div>
        """, unsafe_allow_html=True)
        
        pdf_files = st.file_uploader("Drag and drop PDF files here", 
                                    type=["pdf"], 
                                    accept_multiple_files=True, 
                                    key="pdf_upload", 
                                    help="Upload trade confirmation PDFs")
        
        if pdf_files:
            # st.markdown(f"""
            # <div class="success-container">
            #     <strong>✅ {len(pdf_files)} PDF file(s) uploaded successfully</strong>
            # </div>
            # """, unsafe_allow_html=True)
            
            for file in pdf_files:
                st.markdown(f"📑 {file.name}")
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">
                <span class="card-icon">📊</span> Upload Booking File
            </div>
        """, unsafe_allow_html=True)
        
        excel_file = st.file_uploader("Drag and drop Excel file here", 
                                    type=["xlsx", "xls"], 
                                    key="excel_upload", 
                                    help="Upload booking Excel file")
        
        if excel_file:
            st.markdown(f"""
            <div class="success-container">
                <strong>✅ {excel_file.name} uploaded successfully</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Option to view sample booking data
        with st.expander("View Default Booking Data"):
            try:
                default_excel_path = r"C:\Users\prana\Downloads\trs_booking.xlsx"
                if os.path.exists(default_excel_path):
                    default_df = pd.read_excel(default_excel_path, engine='openpyxl')
                    # Format dates
                    for col in default_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(default_df[col]):
                            default_df[col] = default_df[col].dt.strftime('%Y-%m-%d')
                    st.dataframe(default_df, use_container_width=True)
                else:
                    st.warning("Default booking file not found. Please upload your own file.")
            except Exception as e:
                st.error(f"Error loading default file: {str(e)}")
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process button with enhanced styling
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        process_button = st.button("🔄 Process Files", use_container_width=True)
    
    # Process files
    if process_button:
        if not pdf_files:
            st.markdown("""
            <div class="error-container">
                <strong>❌ Please upload at least one PDF file</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Load Excel data
            if excel_file:
                dl_suite_excel = pd.read_excel(excel_file, engine='openpyxl')
            else:
                try:
                    default_excel_path = r"C:\Users\prana\Downloads\trs_booking.xlsx"
                    dl_suite_excel = pd.read_excel(default_excel_path, engine='openpyxl')
                    # st.markdown("""
                    # <div class="warning-container">
                    #     <strong>ℹ️ Using default booking file</strong>
                    # </div>
                    # """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-container">
                        <strong>❌ Error loading default file: {str(e)}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
            
            # Process each PDF
            for pdf_file in pdf_files:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="card-title">
                    <span class="card-icon">📋</span> Results for {pdf_file.name}
                </div>
                """, unsafe_allow_html=True)
                
                # Process the PDF
                pdf_content = pdf_file.read()
                unmatched_values, df_final, matching = process_pdf(pdf_content, dl_suite_excel, pdf_file.name)
                
                if df_final is not None:
                    # Show validation results if available
                    if unmatched_values is not None and matching is not None:
                        # Get validation status
                        _, validation_status = validate_match_conditions(matching, df_final)
                        
                        # Create validation summary
                        if unmatched_values:
                            st.markdown("""
                            <div class="error-container">
                                <strong>❌ Validation issues detected</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            for issue in unmatched_values:
                                st.warning(issue)
                        else:
                            st.markdown("""
                            <div class="success-container">
                                <strong>✅ All fields matched successfully!</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display confirmation table
                        st.markdown("""
                        <div class="dataframe-container">
                            <div class="card-title">
                                <span class="card-icon">📄</span> Confirmation Details
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Apply cell background styling based on validation status
                        styled_df = df_final.copy()
                        st.dataframe(styled_df, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display booking table
                        st.markdown("""
                        <div class="dataframe-container">
                            <div class="card-title">
                                <span class="card-icon">📊</span> Booking Details
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Apply cell background styling based on validation status
                        styled_matching = matching.style.apply(lambda _: style_dataframe(matching, validation_status), axis=None)
                        st.dataframe(styled_matching, use_container_width=True)

                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="error-container">
                            <strong>❌ No matching booking found!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="dataframe-container">
                            <div class="card-title">
                                <span class="card-icon">📄</span> Confirmation Details
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.dataframe(df_final, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-container">
                        <strong>❌ No data could be extracted from this PDF</strong>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()