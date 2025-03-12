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
from PIL import Image

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
    prompt = f"""Extract data from the given trade confirmation.

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
        st.write("Starting PDF processing...")
        text = extract_text_from_pdf(pdf_path)
        st.write(f"Extracted text length: {len(text)}")
        
        # Save extracted text to a file
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        response_path = os.path.join(downloads_path, f'response_text.txt')
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(text)

        prompt1 = generate_prompt(text)
        st.write("Generated prompt")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )

        st.write("Making API call...")
        with st.spinner('Processing PDF...'):
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
        
        st.write("Received API response")
        if completion is None or completion.choices is None or len(completion.choices) == 0:
            st.error("Error: No completion received from API")
            return None, None, None
            
        data = completion.choices[0].message.content
        st.write(f"API response received")  

        # Save response to a file
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        response_path = os.path.join(downloads_path, f'response_info_{filename}.txt')
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(data)
            st.write(f"Response saved to: {response_path}")

        try:
            # Find the JSON part of the response
            st.write("Attempting to parse JSON...")
            start_idx = data.find('{')
            end_idx = data.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                st.error("Error: No JSON found in response")
                st.code(data)
                return None, None, None
                
            json_string = data[start_idx:end_idx + 1]
            
            parsed_data = json.loads(json_string)
            st.write("Successfully parsed JSON")
            
            df1 = pd.DataFrame([parsed_data])
            df = df1

            # Rest of the processing...
            st.write("Processing dates...")
            date_columns = ['Final Valuation Date', 'Last Valuation Date', 
                          'Floating Amount Payment Date', 'Cash Settlement Payment Date']
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        st.error(f"Error converting {col} to datetime: {str(e)}")

            st.write("Calculating day lags...")
            if all(col in df.columns for col in ['Final Valuation Date', 'Floating Amount Payment Date']):
                df['Market day lag'] = df.apply(
                    lambda row: len(pd.bdate_range(start=row['Final Valuation Date'], 
                                                 end=row['Floating Amount Payment Date'])) - 1 
                    if pd.notnull(row['Final Valuation Date']) and pd.notnull(row['Floating Amount Payment Date']) 
                    else None, 
                    axis=1
                )

            if all(col in df.columns for col in ['Final Valuation Date', 'Cash Settlement Payment Date']):
                df['Payment day lag'] = df.apply(
                    lambda row: len(pd.bdate_range(start=row['Final Valuation Date'], 
                                                 end=row['Cash Settlement Payment Date'])) - 1
                    if pd.notnull(row['Final Valuation Date']) and pd.notnull(row['Cash Settlement Payment Date'])
                    else None,
                    axis=1
                )

            st.write("Processing DataFrame...")
            df_merged = df
            if 'Index' in df_merged.columns:
                df_merged['Index'] = df_merged['Index'].astype('string')

            st.write("Matching units...")
            if 'Number of units' not in df_merged.columns:
                st.error("Error: 'Number of units' column not found")
                st.write(f"Available columns: {df_merged.columns.tolist()}")
                return None, None, None

            df_list = df_merged['Number of units'].tolist()
            dl_suite = d1_suite_excel
            
            if 'Units' not in dl_suite.columns:
                st.error("Error: 'Units' column not found in excel file")
                st.write(f"Available columns in excel: {dl_suite.columns.tolist()}")
                return None, None, None

            matching_index = dl_suite[dl_suite['Units'].isin(df_list)]

            st.write("Creating final DataFrame...")
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

            st.write("Converting dates...")
            for date_col in ['Strike Date', 'Trade Date']:
                if date_col in df_final.columns:
                    df_final[date_col] = pd.to_datetime(df_final[date_col], errors='coerce')

            if matching_index.empty:
                st.warning("No matching records found")
                return None, df_final, None

            matching = matching_index.copy()
            if 'Strike Date' in matching.columns:
                matching['Strike Date'] = pd.to_datetime(matching['Strike Date'], errors='coerce')

            st.write("Validating matches...")
            unmatched_values, validation_status = validate_match_conditions(matching, df_final)

            return unmatched_values, df_final, matching

        except json.JSONDecodeError as e:
            st.error(f"JSON decode error: {str(e)}")
            st.code(json_string)
            return None, None, None
            
        except Exception as e:
            st.error(f"Error in data processing: {str(e)}")
            st.exception(e)
            return None, None, None

    except Exception as e:
        st.error(f"Error in process_pdf: {str(e)}")
        st.exception(e)
        return None, None, None

def main():
    # Configure the page
    st.set_page_config(
        page_title="TRS Booking Validation Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.2);
    }
    .file-uploader {
        border: 1px dashed #E2E8F0;
        border-radius: 6px;
        padding: 10px;
        background-color: #F8FAFC;
    }
    .validation-success {
        background-color: #e6f4ea;
    }
    .validation-error {
        background-color: #ffebee;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([1, 9])
    with col1:
        st.markdown("📊")
    with col2:
        st.title("TRS Booking Validation Tool")
    
    st.markdown("---")

    # Upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Trade Confirmations")
        pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, 
                                   key="pdf_upload", help="Upload trade confirmation PDFs")
        
        if pdf_files:
            st.success(f"{len(pdf_files)} PDF file(s) uploaded")
            for file in pdf_files:
                st.write(f"📑 {file.name}")
    
    with col2:
        st.subheader("📊 Upload Booking File")
        excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], 
                                    key="excel_upload", help="Upload booking Excel file")
        
        if excel_file:
            st.success(f"✅ {excel_file.name} uploaded")
        
        # Option to view sample booking data
        with st.expander("View Default Booking Data"):
            try:
                default_excel_path = r"C:\Users\prana\Downloads\trs_booking.xlsx"
                if os.path.exists(default_excel_path):
                    default_df = pd.read_excel(default_excel_path, engine='openpyxl')
                    st.dataframe(default_df, use_container_width=True)
                else:
                    st.warning("Default booking file not found. Please upload your own file.")
            except Exception as e:
                st.error(f"Error loading default file: {str(e)}")
    
    # Process button
    st.markdown("---")
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        process_button = st.button("🔄 Process Files", use_container_width=True)
    
    # Process files
    if process_button:
        if not pdf_files:
            st.error("⚠️ Please upload at least one PDF file")
        else:
            # Load Excel data
            if excel_file:
                dl_suite_excel = pd.read_excel(excel_file, engine='openpyxl')
            else:
                try:
                    default_excel_path = r"C:\Users\prana\Downloads\trs_booking.xlsx"
                    dl_suite_excel = pd.read_excel(default_excel_path, engine='openpyxl')
                    st.info("Using default booking file")
                except Exception as e:
                    st.error(f"Error loading default file: {str(e)}")
                    st.stop()
            
            # Process each PDF
            for pdf_file in pdf_files:
                st.markdown("---")
                st.subheader(f"Results for {pdf_file.name}")
                
                # Process the PDF
                pdf_content = pdf_file.read()
                unmatched_values, df_final, matching = process_pdf(pdf_content, dl_suite_excel, pdf_file.name)
                
                if df_final is not None:
                    # Show validation results if available
                    if unmatched_values is not None and matching is not None:
                        # Create DataFrame from validation results
                        if unmatched_values:
                            st.subheader("Validation Results")
                            for issue in unmatched_values:
                                st.error(issue)
                        else:
                            st.success("✅ All fields matched successfully!")
                        
                        # Get validation status
                        _, validation_status = validate_match_conditions(matching, df_final)
                        
                        # Display tables with validation status
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Confirmation Details")
                            
                            # Apply styling to highlight matched/unmatched fields
                            styled_df = df_final.copy()
                            for col, status in validation_status.items():
                                if col in styled_df.columns:
                                    if status:
                                        styled_df[col] = styled_df[col].apply(lambda x: f"✅ {x}")
                                    else:
                                        styled_df[col] = styled_df[col].apply(lambda x: f"❌ {x}")
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                        with col2:
                            st.subheader("Booking Details")
                            
                            # Apply styling to highlight matched/unmatched fields
                            styled_matching = matching.copy()
                            for col, status in validation_status.items():
                                if col in styled_matching.columns:
                                    if status:
                                        styled_matching[col] = styled_matching[col].apply(lambda x: f"✅ {x}")
                                    else:
                                        styled_matching[col] = styled_matching[col].apply(lambda x: f"❌ {x}")
                            
                            st.dataframe(styled_matching, use_container_width=True)
                    else:
                        st.error("❌ No matching booking found!")
                        st.subheader("Confirmation Details")
                        st.dataframe(df_final, use_container_width=True)
                else:
                    st.error("⚠️ No data extracted from PDF")

if __name__ == "__main__":
    main()