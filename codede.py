# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
# from xbbg import blp
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from datetime import datetime
import plotly.graph_objects as go
import traceback
# Set page configuration
st.set_page_config(page_title="HedgeFinder", page_icon=":chart_with_upwards_trend:", layout="wide")

# Define function to create default universe
def create_default_universe():
    # Create a dummy dataframe with sample stocks
    data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CSCO'],
        'sector': ['Technology'] * 10,
        'country': ['US'] * 10
    }
    return pd.DataFrame(data)

# Define functions for analysis
def calculate_z_score(spread):
    """Calculate z-score of a spread series"""
    if len(spread) == 0:
        return 0
    # Check for std being zero to avoid division by zero
    std = np.std(spread)
    if std == 0:
        return 0
    # Calculate z-scores for the entire series
    z_scores = (spread - np.mean(spread)) / std
    return z_scores

def perform_cointegration_test(stock_a, stock_b):
    """Perform cointegration test between two stock price series"""
    # First create a model to find the hedge ratio
    # Add a constant to the model
    stock_b_with_const = sm.add_constant(stock_b)
    model = sm.OLS(stock_a, stock_b_with_const).fit()
    
    # Extract the coefficient (excluding the constant)
    hedge_ratio = model.params[1]  # Changed from model.params[0]
    
    # Calculate the spread
    spread = stock_a - hedge_ratio * stock_b
    
    # Use Augmented Dickey-Fuller test for stationarity (cointegration)
    adf_result = adfuller(spread)[1]  # Return p-value
    return adf_result, spread, hedge_ratio

def calculate_half_life(spread):
    """Calculate half-life of mean reversion"""
    if len(spread) <= 1:
        return np.nan
        
    delta_spread = np.diff(spread)
    lagged_spread = spread[:-1]
    model = sm.OLS(delta_spread, sm.add_constant(lagged_spread)).fit()
    
    # Fixed: Handle parameter access more robustly
    # Attempt to get the slope parameter either by index or column name
    try:
        # First try to access by index
        beta = model.params[1]
    except (KeyError, IndexError):
        try:
            # If that fails, try to access by column name
            beta = model.params['x1']
        except (KeyError, IndexError):
            # If both fail, try to extract the second parameter regardless of index
            if len(model.params) >= 2:
                beta = model.params.iloc[1]
            else:
                # If there's somehow only one parameter, use that (the slope)
                beta = model.params.iloc[0]
    
    if beta >= 0:
        return np.nan
    half_life = -np.log(2) / beta
    return half_life

def winsorize_returns(returns, threshold=0.05):
    """Winsorize returns to cap extreme values"""
    return returns.clip(lower=returns.quantile(threshold), upper=returns.quantile(1-threshold))

def filter_stocks(df, ticker, sector='Y', domicility='Y'):
    """Filter stocks based on sector and domicility"""
    # Check if ticker exists in the dataframe
    if ticker not in df['ticker'].values:
        st.write(f"Ticker {ticker} not found in the database.")
        return pd.DataFrame()

    # Get the sector and domicility of the input ticker
    stock_info = df[df['ticker'] == ticker].iloc[0]
    stock_sector = stock_info['sector']
    stock_domicility = stock_info['country']

    # Filter the dataframe based on the conditions
    filtered_df = df.copy()

    if sector == 'Y':
        filtered_df = filtered_df[filtered_df['sector'] == stock_sector]

    if domicility == 'Y':
        filtered_df = filtered_df[filtered_df['country'] == stock_domicility]

    # Remove the input ticker from the results
    filtered_df = filtered_df[filtered_df['ticker'] != ticker]
    
    return filtered_df

# def fetch_stock_data(tickers, start_date, end_date):
#     """Fetch historical stock data using yfinance"""
#     if isinstance(tickers, str):
#         tickers = [tickers]
        
#     start_dt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
#     end_dt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    
#     data = {}
#     for ticker in tickers:
#         try:
#             # Request historical data from Bloomberg
#             stock_data = blp.bdh(ticker, 'PX_LAST', start_dt, end_dt, Fill='P', Days='T')
            
#             if not stock_data.empty:
#                 # Rename column to match our existing code expectations
#                 stock_data.columns = [c.replace('PX_LAST', 'Adj Close') for c in stock_data.columns]
#                 data[ticker] = stock_data['Adj Close']
#         except Exception as e:
#             st.error(f"Error fetching Bloomberg data for {ticker}: {e}")
    
#     return data

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data using yfinance"""
    if isinstance(tickers, str):
        tickers = [tickers]
        
    data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Use Adjusted Close for analysis
            if not stock_data.empty:
                data[ticker] = stock_data['Close']
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    
    return data
def find_best_hedge(data, target_ticker, filtered_tickers, action='BUY', max_results=5):
    """Find the best hedge stocks based on multiple metrics"""
    if target_ticker not in data:
        return []
    
    target_prices = data[target_ticker]
    results = []
    
    for hedge_ticker in filtered_tickers:
        if hedge_ticker not in data or hedge_ticker == target_ticker:
            continue
            
        hedge_prices = data[hedge_ticker]
        
        # Make sure both price series have the same dates
        common_dates = target_prices.index.intersection(hedge_prices.index)
        if len(common_dates) < 30:  # Need sufficient data points
            continue
            
        target_common = target_prices.loc[common_dates]
        hedge_common = hedge_prices.loc[common_dates]
        
        # Calculate correlation
        # Convert to Series explicitly to avoid DataFrame ambiguity
        if isinstance(target_common, pd.DataFrame):
            target_common = target_common.iloc[:, 0]
        if isinstance(hedge_common, pd.DataFrame):
            hedge_common = hedge_common.iloc[:, 0]
            
        target_returns = target_common.pct_change().dropna()
        hedge_returns = hedge_common.pct_change().dropna()
        
        # Check if returns series are empty
        if len(target_returns) == 0 or len(hedge_returns) == 0:
            continue
        
        # Make sure we're working with Series, not DataFrames for correlation
        if isinstance(target_returns, pd.DataFrame):
            target_returns = target_returns.iloc[:, 0]
        if isinstance(hedge_returns, pd.DataFrame):
            hedge_returns = hedge_returns.iloc[:, 0]
            
        # Calculate correlation between Series objects
        try:
            correlation = target_returns.corr(hedge_returns)
        except Exception as e:
            st.error(f"Error calculating correlation for {ticker}: {e}")
            continue
        
        # Check if correlation is NaN (don't filter based on correlation threshold)
        if pd.isna(correlation):
            continue
            
        # Perform cointegration test first to get spread values
        try:
            coint_p_value, spread_values, hedge_ratio = perform_cointegration_test(target_common, hedge_common)
        except Exception as e:
            st.error(f"Error in cointegration test for {ticker}: {e}")
            continue
        
        # Remove the p-value threshold filter
        if pd.isna(coint_p_value):
            continue
            
        # Calculate z-scores from the spread values
        z_scores = calculate_z_score(spread_values)
        
        # Add debug statements to see what's happening with z-scores
        print(f"Z-scores for {ticker}: min={z_scores.min()}, max={z_scores.max()}, mean={z_scores.mean()}")
        
        # Calculate blended z-score (weighted average of recent values)
        if isinstance(z_scores, pd.Series) and len(z_scores) > 2:
            # Explicitly convert to numpy array if needed
            z_values = z_scores.values if hasattr(z_scores, 'values') else z_scores
            blended_z = [0.5 * z_values[i] + 0.3 * z_values[i-1] + 0.2 * z_values[i-2] 
                        for i in range(2, len(z_values))]
            current_z = blended_z[-1] if blended_z else 0
        else:
            # If z_scores is a pandas Series, get the last value
            if isinstance(z_scores, pd.Series) and len(z_scores) > 0:
                current_z = z_scores.iloc[-1]
            # If z_scores is a numpy array, get the last value
            elif isinstance(z_scores, np.ndarray) and len(z_scores) > 0:
                current_z = z_scores[-1]
            else:
                current_z = 0
                
        print(f"Current z-score for {ticker}: {current_z}")
        
        # Check action condition - ensure we have the right action based on z-score
        if (action.upper() == 'BUY' and current_z > 0) or (action.upper() == 'SELL' and current_z < 0):
            continue
            
        # Calculate half-life of mean reversion
        half_life = calculate_half_life(spread_values)
        
        # Calculate blended metrics
        blended_cointegration = (0.4 * coint_p_value + 0.35 * coint_p_value * 2 + 0.25 * coint_p_value * 3)
        blended_correlation = (0.4 * correlation + 0.35 * correlation * 2 + 0.25 * correlation * 3)
        
        if not np.isnan(half_life):
            blended_half_life = (0.4 * half_life + 0.35 * half_life * 2 + 0.25 * half_life * 3)
        else:
            blended_half_life = np.nan
        
        # Final score calculation
        half_life_score = 0
        if not np.isnan(blended_half_life) and blended_half_life > 0:
            half_life_score = 1/blended_half_life
            
        final_score = (0.5 * abs(current_z) + 
                      0.2 * blended_correlation + 
                      0.2 * (1 - blended_cointegration) + 
                      0.1 * half_life_score)
        
        results.append({
            'ticker': ticker,
            'correlation': correlation,
            'cointegration_p': coint_p_value,
            'z_score': current_z,  # Make sure this is a scalar, not a Series or array
            'half_life': half_life,
            'hedge_ratio': hedge_ratio,
            'score': final_score
        })
    
    # Sort by score and return the top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results] if results else []


# Title
st.title(":chart: HedgeFinder")

# Summary or Description of the App
st.markdown("""
HedgeFinder is a tool designed to identify the best hedge(s) for a given stock. 
It leverages metrics like **correlation**, **co-integration**, **mean reversion**, and **recent price trends** to determine 
the stocks most suitable for hedging against the target stock.
""")

# File upload option
st.subheader("Stock Universe")
upload_option = st.radio("Choose stock universe source:", ["Use default example universe", "Upload custom universe file"])

# Initialize df variable outside the conditional blocks
df = None

if upload_option == "Upload custom universe file":
    uploaded_file = st.file_uploader("Upload your stock universe CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded universe with {len(df)} stocks.")
            # Display a preview of the uploaded file
            st.write("Preview of uploaded universe:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading the file: {str(e)}")
            # Create default universe if upload fails
            df = create_default_universe()
    else:
        st.info("Please upload a CSV file containing columns for 'ticker', 'sector', and 'country'.")
        # Create default universe if no file is uploaded yet
        df = create_default_universe()
else:
    # Use default universe
    df = create_default_universe()
    st.info("Using example stock universe with common tech stocks.")

# Simplified Main Content Layout - Just ticker and action inputs
with st.container():
    st.subheader("Input Parameters")
    
    # Only keep stock ticker and buy/sell fields as requested
    ticker = st.text_input("Enter the stock ticker for the target stock", placeholder="e.g., AAPL")
    action = st.radio("What is the action on target stock?", ["BUY", "SELL"])
    
    # Hidden defaults (not shown in UI)
    num_hedge_stocks = 3
    
    # Date range for analysis
    one_year_ago = datetime.now().date().replace(year=datetime.now().year - 1)
    start_date = one_year_ago.strftime('%Y-%m-%d')
    end_date = datetime.now().date().strftime('%Y-%m-%d')

# Button to trigger calculation
if st.button('Find Hedge Basket'):
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        original_ticker = ticker
        with st.spinner('Finding the best hedge basket...'):
            try:
                # Use the df that was already loaded above
                if df is None:
                    st.error("Problem with stock universe data. Please check your file or use the default universe.")
                else:
                    # Check if required columns exist
                    required_columns = ['ticker', 'sector', 'country']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns in universe file: {', '.join(missing_columns)}")
                    else:
                        # Filter stocks based on ticker
                        filtered_stocks = filter_stocks(df, ticker)
                        
                        if filtered_stocks.empty:
                            st.error(f"No stocks in the same sector/country as {ticker} found in the database.")
                        else:
                            # Continue with existing analysis code...
                            # Get all tickers for analysis
                            all_tickers = [ticker] + filtered_stocks['ticker'].tolist()
                            
                            # Fetch historical data
                            st.info(f"Fetching price data for {ticker} and {len(filtered_stocks)} potential hedge stocks...")
                            historical_data = fetch_stock_data(all_tickers, start_date, end_date)
                            
                            if ticker not in historical_data:
                                st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
                            else:
                                # Find the best hedge stocks - removed p_value and correlation threshold parameters
                                results = find_best_hedge(
                                    historical_data, 
                                    ticker, 
                                    filtered_stocks['ticker'].tolist(),
                                    action,
                                    num_hedge_stocks
                                )
                                
                                if not results:
                                    st.warning("No suitable hedge stocks found. Try adjusting the parameters.")
                                else:
                                    # Display results
                                    st.success(f"Found {len(results)} potential hedge stocks for {ticker}!")
                                    
                                    st.subheader("Top Hedge Stocks")
                                    
                                    # Convert results to DataFrame for display
                                    results_df = pd.DataFrame(results)
                                    results_df['correlation'] = results_df['correlation'].round(3)
                                    results_df['cointegration_p'] = results_df['cointegration_p'].round(3)
                                    results_df['z_score'] = results_df['z_score'].round(3)
                                    results_df['half_life'] = results_df['half_life'].round(1)
                                    results_df['hedge_ratio'] = results_df['hedge_ratio'].round(3)
                                    results_df['score'] = results_df['score'].round(3)
                                    
                                    st.dataframe(results_df)
                                    
                                    # Plot price ratio chart
                                    hedge_tickers = [r['ticker'] for r in results]
                                    hedge_ratios = [r['hedge_ratio'] for r in results]
                                    
                                    # fig = plot_price_ratio(historical_data, ticker, hedge_tickers, hedge_ratios)
                                    
                                                                        # After finding the best hedge stocks, add this code to create and display the chart

                                    # Extract the selected hedge stocks
                                    sel_lst = []
                                    for i, result in enumerate(results):
                                        if i < num_hedge_stocks:  # Use the top n stocks based on your parameter
                                            sel_lst.append(result['ticker'])

                                    if sel_lst:
                                        # Get the last price for each hedge stock
                                        h = {}
                                        for ticker in sel_lst:
                                            if ticker in historical_data:
                                                h[ticker] = historical_data[ticker][-1:]
                                        
                                        # Get the last price for the target stock
                                        sp = historical_data[ticker][-1:]
                                        
                                        # Create a dictionary to store the price ratio data
                                        h_data = {}
                                        d = []

                                        for i in range(len(historical_data[ticker]) - 1, -1, -1):
                                            try:
                                                cp = historical_data[ticker].iloc[i]
                                                dtt = historical_data[ticker].index[i]
                                                dt = str(dtt)
                                                h_data[dt] = {'date': dt}

                                                # ✅ Safely extract scalar from sp
                                                try:
                                                    if isinstance(sp, pd.DataFrame):
                                                        sp_val = float(sp.iloc[0, 0])
                                                    elif isinstance(sp, pd.Series):
                                                        sp_val = float(sp.iloc[0])
                                                    else:
                                                        sp_val = float(sp)
                                                except Exception as e:
                                                    st.write(f"Error extracting sp_val: {e}")
                                                    continue

                                                # 🧼 Skip if cp or sp_val is NaN
                                                if isinstance(cp, pd.Series):
                                                    cp = float(cp.iloc[0])
                                                else:
                                                    cp = float(cp)

                                                if pd.isna(cp) or pd.isna(sp_val):
                                                    st.write(f"Skipping index {i} due to NaN in cp or sp_val")
                                                    continue

                                                cp = float(cp)
                                                ret = cp * ((cp - sp_val) / sp_val)

                                                ret1 = 0
                                                count = 0



                                                for s in sel_lst:
                                                    try:
                                                        if s not in historical_data or s not in h:
                                                            continue
                                                        if i >= len(historical_data[s]):
                                                            continue

                                                        cp_s = historical_data[s].iloc[i]
                                                        sp_s = h[s]

                                                        # Safely extract scalar from sp_s
                                                        if isinstance(sp_s, pd.DataFrame):
                                                            sp_s_val = float(sp_s.iloc[0, 0])
                                                        elif isinstance(sp_s, pd.Series):
                                                            sp_s_val = float(sp_s.iloc[0])
                                                        else:
                                                            sp_s_val = float(sp_s)

                                                        # Safely extract scalar from cp_s
                                                        if isinstance(cp_s, pd.Series):
                                                            cp_s = float(cp_s.iloc[0])
                                                        else:
                                                            cp_s = float(cp_s)

                                                        # Skip if prices are NaN
                                                        if pd.isna(cp_s) or pd.isna(sp_s_val):
                                                            continue

                                                        r1 = cp_s * ((cp_s - sp_s_val) / sp_s_val)
                                                        ret1 += r1
                                                        count += 1

                                                    except Exception as e:
                                                        st.write(f"Error processing hedge {s} at index {i}: {e}")
                                                        continue

                                                if count == 0:
                                                    st.write(f"No valid hedge returns at index {i}")
                                                    continue

                                                ret1 = float(ret1) / count

                                                try:
                                                    ratio = float(ret) / float(ret1)
                                                except ZeroDivisionError:
                                                    ratio = 0
                                                except Exception as e:
                                                    st.write(f"Error dividing ret by ret1 at index {i}: {e}")
                                                    continue
                                                h_data[dt]['ratio'] = ratio
                                                d.append(ratio)

                                           
                                            except Exception as e:
                                                st.write(f"Final unexpected error at index {i}: {e}")
                                                st.code(traceback.format_exc())  # 🔍 shows the full traceback
                                                continue


                                        st.write("Collected price ratio points:", len(d), "| Sample:", d[:5])
                                        # Take the last 21 days if available
                                        if len(d) > 21:
                                            d1 = d[-21:]
                                        else:
                                            d1 = d.copy()
                                        
                                        # Calculate statistics
                                        import statistics
                                        
                                        # Filter out zeros and None values
                                        d1_filtered = [x for x in d1 if x != 0 and x is not None]
                                        
                                        if d1_filtered:
                                            mean = statistics.mean(d1_filtered)
                                            std_numpy = np.std(d1_filtered)
                                            
                                            # Calculate standard deviation bands
                                            std_1 = mean + std_numpy
                                            std_2 = mean + (2 * std_numpy)
                                            std_3 = mean - std_numpy
                                            std_4 = mean - (2 * std_numpy)
                                            
                                            # Add statistics to each data point
                                            for i, dt in enumerate(list(h_data.keys())):
                                                if i < len(d):
                                                    ratio = d[i]
                                                    h_data[dt]['ratio'] = ratio
                                                    h_data[dt]['MEAN'] = mean
                                                    h_data[dt]['+1STD_DEV'] = std_1
                                                    h_data[dt]['+2STD_DEV'] = std_2
                                                    h_data[dt]['-1STD_DEV'] = std_3
                                                    h_data[dt]['-2STD_DEV'] = std_4
                                            
                                            # Create a DataFrame from the dictionary
                                            df_clean_ret = pd.DataFrame(h_data).T
                                            
                                            # Sort the DataFrame by date
                                            df_clean_ret = df_clean_ret.sort_values(by='date', ascending=True)
                                            
                                            # Calculate min and max for y-axis scaling
                                            # Filter out None/NaN values first
                                            numeric_columns = ['+2STD_DEV', '+1STD_DEV', 'MEAN', '-1STD_DEV', '-2STD_DEV', 'ratio']
                                            y_values = []
                                            for col in numeric_columns:
                                                if col in df_clean_ret.columns:
                                                    values = df_clean_ret[col].dropna().tolist()
                                                    y_values.extend(values)
                                            
                                            if y_values:
                                                y_min = min(y_values) * 0.9
                                                y_max = max(y_values) * 1.1
                                                
                                                # Create the figure
                                                fig = go.Figure()
                                                
                                                # Add ratio line
                                                fig.add_trace(go.Scatter(
                                                    x=pd.to_datetime(df_clean_ret['date']),
                                                    y=df_clean_ret['ratio'],
                                                    mode='lines', name='Basket Price Ratio',
                                                    line=dict(color='blue', width=2)
                                                ))
                                                
                                                # Add mean line
                                                fig.add_trace(go.Scatter(
                                                    x=pd.to_datetime(df_clean_ret['date']),
                                                    y=df_clean_ret['MEAN'],
                                                    mode='lines', name='Mean',
                                                    line=dict(color='green', width=1)
                                                ))
                                                
                                                # Add standard deviation lines
                                                fig.add_trace(go.Scatter(
                                                    x=pd.to_datetime(df_clean_ret['date']),
                                                    y=df_clean_ret['+1STD_DEV'],
                                                    mode='lines', name='+1 Std Dev',
                                                    line=dict(color='red', width=1, dash='dot')
                                                ))
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=pd.to_datetime(df_clean_ret['date']),
                                                    y=df_clean_ret['-1STD_DEV'],
                                                    mode='lines', name='-1 Std Dev',
                                                    line=dict(color='red', width=1, dash='dot')
                                                ))
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=pd.to_datetime(df_clean_ret['date']),
                                                    y=df_clean_ret['+2STD_DEV'],
                                                    mode='lines', name='+2 Std Dev',
                                                    line=dict(color='purple', width=1, dash='dot')
                                                ))
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=pd.to_datetime(df_clean_ret['date']),
                                                    y=df_clean_ret['-2STD_DEV'],
                                                    mode='lines', name='-2 Std Dev',
                                                    line=dict(color='purple', width=1, dash='dot')
                                                ))
                                                
                                                # Update layout
                                                fig.update_layout(
                                                    title={
                                                        'text': "Long Short Basket Price Ratio",
                                                        'x': 0.5,
                                                        'xanchor': 'center',
                                                        'yanchor': 'top'
                                                    },
                                                    xaxis_title="Date",
                                                    yaxis_title="Price Ratio",
                                                    legend_title="Legend",
                                                    hovermode="x unified",
                                                    template="plotly_white",
                                                    yaxis_range=[y_min, y_max]
                                                )
                                                
                                                # Add a divider
                                                st.divider()
                                                
                                                # Display information about the hedge stocks
                                                sentence = ", ".join(sel_lst)
                                                st.write(f"**There are** {len(sel_lst)} **hedge stocks in the basket.**")
                                                st.write(f"**Chart of price ratio of** {ticker} **and equal weighted basket of filtered hedges** **with positive coefficient: {sentence}.**")
                                                
                                                # Display the chart
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Add interpretation information
                                                st.markdown("""
                                                **How to interpret the chart:**
                                                - When the blue line (Price Ratio) is above the mean, the target stock is relatively expensive compared to the hedge basket
                                                - When below the mean, the target stock is relatively cheap compared to the hedge basket
                                                - Trading signals often occur when the ratio crosses the standard deviation bands
                                                """)
                                            else:
                                                st.write("**Insufficient data to calculate price ratios.**")
                                        else:
                                            st.write("**Not enough valid data points to calculate statistics.**")
                                    else:
                                        st.write("**There are no hedge stocks with positive coefficient.**")


            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Add detailed error information for debugging
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")