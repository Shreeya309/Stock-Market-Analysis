import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .company-info {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86C1;
    }
    .stock-metric {
        text-align: center;
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E86C1;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📈 Stock Market Dashboard</h1>', unsafe_allow_html=True)
    
    # Stock selection
    st.subheader("Select a Company")
    
    # Comprehensive list of companies across sectors
    popular_stocks = {
        # Technology Giants
        'Apple Inc. (AAPL)': 'AAPL',
        'Microsoft Corporation (MSFT)': 'MSFT',
        'Alphabet Inc. (GOOGL)': 'GOOGL',
        'Amazon.com Inc. (AMZN)': 'AMZN',
        'Meta Platforms Inc. (META)': 'META',
        'Tesla Inc. (TSLA)': 'TSLA',
        'NVIDIA Corporation (NVDA)': 'NVDA',
        'Netflix Inc. (NFLX)': 'NFLX',
        'Adobe Inc. (ADBE)': 'ADBE',
        'Salesforce Inc. (CRM)': 'CRM',
        'Oracle Corporation (ORCL)': 'ORCL',
        'Intel Corporation (INTC)': 'INTC',
        'Advanced Micro Devices (AMD)': 'AMD',
        'Cisco Systems Inc. (CSCO)': 'CSCO',
        'IBM Corporation (IBM)': 'IBM',
        'PayPal Holdings Inc. (PYPL)': 'PYPL',
        'Zoom Video Communications (ZM)': 'ZM',
        'Slack Technologies (CRM)': 'WORK',
        'ServiceNow Inc. (NOW)': 'NOW',
        'Snowflake Inc. (SNOW)': 'SNOW',
        
        # Financial Services
        'JPMorgan Chase & Co. (JPM)': 'JPM',
        'Bank of America Corp (BAC)': 'BAC',
        'Wells Fargo & Company (WFC)': 'WFC',
        'Goldman Sachs Group (GS)': 'GS',
        'Morgan Stanley (MS)': 'MS',
        'Citigroup Inc. (C)': 'C',
        'American Express Company (AXP)': 'AXP',
        'Visa Inc. (V)': 'V',
        'Mastercard Inc. (MA)': 'MA',
        'Berkshire Hathaway Inc. (BRK.B)': 'BRK.B',
        
        # Healthcare & Pharmaceuticals
        'Johnson & Johnson (JNJ)': 'JNJ',
        'Pfizer Inc. (PFE)': 'PFE',
        'Moderna Inc. (MRNA)': 'MRNA',
        'AbbVie Inc. (ABBV)': 'ABBV',
        'Merck & Co Inc. (MRK)': 'MRK',
        'Bristol Myers Squibb (BMY)': 'BMY',
        'Eli Lilly and Company (LLY)': 'LLY',
        'UnitedHealth Group Inc. (UNH)': 'UNH',
        'Thermo Fisher Scientific (TMO)': 'TMO',
        'Danaher Corporation (DHR)': 'DHR',
        
        # Consumer Goods & Retail
        'Procter & Gamble Co. (PG)': 'PG',
        'Coca-Cola Company (KO)': 'KO',
        'PepsiCo Inc. (PEP)': 'PEP',
        'Nike Inc. (NKE)': 'NKE',
        'McDonald\'s Corporation (MCD)': 'MCD',
        'Starbucks Corporation (SBUX)': 'SBUX',
        'Walmart Inc. (WMT)': 'WMT',
        'Target Corporation (TGT)': 'TGT',
        'Costco Wholesale Corp (COST)': 'COST',
        'Home Depot Inc. (HD)': 'HD',
        'Lowe\'s Companies Inc. (LOW)': 'LOW',
        
        # Energy & Utilities
        'Exxon Mobil Corporation (XOM)': 'XOM',
        'Chevron Corporation (CVX)': 'CVX',
        'ConocoPhillips (COP)': 'COP',
        'NextEra Energy Inc. (NEE)': 'NEE',
        'Enphase Energy Inc. (ENPH)': 'ENPH',
        'FirstSolar Inc. (FSLR)': 'FSLR',
        
        # Industrial & Manufacturing
        'Boeing Company (BA)': 'BA',
        'Caterpillar Inc. (CAT)': 'CAT',
        '3M Company (MMM)': 'MMM',
        'General Electric Co. (GE)': 'GE',
        'Honeywell International (HON)': 'HON',
        'Lockheed Martin Corp (LMT)': 'LMT',
        'Raytheon Technologies (RTX)': 'RTX',
        
        # Automotive
        'Ford Motor Company (F)': 'F',
        'General Motors Company (GM)': 'GM',
        'Ferrari N.V. (RACE)': 'RACE',
        'Lucid Group Inc. (LCID)': 'LCID',
        'Rivian Automotive Inc. (RIVN)': 'RIVN',
        
        # Media & Entertainment
        'Walt Disney Company (DIS)': 'DIS',
        'Comcast Corporation (CMCSA)': 'CMCSA',
        'Warner Bros Discovery (WBD)': 'WBD',
        'Spotify Technology SA (SPOT)': 'SPOT',
        
        # Telecommunications
        'Verizon Communications (VZ)': 'VZ',
        'AT&T Inc. (T)': 'T',
        'T-Mobile US Inc. (TMUS)': 'TMUS',
        
        # Real Estate & REITs
        'American Tower Corp (AMT)': 'AMT',
        'Crown Castle Inc. (CCI)': 'CCI',
        'Realty Income Corp (O)': 'O',
        
        # Emerging Companies
        'Palantir Technologies (PLTR)': 'PLTR',
        'Robinhood Markets Inc. (HOOD)': 'HOOD',
        'Square Inc. (SQ)': 'SQ',
        'Peloton Interactive (PTON)': 'PTON',
        'Beyond Meat Inc. (BYND)': 'BYND',
        'DoorDash Inc. (DASH)': 'DASH',
        'Uber Technologies Inc. (UBER)': 'UBER',
        'Lyft Inc. (LYFT)': 'LYFT',
        'Airbnb Inc. (ABNB)': 'ABNB',
        'Twilio Inc. (TWLO)': 'TWLO'
    }
    
    # Add search functionality
    search_term = st.text_input("🔍 Search for a company (optional):", placeholder="Type company name or ticker symbol...")
    
    # Filter companies based on search term
    if search_term:
        filtered_stocks = {k: v for k, v in popular_stocks.items() 
                          if search_term.lower() in k.lower() or search_term.lower() in v.lower()}
        if not filtered_stocks:
            st.warning(f"No companies found matching '{search_term}'. Showing all companies.")
            filtered_stocks = popular_stocks
    else:
        filtered_stocks = popular_stocks
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_stock_name = st.selectbox(
            f"Choose from {len(filtered_stocks)} available companies:",
            list(filtered_stocks.keys()),
            index=0
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period:",
            ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years'],
            index=2
        )
    
    # Map time period to yfinance period
    period_mapping = {
        '1 Month': '1mo',
        '3 Months': '3mo', 
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y'
    }
    
    selected_stock = filtered_stocks[selected_stock_name]
    selected_period = period_mapping[time_period]
    
    # Fetch and display stock data
    try:
        with st.spinner(f"Loading {selected_stock_name} data..."):
            stock = yf.Ticker(selected_stock)
            info = stock.info
            hist = stock.history(period=selected_period)
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                start_price = hist['Close'].iloc[0]
                price_change = current_price - start_price
                price_change_percent = (price_change / start_price) * 100
                
                # Company information card
                st.markdown(f"""
                <div class="company-info">
                    <h3>{info.get('longName', selected_stock_name)}</h3>
                    <p><strong>Sector:</strong> {info.get('sector', 'N/A')} | <strong>Industry:</strong> {info.get('industry', 'N/A')}</p>
                    <p><strong>Market Cap:</strong> ${info.get('marketCap', 0) / 1e9:.1f}B | <strong>Employees:</strong> {info.get('fullTimeEmployees', 'N/A'):,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Stock metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change_percent:+.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Day High",
                        f"${hist['High'].iloc[-1]:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Day Low", 
                        f"${hist['Low'].iloc[-1]:.2f}"
                    )
                
                with col4:
                    avg_volume = hist['Volume'].mean()
                    st.metric(
                        "Avg Volume",
                        f"{avg_volume/1e6:.1f}M"
                    )
                
                with col5:
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric(
                        "Volatility",
                        f"{volatility:.1f}%"
                    )
                
                # Price chart
                st.subheader(f"{selected_stock_name} Price Chart")
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price',
                    increasing_line_color='#00D4AA',
                    decreasing_line_color='#FF6B6B'
                ))
                
                # Add moving averages
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                hist['MA50'] = hist['Close'].rolling(window=50).mean()
                
                if len(hist) >= 20:
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['MA20'],
                        mode='lines',
                        name='20-Day MA',
                        line=dict(color='orange', width=1)
                    ))
                
                if len(hist) >= 50:
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='purple', width=1)
                    ))
                
                fig.update_layout(
                    title=f"{selected_stock_name} Stock Performance ({time_period})",
                    yaxis_title="Price (USD)",
                    xaxis_title="Date",
                    height=500,
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    xaxis_rangeslider_visible=False
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                st.subheader("Trading Volume")
                
                fig_volume = go.Figure()
                
                fig_volume.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                fig_volume.update_layout(
                    title=f"{selected_stock_name} Trading Volume",
                    yaxis_title="Volume",
                    xaxis_title="Date",
                    height=300,
                    showlegend=False,
                    plot_bgcolor='white'
                )
                
                fig_volume.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig_volume.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Performance summary
                st.subheader("Performance Summary")
                
                returns = hist['Close'].pct_change().dropna()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_return = (current_price - start_price) / start_price * 100
                    st.metric(f"{time_period} Return", f"{total_return:+.1f}%")
                
                with col2:
                    best_day = returns.max() * 100
                    st.metric("Best Day", f"+{best_day:.1f}%")
                
                with col3:
                    worst_day = returns.min() * 100
                    st.metric("Worst Day", f"{worst_day:.1f}%")
                
    except Exception as e:
        st.error(f"Unable to fetch data for {selected_stock_name}. Please try selecting a different company.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
