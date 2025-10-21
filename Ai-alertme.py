import os
import json
import streamlit as st
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime, timedelta
import pandas as pd
from groq import Groq
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AI Alert Me", page_icon="🤖", layout="wide")

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = {}

# --- Helper Functions ---
def get_crypto_price(symbol, max_retries=3):
    """Fetch cryptocurrency price from CoinGecko API with retry logic"""
    for attempt in range(max_retries):
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd&include_24hr_change=true"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                'price': data[symbol]['usd'],
                'change_24h': data[symbol].get('usd_24h_change', 0),
                'status': 'success'
            }
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            return {'status': 'error', 'message': f'Timeout - will retry in next cycle'}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            return {'status': 'error', 'message': f'Network error - retrying in next cycle'}
        except (KeyError, json.JSONDecodeError) as e:
            return {'status': 'error', 'message': f'Invalid data received'}
        except Exception as e:
            return {'status': 'error', 'message': f'Error: {str(e)[:50]}'}
    return {'status': 'error', 'message': 'Max retries reached'}

def get_stock_price(symbol, max_retries=3):
    """Fetch stock price from Yahoo Finance with retry logic"""
    for attempt in range(max_retries):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            prev_close = data['chart']['result'][0]['meta']['previousClose']
            change = ((price - prev_close) / prev_close) * 100
            return {
                'price': price,
                'change_24h': change,
                'status': 'success'
            }
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            return {'status': 'error', 'message': f'Timeout - will retry in next cycle'}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            return {'status': 'error', 'message': f'Network error - retrying in next cycle'}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {'status': 'error', 'message': f'Invalid data or symbol not found'}
        except Exception as e:
            return {'status': 'error', 'message': f'Error: {str(e)[:50]}'}
    return {'status': 'error', 'message': 'Max retries reached'}

def update_price_history(asset, price):
    """Track price history for AI analysis"""
    if asset not in st.session_state.price_history:
        st.session_state.price_history[asset] = []
    
    st.session_state.price_history[asset].append({
        'timestamp': datetime.now().isoformat(),
        'price': price
    })
    
    # Keep only last 100 data points for better charts
    if len(st.session_state.price_history[asset]) > 100:
        st.session_state.price_history[asset] = st.session_state.price_history[asset][-100:]

def create_price_chart(asset, target_price=None, alert_type=None):
    """Create interactive price chart with target line and colored bars"""
    history = st.session_state.price_history.get(asset, [])
    
    if len(history) < 2:
        return None
    
    # Prepare data
    timestamps = [datetime.fromisoformat(h['timestamp']) for h in history]
    prices = [h['price'] for h in history]
    
    # Calculate price changes
    price_changes = [0]  # First bar is neutral
    for i in range(1, len(prices)):
        price_changes.append(prices[i] - prices[i-1])
    
    # Color bars based on price movement
    colors = ['green' if change > 0 else 'red' if change < 0 else 'gray' for change in price_changes]
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=(f'{asset.upper()} Price Chart', 'Price Change (Bar Chart)')
    )
    
    # Add line chart
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='#00D9FF', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Add target line if specified
    if target_price is not None:
        fig.add_hline(
            y=target_price,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Target: ${target_price:.2f}",
            annotation_position="right",
            row=1, col=1
        )
    
    # Add bar chart for price changes
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=price_changes,
            name='Price Change',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add horizontal line at 0 for bar chart
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Change ($)", row=2, col=1)
    
    return fig

def create_alert_status_chart():
    """Create bar chart showing alert status (above/below target)"""
    if not st.session_state.alerts:
        return None
    
    alerts_data = []
    errors = []
    
    for alert in st.session_state.alerts:
        if alert['type'] == "Cryptocurrency":
            current_data = get_crypto_price(alert['asset'])
        else:
            current_data = get_stock_price(alert['asset'])
        
        if current_data and current_data.get('status') == 'success':
            current_price = current_data['price']
            target_price = alert['target_price']
            difference = current_price - target_price
            percentage_diff = (difference / target_price) * 100
            
            alerts_data.append({
                'asset': alert['asset'],
                'difference': difference,
                'percentage': percentage_diff,
                'alert_type': alert['alert_type'],
                'triggered': alert['triggered']
            })
        else:
            errors.append(f"{alert['asset']}: {current_data.get('message', 'Error')}")
    
    # Show errors if any
    if errors:
        st.warning("⚠️ Some assets failed to load:\n" + "\n".join(errors))
    
    if not alerts_data:
        return None
    
    df = pd.DataFrame(alerts_data)
    
    # Create color based on alert type and difference
    colors = []
    for idx, row in df.iterrows():
        if row['triggered']:
            colors.append('gold')
        elif row['alert_type'] == 'Above':
            colors.append('green' if row['difference'] > 0 else 'red')
        else:  # Below
            colors.append('red' if row['difference'] > 0 else 'green')
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['asset'],
            y=df['percentage'],
            marker_color=colors,
            text=[f"{p:.2f}%" for p in df['percentage']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                          'Difference: %{y:.2f}%<br>' +
                          '<extra></extra>'
        )
    ])
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
    
    fig.update_layout(
        title="Alert Status: Distance from Target (%)",
        xaxis_title="Asset",
        yaxis_title="Percentage Difference from Target",
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def ai_analyze_price_trend(asset, current_data, groq_client, model_name):
    """Use AI to analyze price trends and provide insights"""
    history = st.session_state.price_history.get(asset, [])
    
    if len(history) < 3:
        return "Not enough data yet for AI analysis. Keep monitoring..."
    
    prompt = f"""
    You are a financial AI analyst. Analyze this price data and provide insights:
    
    Asset: {asset}
    Current Price: ${current_data['price']:.2f}
    24h Change: {current_data['change_24h']:.2f}%
    
    Recent Price History (last {len(history)} data points):
    {json.dumps(history[-10:], indent=2)}
    
    Provide:
    1. Brief trend analysis (bullish/bearish/neutral)
    2. Key observations about volatility
    3. Short recommendation (1-2 sentences)
    
    Keep response under 150 words and professional.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Analysis unavailable: {str(e)}"

def ai_suggest_alert(asset, current_price, groq_client, model_name):
    """AI suggests optimal alert prices based on current market"""
    prompt = f"""
    You are a financial advisor AI. Based on this data, suggest smart alert prices:
    
    Asset: {asset}
    Current Price: ${current_price:.2f}
    
    Suggest:
    1. A reasonable upper alert price (for taking profits)
    2. A reasonable lower alert price (for buying opportunity)
    
    Return ONLY valid JSON in this format:
    {{
        "upper_alert": <number>,
        "lower_alert": <number>,
        "reasoning": "<brief explanation>"
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        result = response.choices[0].message.content.strip()
        # Extract JSON from response
        import re
        match = re.search(r'\{[\s\S]*\}', result)
        if match:
            return json.loads(match.group(0))
        return None
    except Exception as e:
        return None

def send_email_alert(recipient_email, sender_email, sender_password, alert_info):
    """Send email alert when price target is hit"""
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"🤖 AI Alert: {alert_info['asset']} Hit Target!"
        
        body = f"""
        AI Price Alert Triggered!
        
        Asset: {alert_info['asset']}
        Current Price: ${alert_info['current_price']:.2f}
        Target Price: ${alert_info['target_price']:.2f}
        Alert Type: {alert_info['alert_type']}
        24h Change: {alert_info.get('change_24h', 'N/A')}%
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        AI Insight: {alert_info.get('ai_insight', 'No analysis available')}
        
        ---
        Powered by AI Alert Me
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# --- Main App ---
def main():
    st.markdown("""
    <style>
        .block-container { padding-left: 2rem; padding-right: 2rem; }
        .stAlert { margin-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖 AI Alert Me - Smart Price Tracker")
    st.markdown("AI-powered price tracking with intelligent insights and recommendations")

    # Sidebar Configuration
    st.sidebar.image("2.png")
    st.sidebar.image("1.png")
    st.sidebar.header("⚙️ Configuration")
    
    # Groq API Key
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    )
    
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar for AI features")
        groq_client = None
    else:
        groq_client = Groq(api_key=groq_api_key)
    
    model_name = st.sidebar.selectbox(
        "AI Model:",
        ["llama-3.3-70b-versatile","openai/gpt-oss-120b", "mixtral-8x7b-32768", "llama-3.1-70b-versatile"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📧 Email Configuration")
    sender_email = st.sidebar.text_input("Gmail Address")
    sender_password = st.sidebar.text_input("Gmail App Password", type="password")
    recipient_email = st.sidebar.text_input("Alert Email")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 AI Features:**")
    st.sidebar.markdown("✓ Smart price trend analysis")
    st.sidebar.markdown("✓ Alert recommendations")
    st.sidebar.markdown("✓ Market insights")
    st.sidebar.markdown("✓ Interactive price charts")
    st.sidebar.caption("Built with 💡 Streamlit + Groq | DW 2025")
    
    # Main Interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "➕ Add Alert", "📈 Charts", "🧠 AI Insights"])
    
    # --- TAB 1: Dashboard ---
    with tab1:
        st.header("Current Prices & Alerts")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            btc_data = get_crypto_price("bitcoin")
            if btc_data and btc_data.get('status') == 'success':
                st.metric(
                    "Bitcoin",
                    f"${btc_data['price']:,.2f}",
                    f"{btc_data['change_24h']:.2f}%"
                )
            else:
                st.error(f"⚠️ Bitcoin: {btc_data.get('message', 'Error')}")
        
        with col2:
            eth_data = get_crypto_price("ethereum")
            if eth_data and eth_data.get('status') == 'success':
                st.metric(
                    "Ethereum",
                    f"${eth_data['price']:,.2f}",
                    f"{eth_data['change_24h']:.2f}%"
                )
            else:
                st.error(f"⚠️ Ethereum: {eth_data.get('message', 'Error')}")
        
        with col3:
            sol_data = get_crypto_price("solana")
            if sol_data and sol_data.get('status') == 'success':
                st.metric(
                    "Solana",
                    f"${sol_data['price']:,.2f}",
                    f"{sol_data['change_24h']:.2f}%"
                )
            else:
                st.error(f"⚠️ Solana: {sol_data.get('message', 'Error')}")
        
        with col4:
            if st.button("🔄 Refresh All"):
                st.rerun()
        
        st.markdown("---")
        
        # Alert Status Chart
        if st.session_state.alerts:
            st.subheader("🎯 Alert Status Overview")
            status_chart = create_alert_status_chart()
            if status_chart:
                st.plotly_chart(status_chart, use_container_width=True)
        
        # Display Active Alerts
        if st.session_state.alerts:
            st.subheader("🔔 Active Alerts")
            
            df = pd.DataFrame(st.session_state.alerts)
            df_display = df[['asset', 'target_price', 'alert_type', 'triggered']].copy()
            df_display.columns = ['Asset', 'Target ($)', 'Type', 'Triggered']
            st.dataframe(df_display, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Monitoring" if not st.session_state.monitoring else "⏸️ Stop Monitoring"):
                    st.session_state.monitoring = not st.session_state.monitoring
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All Alerts"):
                    st.session_state.alerts = []
                    st.rerun()
        else:
            st.info("📝 No active alerts. Add one in the 'Add Alert' tab!")
    
    # --- TAB 2: Add Alert ---
    with tab2:
        st.header("➕ Create New Alert")
        
        asset_type = st.selectbox("Asset Type", ["Cryptocurrency", "Stock"])
        
        if asset_type == "Cryptocurrency":
            asset_symbol = st.selectbox(
                "Select Crypto",
                ["bitcoin", "ethereum", "cardano", "solana", "dogecoin", "ripple", "polkadot"]
            )
        else:
            asset_symbol = st.text_input("Stock Symbol (e.g., AAPL, TSLA)").upper()
        
        # AI Suggestion Feature
        if groq_client and asset_symbol:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🤖 AI Suggest"):
                    with st.spinner("AI analyzing..."):
                        if asset_type == "Cryptocurrency":
                            current_data = get_crypto_price(asset_symbol)
                        else:
                            current_data = get_stock_price(asset_symbol)
                        
                        if current_data:
                            suggestion = ai_suggest_alert(
                                asset_symbol,
                                current_data['price'],
                                groq_client,
                                model_name
                            )
                            if suggestion:
                                st.success("AI Recommendations:")
                                st.write(f"**Upper Alert:** ${suggestion['upper_alert']:.2f}")
                                st.write(f"**Lower Alert:** ${suggestion['lower_alert']:.2f}")
                                st.info(suggestion['reasoning'])
                        else:
                            st.error(f"⚠️ {current_data.get('message', 'Could not fetch price')}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            target_price = st.number_input("Target Price ($)", min_value=0.0, step=0.01)
        with col_b:
            alert_type = st.selectbox("Alert When", ["Above", "Below"])
        
        if st.button("➕ Add Alert", type="primary"):
            if asset_symbol and target_price > 0:
                alert = {
                    'type': asset_type,
                    'asset': asset_symbol,
                    'target_price': target_price,
                    'alert_type': alert_type,
                    'triggered': False,
                    'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.alerts.append(alert)
                st.success(f"✅ Alert added for {asset_symbol}!")
            else:
                st.error("Please fill all fields correctly.")
    
    # --- TAB 3: Charts ---
    with tab3:
        st.header("📈 Price Charts")
        
        if not st.session_state.price_history:
            st.info("📊 Start monitoring to see price charts. Price history will appear here.")
        else:
            # Asset selector
            available_assets = list(st.session_state.price_history.keys())
            selected_asset = st.selectbox("Select Asset to View", available_assets)
            
            if selected_asset:
                # Find if there's an alert for this asset
                target_price = None
                alert_type = None
                for alert in st.session_state.alerts:
                    if alert['asset'] == selected_asset:
                        target_price = alert['target_price']
                        alert_type = alert['alert_type']
                        break
                
                # Create and display chart
                chart = create_price_chart(selected_asset, target_price, alert_type)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Show statistics
                    history = st.session_state.price_history[selected_asset]
                    prices = [h['price'] for h in history]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current", f"${prices[-1]:.2f}")
                    with col2:
                        st.metric("High", f"${max(prices):.2f}")
                    with col3:
                        st.metric("Low", f"${min(prices):.2f}")
                    with col4:
                        change = prices[-1] - prices[0]
                        change_pct = (change / prices[0]) * 100
                        st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
    
    # --- TAB 4: AI Insights ---
    with tab4:
        st.header("🧠 AI Market Insights")
        
        if not groq_client:
            st.warning("Please enter your Groq API key in the sidebar to use AI features")
        else:
            insight_asset_type = st.selectbox("Select Asset Type", ["Cryptocurrency", "Stock"], key="insight_type")
            
            if insight_asset_type == "Cryptocurrency":
                insight_asset = st.selectbox(
                    "Choose Asset to Analyze",
                    ["bitcoin", "ethereum", "cardano", "solana"],
                    key="insight_crypto"
                )
            else:
                insight_asset = st.text_input("Enter Stock Symbol", key="insight_stock").upper()
            
            if st.button("🔍 Generate AI Analysis") and insight_asset:
                with st.spinner("AI analyzing market data..."):
                    if insight_asset_type == "Cryptocurrency":
                        current_data = get_crypto_price(insight_asset)
                    else:
                        current_data = get_stock_price(insight_asset)
                    
                    if current_data and current_data.get('status') == 'success':
                        update_price_history(insight_asset, current_data['price'])
                        
                        analysis = ai_analyze_price_trend(
                            insight_asset,
                            current_data,
                            groq_client,
                            model_name
                        )
                        
                        st.subheader(f"📊 Analysis for {insight_asset.upper()}")
                        st.info(analysis)
                        
                        st.session_state.ai_insights[insight_asset] = {
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        st.error(f"⚠️ {current_data.get('message', 'Could not fetch price data')}")
            
            # Show historical insights
            if st.session_state.ai_insights:
                st.markdown("---")
                st.subheader("📜 Recent Insights")
                for asset, data in st.session_state.ai_insights.items():
                    with st.expander(f"{asset.upper()} - {data['timestamp'][:19]}"):
                        st.write(data['analysis'])
    
    # Monitoring Loop
    if st.session_state.monitoring:
        st.markdown("---")
        st.info("🔍 Monitoring active... Checking every 60 seconds")
        
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            st.write(f"**Last Check:** {datetime.now().strftime('%H:%M:%S')}")
            
            for idx, alert in enumerate(st.session_state.alerts):
                if not alert['triggered']:
                    # Get current price
                    if alert['type'] == "Cryptocurrency":
                        current_data = get_crypto_price(alert['asset'])
                    else:
                        current_data = get_stock_price(alert['asset'])
                    
                    if current_data and current_data.get('status') == 'success':
                        current_price = current_data['price']
                        update_price_history(alert['asset'], current_price)
                        
                        st.write(f"**{alert['asset']}:** ${current_price:.2f} → Target: ${alert['target_price']:.2f} ({alert['alert_type']})")
                        
                        # Check if alert triggered
                        should_trigger = False
                        if alert['alert_type'] == "Above" and current_price >= alert['target_price']:
                            should_trigger = True
                        elif alert['alert_type'] == "Below" and current_price <= alert['target_price']:
                            should_trigger = True
                        
                        if should_trigger:
                            # Get AI insight
                            ai_insight = ""
                            if groq_client:
                                ai_insight = ai_analyze_price_trend(
                                    alert['asset'],
                                    current_data,
                                    groq_client,
                                    model_name
                                )
                            
                            alert_info = {
                                'asset': alert['asset'],
                                'current_price': current_price,
                                'target_price': alert['target_price'],
                                'alert_type': alert['alert_type'],
                                'change_24h': current_data.get('change_24h', 0),
                                'ai_insight': ai_insight
                            }
                            
                            st.success(f"🎯 Alert triggered for {alert['asset']}!")
                            st.write(f"**AI Insight:** {ai_insight}")
                            
                            # Send email
                            if sender_email and sender_password and recipient_email:
                                if send_email_alert(recipient_email, sender_email, sender_password, alert_info):
                                    st.success(f"📧 Email sent to {recipient_email}")
                            
                            st.session_state.alerts[idx]['triggered'] = True
                    else:
                        # Show error and retry message
                        error_msg = current_data.get('message', 'Unknown error') if current_data else 'Connection failed'
                        st.warning(f"⚠️ **{alert['asset']}:** {error_msg} - Will retry in 60 seconds")
        
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()