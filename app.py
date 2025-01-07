import streamlit as st
import pandas as pd
import numpy as np
import vectorbt as vbt
import os

def load_data(file_path):
    """Load the CSV data into a DataFrame."""
    return pd.read_csv(file_path, index_col=0)

def filter_date_range(df):
    """Filter the DataFrame for the given date range."""
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    return df

def generate_signals(df, trade_position):
    """Generate long and short signals based on the trade position."""
    df['long_signal'] = 0
    df['short_signal'] = 0

    df.loc[df['position'] == trade_position, 'long_signal'] = df.apply(lambda row: 1 if row['Autoformer'] > row['cutoff_value'] else 0, axis=1)
    df['long_signal'] = df['long_signal'].shift(-(trade_position+1)) # Entry
    df.loc[df['position'] == trade_position, 'long_signal'] = df.apply(lambda row: -1 if row['Autoformer'] > row['cutoff_value'] else 0, axis=1) # Exit

    df.loc[df['position'] == trade_position, 'short_signal'] = df.apply(lambda row: 1 if row['Autoformer'] < row['cutoff_value'] else 0, axis=1)
    df['short_signal'] = df['short_signal'].shift(-(trade_position+1)) # Entry
    df.loc[df['position'] == trade_position, 'short_signal'] = df.apply(lambda row: -1 if row['Autoformer'] < row['cutoff_value'] else 0, axis=1) # Exit

    return df

def resample_and_interpolate(df, interval='30s'):
    """Resample the DataFrame to the given interval and interpolate 'y' values."""
    df_resampled = df.resample(interval).asfreq()
    df_resampled['y'] = df_resampled['y'].interpolate()
    df_resampled.loc[df_resampled.index.difference(df.index), df.columns.difference(['y'])] = np.nan
    return df_resampled

def create_portfolio(price, long_signals, short_signals, stop_loss_pct=0.001):
    """Create long and short portfolios with stop-loss."""
    entries = (long_signals == 1) & (price.index.time >= pd.Timestamp('09:00').time()) & (price.index.time <= pd.Timestamp('15:59').time())
    exits = (long_signals == -1) & (price.index.time >= pd.Timestamp('09:00').time()) & (price.index.time <= pd.Timestamp('15:59').time())

    short_entries = (short_signals == 1) & (price.index.time >= pd.Timestamp('09:00').time()) & (price.index.time <= pd.Timestamp('15:59').time())
    short_exits = (short_signals == -1) & (price.index.time >= pd.Timestamp('09:00').time()) & (price.index.time <= pd.Timestamp('15:59').time())

    long_portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100, sl_stop=stop_loss_pct)
    short_portfolio = vbt.Portfolio.from_signals(price, short_entries=short_entries, short_exits=short_exits, init_cash=100, sl_stop=stop_loss_pct)

    return long_portfolio, short_portfolio

def plot_portfolio(portfolio, title):
    """Plot the portfolio performance and return statistics."""
    fig = portfolio.plot()
    fig.update_layout(title=title)
    st.plotly_chart(fig)
    return portfolio.stats()

# Streamlit app
st.title("Stock Portfolio Analysis")

# Dropdown for selecting stock
stock_files = {
    "NVDA": "./data/month/multiple_models_Mnthprediction_NVDA (1).csv",
    "AAPL": "./data/month/multiple_models_Mnthprediction_AAPL.csv",
    "MSFT": "./data/month/multiple_models_Mnthprediction_MSFT.csv"
}
stock = st.selectbox("Select Stock", list(stock_files.keys()))

# Load data
file_path = stock_files[stock]
df = load_data(file_path)

# Filter date range
df = filter_date_range(df)

# Generate signals
trade_position = st.slider("Trade Position", min_value=1, max_value=6, value=3)
df = generate_signals(df, trade_position)

# Resample and interpolate
df = resample_and_interpolate(df)

# Set stop-loss percentage
stop_loss_pct = 0.001

# Create and plot portfolios
long_portfolio, short_portfolio = create_portfolio(df['y'], df['long_signal'], df['short_signal'], stop_loss_pct)
st.write("### Long Portfolio Stats:")
st.write(plot_portfolio(long_portfolio, "Long Portfolio Performance"))
st.write("### Short Portfolio Stats:")
st.write(plot_portfolio(short_portfolio, "Short Portfolio Performance"))

# Combine statistics
combined_stats = pd.DataFrame({
    "Long Portfolio": long_portfolio.stats(),
    "Short Portfolio": short_portfolio.stats()
})

# Display combined statistics
st.write("### Combined Portfolio Statistics")
st.write(combined_stats)
