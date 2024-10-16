import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st

# Function to get the previous available date
def get_previous_available_date(date, date_list):
    date_list = pd.to_datetime(date_list, errors='coerce')
    date = pd.to_datetime(date)
    date_list = date_list.dropna()  # Remove NaT values
    date_list_sorted = date_list.sort_values().reset_index(drop=True)
    idx = date_list_sorted.searchsorted(date, side='right') - 1
    if idx >= 0 and idx < len(date_list_sorted):
        return date_list_sorted.iloc[idx]
    else:
        return None

# Function to check if the support line is too close to another support line
def is_line_too_close(new_support_price, existing_support_lines, buffer):
    """Checks if the new support line is too close to any existing support lines."""
    return any(abs(new_support_price - support) < buffer for support in existing_support_lines)

# Streamlit Title
st.title("Candlestick Chart and Market Trend Analysis with Support Line Logic")

# Read data1.csv
data1 = pd.read_csv('data1.csv')
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data1 = data1.dropna(subset=['date'])  # Remove rows with invalid dates

# Ensure numerical data types for high, low, close, and open columns
data1[['open', 'high', 'low', 'close']] = data1[['open', 'high', 'low', 'close']].astype(float)

# Get min and max dates from data1.csv
min_date = data1['date'].min()
max_date = data1['date'].max()

# Initialize session state for storing the random date
if 'random_date' not in st.session_state:
    st.session_state.random_date = None

# User Input for Date
selected_date = st.date_input(
    "Select a date",
    value=st.session_state.random_date.date() if st.session_state.random_date else max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date(),
    key="selected_date"
)

# Add a button to select a random date
if st.button("Select Random Date"):
    st.session_state.random_date = pd.to_datetime(np.random.choice(pd.date_range(min_date, max_date).date))
    selected_date = st.session_state.random_date
    st.write(f"Random date selected: {selected_date.date()}")
else:
    if st.session_state.random_date:
        selected_date = st.session_state.random_date

selected_date = pd.to_datetime(selected_date)

# Function to get the previous available date in data1.csv
dates_in_data1 = data1['date']
selected_date_in_data1 = get_previous_available_date(selected_date, dates_in_data1)

# First Graph: Market Trend Analysis using Linear Regression
st.header("Market Trend of the XAUUSD Market using Least Squares Method")
if selected_date_in_data1 is None:
    st.warning(f"No available date before or on {selected_date.date()} in data1.csv")
else:
    selected_date_data1 = selected_date_in_data1
    # Always use a fixed lookback period of 31 days
    lookback_days_first_graph = 31
    start_date = selected_date_data1 - pd.Timedelta(days=lookback_days_first_graph - 1)
    end_date = selected_date_data1

    # Filter data between start_date and end_date
    filtered_data1 = data1[(data1['date'] >= start_date) & (data1['date'] <= end_date)]

    if filtered_data1.empty:
        st.warning(f"No data available from {start_date.date()} to {end_date.date()} in data1.csv.")
    else:
        # Prepare data for regression
        filtered_data1 = filtered_data1.sort_values('date')
        filtered_data1['day_number'] = (filtered_data1['date'] - filtered_data1['date'].min()).dt.days

        # Select the price option
        price_option = st.selectbox("Select Price Type for Analysis", ["Closing Price", "Average Price (High + Low) / 2"])
        if price_option == "Closing Price":
            y = filtered_data1['close'].values
        else:
            y = ((filtered_data1['high'] + filtered_data1['low']) / 2).values

        x = filtered_data1['day_number'].values

        # Calculate sums for linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_squared = np.sum(x ** 2)
        denominator = n * sum_x_squared - sum_x ** 2

        if denominator == 0:
            st.error("Cannot compute linear regression due to zero denominator.")
        else:
            # Calculate slope (m) and intercept (b)
            m = (n * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator

            # Display the equation of the line
            st.write(f"The equation of the line is: y = {m:.2f}x + {b:.2f}")

            # Generate predictions
            predicted_prices = m * x + b

            # Plot the results using Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(filtered_data1['date'], y, color='blue', label='Original Data')
            ax.plot(filtered_data1['date'], predicted_prices, color='red', label=f'Linear fit: y = {m:.2f}x + {b:.2f}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Linear Regression on {price_option} from {start_date.date()} to {end_date.date()}')
            ax.legend()
            ax.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display the trend message
            if m > 1:
                st.success(f"The market is up trending (Bullish) from {start_date.date()} to {end_date.date()}. Prioritize buying at this moment.")
            elif m < -1:
                st.error(f"The market is down trending from {start_date.date()} to {end_date.date()}. Prioritize selling at this moment.")
            else:
                st.info(f"The market is ranging from {start_date.date()} to {end_date.date()}. It is better to wait until a clear trend forms.")

# Second Graph: Candlestick Chart with Support, Resistance, and Trendline Analysis
st.header("Trendline Analysis with Support/Resistance Lines and Linear Regression")

# User input for minimum distance between two support lines
min_distance_between_supports = st.number_input(
    "Enter the minimum distance between two support lines (e.g., 2% of price range):",
    min_value=0.01,  # minimum value is 0.01 to avoid too small numbers
    value=0.02  # default value is 2%
)

if selected_date_in_data1 is None:
    st.warning(f"No available date before or on {selected_date.date()} in data1.csv")
else:
    selected_date_data1 = selected_date_in_data1
    lookback_days = st.number_input(
        "Enter the number of days to include before the chosen date (for Trendline Analysis)",
        min_value=1,
        max_value=365,
        value=31,
        key="lookback_days"
    )
    start_date1 = selected_date_data1 - pd.Timedelta(days=lookback_days - 1)
    end_date1 = selected_date_data1

    # Filter data between start_date1 and end_date1
    data1_filtered = data1[(data1['date'] >= start_date1) & (data1['date'] <= end_date1)]

    if data1_filtered.empty:
        st.warning(f"No data available from {start_date1.date()} to {end_date1.date()} in data1.csv.")
    else:
        # Sort the data by date in ascending order
        data1_filtered = data1_filtered.sort_values('date')

        # Prepare data
        data1_filtered = data1_filtered[['date', 'open', 'high', 'low', 'close']]
        data1_filtered = data1_filtered.set_index('date')

        # Ensure numerical data types
        data1_filtered = data1_filtered.astype(float)

        # Prepare list to store lines and track support lines
        lines = []
        support_lines = []

        # Variables to track state
        red_candle_found = False
        red_candle_high = None
        red_candle_low = None
        last_red_candle_low = None

        # Iterate over the filtered data to find patterns
        for i in range(2, len(data1_filtered)):
            current_candle = data1_filtered.iloc[i]
            previous_candle_1 = data1_filtered.iloc[i - 1]
            previous_candle_2 = data1_filtered.iloc[i - 2]

            # Check for at least two consecutive green candles
            if (previous_candle_1['close'] > previous_candle_1['open'] and
                previous_candle_2['close'] > previous_candle_2['open']):
                # Find a valid red candle after the green sequence
                if current_candle['close'] < current_candle['open']:
                    red_candle_found = True
                    red_candle_high = current_candle['open']
                    red_candle_low = current_candle['close']
                    last_red_candle_low = red_candle_low  # Store the low of this red candle
                    continue  # Move to the next candle to look for the green one

            # If another red candle is found after the first, update the red candle tracking
            if red_candle_found and current_candle['close'] < current_candle['open']:
                red_candle_high = current_candle['open']
                red_candle_low = current_candle['close']
                last_red_candle_low = red_candle_low  # Update to the most recent red candle

            # Once a red candle is found, look for the next green candle that breaks its high
            if red_candle_found:
                if current_candle['close'] > current_candle['open'] and current_candle['close'] > red_candle_high:
                    # Check if the support line is too close to another support line
                    if not is_line_too_close(last_red_candle_low, support_lines, min_distance_between_supports):
                        # Add the green horizontal line at the low of the most recent red candle's body
                        lines.append(mpf.make_addplot([last_red_candle_low]*len(data1_filtered), color='green', linestyle='--'))
                        support_lines.append(last_red_candle_low)  # Store the support line
                        st.write(f"Green line drawn at {last_red_candle_low} from the red candle on {data1_filtered.index[i-1].date()}.")
                    red_candle_found = False  # Reset the red candle tracking

        # If no green candle breaks the red candle's high, draw the line on the low of the last red candle
        if red_candle_found and last_red_candle_low:
            if not is_line_too_close(last_red_candle_low, support_lines, min_distance_between_supports):
                lines.append(mpf.make_addplot([last_red_candle_low]*len(data1_filtered), color='green', linestyle='--'))
                support_lines.append(last_red_candle_low)
                st.write(f"Green line drawn at {last_red_candle_low} from the last red candle's low.")

        # Plot the candlestick chart with the trendlines and regression line
        fig, axlist = mpf.plot(
            data1_filtered,
            type='candle',
            style='charles',
            title=f"Candlestick with Support, Resistance, and Regression Line from {start_date1.date()} to {end_date1.date()}",
            figsize=(10, 6),
            returnfig=True,
            addplot=lines if lines else None  # Add support lines if they exist
        )

        # Display the plot in Streamlit
        st.pyplot(fig)
