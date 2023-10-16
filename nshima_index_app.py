import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm

# Load data from Excel sheet
@st.cache_data
def load_data():
    # Replace 'data.xlsx' with the actual file path
    df = pd.read_excel('data.xlsx')
    return df

# Fill NaN values with the mean
def fill_nan_with_mean(df):
    df['Maize_50kgs'].fillna(df['Maize_50kgs'].mean(), inplace=True)
    return df

# Calculate descriptive statistics
def calculate_stats(data):
    stats = data['Maize_50kgs'].describe()
    return stats

# Calculate moving averages
def calculate_moving_averages(data, window_sizes):
    moving_averages = {}
    for window_size in window_sizes:
        column_name = f'MA{window_size}'
        data[column_name] = data['Maize_50kgs'].rolling(window=window_size).mean()
        moving_averages[column_name] = data[column_name]
    return moving_averages

# Calculate Nshima Index
def calculate_nshima_index(data):
    nshima_index = (data['Maize_50kgs'].mean() - data['Maize_50kgs']) / data['Maize_50kgs'].mean()
    return nshima_index

# Create SARIMA forecast
def create_sarima_forecast(data, forecast_days, seasonal_order):
    model = sm.tsa.SARIMAX(data['Maize_50kgs'], order=(30, 1, 0), seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=forecast_days)
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    return forecast_mean, forecast_conf_int

# Create Streamlit app
def main():
    st.title('Maize Price Analysis and Nshima Index')

    # Load data
    df = load_data()

    # Fill NaN values with the mean
    df = fill_nan_with_mean(df)

    # Ensure a datetime index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Filter data after the year 2013
    df = df[df.index.year >= 2013]

    # Sidebar to select pages
    page = st.sidebar.radio("Select a Page", ["Data Visualization", "Nshima Index"])

    if page == "Data Visualization":
        # Calculate and display descriptive statistics
        st.subheader('Descriptive Statistics')
        stats = calculate_stats(df)
        st.write(stats)

        # Calculate and display moving averages (30, 90, 180 days)
        st.subheader('Moving Averages')
        window_sizes = [30, 90, 180]
        moving_averages = calculate_moving_averages(df, window_sizes)

        for window_size, ma_data in moving_averages.items():
            st.write(f'Moving Average ({window_size} days):')
            st.line_chart(ma_data)

    elif page == "Nshima Index":
        # Calculate Nshima Index
        nshima_index = calculate_nshima_index(df)

        # Create SARIMA forecast for the next 180 days
        forecast_days = 180
        seasonal_order = (0, 1, 1, 30)  # Modify seasonal order as needed
        forecast_mean, forecast_conf_int = create_sarima_forecast(df, forecast_days, seasonal_order)

        # Plot Nshima Index over time
        st.subheader('Nshima Index Over Time')
        fig_nshima_index = px.line(df, x=df.index, y=nshima_index, title='Nshima Index Over Time')
        st.plotly_chart(fig_nshima_index)

        # Plot SARIMA forecast
        st.subheader(f'SARIMA Forecast for Next {forecast_days} Days')
        forecast_dates = pd.date_range(start=df.index.max() + pd.DateOffset(days=1), periods=forecast_days, freq='D')
        forecast_df = pd.DataFrame({'date': forecast_dates, 'Forecast': forecast_mean})
        forecast_df.set_index('date', inplace=True)
        forecast_conf_int = forecast_conf_int.rename(columns={'lower Maize_50kgs': 'Lower Bound', 'upper Maize_50kgs': 'Upper Bound'})
        fig_sarima_forecast = go.Figure()
        fig_sarima_forecast.add_trace(go.Scatter(x=df.index, y=df['Maize_50kgs'], mode='lines', name='Maize Price'))
        fig_sarima_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast'))
        fig_sarima_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_conf_int['Lower Bound'], fill=None, mode='lines', line_color='rgba(0, 0, 255, 0.2)', name='Lower Bound'))
        fig_sarima_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_conf_int['Upper Bound'], fill='tonexty', mode='lines', line_color='rgba(0, 0, 255, 0.2)', name='Upper Bound'))
        st.plotly_chart(fig_sarima_forecast)

if __name__ == '__main__':
    main()
