# ai_fbprophet model

# stock prediction - streamlit app

import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Option Time Waves - Future")

stocks = ("AAPL","GOOG","MSFT","XOM","INTC","IBM","MAR","CSCO")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
	data = yf.download(ticker, START, TODAY)
	data.reset_index(inplace=True)
	return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plow_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='stock_high'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='stock_low'))
	fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()


# Forcasting
# need to look up symbol ref for additional items open,high,low

df_train = data['Date','Close']
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

