## Stock Price Prediction using LSTM on IDFC Bank Data
- **Overview:**
  - This project demonstrates the prediction of stock prices using the LSTM (Long Short-Term Memory) model on the historical stock price data of IDFC Bank from the National Stock Exchange (NSE) of India. The dataset includes the stock prices from the year 2016 to 2017. The objective is to predict the future stock prices using LSTM, a type of recurrent neural network (RNN) that is particularly effective for time-series forecasting.

- **Key Components:**
  - **Data:** Historical stock price data for IDFC Bank from the NSE.
  - **Model:** LSTM (Long Short-Term Memory) neural network to predict future stock prices based on historical prices.
  - **Evaluation Metrics:** RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² score to evaluate model performance.
- **Dataset**
  The dataset used for this project is from the NSE stock data and includes:

  - **Timestamp:** The date and time of the stock transaction.
  - **SYMBOL:** Stock symbol, filtered for IDFCBANK.
  - **SERIES:** Trading series.
OPEN, CLOSE, HIGH, LOW: Stock prices at different times of the day.
  - **PREVCLOSE:** Previous day closing price.
- **This data was preprocessed by:**
  
  - Dropping irrelevant columns (SYMBOL, SERIES, etc.).
  Filling missing values using forward-fill method (ffill).
  Normalizing the data using MinMaxScaler to scale the values between 0 and 1.
  Creating a sliding window of 60 time-steps to prepare data for LSTM.
- **Approach**
  - **Data Preprocessing:**
  
    - Loaded and cleaned the stock price data.
  Dropped weakly correlated features.
  Handled missing values using forward-fill (ffill).
  Scaled the data for model input using MinMaxScaler.
  - **LSTM Model:**
  
    - Used an LSTM model with 50 units and a dropout layer to prevent overfitting.
  Trained the model for 50 epochs on the scaled data.
  Predicted stock prices for both training and test datasets.
  - **Evaluation:**
  
    - Evaluated the model's performance using RMSE, MAE, and R² score.
  Plotted the predicted vs actual stock prices.
  - **Future Price Prediction:**
  
    - Used the last 60 days of the stock prices to predict the next 30 days.
  Plotted the predicted stock prices for the next 30 business days.
# Results
    - RMSE (Root Mean Squared Error): 0.60
    - MAE (Mean Absolute Error): 0.46
    - R² Score for Training Data: 0.97
    - R² Score for Test Data: ~ 0.73



📓 Colab Notebook
Explore the implementation in this project using the[(https://colab.research.google.com/drive/1A3sKONzKgz0o_xZFCRECpicKNpQTLraQ?usp=sharing]).

- Dataset link: https://www.kaggle.com/datasets/minatverma/nse-stocks-data	
