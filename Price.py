import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU , Dropout, Flatten , Conv1D ,MaxPooling1D
import matplotlib.pyplot as plt
from datetime import date
import pandas_datareader as pdr

from math import sqrt
import datetime
import tensorflow
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
def price():
    # Title
    global look_back
    global target_day
    global feature
    st.title("Nuggets!")
    # Subheader
    st.subheader("Smart way to invest money")
    # Diaplay Images
      
    # import Image from pillow to open images
    from PIL import Image
    #img = Image.open(r"C:/Users/Hp/nuggets/title.png")
    img = Image.open(r"title.png")  
    # display image using streamlit
    # width is used to set the width of an image
    st.image(img, width=200)
  
    stock_symbol = str(st.text_input("Stock Symbol(ticker)"))
    st.write("Historical data to be loaded:")   
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days = 1000)
    start_date = st.date_input('Start date', yesterday)
    end_date = st.date_input('End date', today)
    if start_date < end_date:
        pass
    else:
        st.error('Error: End date must fall after start date.')
    look_back = int(st.text_input("Enter a number of previous days to be considered for the prediction of the data:", 1))
    
    target_day = int(st.text_input("Enter the number of days for long term prediction:", 1))
    
    model_name=str(st.selectbox("Select a model for prediction:",['LSTM MODEL','GRU-LSTM MODEL','CONV1D-LSTM MODEL']))
    
    if(st.button('Predict price')):            
            # Load data from yahoo 
            stock=yf.Ticker(stock_symbol)
            df = stock.history(start=str(start_date),end=str(end_date))
          
            
            lst = ['High', 'Low', 'Open', 'Close']

           
            for i in range(len(lst)):
              feature = lst[i]
              temp1=str(feature)+" Price"
              st.success(temp1)
              trainX, trainY, testX, testY, train, test = preprocess_data(df, look_back, feature)
              if(model_name=='LSTM MODEL'):
                 model = get_lstm_model()
              elif(model_name=='GRU-LSTM MODEL'):
                 model=get_gru_lstm_model()
              else:
                 model=get_conv_lstm_model()
              history = train_model(model, trainX, trainY, testX, testY)
            
              #plot_loss_graph(history)
              yhat = plot_prediction_graph(model, testX, testY)
            
              yhat_inverse, testY_inverse = inv_trasnform(yhat, testY)
              cal_metric_results(feature, yhat_inverse, testY_inverse)
              st.write("")
            
              predict_future_short_term(feature, look_back, model, df)
              st.write("")
            
              predict_future_long_term(test, target_day, feature, model)
              st.write("")
              st.write("")
              st.write("---------------------------------------------------------------")
    
def train_model(model, trainX, trainY, testX, testY):
  model.compile(loss='mse', optimizer='adam')
  # Training the model
  history = model.fit(trainX, trainY, epochs= 100, batch_size=256, validation_data=(testX, testY), verbose=False, shuffle=False)
  return history


def plot_loss_graph(history):
  # Plot the graph loss observed for training and testing
  fig, ax = plt.subplots(figsize=(5, 2))
  ax.plot(history.history['loss'], label='train')
  ax.plot(history.history['val_loss'], label='test')
  plt.yticks(fontsize=5)
  plt.xticks(fontsize=5)
  ax.legend(prop={"size":5})
  plt.title("Loss graph of " + feature,fontsize=8)
  st.pyplot(fig)
  
  
def plot_prediction_graph(model, testX, testY):
  yhat = model.predict(testX)
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(yhat, label='Predict')
  ax.plot(testY, label='True')
  plt.yticks(fontsize=5)
  plt.xticks(fontsize=5)
  ax.legend(prop={"size":5})
  plt.title("Past Prediction graph of " + feature,fontsize=8)
  st.pyplot(fig)
  return yhat


def inv_trasnform(yhat, testY):
  yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
  testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
  return yhat_inverse, testY_inverse


def cal_metric_results(feature, yhat_inverse, testY_inverse):
  # Calculate the RMSE Value
  mse = mean_squared_error(testY_inverse, yhat_inverse)
  rmse = sqrt(mse)
  mae = mean_absolute_error(testY_inverse, yhat_inverse)
  st.write(feature + ' Test RMSE: %.3f' % rmse)
  st.write(feature + ' Test MSE: %.3f' % mse)
  st.write(feature + ' Test MAE: %.3f' % mae)
  
  
def predict_future_short_term(feature, look_back, model, df):
  # Select the sequence of steps (days) from databse to make Predictions

  new_df =df.filter([feature])
  last_days=new_df[-look_back:].values

  # Scale the features
  last_days_scaled = scaler.transform(last_days)

  # Create array of input
  X_test=[]
  X_test.append(last_days_scaled)
  X_test =np.array(X_test)

  # reshape input to be (samples, features, timestamps) 
  # reshaping is required for training the model
  X_test =np.reshape(X_test,(X_test.shape[0],1, X_test.shape[1]))
  pred_price = model.predict(X_test)

  # Next days predicted Price
  pred_price =scaler.inverse_transform(pred_price)
  pred_price = pred_price. flatten()
  temp_p= round(pred_price[0],2)
  temp="The "+ feature +" predicted price for next day is "+ str(temp_p)
  st.write(temp)
  
  
def predict_future_long_term(test, target_day, feature, model):

  del_val = len(test) - look_back
  x_input = test[del_val:].reshape(1, -1)
  temp_input = list(x_input)
  temp_input = temp_input[0].tolist()

  lst_preds = []
  i = 0
  while(i < target_day): # prediction for next n days
    if(len(temp_input) > look_back):
      x_input = np.array(temp_input[1:])
      # print("Day {} input {}".format(i, x_input))
      x_input = x_input.reshape(1, -1)
      x_input = np.reshape(x_input, (x_input.shape[0], 1, x_input.shape[1]))
          
      yhat = model.predict(x_input, verbose=0)
      # print("Day {} output {}".format(i, yhat))
      temp_input.extend(yhat[0].tolist())
      temp_input = temp_input[1:]
      lst_preds.extend(yhat.tolist())
      i = i+1
      # print("")
    else:
      x_input = np.reshape(x_input, (x_input.shape[0], 1, x_input.shape[1]))
      yhat = model.predict(x_input, verbose=0)
      # print(yhat[0])
      temp_input.extend(yhat[0].tolist())
      # print(len(temp_input))
      lst_preds.extend(yhat.tolist())
      i=i+1
      # print("")

  lst_preds = scaler.inverse_transform(lst_preds)
  lst_preds.flatten()

  target_day_price = lst_preds[-1]
  temp_p=round(target_day_price[0],2)
  temp="The predicted {} price after {} days is {}".format(feature, target_day,temp_p )
  st.write(temp)
  
def preprocess_data(df, look_back, feature):

  df = df[df[feature].notna()]

  # Reshaping it from 1D to 2D
  values = df[feature].values.reshape(-1,1)
  values = values.astype('float32')

  # scaling each feature to a given range
  scaled = scaler.fit_transform(values)

  # Split the dataset for training and testing
  train_size = int(len(scaled) * 0.8)
  #test_size = len(scaled) - train_size
  train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
  # print(len(train), len(test))

  # converted into X = t, t+1, t+2, t+3 ... t+(n-1) and Y = t+n
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)

  # reshape input to be (samples, features, timestamps) 
  # reshaping is required for training the model
  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  return trainX, trainY, testX, testY, train, test
# Converting data into dependent and independent features based in the Time step
# Convert an array of the values in the dataset matrix

def create_dataset(dataset, look_back):
  
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
  
    return np.array(dataX), np.array(dataY)


def get_lstm_model():

  model = Sequential()

  model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))
  model.add(LSTM(50, return_sequences=True))

  model.add(LSTM(150, return_sequences=True))
  model.add(LSTM(100, return_sequences=True))

  model.add(Flatten())
  model.add(Dense(units=1))

  return model


def get_conv_lstm_model():
   
  model = Sequential()

  model.add(Conv1D(32, 1, input_shape=(1, look_back), padding="same"))
  model.add(Conv1D(64, 1, padding="same"))

  model.add(LSTM(150, return_sequences=True))

  model.add(Conv1D(128, 1, padding="same"))

  model.add(LSTM(100, return_sequences=True))
  model.add(Flatten())
  model.add(Dense(units=1))

  return model


def get_gru_lstm_model():

  model = Sequential()
  model.add(GRU(256 , input_shape = (1 , look_back) , return_sequences=True))
  model.add(Dropout(0.4))
  model.add(LSTM(256))
  model.add(Dropout(0.4))
  model.add(Dense(64 ,  activation = 'relu'))
  model.add(Flatten())
  model.add(Dense(1))

  return model

  
