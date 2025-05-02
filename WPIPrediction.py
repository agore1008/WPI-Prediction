#Importing libraries
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px


#The data is not ready to use. We must prepare it first.
#Source Of Data:-https://eaindustry.nic.in/indx_download_1112/monthly_index_202502.xls

def prepare_data(file_path):
    try:
        # URL of the Excel file
        response = requests.get(file_path)
        if response.status_code == 200:
            p_index_data = pd.read_excel(BytesIO(response.content), engine="xlrd")
        else:
            print("Failed to download the file.")
    
        p_index_data["COMM_NAME"] = p_index_data["COMM_NAME"].astype(str).str.strip()

        p_index_dict={"CP&NG":"(D). CRUDE PETROLEUM & NATURAL GAS","CP":"Crude Petroleum","NG":"Natural Gas"}
        df_1=pd.DataFrame()
        df_2=pd.DataFrame()
        df_3=pd.DataFrame()

        for k,v in p_index_dict.items():
            p_index_domain_data=p_index_data[p_index_data["COMM_NAME"]==v]
            req_cols=[col for col in p_index_domain_data.columns if "INDX" in col]
            p_index_domain_data=p_index_domain_data[req_cols]
            p_index_domain_data_T=p_index_domain_data.T
            p_index_domain_data_T.columns=[k]
            p_index_domain_data_T.index=pd.Series(p_index_domain_data_T.index).apply(lambda x:pd.to_datetime(f"{x[4:6]}/01/{x[-4:]}",format="%m/%d/%Y"))
            p_index_domain_data_T=p_index_domain_data_T.reset_index()
            p_index_dt_name=[col for col in p_index_domain_data_T.columns if k not in col][0]
            p_index_domain_data_T = p_index_domain_data_T.rename(columns={p_index_dt_name: 'Year_Month'})
            if k=="CP&NG":
                df_1=p_index_domain_data_T
            elif k=="CP":
                df_2=p_index_domain_data_T
            else:
                df_3=p_index_domain_data_T
        dataset=df_1.merge(df_2, on='Year_Month').merge(df_3, on='Year_Month') 
        dataset.set_index('Year_Month',inplace=True) 

    except Exception as e:
        print("Error While Preparing Data",e)
        dataset=pd.DataFrame()
    return dataset


def plot_line_plot(df):
    try:
        for column in df.columns:
            plt.figure(figsize=(12, 6))  # Adjust figure size as needed
            plt.plot(df.index, df[column])
            plt.title(f"{column} over Time")
            plt.xlabel("Date")
            plt.ylabel(column)
            plt.grid(True)
            plt.show()
        
    except Exception as e:
        print(e)



def train_test_split(data, n_past, target_col=""):
    try:
        def createXY(dataset, n_past):
            dataX = []
            dataY = []
            for i in range(n_past, len(dataset)):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i, 0])
            return np.array(dataX), np.array(dataY)
        
        print(f"Shape of Data :- {data.shape} {data.columns}")
        
        if target_col != "": 
            new_col_order = [target_col] + [col for col in data.columns.tolist() if col != target_col] 
            print(f"Column Order :- {new_col_order}")
            data = data[new_col_order]
        
        # No test split - we'll use TimeSeriesSplit for validation
        global scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences from full dataset
        X, Y = createXY(data_scaled, n_past)
        
        print("Full dataset X Shape -- ", X.shape)
        print("Full dataset Y Shape -- ", Y.shape)
        print("Sample X[0] -- \n", X[0])
        print("Corresponding Y[0] -- ", Y[0])
        
        return X, Y, None, None  # Returning None for testX, testY since we're using full series
    
    except Exception as e:
        print(f"Error While Processing Data: {e}")
        raise
    




def create_model(trainX,trainY,testX,testY):
    inp_shape=(trainX.shape[1],trainX.shape[2])
    def build_model(optimizer):
        grid_model = Sequential()
        grid_model.add(LSTM(50,return_sequences=True,input_shape=inp_shape))
        grid_model.add(LSTM(50,return_sequences=True,input_shape=inp_shape))

        grid_model.add(LSTM(50))
        grid_model.add(Dropout(0.2))
        grid_model.add(Dense(1))
        grid_model.compile(loss = 'mse',optimizer = optimizer)
        return grid_model
     # More robust TimeSeriesSplit configuration
    tscv = TimeSeriesSplit(n_splits=6)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=100,
        restore_best_weights=True,
        verbose=1
    )
    
    grid_model = KerasRegressor(
        build_fn=build_model,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    parameters = {
        'batch_size': [12, 24],
        'epochs': [1000],  # Will likely stop earlier due to early_stopping
        'optimizer': ['adam', 'Adadelta']
    }
    
    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=parameters,
        cv=tscv,
        n_jobs=1,
        scoring='neg_mean_squared_error'
    )
    
    # Using full training data with time-based cross-validation
    grid_search.fit(trainX, trainY)
    
    print("Best Parameters:", grid_search.best_params_)
    my_model = grid_search.best_estimator_.model
    
    return my_model



def test_prediction(my_model,trainX,trainY,testX,testY):
    try:
         # Since we're using full series, we'll predict on the last 20% as a pseudo-test
        test_size = int(len(trainX) * 0.2)
        pseudo_testX = trainX[-test_size:]
        pseudo_testY = trainY[-test_size:]
        
        print("Shape of pseudo test X", pseudo_testX.shape)
        prediction = my_model.predict(pseudo_testX)
        print("\nPrediction Shape -", prediction.shape)
        
        prediction_copies_array = np.repeat(prediction, trainX.shape[2], axis=-1)
        pred = scaler.inverse_transform(np.reshape(prediction_copies_array, 
                                                (len(prediction), trainX.shape[2])))[:, 0]
        
        original_copies_array = np.repeat(pseudo_testY, trainX.shape[2], axis=-1)
        original = scaler.inverse_transform(np.reshape(original_copies_array, 
                                                   (len(pseudo_testY), trainX.shape[2])))[:, 0]

        plt.figure(figsize=(12, 6))
        plt.plot(original, color='red', label='Actual Price Index')
        plt.plot(pred, color='blue', label='Predicted Price Index')
        plt.title('Time Series Prediction (Last 20% as Validation)')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.show()
        
        # Calculate and print metrics
        mse = mean_squared_error(original, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(original, pred)
        print(f"\nValidation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    except Exception as e:
        print("Error While Testing Predictions",e)
        pred=None
    return pred





if __name__ == '__main__': 

    print("OK")
    url = "https://eaindustry.nic.in/indx_download_1112/monthly_index_202502.xls"
    dataset=prepare_data(url)
    n_past=6
    trainX,trainY,testX,testY=train_test_split(dataset,n_past,"CP&NG")
    my_model=create_model(trainX,trainY,testX,testY)
    test_prediction(my_model,trainX,trainY,testX,testY)

    