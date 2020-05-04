
"""
Created on Sat Nov 16 17:33:11 2019

@author: aparn
@author: nachi
@author: sarvani
@author: anjali
"""

import random
import math
import pandas as pd
import numpy as np
from pandas import datetime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
#change this path to the path in your local drive
path  = "C:/Users/aparn/OneDrive/Desktop/FALL_2019/ML/project/walmart-recruiting-store-sales-forecasting/"

#the files stores.csv,features.csv,train.csv and test,csv are attached in the zip file
store_dataframe = pd.read_csv(path+"stores.csv")
feature_dataframe = pd.read_csv(path+"features.csv")
train_dataframe = pd.read_csv(path+"train.csv")
test_dataframe = pd.read_csv(path+"test.csv")


feature_dataframe['Date'] = pd.to_datetime(feature_dataframe['Date'])
train_dataframe['Date'] = pd.to_datetime(train_dataframe['Date'],format = "%Y-%m-%d")
test_dataframe['Date'] = pd.to_datetime(test_dataframe['Date'],format = "%Y-%m-%d")

def sigmoid_function(dataValue):
    return 1. / (1 + np.exp(-dataValue))

def sigmoid_derivative(dataValue):
    return dataValue*(1-dataValue)

def tanh_derivative(dataValue):
    return 1. - dataValue ** 2

def generate_random_array(startValue, endValue, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (endValue - startValue) + startValue

class Param:
    def __init__(self, curr_vals, inp_dimension):
        self.curr_vals = curr_vals
        self.inp_dimension = inp_dimension
        total_length = inp_dimension + curr_vals
        
        self.weight_input = generate_random_array(-0.1, 0.1, curr_vals, total_length)
        self.weight_current = generate_random_array(-0.1, 0.1, curr_vals, total_length)
        self.weight_output = generate_random_array(-0.1, 0.1, curr_vals, total_length)
        self.weight_forget = generate_random_array(-0.1, 0.1, curr_vals, total_length)

        
        self.b_input = generate_random_array(-0.1, 0.1, curr_vals)
        self.b_current = generate_random_array(-0.1, 0.1, curr_vals)
        self.b_output = generate_random_array(-0.1, 0.1, curr_vals)
        self.b_forget = generate_random_array(-0.1, 0.1, curr_vals)

       
        self.weight_input_diff = np.zeros((curr_vals, total_length))
        self.weight_current_diff = np.zeros((curr_vals, total_length))
        self.weight_output_diff = np.zeros((curr_vals, total_length))
        self.weight_forget_diff = np.zeros((curr_vals, total_length))
        self.b_input_diff = np.zeros(curr_vals)
        self.b_current_diff = np.zeros(curr_vals)
        self.b_output_diff = np.zeros(curr_vals)
        self.b_forget_diff = np.zeros(curr_vals)


    def set_difference(self, learning_rate_val = 1):
        self.weight_input -= learning_rate_val * self.weight_input_diff
        self.weight_current -= learning_rate_val * self.weight_current_diff
        self.weight_output -= learning_rate_val * self.weight_output_diff
        self.weight_forget -= learning_rate_val * self.weight_forget_diff
        self.b_input -= learning_rate_val * self.b_input_diff
        self.b_current -= learning_rate_val * self.b_current_diff
        self.b_output -= learning_rate_val * self.b_output_diff
        self.b_forget -= learning_rate_val * self.b_forget_diff

        self.weight_input_diff = np.zeros_like(self.weight_input)
        self.weight_current_diff = np.zeros_like(self.weight_current)
        self.weight_output_diff = np.zeros_like(self.weight_output)
        self.weight_forget_diff = np.zeros_like(self.weight_forget)
        self.b_input_diff = np.zeros_like(self.b_input)
        self.b_current_diff = np.zeros_like(self.b_current)
        self.b_output_diff = np.zeros_like(self.b_output)
        self.b_forget_diff = np.zeros_like(self.b_forget) 


class L_Curr_St:
    def __init__(self, curr_vals, inp_dimension):
        self.input_mat = np.zeros(curr_vals)
        self.current_mat = np.zeros(curr_vals)
        self.output_mat = np.zeros(curr_vals)
        self.forget_mat = np.zeros(curr_vals)
        self.y_mat = np.zeros(curr_vals)
        self.x_mat = np.zeros(curr_vals)
        self.x_output_diff = np.zeros_like(self.x_mat)
        self.y_output_diff = np.zeros_like(self.y_mat)
    
class Node:
    def __init__(self, param_val, l_curr_st):
        self.l_curr_st = l_curr_st
        self.param_val = param_val
        self.x_current = None

    def bottom_values(self, inp, y_mat_previous = None, x_mat_previous = None):
        if x_mat_previous is None: x_mat_previous = np.zeros_like(self.l_curr_st.x_mat)
        if y_mat_previous is None: y_mat_previous = np.zeros_like(self.l_curr_st.y_mat)
        self.x_mat_previous = x_mat_previous
        self.y_mat_previous = y_mat_previous
        x_current = np.hstack((inp,  x_mat_previous))
        self.l_curr_st.input_mat = sigmoid_function(np.dot(self.param_val.weight_input, x_current) + self.param_val.b_input)
        self.l_curr_st.current_mat = np.tanh(np.dot(self.param_val.weight_current, x_current) + self.param_val.b_current)
        self.l_curr_st.output_mat = sigmoid_function(np.dot(self.param_val.weight_output, x_current) + self.param_val.b_output)
        self.l_curr_st.forget_mat = sigmoid_function(np.dot(self.param_val.weight_forget, x_current) + self.param_val.b_forget)
        self.l_curr_st.x_mat = self.l_curr_st.y_mat * self.l_curr_st.output_mat
        self.l_curr_st.y_mat = self.l_curr_st.current_mat * self.l_curr_st.input_mat + y_mat_previous * self.l_curr_st.forget_mat
        self.x_current = x_current
    
    def find_top_diff(self, top_diff_x, top_diff_y):
       
        diff_out = self.l_curr_st.y_mat * top_diff_x
        diff_y = self.l_curr_st.output_mat * top_diff_x + top_diff_y
        diff_current = self.l_curr_st.input_mat * diff_y
        diff_input = self.l_curr_st.current_mat * diff_y
        diff_forget = self.y_mat_previous * diff_y

        
        diff_forget_input = sigmoid_derivative(self.l_curr_st.forget_mat) * diff_forget
        diff_input_input = sigmoid_derivative(self.l_curr_st.input_mat) * diff_input
        diff_current_input = tanh_derivative(self.l_curr_st.current_mat) * diff_current
        diff_output_input = sigmoid_derivative(self.l_curr_st.output_mat) * diff_out

        
        self.param_val.weight_forget_diff += np.outer(diff_forget_input, self.x_current)
        self.param_val.weight_input_diff += np.outer(diff_input_input, self.x_current)
        self.param_val.weight_current_diff += np.outer(diff_current_input, self.x_current)
        self.param_val.weight_output_diff += np.outer(diff_output_input, self.x_current)
        self.param_val.b_forget_diff += diff_forget_input
        self.param_val.b_input_diff += diff_input_input
        self.param_val.b_current_diff += diff_current_input
        self.param_val.b_output_diff += diff_output_input
        
        diff_x_current = np.zeros_like(self.x_current)
        diff_x_current += np.dot(self.param_val.weight_forget.T, diff_forget_input)
        diff_x_current += np.dot(self.param_val.weight_input.T, diff_input_input)
        diff_x_current += np.dot(self.param_val.weight_current.T, diff_current_input)
        diff_x_current += np.dot(self.param_val.weight_output.T, diff_output_input)

        
        self.l_curr_st.bottom_diff_x = diff_x_current[self.param_val.inp_dimension:]
        self.l_curr_st.bottom_diff_y = diff_y * self.l_curr_st.forget_mat


class Network():
    def __init__(self, param_val):
        self.lstm_param_val = param_val
        self.lstm_values_l = []
        self.inp_values_list = []

    def y_list_is(self, list_y, current_layer_w_loss):
        
        assert len(list_y) == len(self.inp_values_list)
        current_index = len(self.inp_values_list) - 1
        
        lossValue = current_layer_w_loss.calculate_loss(self.lstm_values_l[current_index].l_curr_st.x_mat, list_y[current_index])
        h_diff = current_layer_w_loss.find_vals_diff_bottom(self.lstm_values_l[current_index].l_curr_st.x_mat, list_y[current_index])
        s_diff = np.zeros(self.lstm_param_val.curr_vals)
        self.lstm_values_l[current_index].find_top_diff(h_diff, s_diff)
        current_index -= 1

        
        while current_index >= 0:
            lossValue += current_layer_w_loss.calculate_loss(self.lstm_values_l[current_index].l_curr_st.x_mat, list_y[current_index])
            h_diff = current_layer_w_loss.find_vals_diff_bottom(self.lstm_values_l[current_index].l_curr_st.x_mat, list_y[current_index])
            h_diff += self.lstm_values_l[current_index + 1].l_curr_st.bottom_diff_x
            s_diff = self.lstm_values_l[current_index + 1].l_curr_st.bottom_diff_y
            self.lstm_values_l[current_index].find_top_diff(h_diff, s_diff)
            current_index -= 1 

        return lossValue

    def inp_list_clear(self):
        self.inp_values_list = []

    def add_inp_to_list(self, inp):
        self.inp_values_list.append(inp)
        if len(self.inp_values_list) > len(self.lstm_values_l):
            lstm_val = L_Curr_St(self.lstm_param_val.curr_vals, self.lstm_param_val.inp_dimension)
            self.lstm_values_l.append(Node(self.lstm_param_val, lstm_val))
        most_recent_ind = len(self.inp_values_list) - 1
        if most_recent_ind == 0:
            self.lstm_values_l[most_recent_ind].bottom_values(inp)
        else:
            previous_s = self.lstm_values_l[most_recent_ind - 1].l_curr_st.y_mat
            previous_h = self.lstm_values_l[most_recent_ind - 1].l_curr_st.x_mat
            self.lstm_values_l[most_recent_ind].bottom_values(inp, previous_s, previous_h)


train_data = pd.merge(train_dataframe, store_dataframe, on="Store", how="left")
test_data = pd.merge(test_dataframe, store_dataframe, on="Store", how="left")


train_data = pd.merge(train_data, feature_dataframe, on=["Date", "Store"], how="inner")
test_data = pd.merge(test_data, feature_dataframe, on=["Date", "Store"], how="inner")


train_data = train_data.drop(["IsHoliday_y"], axis=1)
test_data = test_data.drop(["IsHoliday_y"], axis=1)



train_data.loc[train_data['MarkDown1'] < 0, ['MarkDown1']] = 0.001
train_data.loc[train_data['MarkDown2'] < 0, ['MarkDown2']] = 0.00
train_data.loc[train_data['MarkDown3'] < 0, ['MarkDown3']] = 0.00
train_data.loc[train_data['MarkDown4'] < 0, ['MarkDown4']] = 0.00
train_data.loc[train_data['MarkDown5'] < 0, ['MarkDown5']] = 0.00
train_data.loc[train_data['Weekly_Sales'] < 0, ['Weekly_Sales']] = 0.00

test_data.loc[test_data['MarkDown1'] < 0, ['MarkDown1']] = 0.00
test_data.loc[test_data['MarkDown2'] < 0, ['MarkDown2']] = 0.00
test_data.loc[test_data['MarkDown3'] < 0, ['MarkDown3']] = 0.00
test_data.loc[test_data['MarkDown4'] < 0, ['MarkDown4']] = 0.00
test_data.loc[test_data['MarkDown5'] < 0, ['MarkDown5']] = 0.00



p_train_data = train_data.fillna(0)
p_test_data = test_data.fillna(0)


lblEncoder = preprocessing.LabelEncoder()
lblEncoder.fit(p_train_data['IsHoliday_x'].values.astype('str'))
p_train_data['IsHoliday_x'] = lblEncoder.transform(p_train_data['IsHoliday_x'].values.astype('str'))
lblEncoder.fit(p_test_data['IsHoliday_x'].values.astype('str'))
p_test_data['IsHoliday_x'] = lblEncoder.transform(p_test_data['IsHoliday_x'].values.astype('str'))
lblEncoder.fit(p_train_data['Type'].values.astype('str'))
p_train_data['Type'] = lblEncoder.transform(p_train_data['Type'].values.astype('str'))
lblEncoder.fit(p_test_data['Type'].values.astype('str'))
p_test_data['Type'] = lblEncoder.transform(p_test_data['Type'].values.astype('str'))

p_train_data = p_train_data[['Store', 'Dept', 'Date', 'Unemployment', 'IsHoliday_x', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Weekly_Sales']]


splitting_date = pd.datetime(2012, 8, 24)
train_data_set = p_train_data.loc[p_train_data['Date'] <= splitting_date]
dev_data_set = p_train_data.loc[p_train_data['Date'] > splitting_date]


split_date_dev = pd.datetime(2012,9,25)
val_data_set = dev_data_set.loc[dev_data_set['Date'] <= split_date_dev]
test_data_set = dev_data_set.loc[dev_data_set['Date'] > split_date_dev]

train_data_set = train_data_set.set_index('Date')
val_data_set = val_data_set.set_index('Date')
test_data_set = test_data_set.set_index('Date')

train_data_set_array = train_data_set.iloc[:, :].values
val_data_set_array = val_data_set.iloc[:, :].values
test_data_set_array = test_data_set.iloc[:, :].values

#print("Shape of train, val and test array:\n",train_set_array.shape,"\n",val_set_array.shape,"\n",test_set_array.shape)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = scaler.fit_transform(train_data_set_array[:, :])
val_set_scaled = scaler.fit_transform(val_data_set_array[:, :])
test_set_scaled = scaler.fit_transform(test_data_set_array[:, :])

#print(train_set_scaled.shape, val_set_scaled.shape, test_set_scaled.shape)

X_train_data = []
y_train_data = []
X_val_data = []
y_val_data = []
X_test_data = []
y_test_data = []

X_train_data, y_train_data = train_set_scaled[:,:-1], train_set_scaled[:,-1]
X_val_data, y_val_data = val_set_scaled[:,:-1], val_set_scaled[:,-1]
X_test_data, y_test_data = test_set_scaled[:,:-1], test_set_scaled[:,-1]
print(y_train_data)

class SalesLossLayer:
   
    @classmethod
    def calculate_loss(self, prediction, labelValue):
        return (prediction[0] - labelValue) ** 2

    @classmethod
    def find_vals_diff_bottom(self, prediction, labelValue):
        bottom_df_val = np.zeros_like(prediction)
        bottom_df_val[0] = 2 * (prediction[0] - labelValue)
        return bottom_df_val

def RNNwithLSTM(epoch):
    
    number_of_cell_count = 14
    inp_dimension = X_train_data.shape[1]
    param_value_lstm = Param(number_of_cell_count, inp_dimension)
    netValueLstm = Network(param_value_lstm)
    
    out_list = y_train_data
    input_data = X_train_data
    predictionsList = []
    for iterValue in range(epoch):
        print("Epoch", "%2s" % str(iterValue+1), end=": ")
        for index_val in range(len(out_list)):
            netValueLstm.add_inp_to_list(input_data[index_val])

        predictionsList.append([netValueLstm.lstm_values_l[index_val].l_curr_st.x_mat[0] for index_val in range(len(out_list))])
        calculated_loss_val = netValueLstm.y_list_is(out_list, SalesLossLayer)
        print("calculated_loss_val during training:", "%.3e" % calculated_loss_val)
        param_value_lstm.set_difference(learning_rate_val=.00000001)
        netValueLstm.inp_list_clear()
    
    out_list = y_test_data
    input_data = X_test_data
    predictionsList = []
    for index_val in range(len(out_list)):
        netValueLstm.add_inp_to_list(input_data[index_val])

    predictionsList.append(["% 2.5f" % netValueLstm.lstm_values_l[index_val].l_curr_st.x_mat[0] for index_val in range(len(out_list))])
    new_prdicted_sales_df = pd.DataFrame(predictionsList)
    new_observed_saled_df = pd.DataFrame(y_test_data)
    print("Prediced_sales", new_prdicted_sales_df)
    print(new_prdicted_sales_df.shape)
    print("Observed_sales", new_observed_saled_df)
    totalLossValue = netValueLstm.y_list_is(out_list, SalesLossLayer)
    print("calculated_loss_val during test:", "%.3e" % totalLossValue)
    param_value_lstm.set_difference(learning_rate_val=.00000001)
    netValueLstm.inp_list_clear()
    # plt.plot(new_prdicted_sales_df, 'g')
    # plt.plot(new_observed_saled_df, 'r')
    # plt.show()

if __name__ == "__main__":
    learning_rate = input("Enter the number of epochs for training the model:")
    RNNwithLSTM(int(learning_rate))