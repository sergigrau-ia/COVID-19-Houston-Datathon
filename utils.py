import numpy as np
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.multioutput import MultiOutputRegressor
from main import *
from sklearn.preprocessing import MinMaxScaler


class Cue:
    # Constructor
    def __init__(self, values=[]): 
        self.values = values
    
    # Remove old element
    def pop(self):
        if(len(self.values) > 0):
            element = self.values[0]
            self.values = self.values[1:]
            return element
            
    # Add new element            
    def push(self, value):
        self.values= np.append(self.values, value)
        
    # Get cue elements 
    def getCueElements(self, shape=None):
        if shape is not None:
            return np.reshape(self.values, shape)
        return self.values  

def reshape_df(df, type_data):
    df = df.reindex(columns=df.columns.tolist())
    columns = ["UID", "iso2", "iso3", "code3", "FIPS", "Admin2",
                          "Province_State", "Country_Region", "Lat", "Long_",
                          "Combined_Key", "Population"]
    target = 'Deaths'
    if type_data == 1:
        columns = ["UID", "iso2", "iso3", "code3", "FIPS", "Admin2",
                          "Province_State", "Country_Region", "Lat", "Long_",
                          "Combined_Key"]
        target = 'Confirmed'
    df = df.melt(id_vars=columns,
                 var_name='Day', value_name=target)
    return df        

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def get_dataset(path, uid, lag, county_name, data_type):
    print('\n#PREPARING DATA#')
    
    '''df_confirmed = pd.read_csv(CONFIRMED_PATH_DATA, sep=",")
    # Select specific country
    df_confirmed = df_confirmed[df_confirmed["UID"] == uid]
    # Reshape the dataset to use date columns as rows
    df_confirmed = reshape_df(df_confirmed, 1)
    df_confirmed['Day'] = pd.to_datetime(df_confirmed['Day'], format="%m/%d/%y")
    df_confirmed = df_confirmed[(df_confirmed.Day >= '2020-04-01') &
                                (df_confirmed.Day <= '2020-09-12')]

    df_confirmed['diff'] = df_confirmed['Confirmed'].diff()
    df_confirmed['diff_confirmed_7'] = df_confirmed['diff'].shift(7)'''

    #df_confirmed['diff_confirmed_7'] = df_confirmed['diff']
    
    if data_type == 'HOSPITALISATIONS':
        # Read hospitalizations file
        df = pd.read_excel('hospitalizations/{}_hosp.xlsx'.format(county_name))
        
        # Calculate total covid hospitalizations per day
        df['Hospitalisations'] = df['CovidBed'] + df['ICUCovidBed']
        #df['diff_confirmed_7'] = df_confirmed['diff_confirmed_7'].values
        
        # Remove innecessary columns
        #df = df[['Date', 'diff_confirmed_7', 'Hospitalisations']]
        df = df[['Date', 'Hospitalisations']]
        
        #df = df[['Date', 'Hospitalisations']]
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y")
        
        # Set date as the index
        df = df.set_index('Date')
        #df = df[df.index < '2020-09-10']
        diff_column = 'Hospitalisations'
        #df[df.confirmed < 0] = 0
        df = df.dropna()
    else:
        # Read dataset
        df = pd.read_csv('deaths_timeseries.csv', sep=",")
        # Select specific country
        df = df[df["UID"] == uid]
        # Reshape the dataset to use date columns as rows
        df = reshape_df(df, 0)
        
        # Get necessary only columns
        df = df[['Day', 'Deaths']]
        # Give datetime format to column Day
        df['Day'] = pd.to_datetime(df['Day'], format="%m/%d/%y")
        df = df[df.Day < '2020-09-14']
        
        # Remove innecessary columns
        df = df[['Day', 'Deaths']]   
        # Set day as the index
        df = df.set_index('Day')
        diff_column = 'Deaths'
         
    # Save a copy of the original dataframe
    df_original = df.copy()
    '''if data_type == 'HOSPITALISATIONS':
        del df_original['confirmed_7']'''
    
    # Obtain the diff column
    if diff_column == 'Deaths':
        df['diff'] = df[diff_column].diff()
    else:
        df['diff'] = df[diff_column].diff()
        
    # Obtain all dates in the dataset
    dates = df.index
    # Remove an unuseful columns
    del df[diff_column]
    df = df.dropna()

    # Transform the dataframe into a n-d array
    data = df.values

    # Normalize data
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    
    # Attributes for live
    attr = []
    '''if ACTUAL_MODE == 'LIVE':
        attr = df_confirmed['diff'].values[-7:]        
        attr = scaler.transform(pd.DataFrame({'a': attr, 'b': attr}).values)[:, 0]'''
    
    # Transform the time series data to a supervised data
    supervised_data = series_to_supervised(data, lag)

    # Obtain target values
    y = supervised_data[supervised_data.columns[-1]].values

    # Remove the target values from the data
    del supervised_data[supervised_data.columns[-1]]

    # Obtain the features
    X = supervised_data.values
    
    return X, y, df_original, dates, scaler, attr


def get_train_test(X, y, mode, test_size=0.05):
    print('#CREATING PARTITIONS#')
    if mode == 'TRAINING' or mode == 'MODEL_TUNNING':
        # Get train and test partitions with no shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        # Shuffle the train data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True)
        # Obtain the final train shuffled data
        X_train = np.append(X_train, X_val, axis=0)
        y_train = np.append(y_train, y_val, axis=0)
    elif mode == 'LIVE':
        # Get train and test partitions shuffling the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        # Obtain the final train shuffled data
        X_train = np.append(X_train, X_test, axis=0)
        y_train = np.append(y_train, y_test, axis=0)
        X_test = X[-1, :]
        y_test = y[-1]
    
    print("\nSETS DIMENSIONS:\n\nX TRAIN:", X_train.shape)
    print("Y TRAIN:", y_train.shape)
    print("\nX TEST:", X_test.shape)
    print("Y TEST:", y_test.shape)    

    return X_train, y_train, X_test, y_test

def train_regressor(X_train, y_train, parameters=None):
    # Instanciate and parameterize XGBRegressor
    if parameters is None:
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3,
            verbosity=1
        )
    else:
        regressor = xgb.XGBRegressor(
        n_estimators=100,
        reg_lambda=parameters['l2'],
        reg_alpha=parameters['l1'],
        eta=parameters['lr'],
        gamma=0,
        max_depth=3,
        verbosity=1
    )
    # Train the regressor
    print('\n#TRAINING THE REGRESSOR#')
    regressor.fit(X_train, y_train)
    #pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=supervised_data.columns[:-1])
    
    return regressor

def predict_live_values(regressor, df_original, X_test, y_test, days_to_predict, 
                        scaler, attr):
    #cue_confirmed = Cue(attr)
    cue = Cue(X_test)
    cue.pop()
    '''cue.pop()
    cue.push(cue_confirmed.pop())'''
    cue.push(y_test)
    predictions = []

    for element in range(days_to_predict):
        pred_scaled_value = regressor.predict(cue.getCueElements((1, X_test.shape[0]))).reshape(-1, 1)
        #pred_scaled = pd.DataFrame({'A': pred_scaled_value[0], 'B': pred_scaled_value[0]}).values
        pred = scaler.inverse_transform(pred_scaled_value)

        predictions.append(pred[0][0])
        cue.pop()
        '''cue.pop()
        value = cue_confirmed.pop()
        cue.push(value)'''
        cue.push(pred_scaled_value[0])
        

        '''pred = regressor.predict(cue.getCueElements((1, X_test.shape[0])))
        predictions.append(pred[0])
        cue.pop()
        cue.push(pred[0])'''

    return transform_live_predictions(df_original, predictions)

def predict_regressor(X_test, mode, regressor, df_original, scaler, attr, 
                      y_test=[], days_to_predict=0):
    print('#PREDICTING VALUES#\n')
    if mode == 'TRAINING' or mode == 'MODEL_TUNNING':
        # Make predictions
        #predictions = regressor.predict(X_test).reshape(-1, 1)
        predictions = regressor.predict(X_test)
        predictions = pd.DataFrame({'A': predictions.tolist(), 'B': predictions.tolist()}).values
        predictions = scaler.inverse_transform(predictions)[:, 1:]
    elif mode == 'LIVE':
        # Make predictions
        return predict_live_values(regressor, df_original, X_test, y_test, 7, scaler, attr)
    
    '''if ACTUAL_TYPE == 'HOSPITALISATIONS':
        return predictions'''
    
    return transform_predictions(df_original, predictions)
    
def transform_live_predictions(df_original, predictions):
    last = df_original.values[-1][0]
    isFirst = True
    real = 0
    real_pred = []

    for element in predictions:
        if isFirst:
            real = int(last + element)
            isFirst = False
            real_pred.append(real)
        else:
            real = int(last + element)
            real_pred.append(real)
        last = real

    return real_pred

def transform_predictions(df_original, predictions):
    # Transform dataframe into n-d array
    if ACTUAL_TYPE == 'HOSPITALISATIONS':
        original_values = df_original['Hospitalisations'].values
    else:
        original_values = df_original['Deaths'].values

    # Get the real values related with the predicted values
    original_values = original_values[(original_values.size) - (predictions.size) - 1:][:-1]
    # Sum the past value to the predicted diff to obtain the real value
    y_pred = []

    for element, element2 in zip(original_values, predictions):
        # Sum the previous real value to the diff prediction and add it 
        # to the list of predictions
        #y_pred.append(int(element[0] + element2))
        y_pred.append(int(element + element2))

    return y_pred

def print_results(y_pred, y_real, indexs, mode, county, data_type):
    df = pd.DataFrame()
    if len(y_real) > 0:
        df[data_type] = y_real
        df['Forecast'] = y_pred
    else:
        df[data_type] = y_pred
    
    df.index = indexs
    df.to_csv('{}_DF/{}.csv'.format(data_type, county))
    print("\n{} {} - {} FORECAST RESULTS: \n\n".format(county, data_type, mode), df.head(len(y_pred)))
    error = None
    if len(y_real) > 0:
        y_pred = np.array(y_pred)
        y_pred[y_pred < 0] = 0
        error = mean_squared_log_error(y_real, y_pred.tolist())
        print("\nMSLE ERROR: ", error)
    
    return error
    
def show_results(dates, y_pred, y_real, mode, county, data_type):
    # Display results 
    indexs = dates[len(dates) - len(y_pred):].strftime('%Y/%m/%d')

    error = print_results(y_pred, y_real, indexs, mode, county, data_type)
    # Create an empty figure
    fig, ax = plt.subplots(figsize=(13,5))
    # Plot a line into the figure
    line_forecast, = ax.plot(indexs, y_pred, color="green")
    line_forecast.set_label('Forecast')
    # Plot a line into the figure
    if len(y_real) > 1:
        line_real, = ax.plot(indexs, y_real, color="blue")
        line_real.set_label('Real')
    ax.legend()
    # Set figure attributes
    ax.set_title('{} - {} FORECAST OF PERIOD {}-{}'.format(county, mode, indexs[0], indexs[-1]))
    ax.set_xlabel('Dates')
    ax.set_ylabel('Deaths')
    # Display the graph
    plt.show()
    return error
    
def check_better_lag(counties_lag, county, error, lag):
    if county not in counties_lag:
        counties_lag.update({
            county : {
                'Error': error,
                'Lag': lag
            }    
        })
        return counties_lag
    value = counties_lag[county]
    if error < value['Error']:
        value['Error'] = error
        value['Lag'] = lag        
        counties_lag.update({
            county : value    
        })
    return counties_lag

def create_submission_file(counties):
    df_submission = None
    for county in counties:
        df_hosp = pd.read_csv('HOSPITALISATIONS_DF/{}.csv'.format(county), index_col=('Unnamed: 0'))
        df_hosp.insert(0, 'ID', df_hosp.index)
        df_hosp['ID'] = pd.to_datetime(df_hosp['ID'], format="%Y-%m-%d")
        df_hosp['ID'] = '{}'.format(county) + df_hosp['ID'].astype(str)
        
        
        df_deaths = pd.read_csv('DEATHS_DF/{}.csv'.format(county), index_col=('Unnamed: 0')) 
        df_deaths.insert(0, 'ID', df_deaths.index)
        df_deaths['ID'] = pd.to_datetime(df_deaths['ID'], format="%Y-%m-%d")
        df_deaths['ID'] = '{}'.format(county) + df_deaths['ID'].astype(str)
        df = pd.concat([df_hosp, df_deaths['DEATHS']], axis=1)
        if df_submission is None:
            df_submission = df
        else:
            df_submission = pd.concat([df_submission, df], axis=0)
        
    df_submission.to_csv('submissions.csv', index=False)
    
def execute_experiment(uid, lag, county, parameters=None):
    X, y, df_original, dates, scaler, attr = get_dataset(PATH_DATA, uid, lag, county, ACTUAL_TYPE)
    X_train, y_train, X_test, y_test = get_train_test(X, y, ACTUAL_MODE)
    regressor = train_regressor(X_train, y_train, parameters)
    predictions = predict_regressor(X_test, ACTUAL_MODE, regressor, 
                                    df_original, scaler, attr, y_test, DAYS_FORECAST)
    if ACTUAL_MODE == 'LIVE':

        return show_results(pd.date_range(start=dates[-1], periods=DAYS_FORECAST+1)[1:], 
                            predictions, [], ACTUAL_MODE, county, ACTUAL_TYPE)
    else:
        return show_results(dates, predictions, transform_predictions(df_original, y_test),
                            ACTUAL_MODE, county, ACTUAL_TYPE)
    
def check_parameters(model_tunning, county, error, lr_element, 
                     l2_element, l1_element):
    if county not in model_tunning:
        model_tunning.update({
            county : {
                'Error': error,
                'lr': lr_element,
                'l1': l1_element,
                'l2': l2_element
                
            }    
        })
        return model_tunning
    value = model_tunning[county]
    if error < value['Error']:
        value['Error'] = error 
        value['lr'] = lr_element
        value['l1'] = l1_element
        value['l2'] = l2_element
        model_tunning.update({
            county : value    
        })
    return model_tunning