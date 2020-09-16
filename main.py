from utils import *
import json

#PATH_DATA = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
PATH_DATA = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
CONFIRMED_PATH_DATA = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"

MODES = ['TRAINING', 'LIVE', 'SUBMISSION', 'MODEL_TUNNING']
ACTUAL_MODE = MODES[2]
DAYS_FORECAST = 7

COUNTIES = {
    'Harris' : 84048201,
    'Fort Bend' : 84048157,
    'Montgomery' : 84048339,
    'Brazoria' : 84048039,
    'Galveston' : 84048167,
    'Liberty' : 84048291,
    'Chambers' : 84048071,
    'Austin' : 84048015   
}

TYPE = ['DEATHS', 'HOSPITALISATIONS']
ACTUAL_TYPE = TYPE[0]

if __name__ == "__main__":
    if ACTUAL_MODE == 'TRAINING':
        counties_lag = {}
        for element in COUNTIES.items():
            county = element[0]
            uid = element[1]
            for lag in range(1, 31):
                error = execute_experiment(uid, lag, county)
                counties_lag = check_better_lag(counties_lag, county, error, lag)
        
        with open('counties_lags_{}.json'.format(ACTUAL_TYPE), 'w') as fp:
            json.dump(counties_lag, fp)
            
    elif ACTUAL_MODE == 'LIVE':
        with open('counties_lags_{}.json'.format(ACTUAL_TYPE), 'r') as fp:
            counties_lag = json.load(fp)
        with open('model_tunning_parameters_{}.json'.format(ACTUAL_TYPE), 'r') as fp:
            model_tunning_parameters = json.load(fp)    
        for element in COUNTIES.items():
            county = element[0]
            uid = element[1]
            lag = counties_lag[county]['Lag']
            parameters = model_tunning_parameters[county]
            execute_experiment(uid, lag, county, None)
    
    elif ACTUAL_MODE == 'SUBMISSION':
        create_submission_file(COUNTIES)
    elif ACTUAL_MODE == 'MODEL_TUNNING':
        model_tunning_parameters = {}
        lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        l2 = range(1, 5)
        l1 = range(1, 5)        
        with open('counties_lags_{}.json'.format(ACTUAL_TYPE), 'r') as fp:
            counties_lag = json.load(fp)
        for element in COUNTIES.items():
            for lr_element in lr:
                for l2_element in l2:
                    for l1_element in l1:
                        county = element[0]
                        uid = element[1]
                        lag = counties_lag[county]['Lag']
                        parameters = {
                            'lr': lr_element,
                            'l1': l1_element,
                            'l2': l2_element
                        }
                        error = execute_experiment(uid, lag, county, parameters)
                        model_tunning_parameters = check_parameters(model_tunning_parameters, 
                                                                    county, error, lr_element, 
                                                                    l2_element, l1_element)
        with open('model_tunning_parameters_{}.json'.format(ACTUAL_TYPE), 'w') as fp:
            json.dump(model_tunning_parameters, fp)
        
             
        