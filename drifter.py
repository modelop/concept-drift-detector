import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import binom_test

#modelop.init
def begin():
    global train, actual_values
    train = pd.read_csv('training_data.csv')
    actual_values = train.loan_status
    pass

#modelop.score
def action(datum):
    yield datum

#modelop.metrics
def metrics(data):
    predicted_values = data.prediction
    empirical_probability = actual_values.sum()/train.shape[0]
    pvalue = binom_test(predicted_values.sum(), 
                        data.shape[0], 
                        empirical_probability)
    
    yield {"pvalue": pvalue}
