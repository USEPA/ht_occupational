#!/usr/bin/env python
# coding: utf-8

"""
plotting.py - load hurdle model results and predictions and create plots
author - Jeffrey Minucci
"""	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
import logging
from plotting_functions import *

# logging settings
level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('output/plotting.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

logging.info('\nBeginning plotting.py script.\n')

# Load model predictions
indexed_osha_test = pd.read_csv('output/osha_test_predictions.csv', dtype={'log_mgm3':float}, na_values = 'ND')
indexed_osha_train = pd.read_csv('output/osha_train_predictions.csv', dtype={'log_mgm3':float}, na_values = 'ND')
indexed_osha_full = pd.concat([indexed_osha_test, indexed_osha_train])

# Load full probabilistic sampling
samples_out_path = 'output/probabilistic_samples/'
test_p1_samples = pd.read_csv(samples_out_path + 'test_p1_samples.csv')
test_p2_samples = pd.read_csv(samples_out_path + 'test_p2_samples.csv')
train_p1_samples = pd.read_csv(samples_out_path + 'train_p1_samples.csv')
train_p2_samples = pd.read_csv(samples_out_path + 'train_p2_samples.csv')
full_p1_samples = pd.read_csv(samples_out_path + 'full_p1_samples.csv')
full_p2_samples = pd.read_csv(samples_out_path + 'full_p2_samples.csv')


#print(indexed_osha_test.dtypes)
                                                                    

# Fig 2 - Plot actual vs predicted for actual positives
actual_tp = indexed_osha_test['log_mgm3'][(indexed_osha_test['pred_detected'] == 1) & (indexed_osha_test['detected'] == 1)]
predicted_tp = indexed_osha_test['pred_log_mgm3'][(indexed_osha_test['pred_detected'] == 1) & (indexed_osha_test['detected'] == 1)]
actual_fn = indexed_osha_test['log_mgm3'][(indexed_osha_test['pred_detected'] == 0) & (indexed_osha_test['detected'] == 1)]
predicted_fn = indexed_osha_test['pred_log_mgm3'][(indexed_osha_test['pred_detected'] == 0) & (indexed_osha_test['detected'] == 1)]
rmse = np.sqrt(mean_squared_error(actual_tp, predicted_tp))
plt.scatter(predicted_fn, actual_fn, s=10, color="red", alpha=1, label='Predicted non-detect')
plt.scatter(predicted_tp, actual_tp, s=10, alpha=1, label='Predicted detect')
plt.plot([-20, 10], [-20, 10], color='black')
plt.xlim([-5,6])
plt.ylim([-5,6])
plt.title("Test set")
plt.xlabel('Predicted air concentration (log mg/m3)')
plt.ylabel('Actual air concentration (log mg/m3)')
plt.text(-4.5, 2.5, 'RMSE = {:.2f}'.format(rmse))
plt.legend()
fig2_path = 'output/figures/fig2.png'
plt.savefig(fig2_path)
logging.info('Figure 2 - actual vs predicted concentration saved to: {} \n'.format(fig2_path))

# For fig 2 left table
logging.info('RMSE for true positives: {:.2f}'.format(rmse))
falsepos = np.where((indexed_osha_test['detected']==0) & (indexed_osha_test['pred_detected']==1), 1, 0).sum() / len(indexed_osha_test)
falseneg = np.where((indexed_osha_test['detected']==1) & (indexed_osha_test['pred_detected']==0), 1, 0).sum() / len(indexed_osha_test)
trueneg = np.where((indexed_osha_test['detected']==0) & (indexed_osha_test['pred_detected']==0), 1, 0).sum() / len(indexed_osha_test)
truepos = np.where((indexed_osha_test['detected']==1) & (indexed_osha_test['pred_detected']==1), 1, 0).sum() / len(indexed_osha_test)
truepos_error_1_order = np.where((indexed_osha_test['detected']==1) & (indexed_osha_test['pred_detected']==1) & 
                                   (abs(indexed_osha_test['log_mgm3'] - indexed_osha_test['pred_log_mgm3']) < 1), 1, 0).sum() / len(indexed_osha_test)
n_truepos = np.where((indexed_osha_test['detected']==1) & (indexed_osha_test['pred_detected']==1), 1, 0).sum()
percent_of_tp_error_1_order = np.where((indexed_osha_test['detected']==1) & (indexed_osha_test['pred_detected']==1) & 
                                   (abs(indexed_osha_test['log_mgm3'] - indexed_osha_test['pred_log_mgm3']) < 1), 1, 0).sum() / n_truepos
logging.info('True negative %: {:.1f}%'.format(trueneg*100))
logging.info('False positive %: {:.1f}%'.format(falsepos*100))
logging.info('False negative %: {:.1f}%'.format(falseneg*100))
logging.info('True positive %: {:.1f}%'.format(truepos*100))
logging.info('True positive AND within 1 order of magnitude concentration: {:.1f}%'.format(truepos_error_1_order*100))
logging.info('% of true positives within 1 order of magnitude concentration: {:.1f}%'.format(percent_of_tp_error_1_order*100))
logging.info('{:.2f}% of the time we correctly predict the air concentration within 1 order of magnitude \n'.format((trueneg + truepos_error_1_order)*100))

true_pos_rate = truepos/(truepos + falseneg)
true_neg_rate = trueneg/(trueneg + falsepos)
logging.info("True positive rate: {:.1f}".format(true_pos_rate*100))
logging.info("True negative rate: {:.1f}".format(true_neg_rate*100))

# For fig 4 
fig4_path = 'output/figures/fig4.png'
plot_predictions_combined(full_p1_samples, full_p2_samples, min_N = 5, save_path=fig4_path)
logging.info('\nFigure 4 - model predictions by NAICS sector - saved to: {} \n'.format(fig4_path))


# For fig S2
figS2_path = 'output/figures/figS2.png'
plot_substance_predictions(test_p1_samples, test_p2_samples, save_path=figS2_path, min_N=5, sortbyair=True, textout=False)
logging.info('\nFigure S2 - model predictions by test set substances - saved to: {} \n'.format(figS2_path))


# For fig S1
figS1_path = 'output/figures/figS1.png'
plot_predictions_combined(full_p1_samples, full_p2_samples, subsector=True, save_path=figS1_path, min_N=5)
plt.savefig(figS1_path)
logging.info('\nFigure S1 - model predictions by NAICS subsector - saved to: {} \n'.format(figS1_path))


logging.info('\nplotting.py completed')
