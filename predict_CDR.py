#!/usr/bin/env python
# coding: utf-8

"""
predict_CDR.py - load hurdle model trained on OSHA data and make predictions for CDR chemicals + workplaces
author - Jeffrey Minucci
"""

from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
import sqlite3
import pymc3 as pm
import numpy as np
import pandas as pd
import zipfile
import datetime
import os
import scipy
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az
from theano import tensor as tt
from theano import shared
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from plotting_functions import *

import logging
level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('output/predict_CDR.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

logging.info('\nBeginning load_hurdle_model.py script.\n')

# Load data files
sector_subsector_indexes_df = pd.read_csv('output/hurdle_model_sector_subsector_domain.csv')
osha_train = pd.read_csv('output/osha_processed_training.csv')
osha_test = pd.read_csv('output/osha_processed_test.csv')
osha_full = pd.read_csv('output/osha_processed_full.csv')


# Define column names for our opera predictors and interaction terms
opera_cols = ['Intercept','logp_pred','bp_pred','loghl_pred','rt_pred','logoh_pred','logkoc_pred']
opera_cols_full = ['intercept','log octanol-water partition coefficient','boiling point', "log henry's law constant",
                 'HPLC retention time', 'log OH rate constant', 'log soil adsorption coefficient']
opera_cols_ints_names = ['logp x bp', 'logp x loghl', 'logp x rt', 'logp x logoh', 'logp x logkoc', 'bp x loghl', 'bp x rt',
                         'bp x logoh', 'bp x logkoc', 'loghl x rt', 'loghl x logoh', 'loghl x logkoc', 'rt x logoh', 'rt x logkoc', 'logoh x logkoc']


# ### Define and load model part 1

sector_count = len(sector_subsector_indexes_df.index_s.unique())
sector_subsector_count = len(sector_subsector_indexes_df.index_ss.unique())

preds = osha_train[opera_cols+opera_cols_ints_names]
npreds = len(opera_cols+opera_cols_ints_names)

detected_shared = shared(osha_train.detected.values)
preds_shared = shared(preds.values)
index_s_shared = shared(osha_train.index_s.values)
index_ss_shared = shared(osha_train.index_ss.values)

# define our Bayesian hierarchical logistic model in pymc3
with pm.Model() as osha_part_1:
    
    global_mu = pm.StudentT('global_mu', nu=5, mu=0, sigma=10)
    global_lam = pm.HalfCauchy('global_lam',beta=25)
    
    sector_int_mu = pm.StudentT("sector_int_mu",nu = 5, mu = global_mu, sigma=global_lam, shape=sector_count)

    sigma_subsector_int = pm.HalfCauchy('sigma_subsector_int',beta=25)
    
    subsector_int = pm.StudentT("subsector_int", 1, mu=sector_int_mu[sector_subsector_indexes_df['index_s']],
                          sigma=sigma_subsector_int, shape=sector_subsector_count)
    beta = pm.StudentT('beta',nu=5,mu=0,sigma=2.5, shape=(npreds))  # recommended by stan wiki, Gelman et al 2008
    μ = subsector_int[index_ss_shared] + tt.dot(preds_shared,beta)
    
    θ = pm.Deterministic('θ',pm.math.sigmoid(μ))
    
    like = pm.Bernoulli(
        'likelihood',
        p = θ,
        observed=detected_shared
    )

# Load pre-fit model part 1
logging.info("Loading previously fit Bayesian hierarchical model - Part 1 object.")
model_part_1_prefit_file = 'output/model_part1_prefit.file'
fit1 = pickle.load(open(model_part_1_prefit_file,'rb'))


# ### Define and load model part 2

osha_train_detects = osha_train[osha_train['detected'] == 1].copy()
osha_train_detects.loc[:,'log_mgm3'] = np.log10(osha_train_detects['conc_mgm3'].copy())
osha_full['log_mgm3'] = np.log10(osha_full['conc_mgm3'].copy())

# Define part 2 of the model (concentration)
preds_pt2 = osha_train_detects[opera_cols+opera_cols_ints_names].copy()
npreds = len(opera_cols+opera_cols_ints_names)

conc_shared = shared(osha_train_detects.log_mgm3.values)
preds_shared_pt2 = shared(preds_pt2.values)
index_s_shared_pt2 = shared(osha_train_detects.index_s.values)
index_ss_shared_pt2 = shared(osha_train_detects.index_ss.values)

with pm.Model() as osha_part_2:
    
    global_mu = pm.StudentT('global_mu', nu=5, mu=0, sigma=10)
    global_lam = pm.HalfCauchy('global_lam',beta=25)
    
    sector_int_mu = pm.StudentT("sector_int_mu",nu = 5, mu = global_mu, sigma=global_lam, shape=sector_count)

    sigma_subsector_int = pm.HalfCauchy('sigma_subsector_int',beta=25)
    
    subsector_int = pm.StudentT("subsector_int", 1, mu=sector_int_mu[sector_subsector_indexes_df['index_s']],
                          sigma=sigma_subsector_int, shape=sector_subsector_count)
    beta = pm.StudentT('beta',nu=5,mu=0,sigma=2.5, shape=(npreds))  # recommended by stan wiki, Gelman et al 2008
    μ = subsector_int[index_ss_shared_pt2] + tt.dot(preds_shared_pt2,beta)
    sampling_error = pm.HalfCauchy('sampling_error',beta=25)
    
    like = pm.Normal(
        'likelihood',
        mu = μ,
        sigma = sampling_error,
        observed=conc_shared
    )

# Load pre-fit model part 2

logging.info("Loading previously fit Bayesian hierarchical model - Part 2 object.")
model_part_2_prefit_file = 'output/model_part2_prefit.file'
fit2 = pickle.load(open(model_part_2_prefit_file,'rb'))


# ### Format and summarize the CDR data

cdr = pd.read_csv('data/cdr/cdr_data_for_predictions.csv', dtype={'naics_code':str},
                  na_values = ["", " ","NaN", "nan", "NA", "na", "Na"])
osha_domain = pd.read_csv('output/osha_chem_workplace_domain.csv')
# collapse CDR data into unique chemical+sector+subsector combinations, count # observations for each
cdr = cdr.groupby(['dtxsid', 'casrn_ord','preferred_name', 'sector_name', 'subsector_name', 'index_s', 'index_ss'] +
                  opera_cols+opera_cols_ints_names).size().reset_index(name='reports')
logging.info("Number of unique chemical+subsector combinations in the CDR: {}".format(len(cdr)))
cdr_chems = pd.Series(cdr['preferred_name'].unique(), name = 'preferred_name')
logging.info("Number of unique chemical+subsector combinations in the OSHA dataset: {}".format(len(osha_domain)))
logging.info("Number of substances in the CDR: {}".format(len(cdr_chems)))
osha_chems = pd.Series(osha_domain['preferred_name'].unique(), name = 'preferred_name')
logging.info("Number of substances in the OSHA dataset: {}".format(len(osha_chems)))
new_chems = pd.Series(cdr_chems[~cdr_chems.isin(osha_chems)], name = 'preferred_name')
duplicate_chems = cdr_chems[cdr_chems.isin(osha_chems)]
logging.info("Number of NEW substances in the CDR: {}".format(len(new_chems)))
logging.info("Number of duplicate substances in the CDR: {}".format(len(duplicate_chems)))

cdr_combos = pd.Series(cdr['preferred_name'] + '_' + cdr['subsector_name']).unique()
osha_combos = pd.Series(osha_domain['preferred_name'] + '_' + osha_domain['subsector_name']).unique()
new_combos = cdr_combos[~pd.Series(cdr_combos).isin(osha_combos)]  
n_new_combos = len(new_combos)
logging.info("Number of NEW chemical+subsector combinations in the CDR: {}".format(n_new_combos))

# Figure out which new substances in the CDR are outside the range of previously seen physicochemical
# properties (extrapolation)
new_chem_properties = pd.merge(new_chems, cdr[['preferred_name']+opera_cols + opera_cols_ints_names])
cdr_properties = cdr[['preferred_name']+opera_cols + opera_cols_ints_names]
osha_chem_properties = pd.merge(osha_chems, osha_full[['preferred_name']+opera_cols + opera_cols_ints_names])
osha_property_domain = osha_chem_properties.agg([min, max]).drop(columns=['preferred_name'])
extrapolation_mask_max = cdr_properties.drop(columns=['preferred_name']) > osha_property_domain.loc['max']
extrapolation_mask_min = cdr_properties.drop(columns=['preferred_name']) < osha_property_domain.loc['min']
extrapolation_mask = extrapolation_mask_max.any(axis=1) | extrapolation_mask_min.any(axis=1)
extrapolation_chems = cdr_properties[extrapolation_mask]
extrapolation_new_chems = pd.Series(extrapolation_chems.preferred_name.unique())[pd.Series(extrapolation_chems.preferred_name.unique()).isin(new_chems)]
logging.info("Number of new chemicals that are outside the OSHA physicochemical property domain: {}".format(len(extrapolation_new_chems)))

# create dataset for predictions
cdr['extrapolation'] = extrapolation_mask
cdr['new_chemical'] = ~cdr['preferred_name'].isin(osha_chems)  #~pd.Series(cdr_combos).isin(osha_combos)
cdr = cdr[~pd.Series(cdr_combos).isin(osha_combos)]  # keep only new substance X workplace combos
cdr_no_extrapolation = cdr[cdr['extrapolation'] == False]
chems_no_extrapolation = pd.Series(cdr_no_extrapolation.preferred_name.unique())[~pd.Series(cdr_no_extrapolation.preferred_name.unique()).isin(osha_chems)]
logging.info("Number of NEW chemical+subsector combinations in the CDR after dropping extrapolation substances: {}".format(len(cdr_no_extrapolation)))
logging.info("Number of NEW substances in the CDR after dropping extrapolation substances: {}".format(len(chems_no_extrapolation)))


# ### Generate predictions for CDR dataset

logging.info('Generating CDR predictions...')
logging.info("CDR predictions - Part 1")
preds_test = cdr[opera_cols+opera_cols_ints_names]
detected_shared.set_value(cdr.Intercept.values) # has no effect
preds_shared.set_value(preds_test.values)
index_s_shared.set_value(cdr.index_s.values)
index_ss_shared.set_value(cdr.index_ss.values)
np.random.seed(5365)
sam = fit1.sample(5000)
ppc_cdr = pm.sample_posterior_predictive(sam, samples=1000, random_seed=4858, model=osha_part_1)
avg_predictions_cdr = np.where(ppc_cdr['likelihood'].mean(axis=0) >= 0.5, 1, 0)
avg_probs_cdr = ppc_cdr['likelihood'].mean(axis=0).round(3)
cdr['detection_prob'] = avg_probs_cdr
cdr['detected_pred'] = avg_predictions_cdr

logging.info("CDR predictions - Part 2")
dummy_conc = np.ones(len(cdr))
conc_shared.set_value(dummy_conc)  # has no effect
preds_shared_pt2.set_value(preds_test.values)
index_s_shared_pt2.set_value(cdr.index_s.values)
index_ss_shared_pt2.set_value(cdr.index_ss.values)
sam = fit2.sample(5000)
ppc_p2_cdr = pm.sample_posterior_predictive(sam, samples=1000, random_seed=12699, model=osha_part_2)
avg_predictions_pt2_cdr = ppc_p2_cdr['likelihood'].mean(axis=0).round(3)
median_predictions_pt2_cdr = np.quantile(ppc_p2_cdr['likelihood'], axis=0, q=0.5).round(3)
u95_predictions_pt2_cdr = np.quantile(ppc_p2_cdr['likelihood'], axis=0, q=0.975).round(3)
l95_predictions_pt2_cdr = np.quantile(ppc_p2_cdr['likelihood'], axis=0, q=0.25).round(3)

cdr['log_mgm3_pred_mean'] = avg_predictions_pt2_cdr
cdr['log_mgm3_pred_50th'] = median_predictions_pt2_cdr
cdr['log_mgm3_pred_97.5th'] = u95_predictions_pt2_cdr
cdr['log_mgm3_pred_2.5th'] = l95_predictions_pt2_cdr

# ### Save CDR predictions
logging.info("Saving CDR predictions")
cdr_final = cdr[cdr['extrapolation'] == False]  # keep only substances with physicochemical predictors within model domain
cdr_final[['dtxsid', 'casrn_ord','preferred_name', 'sector_name', 'subsector_name', 'detection_prob',
                'log_mgm3_pred_mean', 'log_mgm3_pred_2.5th', 'log_mgm3_pred_50th', 'log_mgm3_pred_97.5th', 'extrapolation', 'new_chemical']].to_csv('output/TableS3.csv',index=False)
cdr[['dtxsid', 'casrn_ord','preferred_name', 'sector_name', 'subsector_name', 'detection_prob',
                'log_mgm3_pred_mean','log_mgm3_pred_2.5th', 'log_mgm3_pred_50th', 'log_mgm3_pred_97.5th', 'extrapolation', 'new_chemical']].to_csv('output/CDR_predictions_with_extrapolation.csv',index=False)
logging.info('Model predictions for chemicals within OSHA model domain: {}'.format('output/TableS3.csv'))
logging.info('Model predictions for ALL chemicals: {}'.format('output/CDR_predictions_with_extrapolation.csv'))

cdr_p1_samples = pd.DataFrame(ppc_cdr['likelihood'].transpose())
cdr_p1_samples['preferred_name'] = cdr['preferred_name'].reset_index(drop=True)
cdr_p1_samples['sector_name'] = cdr['sector_name'].reset_index(drop=True)
cdr_p1_samples['subsector_name'] = cdr['subsector_name'].reset_index(drop=True)
cdr_p1_samples['extrapolation'] = cdr['extrapolation'].reset_index(drop=True)
cdr_p1_samples['combo'] = cdr_p1_samples['preferred_name'] + cdr_p1_samples['subsector_name'] 
cdr_p1_samples = cdr_p1_samples[cdr_p1_samples['extrapolation'] == False]

cdr_p2_samples = pd.DataFrame(ppc_p2_cdr['likelihood'].transpose())
cdr_p2_samples['preferred_name'] = cdr['preferred_name'].reset_index(drop=True)
cdr_p2_samples['sector_name'] = cdr['sector_name'].reset_index(drop=True)
cdr_p2_samples['subsector_name'] = cdr['subsector_name'].reset_index(drop=True)
cdr_p2_samples['extrapolation'] = cdr['extrapolation'].reset_index(drop=True)
cdr_p2_samples['combo'] = cdr_p2_samples['preferred_name'] + cdr_p2_samples['subsector_name'] 
cdr_p2_samples = cdr_p2_samples[cdr_p2_samples['extrapolation'] == False]


# ###Plot CDR predictions
colors = {True: 'C1', False: 'C0'}
plt.scatter(cdr['log_mgm3_pred_mean'],cdr['detection_prob'], c=cdr['extrapolation'].map(colors))
plt.xlabel('Predicted air conc. (log mg/m3)')
plt.ylabel('Predicted detection probability')
plt.savefig('output/figures/figS3_with_extrapolation.png')
plt.close()

figS3_path = 'output/figures/figS3.png'
plt.scatter(cdr_final['log_mgm3_pred_mean'],cdr_final['detection_prob'], s=10)
plt.xlabel('Predicted air conc. (log mg/m3)')
plt.ylabel('Predicted detection probability')
plt.savefig(figS3_path)
plt.close()

plt.hist(cdr_final['log_mgm3_pred_mean'], bins=20)
plt.xlabel('Predicted air conc. (log mg/m3)')
plt.savefig('output/figures/cdr2.png')
plt.close()

plt.hist(cdr_final['detection_prob'], bins=20)
plt.xlabel('Predicted detection probability')
plt.savefig('output/figures/cdr3.png')
plt.close()

logging.info('CDR predictions scatterplot saved to: {}'.format(fig6_path))
logging.info('\n predict_CDR.py completed')
