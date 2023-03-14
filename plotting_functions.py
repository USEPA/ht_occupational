#!/usr/bin/env python
# coding: utf-8

"""
plotting_functions.py - helper functions for plotting
author - Jeffrey Minucci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
import logging

# logging settings
level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('output/plotting.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

def plot_substance_predictions(detect_probs, concs, save_path = None, min_N = None, title = '', color1='C0', color2='C1', sortbyair=False, textout=False):
    predictions_c = concs.copy()
    predictions_d = detect_probs.copy()
    median = [None,None]
    size = [None,None]
    error_95_upper = [None,None]
    error_95_lower = [None,None]
    order = None
    for i, predictions in enumerate([predictions_d, predictions_c]):
        if min_N:
            predictions = predictions[predictions['preferred_name'].isin(predictions.preferred_name.value_counts().index[predictions.preferred_name.value_counts()>min_N])]
        conc = predictions.groupby('preferred_name',axis=0).mean()
        med = conc.quantile(axis=1,q=0.5)
        pi_95h = conc.quantile(axis=1,q=0.975)
        pi_95l = conc.quantile(axis=1,q=0.025)
        if i == 0:
            order = med.sort_values(ascending=True)  # order for plotting
            if sortbyair:
                if min_N:
                    predictions = predictions_c[predictions_c['preferred_name'].isin(predictions_c.preferred_name.value_counts().index[predictions_c.preferred_name.value_counts()>min_N])]
                    order = predictions.groupby('preferred_name',axis=0).mean().quantile(axis=1,q=0.5).sort_values(ascending=True)
        size = (11,0.35*len(order))
        median[i] = med
        error_95_upper[i] = (pi_95h-med).loc[order.index]
        error_95_lower[i] = (med-pi_95l).loc[order.index]
        if textout:
            print(med.sort_values())
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=size)
    ylabels = order.index.tolist()
    ax1.errorbar(x=median[0].loc[order.index], y=ylabels,
                 xerr=(error_95_lower[0], error_95_upper[0]),
                fmt='o', color=color1)
    fig.suptitle('{}'.format(title), size = 18)
    ax1.set_xlabel('Probability of detection', size=14)
    ax1.set_ylim(-1,len(order))
    ax2.set_xlabel('Air concentration (log mg/m3)', size=14)  # we already handled the x-label with ax1
    ax2.errorbar(x=median[i].loc[order.index], y=ylabels,
                 xerr=(error_95_lower[i], error_95_upper[i]),
                fmt='o', color=color2)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_predictions_combined(predictions_detect_in, predictions_conc_in, save_path = None, min_N = None, subsector=False, kingdom=None, superclass=None, _class=None, title = '', color='C0'):
    predictions_c = predictions_conc_in.copy()
    predictions_d = predictions_detect_in.copy()
    median = [None,None]
    size = [None,None]
    error_95_upper = [None,None]
    error_95_lower = [None,None]
    order = None
    for i, predictions in enumerate([predictions_d, predictions_c]):
        if kingdom:
            predictions = predictions[predictions['kingdom'] == kingdom]
        if superclass:
            predictions = predictions[predictions['superclass'] == superclass]
        if _class:
            predictions = predictions[predictions['class'] == _class]
        if not subsector:
            if min_N:
                predictions = predictions[predictions['sector_name'].isin(predictions.sector_name.value_counts().index[predictions.sector_name.value_counts()>=min_N])]
            conc = predictions.groupby('sector_name',axis=0).mean()
            med = conc.quantile(axis=1,q=0.5)
            pi_95h = conc.quantile(axis=1,q=0.975)
            pi_95l = conc.quantile(axis=1,q=0.025)
            if i == 0:
                order = med.sort_values(ascending=True)  # order for plotting
            size = (11,0.35*len(order))
        else:
            if min_N:
                predictions = predictions[predictions['subsector_name'].isin(predictions.subsector_name.value_counts().index[predictions.subsector_name.value_counts()>=min_N])]
            conc = predictions.groupby('subsector_name',axis=0).mean()
            med = conc.quantile(axis=1,q=0.5)
            pi_95h = conc.quantile(axis=1,q=0.975)
            pi_95l = conc.quantile(axis=1,q=0.025)
            if i == 0:
                order = med.sort_values(ascending=True) # order for plotting
            size=(11,0.20*len(order))
        median[i] = med
        error_95_upper[i] = (pi_95h-med).loc[order.index]
        error_95_lower[i] = (med-pi_95l).loc[order.index]
        logging.info('\nSector/subsector values: ')
        logging.info(med.sort_values())
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=size)
    ylabels = order.index.tolist()
    ylabels[5] = 'Admin., Support, Waste Manage. and Remediation Services'
    ax1.errorbar(x=median[0].loc[order.index], y=ylabels,
                 xerr=(error_95_lower[0], error_95_upper[0]),
                fmt='o', color=color)
    fig.suptitle('{}'.format(title), size = 18)
    ax1.set_xlabel('Probability of detection', size=14)
    ax1.set_ylim(-1,len(order))
    ax2.set_xlabel('Air concentration (log mg/m3)', size=14)  # we already handled the x-label with ax1
    ax2.errorbar(x=median[i].loc[order.index], y=ylabels,
                 xerr=(error_95_lower[i], error_95_upper[i]),
                fmt='o', color='C1')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_substance_predictions_cdr(detect_probs, concs, save_path = None, min_N = None, title = '', color1='C0', color2='C1', sortbyair=False, textout=False):
    predictions_c = concs.copy()
    predictions_d = detect_probs.copy()
    median = [None,None]
    size = [None,None]
    error_95_upper = [None,None]
    error_95_lower = [None,None]
    order = None
    for i, predictions in enumerate([predictions_d, predictions_c]):
        if min_N:
            predictions = predictions[predictions['preferred_name'].isin(predictions.preferred_name.value_counts().index[predictions.preferred_name.value_counts()>min_N])]
        predictions['med'] = predictions.quantile(axis=1,q=0.5).copy()
        predictions['pi_95h'] = predictions.quantile(axis=1,q=0.975).copy()
        predictions['pi_95l'] = predictions.quantile(axis=1,q=0.025).copy()
        conc = predictions.groupby('preferred_name',axis=0).mean()
        print(conc)
        med = conc['med']
        pi_95h = conc['pi_95h']
        pi_95l = conc['pi_95l']
        if i == 0:
            order = med.sort_values(ascending=True)  # order for plotting
            if sortbyair:
                if min_N:
                    predictions = predictions_c[predictions_c['preferred_name'].isin(predictions_c.preferred_name.value_counts().index[predictions_c.preferred_name.value_counts()>min_N])]
                    order = predictions.groupby('preferred_name',axis=0).mean()['med'].sort_values(ascending=True)
        size = (11,0.35*len(order))
        median[i] = med
        error_95_upper[i] = (pi_95h-med).loc[order.index]
        error_95_lower[i] = (med-pi_95l).loc[order.index]
        if textout:
            print(med.sort_values())
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=size)
    ylabels = order.index.tolist()
    ax1.errorbar(x=median[0].loc[order.index], y=ylabels,
                 xerr=(error_95_lower[0], error_95_upper[0]),
                fmt='o', color=color1)
    fig.suptitle('{}'.format(title), size = 18)
    ax1.set_xlabel('Probability of detection', size=14)
    ax1.set_ylim(-1,len(order))
    ax2.set_xlabel('Air concentration (log mg/m3)', size=14)  # we already handled the x-label with ax1
    ax2.errorbar(x=median[i].loc[order.index], y=ylabels,
                 xerr=(error_95_lower[i], error_95_upper[i]),
                fmt='o', color=color2)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
