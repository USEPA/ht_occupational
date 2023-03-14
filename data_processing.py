#!/usr/bin/env python
# coding: utf-8

"""
data_processing.py - load the OSHA dataset, standardize NAICS codes and units, assign properties
author - Jeffrey Minucci
"""	

# ### Loading data

from sklearn.preprocessing import StandardScaler, scale, PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from numba import jit
import datetime
import arviz as az
import os
import scipy
import pickle
from functions_utils.naics_convert import naics_to_2017
from functions_utils.naics_definitions import industries, get_naics_subcode, get_naics_industry, get_naics_definition
import logging

# Set up logging to file
level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('output/data_processing.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

logging.info('Beginning data_processing.py script.\n')

# Load the raw OSHA monitoring data
logging.info('Loading raw OSHA data 1984-2018...')
if not os.path.exists('data/osha/osha_monitoring_1984_2018.csv'):
    logging.info('Unzipping osha monitoring data...\n')
    with zipfile.ZipFile('data/osha/osha_monitoring_1984_2018.zip','r') as zfile:
        zfile.extractall('data/osha')
osha = pd.read_csv('data/osha/osha_monitoring_1984_2018.csv', dtype={'naics_code':pd.Int32Dtype(), 'sic_code':pd.Int32Dtype()},
                  na_values = ["", " ","NaN", "nan", "NA", "na", "Na"])
osha.naics_code = osha.naics_code.astype(str)
osha.sic_code = osha.sic_code.astype(str)

# Load workplace code crosswalk files
logging.info('Loading SIC and NAICS crosswalk files...')
sic_naics_2002 = pd.read_csv('data/naics/1987_SIC_to_2002_NAICS_one_to_one.csv', 
                            dtype={'SIC': str, '2002 NAICS': str},
                             na_values = ["", " ", "0", "AUX", "Aux", "NaN", "nan"])

# ### record the original 2017 NAICS sector/subsector domain before data cleaning/pre-processing
# ### But after removing non-air samples

osha_pre_processing = osha.copy()
osha_pre_processing = osha_pre_processing [osha_pre_processing ['sample_type'].isin(['P','A'])]
naics_missing = np.where(pd.to_numeric(osha_pre_processing ['naics_code'], errors='coerce') > 1000,0,1)
sic = osha_pre_processing ['sic_code'].astype(int, errors='ignore').astype(str)
naics_from_sic = pd.DataFrame(sic).merge(sic_naics_2002[['SIC', '2002 NAICS']].dropna(), left_on='sic_code',
                         right_on='SIC', how='left')
osha_pre_processing ['naics_from_sic'] = np.where(naics_missing, naics_from_sic['2002 NAICS'], osha_pre_processing ['naics_code']).astype(str)
to_2017 = pd.Series(naics_to_2017(osha_pre_processing ['naics_from_sic']))
osha_pre_processing ['naics_unified'] = to_2017
osha_pre_processing['sector_name'] = [get_naics_definition(x,2) for x in osha_pre_processing['naics_unified']]
osha_pre_processing['subsector_name'] = [get_naics_definition(x,3)  for x in osha_pre_processing['naics_unified']]
original_workplace_domain = osha_pre_processing.groupby(['sector_name','subsector_name']).size().reset_index()

# Load the DSSTox chemical data and OPERA predictions for chemicals in the OSHA dataset
logging.info('Loading DSSTox chemical data and OPERA predictions for chemicals in the OSHA dataset...')
chem_data = pd.read_csv('data/osha/osha_chem_properties.csv')
chem_data[chem_data.columns[8:]] = chem_data[chem_data.columns[8:]].apply(pd.to_numeric,errors='coerce')
chem_data.columns = [x.lower() for x in chem_data.columns]
ids = pd.read_csv('data/osha/chemical_metadata/osha_substance_ids.csv')
opera = pd.read_csv('data/osha/chemical_metadata/opera2_physical.csv')

# Load which substances have been randomly assigned to the test set
test_substances = pd.read_csv('data/test_substances.csv',names = ['i', 'substance']).substance

osha.drop(osha.columns[0], axis=1, inplace=True)  # drop first column of old indices
osha['imis_substance_code'] = osha['imis_substance_code'].astype('category')
osha['sample_result'] = pd.to_numeric(osha['sample_result'], errors='coerce')
osha['air_volume_sampled'] = pd.to_numeric(osha['air_volume_sampled'], errors='coerce')

# keep only samples with units of concentration, or that we can convert to concentrations
logging.info('Dropping bulk samples, blanks and invalid concentration results...')
osha = osha[osha['unit_of_measurement'].isin(['M', 'P', "%"]) | ((osha['unit_of_measurement'].isin(['X', 'Y'])) & osha['air_volume_sampled']>0)]

# keep only sample types P=personal, A=area, drop B=Bulk (samples that are of pure substances, not air monitoring)
osha = osha[osha['sample_type'].isin(['P','A'])]
osha = osha[~osha.qualifier.str.lower().str.contains("b.*k|b.*l",regex=True,na=False)]  # drop blanks and bulk samples mislabeled as P or A type

#drop rows with measurements that are NaN or Inf
osha = osha[np.isfinite(osha.sample_result)]

# initial db size
logging.info('Initial N, all air samples: {}'.format(len(osha)))
logging.info('Initial N, personal air samples: {}'.format(len(osha[osha['sample_type'] == 'P'])))
logging.info('Initial N, area air samples: {}'.format(len(osha[osha['sample_type'] == 'A'])))

# ### add average mol mass of the substances by matching with DSSTox

logging.info('Dropping substances with no DSSTox match...')
a = len(osha)
osha = osha.merge(chem_data[['input', 'average_mass', 'preferred_name', 'dtxsid', 'casrn']], left_on='substance', right_on='input')
osha = osha[osha['preferred_name'] != '-']
b = len(osha)
logging.info('{} rows before adding dsstox ID, {} after'.format(a,b))
osha = osha.reset_index()

# ### converting units to mg/m3

logging.info('Converting all units to mg/m3...')
# convert all units to mg/m3
@jit(parallel=True, forceobj=True)
def convert_mg_m3(result, units, air_vol, mol_wt):
    n = len(result)
    concentrations = np.empty(n)
    for i in range(n):
        unit = units[i]
        x = result[i]
        vol = air_vol[i]
        mw = mol_wt[i]
        if unit == 'Y': # mg
            val = x/(vol*0.001)
        elif unit == 'X': # mcg
            val = (x/1000)/(vol*0.001)
        elif unit == '%': # percent
            val = (x*10000) * mw/24.45  # based on 25C and 1 atm
        elif unit == 'P':  
            val = x * mw/24.45  # based on 25C and 1 atm
        else:
            val = x
        concentrations[i] = val
    return concentrations

osha['conc_mgm3'] = convert_mg_m3(osha['sample_result'].to_numpy(), osha['unit_of_measurement'].to_numpy(dtype=str),
                                  osha['air_volume_sampled'].to_numpy(), osha['average_mass'].to_numpy())

osha['conc_mgm3'] = np.where(osha['qualifier'].str.contains('ND',na=False), 0, osha['conc_mgm3'])  # if non-detect, set conc to 0


# ### update and assign workplace codes

# Convert SIC to NAICS codes
logging.info('Converting SIC and old NAICS codes to NAICS 2017 and assigning sector/subsector names...')
naics_missing = np.where(pd.to_numeric(osha['naics_code'], errors='coerce') > 1000,0,1)
sic = osha['sic_code'].astype(int, errors='ignore').astype(str)
naics_from_sic = pd.DataFrame(sic).merge(sic_naics_2002[['SIC', '2002 NAICS']].dropna(), left_on='sic_code',
                         right_on='SIC', how='left')
osha['naics_from_sic'] = np.where(naics_missing, naics_from_sic['2002 NAICS'], osha['naics_code']).astype(str)

# Update all NAICS codes to 2017 version
to_2017 = pd.Series(naics_to_2017(osha['naics_from_sic']))
osha['naics_unified'] = to_2017

# Assign sector and subsector names
osha['sector'] = osha['naics_unified'].apply(lambda x: get_naics_subcode(x,digits=2) if not pd.isnull(x) else x)
osha['sector_name'] = [get_naics_definition(x,2) for x in osha['naics_unified']]
osha['subsector_name'] = [get_naics_definition(x,3)  for x in osha['naics_unified']]
osha['industry_group_name'] = [get_naics_definition(x,4)  for x in osha['naics_unified']]

# Drop data where the sector or subsector couldn't be defined
x = len(osha)
osha_undefined_naics = osha[osha['sector_name'].isin(["Undefined/Multiple"]) | osha['subsector_name'].isin(["Undefined/Multiple"])].copy()
#osha_undefined_naics.to_csv('output/osha_naics_undefined.csv')
osha_assigned = osha[~osha['sector_name'].isin(["Undefined/Multiple"])]
osha_assigned = osha_assigned[~osha_assigned['subsector_name'].isin(["Undefined/Multiple"])]
osha_assigned = osha_assigned.dropna(subset=['subsector_name', 'sector_name'])
logging.info('Rows dropped for undefined NAICS sector/subsector: {} out of {}'.format(x-len(osha_assigned),x))


# #### remove high extreme outliers + undefined concentrations

# this function scans for extreme high outliers using untransformed mg/m3 data which includes non-detects
#  in order to consider the entire distribution. Using log-transformed values, we would lose the non-detects.
def outlier_scan(df, z_cutoff):
    output = df.copy()
    chem_list = df['preferred_name'].unique().copy()
    for i, chem in enumerate(chem_list):
        vals = df[df['preferred_name'] == chem].conc_mgm3.copy()
        if len(vals) < 1:
            continue
        zs = scipy.stats.zscore(vals)
        if np.any(zs > z_cutoff):
            cutoff = vals[zs>z_cutoff].min()
            output = output.drop(output[(output['preferred_name']==chem) & (output['conc_mgm3'] >= cutoff)].index).copy()
    return output


osha_raw = osha_assigned[['date_sampled','inspection_number', 'establishment_name', 'preferred_name','naics_unified', 'sector_name', 'subsector_name',
             'industry_group_name', 'conc_mgm3', 'sample_type', 'qualifier']]  # keep only cols we need
osha_raw.loc[:,'date_sampled'] = osha_raw['date_sampled'].str.slice(0,4).astype(int).copy()
x = len(osha_raw)
osha_raw = osha_raw.dropna(subset=['conc_mgm3'])
logging.info('rows dropped for undefined sample concentration: {} out of {}'.format(x-len(osha_raw),x))
osha_raw.drop(osha_raw[(osha_raw.inspection_number == '799890') & osha_raw.preferred_name.isin(['2-Chlorotoluene', '4-Chlorotoluene'])].index)  # drop 2 probable errors
osha_raw['detected'] = np.where(osha_raw['conc_mgm3'] > 0, 1, 0)

# ### Aggregate by inspection number + do extreme outlier scan

logging.info('Aggregating observations by inspection number and taking max...')
n_per_inspection = osha_raw.groupby(['inspection_number', 'establishment_name', 'preferred_name','naics_unified', 'sector_name',
                     'subsector_name', 'industry_group_name', 'sample_type'], as_index=False).size()['size']
logging.info('Max, median and min number of obs per inspection: {}, {} and {}'.format(n_per_inspection.max(), n_per_inspection.median(), n_per_inspection.min()))
osha_agg = osha_raw.groupby(['inspection_number', 'establishment_name', 'preferred_name','naics_unified', 'sector_name',
                     'subsector_name', 'industry_group_name', 'sample_type'], as_index=False).max(numeric_only=True)
logging.info('{} rows before aggregating by inspection, {} rows after.'.format(len(osha_raw), len(osha_agg)))
logging.info('Detecting and removing extreme outliers (z score > 4)...')
osha = outlier_scan(osha_agg, 4)  # remove extreme outliers
logging.info('outliers removed: {} out of {}'.format(len(osha_agg)- len(osha), len(osha_agg)))


# ### Filter out subsectors that don't have sufficient data

x = len(osha)
less_than_10_detected = osha.groupby('subsector_name')['detected'].sum().index[osha.groupby('subsector_name')['detected'].sum() < 10]
osha_less_than_10_detects = osha[osha.subsector_name.isin(less_than_10_detected)].groupby(['sector_name','subsector_name']).agg({'detected':['size','sum']})
osha_less_than_10_detects.to_csv('output/subsectors_below_10_detects.csv')
osha = osha[~osha.subsector_name.isin(less_than_10_detected)]
#logging.info('ubsectors dropped for <10 detects:\n{}\n'.format(less_than_10_detected))
logging.info('Rows dropped for subsector insufficient data (less than 10 observations): {} out of {}'.format(x-len(osha),x))


# ### Assign OPERA predictions

logging.info('Assigning OPERA predictions and calculating interaction terms...')
opera = opera.drop(columns=['MoleculeID'])
opera.columns = [x.lower() for x in opera.columns]
opera_cols =  opera.columns[1:]
opera_s = opera.copy()
scaler = StandardScaler()
scaler.fit(opera_s[opera_cols])
scaler.transform(opera_s[opera_cols])
opera_s[opera_cols] =  scale(opera_s[opera_cols])    # center and scale opera predictors

opera_s = opera_s.drop(columns=['logkoa_pred','mp_pred', 'logd55_pred','logd74_pred', 'logvp_pred', 'logws_pred']) # drop highly correlated predictors
opera_cols = opera_s.columns.tolist()
opera_cols.remove('preferred_name')
opera_cols_med = ['log octanol-water partition coefficient','boiling point', "log henry's law constant",
                 'HPLC retention time', 'log OH rate constant', 'log soil adsorption coefficient']
opera_s = opera_s.merge(ids, on='preferred_name')
opera_s['Intercept'] = 1

x = len(osha)
chem_osha = osha.merge(opera_s, left_on='preferred_name', right_on='preferred_name')
chem_osha = chem_osha.dropna(subset=opera_cols)  # drop rows missing opera predictions
osha_enc = chem_osha.copy()
osha_enc['sector_enc'] = osha_enc['sector_name'].astype('category').cat.codes  # re-factor
osha_enc['subsector_enc'] = osha_enc ['subsector_name'].astype('category').cat.codes  # re-factor
logging.info('Rows dropped for no OPERA predictions (e.g. inorganics): {} out of {}'.format(x-len(osha_enc),x))

# generate interaction terms
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
opera_s_ints = poly.fit_transform(osha_enc[opera_cols])
opera_cols_ints = poly.get_feature_names()[len(opera_cols):]
opera_cols_ints_names = poly.get_feature_names(opera_cols)[len(opera_cols):]
opera_cols_ints_names = [x.replace("_pred","") for x in opera_cols_ints_names]
opera_cols_ints_names = [x.replace(" "," x ") for x in opera_cols_ints_names]
opera_interactions = pd.DataFrame(opera_s_ints[:,len(opera_cols):], columns=opera_cols_ints_names, index=osha_enc.index)
osha_enc = osha_enc.join(opera_interactions)

opera_cols_med = ['Intercept'] + opera_cols_med
opera_cols = ['Intercept'] + opera_cols


# ### Index categorical factors for pyMC3

sector_index = osha_enc.groupby(['sector_name']).all().reset_index().reset_index()[['index', 'sector_name']]
sector_subsector_index = osha_enc.groupby(['sector_name', 'subsector_name']).all().reset_index().reset_index()[['index', 'sector_name', 'subsector_name']]
sector_subsector_indexes_df = pd.merge(sector_index, sector_subsector_index, how='inner', on='sector_name', suffixes=('_s', '_ss'))
indexed_osha = pd.merge(osha_enc, sector_subsector_indexes_df, how = 'inner', on=['sector_name', 'subsector_name']).reset_index()

sector_indexes = sector_index['index'].values
sector_count = len(sector_indexes)
sector_subsector_indexes = sector_subsector_indexes_df['index_ss'].values
sector_subsector_count = len(sector_subsector_indexes)

indexed_osha_full = indexed_osha.copy()


# ### Split into training and test datasets based on substance

logging.info('Splitting into training and test sets based on randomly selected substances...')

indexed_osha_test = indexed_osha[indexed_osha['preferred_name'].isin(test_substances)]
indexed_osha_train = indexed_osha[~(indexed_osha['preferred_name'].isin(test_substances))]
x = len(indexed_osha_test)/(len(indexed_osha)+len(indexed_osha_test))
n_test_substances = len(pd.unique(indexed_osha_test['preferred_name']))
n_train_substances = len(pd.unique(indexed_osha_train['preferred_name']))
n_full_substances = len(pd.unique(indexed_osha['preferred_name']))
logging.info('The test set is {:.2f}% of the data'.format(x*100))
logging.info('The processed full dataset is {} rows.'.format(len(indexed_osha_full)))
logging.info('The processed full dataset is {} substances.'.format(n_full_substances))
logging.info('The processed training dataset is {} rows.'.format(len(indexed_osha_train)))
logging.info('The processed training dataset is {} substances.'.format(n_train_substances))
logging.info('The processed test dataset is {} rows.'.format(len(indexed_osha_test)))
logging.info('The processed test dataset is {} substances.'.format(n_test_substances))



# ### Output our data files

indexed_osha_full.drop('index',axis=1).to_csv('output/osha_processed_full.csv', index=False)
indexed_osha_train.drop('index',axis=1).to_csv('output/osha_processed_training.csv', index=False)
indexed_osha_test.drop('index',axis=1).to_csv('output/osha_processed_test.csv', index=False)
sector_subsector_indexes_df.to_csv('output/hurdle_model_sector_subsector_domain.csv',index=False)
#indexed_osha_full.groupby(['sector_name','subsector_name']).agg({'detected': ['size','sum']}).to_csv('output/osha_model_workplace_domain.csv')
indexed_osha_full.groupby(['preferred_name','sector_name','subsector_name']).size().reset_index(name='observations').to_csv('output/osha_chem_workplace_domain.csv', index=False)
logging.info('The processed full dataset has been output to: {}.'.format(os.path.abspath('output/osha_processed_full.csv')))
logging.info('The processed training dataset has been output to: {}'.format(os.path.abspath('output/osha_processed_training.csv')))
logging.info('The processed test dataset has been output to: {}'.format(os.path.abspath('output/osha_processed_test.csv')))
logging.info('A list of sectors and subsectors present in the full dataset has been output to: {}'.format(os.path.abspath('output/hurdle_model_sector_subsector_domain.csv')))
logging.info('A list of chemical X sector X subsector combos present in the full dataset has been output to: {}'.format(os.path.abspath('output/osha_chem_workplace_domain.csv')))

# ### Define which 2017 NAICS sectors and subsectors were present in the final model and which were dropped in pre-processing
final_workplace_domain = indexed_osha_full.groupby(['sector_name','subsector_name']).agg({'detected': ['size','sum']}).reset_index()
original_workplace_domain['Included in final model'] = original_workplace_domain['subsector_name'].isin(final_workplace_domain['subsector_name'])
original_workplace_domain.columns = ['Sector Name', 'Subsector Name', 'Records', 'Included in final model']
original_workplace_domain.to_csv('output/osha_workplace_domain.csv', index=False)
logging.info('A table of 2017 NAICS sectors/subsectors present in the OSHA data has been output to: {}'.format(os.path.abspath('output/osha_workplace_domain.csv')))

logging.info('Completed data_processing.py script.\n')
