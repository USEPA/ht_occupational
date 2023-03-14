import numpy as np
import pandas as pd

naics_defs= pd.read_csv('data/naics/2_6_digit_naics_2017.csv', dtype={'2017_naics_code': str}, index_col='2017_naics_code')


industries = {'11':'Agriculture, Forestry, Fishing and Hunting', '21':'Mining, Quarrying, and Oil and Gas Extraction', '22':'Utilities', '23':'Construction', '31':'Manufacturing', '32':'Manufacturing', '33':'Manufacturing', '3*': 'Manufacturing', '42':'Wholesale Trade', '44':'Retail Trade', '45':'Retail Trade', '4(4|5)':'Retail Trade', '48':'Transportation and Warehousing', '49':'Transportation and Warehousing', '4(8|9)':'Transportation and Warehousing', '51':'Information', '52':'Finance and Insurance', '53':'Real Estate and Rental and Leasing', '54':'Professional, Scientific, and Technical Services', '55':'Management of Companies and Enterprises', '56':'Administrative and Support and Waste Management and Remediation Services', '61':'Educational Services', '62':'Health Care and Social Assistance', '71':'Arts, Entertainment, and Recreation', '72':'Accomodation and Food Services', '81':'Other Services', '92':'Public Administration'}

def get_naics_subcode(code, digits):
    if pd.isnull(code):
        return np.nan
    subcode = code[0:digits]
    if '(' in subcode:  # Check to see if this is an industry that covers multiple 2 digit codes
        subcode = code[0:digits+4]
    return subcode

#get_naics_subcode('3*****', 2)

def get_naics_industry(code):
    if pd.isnull(code):
        return np.nan
    subcode = get_naics_subcode(code, digits=2)
    try:
        name = industries[subcode]
    except KeyError as e:
        return "Undefined/Multiple"
    return name


def get_naics_definition(code, digits):
    if (pd.isnull(code)) or (digits > 6):
        return np.nan
    subcode = get_naics_subcode(code, digits)
    if digits == 2:
        try:
            name = industries[subcode]
        except KeyError as e:
            return "Undefined/Multiple"
    else:
        try:
            name = naics_defs.loc[subcode, '2017_naics_title']
        except (KeyError, IndexError) as e:
            return "Undefined/Multiple"
    return name