### Code to wildcard SIC code to NAICS code conversions that map to more than one NAICS code
### By: Jeff Minucci 11/5/19

import pandas as pd
import numpy as np


def wildcard_sic_naics(df_in, sic_col="SIC", naics_col="2002 NAICS", naics_title="2002 NAICS Title"):
    df = df_in.copy()
    df_out = df.copy()
    last_sic = None
    matches = []
    for i, row in df.iterrows():
        matches.append(row[naics_col])
        if row[sic_col] != last_sic:
            if len(matches) > 1:
                new_label = wildcard_set(matches)
                df_out.loc[df_out.SIC == last_sic, naics_col] = new_label
                df_out.loc[df_out.SIC == last_sic, naics_title] = "Multiple"
            matches = []
            last_sic = row[sic_col]
    return df_out


def wildcard_set(strings = [], str_len=6):
    common_len = 0
    second_digit = []
    for x in range(0,str_len):
        common = True
        previous = None
        for s in strings:
            if x == 1:
                second_digit.append(s[x])
            if previous:
                if s[x] != previous:
                    common = False
            else:
                previous = s[x]
        if common:
            common_len += 1
        else:
            break
    common_str = s[0:common_len] + ((str_len - common_len))*"*"
    if common_str == "4*****":  # differentiate 44-45 (retail) and 48-49 (transportation and warehousing) if possible
        if all(digit in ['4', '5'] for digit in second_digit):
            common_str = "4(4|5)****"
        elif all(digit in ['8', '9'] for digit in second_digit):
            common_str = "4(8|9)****"
    #print("Converted {} to {}".format(strings, common_str))
    return common_str


def wildcard_series(in_series, str_len=6):
    return wildcard_set(in_series.values.tolist(), str_len)


if __name__ == '__main__':
    # Unit tests for the wildcart set function
    ex1 = ['117899', '117858', '117859']  # *1178**
    print(wildcard_set(ex1))
    ex2= ['127899', '117858', '117859']  # 1*****
    print(wildcard_set(ex2))
    ex3 = ['117899', '117898', '117899']  # 11789*
    print(wildcard_set(ex3))
    ex4 = ['446876', '452345', '452344']  # all in retail (44-45), should output 4(4|5)****
    print(wildcard_set(ex4))
    ex5 = ['123456']
    print(wildcard_set(ex5))  # 123456

    # Generate a one to one sic to naics crosswalk
    #
    #sic_naics_2002 = pd.read_csv('../../data/naics/1987_SIC_to_2002_NAICS.csv',
    #                            dtype={'SIC': str, '2002 NAICS': str},
    #                             na_values = ["", " ", "0", "AUX", "Aux", "NaN", "nan"])

    #wildcarded_converter = wildcard_sic_naics(sic_naics_2002)
    #wildcarded_converter.drop_duplicates(subset=['SIC', '2002 NAICS'], inplace=True)
    #wildcarded_converter.to_csv('../../data/naics/1987_SIC_to_2002_NAICS_one_to_one.csv', index=False)