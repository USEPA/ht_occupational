import numpy as np
import pandas as pd

def naics_to_2017(in_series):
    """
    Takes a Series of NAICS codes that is a mix of 2002, 2007, 2012 and 2017 formats
    and returns a series unified to 2017 formats. When conversions are one-to-many, 
    wildcard digits are used. Once wildcard digits are introduced, they cannot be converted 
    further. """
    codes = in_series.copy()
    col_name = codes.name
    is_undefined = codes.str.contains('\*', na=True)
    is_2017 = codes.isin(naics1217['2017 NAICS Code'])
    is_2012 = codes.isin(naics1217['2012 NAICS Code'])
    is_2007 = codes.isin(naics0712['2007 NAICS Code'])
    is_2002 = codes.isin(naics0207['2002 NAICS Code'])
    con_12_17 = pd.merge(codes.to_frame(),naics1217w,left_on=col_name, right_on='2012 NAICS Code', how='left')
    con_12_17 = con_12_17[col_name].where(con_12_17[col_name].str.contains('\*',na=False), con_12_17['2017 NAICS Code'])
    con_07_12 = pd.merge(codes.to_frame(),naics0712w,left_on=col_name, right_on='2007 NAICS Code', how='left')
    con_07_12 = con_07_12[col_name].where(con_07_12[col_name].str.contains('\*',na=False), con_07_12['2012 NAICS Code'])
    con_07_17 = pd.merge(con_07_12.to_frame(),naics1217w,left_on=col_name, right_on='2012 NAICS Code', how='left')
    con_07_17 = con_07_17[col_name].where(con_07_17[col_name].str.contains('\*',na=False), con_07_17['2017 NAICS Code'])
    con_02_07 = pd.merge(codes.to_frame(),naics0207w,left_on=col_name, right_on='2002 NAICS Code', how='left')
    con_02_07 = con_02_07[col_name].where(con_02_07[col_name].str.contains('\*',na=False), con_02_07['2007 NAICS Code'])
    con_02_12 = pd.merge(con_02_07.to_frame(),naics0712w,left_on=col_name, right_on='2007 NAICS Code', how='left')
    con_02_12 = con_02_12[col_name].where(con_02_12[col_name].str.contains('\*',na=False),  con_02_12['2012 NAICS Code'])
    con_02_17 = pd.merge(con_02_12.to_frame(),naics1217w,left_on=col_name, right_on='2012 NAICS Code', how='left')
    con_02_17 = con_02_17[col_name].where(con_02_17[col_name].str.contains('\*',na=False), con_02_17['2017 NAICS Code'])

    converted = np.where((is_2017 | is_undefined),codes,
                        np.where(is_2012, con_12_17,
                                np.where(is_2007, con_07_17,
                                        np.where(is_2002, con_02_17, np.nan))))
    
    return(converted)
    

if __name__ == '__main__':
    import wildcard_sic_naics as wc
    naics0207 = pd.read_csv('data/naics/2002_to_2007_NAICS.csv', skiprows=2, dtype={'2002 NAICS Code': str, '2007 NAICS Code': str})
    naics0712 = pd.read_csv('data/naics/2007_to_2012_NAICS.csv', skiprows=2, dtype={'2007 NAICS Code': str, '2012 NAICS Code': str})
    naics1217 = pd.read_csv('data/naics/2012_to_2017_NAICS.csv', skiprows=2, dtype={'2012 NAICS Code': str, '2017 NAICS Code': str})
    naics0207w = naics0207.iloc[:,[0,2]].copy().groupby('2002 NAICS Code').agg({'2007 NAICS Code':wc.wildcard_series}).reset_index()
    naics0712w = naics0712.iloc[:,[0,2]].copy().groupby('2007 NAICS Code').agg({'2012 NAICS Code':wc.wildcard_series}).reset_index()
    naics1217w = naics1217.iloc[:,[0,2]].copy().groupby('2012 NAICS Code').agg({'2017 NAICS Code':wc.wildcard_series}).reset_index()
    
    test = pd.Series(['111110', '11112*', '212234', '339111', '999999', '516110'], name = 'test_codes')
    # expected output: ['111110', '11112*', '212230', '33****', nan, '519130']
    print(naics_to_2017(test))

else:
    import functions_utils.wildcard_sic_naics as wc
    naics0207 = pd.read_csv('data/naics/2002_to_2007_NAICS.csv', skiprows=2, dtype={'2002 NAICS Code': str, '2007 NAICS Code': str})
    naics0712 = pd.read_csv('data/naics/2007_to_2012_NAICS.csv', skiprows=2, dtype={'2007 NAICS Code': str, '2012 NAICS Code': str})
    naics1217 = pd.read_csv('data/naics/2012_to_2017_NAICS.csv', skiprows=2, dtype={'2012 NAICS Code': str, '2017 NAICS Code': str})
    naics0207w = naics0207.iloc[:,[0,2]].copy().groupby('2002 NAICS Code').agg({'2007 NAICS Code':wc.wildcard_series}).reset_index()
    naics0712w = naics0712.iloc[:,[0,2]].copy().groupby('2007 NAICS Code').agg({'2012 NAICS Code':wc.wildcard_series}).reset_index()
    naics1217w = naics1217.iloc[:,[0,2]].copy().groupby('2012 NAICS Code').agg({'2017 NAICS Code':wc.wildcard_series}).reset_index()
