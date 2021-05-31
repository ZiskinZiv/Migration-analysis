#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:44:45 2021

@author: ziskin
"""
from MA_paths import work_david


# def convert_headers_to_dict(filepath):
#     f = open(filepath, "r")
#     lines = f.readlines()
#     headers = {}
#     for old_line in lines:
#         line = old_line.replace("\n", "")
#         split_here = line.find(":")
#         headers[line[:split_here]] = line[split_here+2:]
#     return headers

def get_all_streets_from_df(df, city_code=5000):
    return df[df['city_code'] == city_code]

def read_street_city_names(path=work_david):
    import pandas as pd
    df = pd.read_csv(path/'street_names_israel.csv', encoding="cp1255")
    df.columns = ['city_code', 'city_name', 'street_code', 'street_name']
    return df


def parse_one_json_nadlan_page_to_pandas(page):
    import pandas as pd
    df = pd.DataFrame(page['AllResults'])
    df.set_index('DEALDATETIME', inplace=True)
    df.index = pd.to_datetime(df.index)
    df['DEALAMOUNT'] = df['DEALAMOUNT'].str.replace(',','').astype(int)
    df['DEALNATURE'] = df['DEALNATURE'].astype(float)
    df['ASSETROOMNUM'] = pd.to_numeric(df['ASSETROOMNUM'])
    return df


def produce_nadlan_rest_request(city='רעננה', street='אחוזה'):
    import requests
    url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetDataByQuery?query={} {}'.format(city, street)
    r = requests.get(url)
    body = r.json()
    if r.status_code != 200:
        raise ValueError('couldnt get a response.')
    if body['PageNo'] == 0:
        body['PageNo'] = 1
    return body


def post_nadlan_rest(body):
    import requests
    url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetAssestAndDeals'
    r = requests.post(url, json=body)
    if r.status_code != 200:
        raise ValueError('couldnt get a response.')
    result = r.json()
    if result['ResultType'] != 1:
        raise TypeError(result['ResultLable'])
    return result


def get_all_historical_nadlan_deals(city='רעננה', street='אחוזה'):
    import pandas as pd
    body = produce_nadlan_rest_request(city=city, street=street)
    page_dfs = []
    cnt = 1
    last_page = False
    while not last_page:
        print('Page : ', cnt)
        result = post_nadlan_rest(body)
        page_dfs.append(parse_one_json_nadlan_page_to_pandas(result))
        cnt += 1
        if result['IsLastPage']:
            last_page = True
        else:
            body['PageNo'] += 1
    df = pd.concat(page_dfs)
    df = df.sort_index()
    return df


def read_periphery_index(path=work_david):
    import pandas as pd
    df = pd.read_excel(
        path/'peripheri_index_cbs_2015_local_cities.xls', skiprows=8)
    cols = ['municipal_status', 'city_code', 'NameHe', 'NameEn', 'Pop2015', 'TLV_proximity_value',
            'TLV_proximity_rank', 'PAI_value', 'PAI_rank', 'P2015_value', 'P2015_rank', 'P2015_cluster', 'P2004_cluster']
    df.columns = cols
    df = df.dropna()
    df['city_code'] = df['city_code'].astype(int)
    df.set_index('city_code', inplace=True)
    df_r = pd.read_excel(
        path/'peripheri_index_cbs_2015_regional_councils.xls', skiprows=6)
    cols = ['municipal_status', 'RC_nameHe', 'RC_nameEn', 'city_code', 'NameHe', 'NameEn', 'Sub_district', 'Pop2015', 'TLV_distance',
            'TLV_proximity_value', 'TLV_proximity_rank', 'PAI_value', 'PAI_rank', 'P2015_value', 'P2015_rank', 'P2015_cluster']
    df_r.columns = cols
    df_r = df_r.dropna()
    df_r.set_index('city_code', inplace=True)
    dff = pd.concat([df, df_r], axis=0)
    dff.index = dff.index.astype(int)
    return dff


def read_cbs_main_indicies(start_date='1997-01', end_date=None,
                           savepath=None):
    import pandas as pd
    # import xml.etree.ElementTree as ET
    import xmltodict
    import requests
    st_date = ''.join(start_date.split('-'))
    if end_date is not None:
        ed_date = ''.join(end_date.split('-'))
    else:
        ed_date = pd.Timestamp.now().strftime('%Y%m')
    url = 'https://api.cbs.gov.il/index/data/price_selected_b?StartDate={}&EndDate={}'.format(
        st_date, ed_date)
    r = requests.get(url)
    xmlDict = xmltodict.parse(r.content)
    cols = []
    data = []
    for item in xmlDict['indices']['ind']:
        cols = [x for x in item.keys()]
        data.append([x for x in item.values()])
    df = pd.DataFrame(data, columns=cols)
    df.index = pd.to_datetime(df['date'], format='%Y-%m')
    dates = pd.to_datetime(df['date'].unique(), format='%Y-%m')
    dff = pd.DataFrame(index=dates)
    names = {}
    for ind_code in df['code'].unique():
        dff[str(ind_code)] = df[df['code'] == ind_code]['index'].astype(float)
        dff[str(ind_code)+'_base'] = df[df['code'] == ind_code]['base']
        names[ind_code] = df[df['code'] == ind_code]['name'].unique().item()
    dff.index.name = 'time'
    ds = dff.to_xarray()
    for da in ds:
        try:
            ds[da].attrs['index_name'] = names[da]
        except KeyError:
            pass
    # root = ET.XML(r.content)
    # data = []
    # cols = []
    # for i, child in enumerate(root):
    #     data.append([subchild.text for subchild in child])
    #     cols.append(child.tag)
    # df = pd.DataFrame(data).T  # Write in DF and transpose it
    # df.columns = cols
    return ds
