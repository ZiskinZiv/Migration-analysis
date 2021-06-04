#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:44:45 2021

@author: ziskin
"""
# TODO: each call to body with street name yields district, city and street
# name and mid-street coords along with neiborhood based upon mid-street coords.
# however, if we request street and building number we get more precise location
# and neighborhood. We should re-run the body request for each line in the DataFrame
# and get the precise coords and nieghborhood
from MA_paths import work_david
nadlan_path = work_david / 'Nadlan_deals'

# def convert_headers_to_dict(filepath):
#     f = open(filepath, "r")
#     lines = f.readlines()
#     headers = {}
#     for old_line in lines:
#         line = old_line.replace("\n", "")
#         split_here = line.find(":")
#         headers[line[:split_here]] = line[split_here+2:]
#     return headers


# def go_over_df_and_add_location_and_neighborhood_data(df):
#     import pandas as pd
#     loc_df = pd.DataFrame()
#     for i, row in df.iterrows():
#         fa = row['FULLADRESS']
#         print('getting data for {} '.format(fa))
#         body = produce_nadlan_rest_request(full_address=fa)
#         loc_df = loc_df.append(parse_body_request_to_dataframe(body))
#     print('Done!')
#     df = pd.concat([df, loc_df], axis=1)
#     return df

# def fill_in_missing_str_in_the_same_col(df, same_val_col='GUSH', missin_col='Neighborhood'):
#     for group, inds in df.groupby([same_val_col]).groups.items():
#         if len(inds) > 1:


def remove_outlier(df_in, col_name, k=1.5):
    """remove outlier using iqr criterion (k=1.5)"""
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1  # Interquartile range
    fence_low = q1 - k * iqr
    fence_high = q3 + k * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) &
                       (df_in[col_name] < fence_high)]
    return df_out


def sleep_between(start=2, end=4):
    """sleep between start and end seconds"""
    from numpy import random
    from time import sleep
    sleeptime = random.uniform(start, end)
    print("sleeping for: {:.2f} seconds".format(sleeptime))
    sleep(sleeptime)
    # print("sleeping is over")
    return


def process_nadlan_deals_df_from_one_city(df):
    """first attempt of proccessing nadlan deals df:
        removal of outliers and calculation of various indices"""
    # first, drop some cols:
    df = df.drop(['NEWPROJECTTEXT', 'PROJECTNAME', 'YEARBUILT', 'KEYVALUE', 'TYPE',
                  'POLYGON_ID', 'TREND_IS_NEGATIVE', 'TREND_FORMAT', 'ObjectID', 'DescLayerID'], axis=1)
    # remove outliers in squared meters per asset:
    df = remove_outlier(df, 'DEALNATURE')
    df = remove_outlier(df, 'ASSETROOMNUM')
    # calculate squared meters per room:
    df['M2_per_ROOM'] = df['DEALNATURE'] / df['ASSETROOMNUM']
    return df


def concat_all_nadlan_deals_from_all_cities_and_save(nadlan_path=work_david/'Nadlan_deals',
                                                     savepath=None,
                                                     delete_files=False):
    """concat all nadlan deals for all the cities in nadlan_path"""
    from Migration_main import path_glob
    import numpy as np
    files = path_glob(nadlan_path, 'Nadlan_deals_city_*_street_*.csv')
    city_codes = [x.as_posix().split('/')[-1].split('_')[3] for x in files]
    city_codes = np.unique(city_codes)
    print('found {} city codes: {}'.format(len(city_codes), ', '.join(city_codes)))
    for city_code in sorted(city_codes):
        concat_all_nadlan_deals_from_one_city_and_save(nadlan_path=nadlan_path,
                                                       city_code=int(city_code),
                                                       savepath=savepath,
                                                       delete_files=delete_files)
    return


def concat_all_nadlan_deals_from_one_city_and_save(nadlan_path=work_david/'Nadlan_deals',
                                                   city_code=8700,
                                                   savepath=None,
                                                   delete_files=False):
    """concat all nadlan deals for all streets in a specific city"""
    import pandas as pd
    from Migration_main import path_glob
    import click
    files = path_glob(
        nadlan_path, 'Nadlan_deals_city_{}_street_*.csv'.format(city_code))
    dfs = [pd.read_csv(file, na_values='None') for file in files]
    df = pd.concat(dfs)
    df.set_index(pd.to_datetime(df['DEALDATETIME']), inplace=True)
    df.drop(['DEALDATETIME', 'DEALDATE', 'DISPLAYADRESS'], axis=1, inplace=True)
    print('concated all {} ({}) csv files.'.format(df['City'].unique()[0] , city_code))
    df = df.sort_index()
    df = df.iloc[:, 2:]
    # first drop records with no full address:
    df = df[~df['FULLADRESS'].isna()]
    # now go over all the unique GUSHs and fill in the geoloc codes if missing
    # and then drop duplicates
    good_district = df['District'].unique()[df['District'].unique() != ''][0]
    good_city = df['City'].unique()[df['City'].unique() != ''][0]
    df = df.copy()
    if not pd.isnull(good_district):
        df.loc[:, 'District'].fillna(good_district, inplace=True)
    df.loc[:, 'City'].fillna(good_city, inplace=True)
    # take care of street_code columns all but duplicates bc of neighhood code/street code:
    df = df.drop_duplicates(subset=df.columns.difference(['street_code', 'Street']))
    # lasty, extract Building number from FULLADRESS and is NaN remove record:
    df['Building'] = df['FULLADRESS'].str.extract('(\d+)')
    df = df[~df['Building'].isna()]
    if savepath is not None:
        yrmin = df.index.min().year
        yrmax = df.index.max().year
        filename = 'Nadlan_deals_city_{}_all_streets_{}-{}.csv'.format(
            city_code, yrmin, yrmax)
        df.to_csv(savepath/filename, na_rep='None')
        print('{} was saved to {}'.format(filename, savepath))
    if delete_files:
        msg = 'Deleting all {} city files, Do you want to continue?'.format(city_code)
        if click.confirm(msg, default=True):
            [x.unlink() for x in files]
            print('{} files were deleted.'.format(city_code))
    return df


def get_all_nadlan_deals_from_one_city(path=work_david, city_code=5000,
                                       savepath=None, sleep_between_streets=[1,  5]):
    """gets all nadlan deals for specific city from nadlan.gov.il"""
    import pandas as pd
    df = read_street_city_names(path=path, filter_neighborhoods=True)
    streets_df = df[df['city_code'] == city_code]
    bad_streets_df = pd.DataFrame()
    all_streets = len(streets_df)
    cnt = 1
    for i, row in streets_df.iterrows():
        city_name = row['city_name']
        street_name = row['street_name']
        street_code = row['street_code']
        print('Fetching Nadlan deals, city: {} , street: {} ({} out of {})'.format(
            city_name, street_name, cnt, all_streets))
        df = get_all_historical_nadlan_deals(
            city=city_name, street=street_name, city_code=city_code,
            street_code=street_code, savepath=savepath, sleep_between_streets=True)
        if df is None:
            bad_df = pd.DataFrame(
                [city_code, city_name, street_code, street_name]).T
            bad_df.columns = ['city_code', 'city_name',
                              'street_code', 'street_name']
            print('no data for street: {} in {}'.format(street_name, city_name))
            bad_streets_df = bad_streets_df.append(bad_df)
        elif df.empty:
            cnt += 1
            continue
        if sleep_between_streets is not None:
            sleep_between(sleep_between_streets[0], sleep_between_streets[1])
        cnt += 1
    print('Done scraping {} city ({}) from nadlan.gov.il'.format(city_name, city_code))

    if savepath is not None:
        filename = 'Nadlan_missing_streets_{}.csv'.format(city_code)
        bad_streets_df.to_csv(savepath/filename)
        print('{} was saved to {}'.format(filename, savepath))
    return


def get_all_city_codes_from_largest_to_smallest(path=work_david):
    """get all city codes and sort using population (2015)"""
    dff = read_periphery_index(work_david)
    city_codes = dff.sort_values('Pop2015', ascending=False)
    return city_codes


def read_street_city_names(path=work_david, filter_neighborhoods=True,
                           fix_alley=True):
    """get street names, codes and city names and codes"""
    import pandas as pd
    df = pd.read_csv(path/'street_names_israel.csv', encoding="cp1255")
    df.columns = ['city_code', 'city_name', 'street_code', 'street_name']
    df['city_name'] = df['city_name'].str.replace(
        'תל אביב - יפו', 'תל אביב יפו')
    if filter_neighborhoods:
        df = df[df['street_code'] <= 5999]
    if fix_alley:
        df['street_name'] = df['street_name'].str.replace('סמ ', 'סמטת ')
    return df


def parse_one_json_nadlan_page_to_pandas(page, city_code=None, street_code=None):
    """parse one request of nadlan deals JSON to pandas"""
    import pandas as pd
    df = pd.DataFrame(page['AllResults'])
    # df.set_index('DEALDATETIME', inplace=True)
    # df.index = pd.to_datetime(df.index)
    df['DEALDATETIME'] = pd.to_datetime(df['DEALDATETIME'])
    df['DEALAMOUNT'] = df['DEALAMOUNT'].str.replace(',','').astype(int)
    df['DEALNATURE'] = pd.to_numeric(df['DEALNATURE'])
    df['ASSETROOMNUM'] = pd.to_numeric(df['ASSETROOMNUM'])
    if city_code is not None:
        df['city_code'] = city_code
    if street_code is not None:
        df['street_code'] = street_code
    return df


def produce_nadlan_rest_request(city='רעננה', street='אחוזה',
                                full_address=None):
    """produce the body of nadlan deals request, also usfull for geolocations"""
    import requests
    if full_address is not None:
        url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetDataByQuery?query={}'.format(full_address)
    else:
        url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetDataByQuery?query={} {}'.format(city, street)
    r = requests.get(url)
    body = r.json()
    if r.status_code != 200:
        raise ValueError('couldnt get a response.')
    if body['ResultType'] != 1 and full_address is None:
        raise TypeError(body['ResultLable'])
    if body['DescLayerID'] == 'NEIGHBORHOODS_AREA':
        raise TypeError('result is a NEIGHBORHOOD!, skipping')
    if body['PageNo'] == 0:
        body['PageNo'] = 1
    return body


def post_nadlan_rest(body):
    """take body from request and post to nadlan.gov.il deals REST API"""
    import requests
    url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetAssestAndDeals'
    r = requests.post(url, json=body)
    if r.status_code != 200:
        raise ValueError('couldnt get a response.')
    result = r.json()
    if not result['AllResults']:
        raise TypeError('no results found')
    return result


def parse_body_request_to_dataframe(body):
    """parse body request to pandas"""
    import pandas as pd

    def parse_body_navs(body):
        navs = body['Navs']
        if navs is not None:
            nav_dict = {}
            order_dict = {1: 'District', 2: 'City', 3: 'Neighborhood', 4: 'Street'}
            for list_item in navs:
                nav_dict[order_dict[list_item['order']]] = list_item['text']
            nav_df = pd.DataFrame.from_dict(nav_dict, orient='index').T
            for order in [x for x in order_dict.values()]:
                if order not in nav_df.columns:
                    nav_df[order] = ''
            # nav_df = pd.Series([navs[x]['text']
                                # for x in range(len(navs))]).to_frame().T
        else:
            nav_df = pd.DataFrame(['', '', '', '']).T
            nav_df.columns = ['District', 'City', 'Neighborhood', 'Street']
        return nav_df
    nav_df = parse_body_navs(body)
    keys_to_keep = ['ObjectID', 'DescLayerID', 'X', 'Y']
    df = pd.Series([body[x] for x in keys_to_keep]).to_frame().T
    df.columns = keys_to_keep
    if not nav_df.empty:
        df = pd.concat([df, nav_df], axis=1)
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    return df


def get_all_historical_nadlan_deals(city='רעננה', street='אחוזה',
                                    city_code=None, street_code=None,
                                    savepath=None,
                                    check_for_downloaded_files=True,
                                    sleep_between_streets=True):
    import pandas as pd
    from json import JSONDecodeError
    from Migration_main import path_glob
    if check_for_downloaded_files and savepath is not None:
        try:
            file = path_glob(savepath, 'Nadlan_deals_city_{}_street_{}_*.csv'.format(city_code, street_code))
            print('{} already found, skipping...'.format(file))
            return pd.DataFrame()
        except FileNotFoundError:
            pass
    try:
        body = produce_nadlan_rest_request(city=city, street=street)
    except TypeError:
        return None
    page_dfs = []
    cnt = 1
    last_page = False
    while not last_page:
        print('Page : ', cnt)
        try:
            result = post_nadlan_rest(body)
        except TypeError:
            return pd.DataFrame()
        except ValueError:
            pass
        page_dfs.append(parse_one_json_nadlan_page_to_pandas(
            result, city_code, street_code))
        cnt += 1
        if result['IsLastPage']:
            last_page = True
        else:
            body['PageNo'] += 1
    df = pd.concat(page_dfs)
    df = df.reset_index()
    df = df.sort_index()
    # now re-run and get all body requests for all street numbers:
    locs = []
    unique_addresses = df['FULLADRESS'].unique()
    print('processing {} unique addresses...'.format(len(unique_addresses)))
    for fa in unique_addresses:
        #         print('getting data for {} '.format(fa))
        rows = len(df[df['FULLADRESS'] == fa])
        ind = df[df['FULLADRESS'] == fa].index
        try:
            body = produce_nadlan_rest_request(full_address=fa)
        except JSONDecodeError:
            body = produce_nadlan_rest_request(full_address=fa)
        loc_df = parse_body_request_to_dataframe(body)
        loc_df_street = loc_df.append([loc_df]*(rows-1), ignore_index=True)
        loc_df_street.index = ind
        locs.append(loc_df_street)
        if sleep_between_streets:
            sleep_between(0.2, 0.6)
    loc_df_street = pd.concat(locs, axis=0)
    df = pd.concat([df, loc_df_street.sort_index()], axis=1)
    # fill in district and city, street:
    try:
        good_district = df['District'].unique()[df['District'].unique() != ''][0]
    except IndexError:
        good_district = ''
    df.loc[df['District'] == '', 'District'] = good_district
    df.loc[df['City'] == '', 'City'] = city
    df.loc[df['Street'] == '', 'Street'] = street
    if savepath is not None:
        yrmin = df['DEALDATETIME'].min().year
        yrmax = df['DEALDATETIME'].max().year
        filename = 'Nadlan_deals_city_{}_street_{}_{}-{}.csv'.format(
            city_code, street_code, yrmin, yrmax)
        df.to_csv(savepath/filename, na_rep='None')
        print('{} was saved to {}.'.format(filename, savepath))
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
