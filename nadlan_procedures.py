#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:37:53 2021

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


def sleep_between(start=2, end=4):
    """sleep between start and end seconds"""
    from numpy import random
    from time import sleep
    sleeptime = random.uniform(start, end)
    print("sleeping for: {:.2f} seconds".format(sleeptime))
    sleep(sleeptime)
    # print("sleeping is over")
    return


def keep_only_historic_changed_assets(df):
    grps = df.groupby('GUSH').groups
    inds = []
    for gush, ind in grps.items():
        if len(ind) > 1:
            inds.append([x for x in ind])
    flat_list = [item for sublist in inds for item in sublist]
    df_slice = df.loc[flat_list]
    return df_slice


def plot_deal_amount_room_number(df, room_min=2, room_max=5,
                                 path=nadlan_path, yrmin='2000', yrmax='2020',
                                 just_with_historic_change=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from cbs_procedures import read_bycode_city_data
    import numpy as np
    sns.set_theme(style='ticks', font_scale=1.5)
    df = df.loc[(df['ASSETROOMNUM'] >= room_min) &
                (df['ASSETROOMNUM'] <= room_max)]
    df.set_index('DEALDATETIME', inplace=True)
    df = df.loc[yrmin:yrmax]
    city_code = df.loc[:, 'city_code'].unique()[0]
    df = df.rename({'ASSETROOMNUM': 'Rooms', 'DEALAMOUNT': 'Price'}, axis=1)
    df['Price'] /= 1000000
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.lineplot(data=df, x=df.index.year, y='Price', hue='Rooms',
                      ci='sd', ax=ax, palette='Set1')
    ax.grid(True)
    ax.set_xlabel('')
    ax.set_xticks(np.arange(int(yrmin), int(yrmax) + 1, 2))
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylabel('Price [millions of NIS]')
    bycode = read_bycode_city_data()
    city_name = bycode[bycode['city_code']==city_code]['NameEn'].values[0]
    fig.suptitle('Real-Estate prices in {}'.format(city_name))
    return ax


def process_nadlan_deals_df_from_one_city(df):
    """first attempt of proccessing nadlan deals df:
        removal of outliers and calculation of various indices"""
    from Migration_main import remove_outlier
    from cbs_procedures import read_statistical_areas_gis_file
    from cbs_procedures import read_social_economic_index
    import geopandas as gpd
    import numpy as np
    df = df.reset_index(drop=True)
    # first, drop some cols:
    df = df.drop(['NEWPROJECTTEXT', 'PROJECTNAME', 'TYPE',
                  'POLYGON_ID', 'TREND_IS_NEGATIVE', 'TREND_FORMAT',
                  'ObjectID', 'DescLayerID'], axis=1)
    # now convert to geodataframe using X and Y:
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
    df.crs = {'init': 'epsg:2039'}
    # now try to add CBS statistical areas for each address coords:
    stats = read_statistical_areas_gis_file()
    stats = stats[stats['city_code'] == df['city_code'].iloc[0]]
    # so iterate over stats geodataframe and check if asset coords are 
    # within the stat area polygon (also all population 2019):
    # use the same loop to to input the social economic index from CBS:
    SEI = read_social_economic_index()
    SEI = SEI[SEI['city_code'] == df['city_code'].iloc[0]]
    for i, row in stats.iterrows():
        poly = row['geometry']
        stat_code = row['stat11']
        pop2019 = row['pop_total']
        inds = df.geometry.within(poly)
        df.loc[inds, 'stat_code'] = stat_code
        df.loc[inds, 'pop2019_total'] = pop2019
        SEI_code = SEI[SEI['stat_code'] == stat_code]
        # print(stat_code)
        try:
            df.loc[inds, 'SEI_value'] = SEI_code['index_value'].values[0]
            df.loc[inds, 'SEI_rank'] = SEI_code['rank'].values[0]
            df.loc[inds, 'SEI_cluster'] = SEI_code['cluster'].values[0]
        except IndexError:
            df.loc[inds, 'SEI_value'] = np.nan
            df.loc[inds, 'SEI_rank'] = np.nan
            df.loc[inds, 'SEI_cluster'] = np.nan

    # remove outliers in squared meters per asset:
    df = remove_outlier(df, 'DEALNATURE')
    df = remove_outlier(df, 'ASSETROOMNUM')
    df = remove_outlier(df, 'DEALAMOUNT')
    # calculate squared meters per room:
    df['M2_per_ROOM'] = df['DEALNATURE'] / df['ASSETROOMNUM']
    df['NIS_per_M2'] = df['DEALAMOUNT'] / df['DEALNATURE']
    # try to guess new buildings:
    df['New'] = df['BUILDINGYEAR'] == df['DEALDATETIME'].dt.year
    df['Age'] = df['DEALDATETIME'].dt.year - df['BUILDINGYEAR']
    # try to guess ground floors apts.:
    df['Ground'] = df['FLOORNO'].str.contains('קרקע')
    return df


def concat_all_nadlan_deals_from_all_cities_and_save(nadlan_path=work_david/'Nadlan_deals',
                                                     savepath=None,
                                                     delete_files=False):
    """concat all nadlan deals for all the cities in nadlan_path"""
    from Migration_main import path_glob
    import numpy as np
    folders = path_glob(nadlan_path, '*/')
    folders = [x for x in folders if any(i.isdigit() for i in x.as_posix())]
    # files = path_glob(nadlan_path, 'Nadlan_deals_city_*_street_*.csv')
    city_codes = [x.as_posix().split('/')[-1] for x in folders]
    city_codes = np.unique(city_codes)
    print('found {} city codes: {}'.format(len(city_codes), ', '.join(city_codes)))
    for city_code in sorted(city_codes):
        concat_all_nadlan_deals_from_one_city_and_save(nadlan_path=nadlan_path,
                                                       city_code=int(city_code),
                                                       savepath=savepath,
                                                       delete_files=delete_files)
    return


def post_nadlan_historic_deals(keyvalue):
    """take keyvalue from body or df fields and post to nadlan.gov.il deals
    REST API historic deals"""
    import requests
    url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetAllAssestHistoryDeals'
    r = requests.post(url, json=str(keyvalue))
    if r.status_code != 200:
        raise ValueError('couldnt get a response ({}).'.format(r.status_code))
    result = r.json()
    if not result:
        raise TypeError('no results found')
    return result


def get_historic_deals_for_city_code(main_path=nadlan_path, city_code=8700):
    from Migration_main import path_glob
    import os
    import pandas as pd
    streets = path_glob(main_path / str(city_code), 'Nadlan_deals_city_{}_street_*.csv'.format(city_code))
    # check for the h suffix and filter these files out (already done historic):
    streets = [x for x in streets if '_h' not in x.as_posix().split('/')[-1]]
    print('Found {} streets to check.'.format(len(streets)))
    cnt = 1
    for street in streets:
        df = pd.read_csv(street, na_values='None')
        df = get_historic_nadlan_deals_from_a_street_df(df, unique=True)
        filename = street.as_posix().split('.')[0] + '_h.csv'
        if not df.empty:
            df.to_csv(filename, na_rep='None', index=False)
            print('File was rewritten ({} out of {}).'.format(cnt, len(streets)))
            # delete old file:
            street.unlink()
        else:
            print('skipping and renaming ({} of out {})'.format(cnt, len(streets)))
            os.rename(street.as_posix(), filename)
        cnt += 1
    print('Done fetching historic deals for {}.'.format(city_code))
    return


def get_historic_nadlan_deals_from_a_street_df(df, unique=True):
    import pandas as pd
    import numpy as np
    # first get the unique dataframe of KEYVALUE (should be the same)
    if unique:
        df = df.sort_index().groupby('KEYVALUE').filter(lambda group: len(group) == 1)
    df['DEALDATETIME'] = pd.to_datetime(df['DEALDATETIME'])
    city = df['City'].iloc[0]
    street = df['Street'].iloc[0]
    total_keys = len(df)
    print('Fetching historic deals on {} street in {} (total {} assets).'.format(street, city, total_keys))
    cnt = 0
    dfs = []
    for i, row in df.iterrows():
        keyvalue = row['KEYVALUE']
        city_code = row['city_code']
        street_code = row['street_code']
        other_fields = row.loc['ObjectID': 'Street']
        try:
            result = post_nadlan_historic_deals(keyvalue)
            dfh_key = parse_one_json_nadlan_page_to_pandas(result, city_code,
                                                           street_code,
                                                           historic=True)
            other_df = pd.DataFrame().append(
                [other_fields.to_frame().T] * len(dfh_key), ignore_index=True)
            dfs.append(pd.concat([dfh_key, other_df], axis=1))
            sleep_between(0.1, 0.15)
            cnt += 1
        except TypeError:
            sleep_between(0.01, 0.05)
            continue
        except ValueError:
            sleep_between(0.01, 0.05)
            continue
    print('Found {} assets with historic deals out of {}.'.format(cnt, total_keys))
    try:
        dfh = pd.concat(dfs)
    except ValueError:
        print('No hitoric deals found for {}, skipping...'.format(street))
        return pd.DataFrame()
    dfh = dfh.reset_index()
    dfh = dfh.sort_index()
    df_new = pd.concat([df, dfh], axis=0)
    df_new.replace(to_replace=[None], value=np.nan, inplace=True)
    df_new = df_new.sort_values('GUSH')
    df_new = df_new.reset_index(drop=True)
    # try to fill in missing description from current deals:
    grps = df_new.groupby('GUSH').groups
    for gush, ind in grps.items():
        if len(ind) > 1:
            desc = df_new.loc[ind, 'DEALNATUREDESCRIPTION'].dropna()
            try:
                no_nan_desc = desc.values[0]
            except IndexError:
                no_nan_desc = np.nan
            df_new.loc[ind, 'DEALNATUREDESCRIPTION'] = df_new.loc[ind,
                                                                  'DEALNATUREDESCRIPTION'].fillna(no_nan_desc)
    # finally make sure that we don't add duplicate deals:
    cols = df_new.loc[:, 'DEALAMOUNT':'TREND_FORMAT'].columns
    df_new = df_new.drop_duplicates(subset=df_new.columns.difference(cols))
    return df_new


def concat_all_nadlan_deals_from_one_city_and_save(nadlan_path=work_david/'Nadlan_deals',
                                                   city_code=8700,
                                                   savepath=None,
                                                   delete_files=False):
    """concat all nadlan deals for all streets in a specific city"""
    import pandas as pd
    from Migration_main import path_glob
    import click
    city_path = nadlan_path / str(city_code)
    files = path_glob(
        city_path, 'Nadlan_deals_city_{}_street_*.csv'.format(city_code))
    dfs = [pd.read_csv(file, na_values='None') for file in files]
    df = pd.concat(dfs)
    df['DEALDATETIME'] = pd.to_datetime(df['DEALDATETIME'])
    df.drop(['DEALDATE', 'DISPLAYADRESS'], axis=1, inplace=True)
    print('concated all {} ({}) csv files.'.format(df['City'].unique()[0] , city_code))
    df = df.sort_index()
    df = df.iloc[:, 2:]
    # first drop records with no full address:
    df = df[~df['FULLADRESS'].isna()]
    # now go over all the unique GUSHs and fill in the geoloc codes if missing
    # and then drop duplicates
    # good_district = df['District'].unique()[df['District'].unique() != ''][0]
    good_district = df['District'].value_counts().index[0]
    good_city = df['City'].value_counts().index[0].strip()
    df = df.copy()
    if not pd.isnull(good_district):
        df.loc[:, 'District'].fillna(good_district, inplace=True)
    df.loc[:, 'City'].fillna(good_city, inplace=True)
    df['City'] = df['City'].str.strip()
    df = df[df['City'] == good_city]
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
                                       savepath=None,
                                       sleep_between_streets=[1,  5],
                                       start_from_street_code=None):
    """Get all nadlan deals for specific city from nadlan.gov.il."""
    import pandas as pd
    import os
    from cbs_procedures import read_street_city_names
    df = read_street_city_names(path=path, filter_neighborhoods=True)
    streets_df = df[df['city_code'] == city_code]
    st_df = streets_df.reset_index(drop=True)
    if city_code == 1061:
        st_df['city_name'] = st_df[st_df['city_code'] == 1061]['city_name'].str.replace(
            'נוף הגליל', 'נצרת עילית')
    if start_from_street_code is not None:
        ind = st_df[st_df['street_code'] == start_from_street_code].index
        st_df = st_df.iloc[ind[0]:]
        print('Starting from street {} for city {}'.format(
            st_df.iloc[0]['street_name'], st_df.iloc[0]['city_name']))
    bad_streets_df = pd.DataFrame()
    all_streets = len(st_df)
    cnt = 1
    for i, row in st_df.iterrows():
        city_name = row['city_name']
        street_name = row['street_name']
        street_code = row['street_code']
        print('Fetching Nadlan deals, city: {} , street: {} ({} out of {})'.format(
            city_name, street_name, cnt, all_streets))
        if not savepath.is_dir():
            print('Folder {} not found...Creating it.'.format(savepath))
            os.mkdir(savepath)
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
    from cbs_procedures import read_periphery_index
    dff = read_periphery_index(work_david)
    city_codes = dff.sort_values('Pop2015', ascending=False)
    return city_codes


def parse_one_json_nadlan_page_to_pandas(page, city_code=None,
                                         street_code=None, historic=False):
    """parse one request of nadlan deals JSON to pandas"""
    import pandas as pd
    if historic:
        df = pd.DataFrame(page)
    else:
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
        raise ValueError('couldnt get a response ({}).'.format(r.status_code))
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
    no_results = False
    while not last_page:
        print('Page : ', cnt)
        try:
            result = post_nadlan_rest(body)
        except TypeError:
            no_results = True
            # if cnt > 1:
                # pass
            # else:
            return pd.DataFrame()
        except ValueError:
            no_results = True
            # if cnt > 1:
            # else:
            # return pd.DataFrame()
        if no_results and cnt > 1:
            last_page = True
        elif no_results and cnt == 1:
            return pd.DataFrame()
        page_dfs.append(parse_one_json_nadlan_page_to_pandas(
            result, city_code, street_code))
        cnt += 1
        if result['IsLastPage']:
            last_page = True
        else:
            body['PageNo'] += 1
        no_results = False
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
            sleep_between(0.1, 0.3)
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

