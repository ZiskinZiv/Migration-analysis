#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:37:53 2021
First download all nadlan deals:
then, do concat_all and save to Nadlan_full_cities folder
then do recover_neighborhoods_for_all_cities(recover_XY=True)
then process each df_city
NEW SEARCH and PROCESS (search by NEIGHBORHOODs):
1) use neighborhood_city_code_counts.csv file to search for cities/neighborhoods
2) for each city_neighborhood: do body post and parse to dataframe
3) now go over each row and try to fill in cols:
    X, Y: from body requests with the FULLADRESS
    if no FULLADRESS try:
        get_XY_coords_using_GUSH and get_address_using_PARCEL_ID
4) now search for historic deals if TREND_FORMAT != ''


@author: ziskin
"""
# TODO: each call to body with street name yields district, city and street
# name and mid-street coords along with neiborhood based upon mid-street coords.
# however, if we request street and building number we get more precise location
# and neighborhood. We should re-run the body request for each line in the DataFrame
# and get the precise coords and nieghborhood
from MA_paths import work_david
nadlan_path = work_david / 'Nadlan_deals'
apts = ['דירה', 'דירה בבית קומות']

intel_kiryat_gat = [31.599645, 34.785265]


def transform_lat_lon_point_to_new_israel(lat, lon):
    from pyproj import Transformer
    from shapely.geometry import Point
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2039")
    p = transformer.transform(lon, lat)
    return Point(p)


intel_kiryat_gat_ITM = transform_lat_lon_point_to_new_israel(
    34.785265, 31.599645)


def produce_dfs_for_circles_around_point(point=intel_kiryat_gat_ITM,
                                         path=work_david, point_name='Intel_kiryat_gat',
                                         nadlan_path=nadlan_path,
                                         min_dis=0, max_dis=10, savepath=None):
    from Migration_main import path_glob
    from cbs_procedures import geo_location_settelments_israel
    import pandas as pd
    print('Producing nadlan deals with cities around point {} with radii of {} to {} kms.'.format(
        point, min_dis, max_dis))
    df = geo_location_settelments_israel(path)
    df_radii = filter_df_with_distance_from_point(df, point,
                                                  min_distance=min_dis, max_distance=max_dis)
    cities_radii = [x for x in df_radii['city_code']]
    files = path_glob(nadlan_path, '*/')
    available_city_codes = [x.as_posix().split('/')[-1] for x in files]
    available_city_codes = [int(x)
                            for x in available_city_codes if x.isdigit()]
    dfs = []
    cnt = 1
    for city_code in cities_radii:
        if city_code in available_city_codes:
            print(city_code)
            # print('found {} city.'.format(city_code))
            df = concat_all_nadlan_deals_from_one_city_and_save(
                city_code=int(city_code))
            if not df.empty:
                dfs.append(df)
                cnt += 1
    print('found total {} cities within {}-{} km radius.'.format(cnt, min_dis, max_dis))
    df = pd.concat(dfs, axis=0)
    if savepath is not None:
        filename = 'Nadlan_deals_around_{}_{}-{}.csv'.format(
            point_name, min_dis, max_dis)
        df.to_csv(savepath/filename, na_rep='None', index=False)
        print('{} was saved to {}.'.format(filename, savepath))
    return df


def produce_n_largest_cities_nadlan_time_series(df, col='NIS_per_M2', n=10):
    import pandas as pd
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    cities = get_all_city_codes_from_largest_to_smallest()
    inds = cities.iloc[0:n].index
    dfs = []
    for ind in inds:
        print('producing time seires for {}'.format(ind))
        city_df = df[df['city_code'] == ind]
        city_mean = city_df.set_index('DEALDATETIME').resample(
            'MS')[col].mean().rolling(12, center=True).mean()
        city_mean = city_mean.to_frame('{}_{}'.format(col, ind))
        city_mean.index.name = ''
        dfs.append(city_mean)
    df_cities = pd.concat(dfs, axis=1)
    return df_cities


def remove_outlier_by_value_counts(df, col, thresh=5, keep_nans=True):
    import pandas as pd
    nans = df[df[col].isnull()]
    vc = df[col].value_counts()
    vc = vc[vc > thresh]
    print('filtering df by allowing for minmum {} value counts in {}.'.format(thresh, col))
    vals = [x for x in vc.index]
    df = df[df[col].isin(vals)]
    if keep_nans:
        df = pd.concat([df, nans], axis=0)
    df = df.reset_index(drop=True)
    return df


def tries_requests_example():
    from requests.exceptions import RequestException, ConnectionError
    tries = 100
    for i in range(tries):
        try:
            for file in files:
                city_code = file.as_posix().split('/')[-1]
                try:
                    get_historic_deals_for_city_code(city_code=city_code)
                except FileNotFoundError:
                    continue
        except ConnectionError:
            if i < tries - 1:
                print('retrying...')
                sleep_between(5, 10)
                continue
            else:
                raise
            break

    # this one worked:
    tries = 100
    for i in range(tries):
        try:
            for c in reversed(cc):
                try:
                    get_all_no_street_nadlan_deals_from_settelment(
                        city_code=c, savepath=nadlan_path/str(c), pop_warn=None)
                except UnboundLocalError:
                    continue
        except RequestException:
            if i < tries - 1:
                print('retrying...')
                sleep_between(5, 10)
                continue
            else:
                raise
        break
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

def load_gush_parcel_shape_file(path=work_david/'gis'):
    import geopandas as gpd
    gdf = gpd.read_file(
        work_david/'gis/PARCEL_ALL - Copy.shp', encoding='cp1255')
    return gdf


def recover_XY_using_gush_parcel_shape_file(df_all, gush_gdf):
    import pandas as pd
    df_all = df_all.copy().reset_index(drop=True)
    city = df_all['City'].unique()[0]
    df = df_all[df_all['DescLayerID'] == 'SETL_MID_POINT']
    if df.empty:
        print('No SETL_MID_POINT XYs were found in dataframe...')
        return df_all
    df['lot'] = [x[0] for x in df['GUSH'].str.split('-')]
    df['parcel'] = [x[1] for x in df['GUSH'].str.split('-')]
    groups = df.groupby(['lot', 'parcel']).groups
    total = len(groups)
    cnt_all = 0
    cnt_gush = 0
    print('Found {} deals with non recovered XY for city: {}.'.format(total, city))
    # for i, (ind, row) in enumerate(df.iterrows()):
    for i, ((lot, parcel), inds) in enumerate(groups.items()):
        print('Fetching XY for {}-{} ({}/{}).'.format(lot, parcel, i+1, total))
        gdf = gush_gdf.query('GUSH_NUM=={} & PARCEL=={}'.format(lot, parcel))
        if gdf.empty:
            print('No XY were recovered for {}-{}, trying only GUSH'.format(lot, parcel))
            gdf = gush_gdf.query('GUSH_NUM=={}'.format(lot))
            if gdf.empty:
                print(
                    'No XY were recovered for {}-{} (even GUSH)...'.format(lot, parcel))
                continue
            df_all.loc[inds, 'X'] = gdf.unary_union.centroid.x
            df_all.loc[inds, 'Y'] = gdf.unary_union.centroid.y
            df_all.loc[inds, 'DescLayerID'] = 'XY_recovered_shape_gush_only'
            cnt_gush += 1
        else:
            df_all.loc[inds, 'X'] = gdf.centroid.x.item()
            df_all.loc[inds, 'Y'] = gdf.centroid.y.item()
            df_all.loc[inds, 'DescLayerID'] = 'XY_recovered_shape'
            cnt_all += 1
    print('Recovered exact {} XYs and {} GUSH XYs out of {} total records.'.format(
        cnt_all, cnt_gush, total))
    return df_all


def recover_neighborhoods_for_all_cities(path=nadlan_path/'Nadlan_full_cities',
                                         gush_file=None):
    from Migration_main import path_glob
    import numpy as np
    # first run concat_all...
    folders = path_glob(nadlan_path, '*/')
    folders = [x for x in folders if any(i.isdigit() for i in x.as_posix())]
    # files = path_glob(nadlan_path, 'Nadlan_deals_city_*_street_*.csv')
    city_codes = [x.as_posix().split('/')[-1] for x in folders]
    city_codes = sorted(np.unique(city_codes))
    print('found {} city codes: {}'.format(
        len(city_codes), ', '.join(city_codes)))

    for city_code in sorted(city_codes):
        try:
            recover_neighborhoods_for_one_city(path, city_code, gdf=gush_file)
        except FileNotFoundError:
            print('{} not found in {}, skipping...'.format(city_code, path))
            continue
    return


def recover_neighborhoods_for_one_city(path=nadlan_path/'Nadlan_full_cities',
                                       city_code=8700, gdf=None):
    from Migration_main import path_glob
    import pandas as pd
    file = path_glob(
        path, 'Nadlan_deals_city_{}_all_streets_*.csv'.format(city_code))
    file = file[0]
    if 'recovered_neighborhoods' in file.as_posix():
        print('{} already recovered_neighborhoods, skipping...'.format(file))
        return
    df = pd.read_csv(file, na_values='None')
    if gdf is not None:
        print('First, recovering XY using shape file:')
        df = recover_XY_using_gush_parcel_shape_file(df, gdf)
    print('Recovering neighborhoods in {}:'.format(city_code))
    df = recover_neighborhood_for_df(df)
    if df.empty:
        print('no empty neighborhoods in {} found.'.format(city_code))
        return
    filename = file.as_posix().split('/')[-1]
    filename = '_'.join(filename.split(
        '_')[:-1]) + '_recovered_neighborhoods' + '_h.csv'
    df.to_csv(path/filename, na_rep='None', index=False)
    print('Successfully saved {}'.format(filename))
    file.unlink()
    return


def recover_neighborhood_for_df(df_all):
    import numpy as np
    import pandas as pd
    df_all = df_all.copy().reset_index(drop=True)
    city = df_all['City'].unique()[0]
    df = df_all[df_all['Neighborhood'].isnull()]
    if df.empty:
        return pd.DataFrame()
    df['lot'] = [x[0] for x in df['GUSH'].str.split('-')]
    df['parcel'] = [x[1] for x in df['GUSH'].str.split('-')]
    groups = df.groupby(['lot', 'parcel']).groups
    total = len(groups)
    df.loc[:, 'Neighborhood'] = 'None'
    df.loc[:, 'Neighborhood_code'] = np.nan
    df.loc[:, 'Neighborhood_id'] = np.nan
    cnt = 0
    print('Found {} deals in empty neighborhoods for city: {}.'.format(total, city))
    for i, ((lot, parcel), inds) in enumerate(groups.items()):
        # for i, (ind, row) in enumerate(df.iterrows()):
        print('Fetching Nieghborhood for {}-{} ({}/{}).'.format(lot, parcel, i+1, total))
        x = df.loc[inds[0], 'X']
        y = df.loc[inds[0], 'Y']
        df_neigh = get_neighborhoods_area_by_XY(x, y)
        if df_neigh.empty:
            continue
        cnt += 1
        df.loc[inds, 'Neighborhood'] = df_neigh['Neighborhood'].item()
        df.loc[inds, 'Neighborhood_code'] = df_neigh['Neighborhood_code'].item()
        df.loc[inds, 'Neighborhood_id'] = df_neigh['Neighborhood_id'].item()
        sleep_between(0.05, 0.1)
    print('Recovered {} Neighborhoods out of {} total no streets records.'.format(
        cnt, total))
    df_all.loc[df.index, 'Neighborhood'] = df['Neighborhood']
    df_all.loc[df.index, 'Neighborhood_code'] = df['Neighborhood_code']
    df_all.loc[df.index, 'Neighborhood_id'] = df['Neighborhood_id']
    return df_all


def recover_XY_for_all_no_streets(nadlan_path=nadlan_path):
    from Migration_main import path_glob
    import numpy as np
    folders = path_glob(nadlan_path, '*/')
    folders = [x for x in folders if any(i.isdigit() for i in x.as_posix())]
    city_codes = [x.as_posix().split('/')[-1] for x in folders]
    city_codes = sorted(np.unique([int(x) for x in city_codes]))
    for city_code in city_codes:
        try:
            recover_XY_for_one_city(nadlan_path, city_code)
        except FileNotFoundError:
            continue
    print('Done recovering XY from no streets.')
    return


def recover_XY_for_one_city(nadlan_path=nadlan_path, city_code=8400):
    from Migration_main import path_glob
    import pandas as pd
    files = sorted(path_glob(nadlan_path/str(city_code),
                             'Nadlan_deals_city_{}_no_streets_*_h.csv'.format(city_code)))
    file = files[0]
    print('processing recovery of XY in city code: {}'.format(city_code))
    if 'recovered' not in file.as_posix() and len(files) == 2:
        file.unlink()
        print('deleted {}'.format(file))
        return
    elif 'recovered' in file.as_posix() and len(files) == 1:
        print('{} already recovered, skipping...'.format(file))
        return
    elif 'recovered' not in file.as_posix() and len(files) == 1:
        df = pd.read_csv(file, na_values='None')
        try:
            df = recover_XY_address_and_neighborhoods_using_GUSH(df)
        except AttributeError:
            print('nan detected in {}, deleting...'.format(file))
            file.unlink()
            return
        filename = file.as_posix().split('/')[-1]
        if '_recovered' not in filename:
            filename = '_'.join(filename.split(
                '_')[:-1]) + '_recovered' + '_h.csv'
        df.to_csv(nadlan_path/str(city_code) /
                  filename, na_rep='None', index=False)
        print('Successfully saved {}'.format(filename))
        # if '_recovered' not in filename:
        # now delete the old file:
        file.unlink()
    return


def recover_XY_address_and_neighborhoods_using_GUSH(df_no_streets):
    import numpy as np
    # assume we deal with no_streets_df
    df = df_no_streets.copy()
    total = len(df)
    city = df['City'].unique()[0]
    df['Neighborhood'] = 'None'
    df['Neighborhood_code'] = np.nan
    df['Neighborhood_id'] = np.nan
    cnt_arr = np.array([0, 0, 0])
    print('Found {} no streets deals for city: {}.'.format(total, city))
    for i, row in df_no_streets.iterrows():
        print('Fetching X, Y for {} ({}/{}).'.format(row['GUSH'], i+1, total))
        try:
            x, y, parcel_id = get_XY_coords_using_GUSH(row['GUSH'])
            df.at[i, 'X'] = x
            df.at[i, 'Y'] = y
            df.at[i, 'ObjectID'] = parcel_id
            sleep_between(0.05, 0.1)
            cnt_arr[0] += 1
            df.at[i, 'DescLayerID'] = 'XY_recovered'
        except ValueError:
            print('No X, Y found for {}.'.format(row['GUSH']))
            continue
        try:
            df_address = get_address_using_PARCEL_ID(parcel_id)
            df.at[i, 'FULLADRESS'] = df_address['FULLADRESS']
            df.at[i, 'Street'] = df_address['Street']
            sleep_between(0.05, 0.1)
            df.at[i, 'DescLayerID'] = 'ADDR_V1_recovered'
            cnt_arr[1] += 1
        except ValueError:
            print('No address found for {}.'.format(row['GUSH']))
            pass
        df_neigh = get_neighborhoods_area_by_XY(x, y)
        if df_neigh.empty:
            continue
        df.at[i, 'Neighborhood'] = df_neigh['Neighborhood'].item()
        df.at[i, 'Neighborhood_code'] = df_neigh['Neighborhood_code'].item()
        df.at[i, 'Neighborhood_id'] = df_neigh['Neighborhood_id'].item()
        sleep_between(0.05, 0.1)
        cnt_arr[2] += 1
    print('Recovered {} Xys, {} Addresses and {} Neighborhoods out of {} total no streets records.'.format(
        cnt_arr[0], cnt_arr[1], cnt_arr[2], total))
    return df


def get_neighborhoods_area_by_XY(X, Y):
    import requests
    import pandas as pd
    body = {'x': X,
            'y': Y,
            'mapTolerance': 39.6875793751587,
            'layers': [{'LayerType': 0,
                        'LayerName': 'neighborhoods_area',
                        'LayerFilter': ''}]}
    url = 'https://ags.govmap.gov.il/Identify/IdentifyByXY'
    r = requests.post(url, json=body)
    if not r.json()['data']:
        print('No nieghborhood found in {}, {}'.format(X, Y))
        return pd.DataFrame()
    results = r.json()['data'][0]['Result'][0]['tabs'][0]['fields']
    df = pd.DataFrame(results).T
    df.columns = ['Neighborhood', 'disc', 'City',
                  'Neighborhood_code', 'Neighborhood_id']
    df.drop('disc', axis=1, inplace=True)
    df.drop(['FieldName', 'FieldType'], axis=0, inplace=True)
    df = df.reset_index(drop=True)
    df['Neighborhood_code'] = pd.to_numeric(
        df['Neighborhood_code'], errors='coerce')
    df['Neighborhood_id'] = pd.to_numeric(
        df['Neighborhood_id'], errors='coerce')
    print('Successfuly recovered neighborhood as {}.'.format(
        df['Neighborhood'].item()))
    return df


def get_address_using_PARCEL_ID(parcel_id='596179', return_first_address=True):
    import requests
    import pandas as pd
    body = {'locateType': 3, 'whereValues': ["PARCEL_ID", parcel_id, "number"]}
    url = 'https://ags.govmap.gov.il/Search/SearchLocate'
    r = requests.post(url, json=body)
    df = pd.DataFrame(r.json()['data']['Values'])
    if df.empty:
        raise ValueError
    city = [x[0] for x in df['Values']]
    street = [x[1] for x in df['Values']]
    building = [int(x[2]) for x in df['Values']]
    df['City'] = city
    df['Street'] = street
    df['Building'] = building
    fa = []
    for i, row in df.iterrows():
        bi = row['Building']
        ci = row['City']
        st = row['Street']
        fa.append('{} {}, {}'.format(st, bi, ci))
    df['FULLADRESS'] = fa
    df.drop(['Values', 'Created', 'IsEditable'], axis=1, inplace=True)
    if return_first_address:
        print('Succsefully recoverd first address as {}'.format(
            df.iloc[0]['FULLADRESS']))
        return df.iloc[0]
    else:
        return df


def get_XY_coords_using_GUSH(GUSH='1533-149-12'):
    import requests
    g = GUSH.split('-')
    lot = g[0]  # גוש
    parcel = g[1]  # חלקה
    url = 'https://es.govmap.gov.il/TldSearch/api/DetailsByQuery?query=גוש {} חלקה {}&lyrs=262144&gid=govmap'.format(
        lot, parcel)
    r = requests.get(url)
    rdict = r.json()
    if rdict['ErrorMsg'] is not None:
        raise ValueError('could not find coords for {}: {}'.format(
            GUSH, rdict['ErrorMsg']))
    else:
        assert rdict['data']['GOVMAP_PARCEL_ALL'][0]['AData']['GUSH_NUM'] == lot
        assert rdict['data']['GOVMAP_PARCEL_ALL'][0]['AData']['PARCEL'] == parcel
        X = rdict['data']['GOVMAP_PARCEL_ALL'][0]['X']
        Y = rdict['data']['GOVMAP_PARCEL_ALL'][0]['Y']
        parcel_id = rdict['data']['GOVMAP_PARCEL_ALL'][0]['ObjectID']
        print('succsesfully recoverd X, Y for {}'.format(GUSH))
    # if get_address_too:
    #     df = get_address_using_PARCEL_ID(parcel_id)
    #     df['X'] = X
    #     df['Y'] = Y
    #     df['ObjectID'] = parcel_id
    #     return df
    # else:
    return X, Y, parcel_id


def filter_df_with_distance_from_point(df, point, min_distance=0,
                                       max_distance=10):
    import geopandas as gpd
    # ditance is in kms, but calculation is in meters:
    if not isinstance(df, gpd.GeoDataFrame):
        print('dataframe is not GeoDataFrame!')
        return
    print('fitering nadlan deals from {} to {} km distance from {}'.format(
        min_distance, max_distance, point))
    df['distance_to_point'] = df.distance(point) / 1000.0
    df = df.loc[(df['distance_to_point'] >= min_distance) &
                (df['distance_to_point'] < max_distance)]
    return df


def sleep_between(start=2, end=4):
    """sleep between start and end seconds"""
    from numpy import random
    from time import sleep
    sleeptime = random.uniform(start, end)
    print("sleeping for: {:.2f} seconds".format(sleeptime))
    sleep(sleeptime)
    # print("sleeping is over")
    return


def concat_all_processed_nadlan_deals_and_save(path=nadlan_path/'Nadlan_full_cities',
                                               savepath=work_david):
    from Migration_main import path_glob
    import pandas as pd
    files = path_glob(path, 'Nadlan_deals_processed_city_*_all_streets_*.csv')
    print('loading and concating {} cities.'.format(len(files)))
    dtypes = {'FULLADRESS': 'object', 'Street': 'object', 'FLOORNO': 'object',
              'NEWPROJECTTEXT': 'object', 'PROJECTNAME': 'object'}
    dfs = [pd.read_csv(file, na_values='None', parse_dates=['DEALDATETIME'],
                       dtype=dtypes) for file in files]
    df = pd.concat(dfs, axis=0)
    yrmin = df['DEALDATETIME'].min().year
    yrmax = df['DEALDATETIME'].max().year
    filename = 'Nadlan_deals_processed_{}-{}'.format(yrmin, yrmax)
    fcsv = filename + '.csv'
    fhdf = filename + '.hdf'
    df.to_csv(savepath/fcsv, na_rep='None', index=False)
    df.to_hdf(savepath/fhdf, complevel=9, mode='w', key='nadlan',
              index=False)
    print('{} was saved to {}.'.format(filename, path))
    return df


def process_nadlan_deals_from_all_cities(path=nadlan_path/'Nadlan_full_cities'):
    from Migration_main import path_glob
    import pandas as pd
    files = path_glob(path, 'Nadlan_deals_city_*_all_streets_*.csv')
    total = len(files)
    for i, file in enumerate(files):
        dtypes = {'FULLADRESS': 'string',
                  'Street': 'string', 'FLOORNO': 'string'}
        df = pd.read_csv(file, na_values='None', parse_dates=['DEALDATETIME'],
                         dtype=dtypes)
        city = df['City'].unique()[0]
        city_code = df['city_code'].unique()[0]
        print('processing {} ({}) ({}/{}).'.format(city, city_code, i+1, total))
        print('Found {} deals.'.format(len(df)))
        df = process_nadlan_deals_df_from_one_city(df)
        filename = 'Nadlan_deals_processed_' + \
            '_'.join(file.as_posix().split('/')[-1].split('_')[2:])
        df.to_csv(path/filename, na_rep='None', index=False)
        print('{} was saved to {}.'.format(filename, path))
    print('Done!')
    return


def process_nadlan_deals_df_from_one_city(df):
    """first attempt of proccessing nadlan deals df:
        removal of outliers and calculation of various indices"""
    from Migration_main import remove_outlier
    from cbs_procedures import read_statistical_areas_gis_file
    from cbs_procedures import read_social_economic_index
    import geopandas as gpd
    import numpy as np
    df = df.reset_index(drop=True)
    df = df[[x for x in df.columns if 'Unnamed' not in x]]
    # first, drop some cols:
    # df = df.drop(['NEWPROJECTTEXT', 'PROJECTNAME', 'TYPE',
    #               'POLYGON_ID', 'TREND_IS_NEGATIVE', 'TREND_FORMAT',
    #               'ObjectID', 'DescLayerID'], axis=1)
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
    if len(df) > 100:
        df = remove_outlier_by_value_counts(df, 'ASSETROOMNUM', thresh=5)
    df = remove_outlier(df, 'DEALAMOUNT')
    # calculate squared meters per room:
    df['M2_per_ROOM'] = df['DEALNATURE'] / df['ASSETROOMNUM']
    df['NIS_per_M2'] = df['DEALAMOUNT'] / df['DEALNATURE']
    # try to guess new buildings:
    df['New'] = df['BUILDINGYEAR'] == df['DEALDATETIME'].dt.year
    df['Age'] = df['DEALDATETIME'].dt.year - df['BUILDINGYEAR']
    # try to guess ground floors apts.:
    df['Ground'] = df['FLOORNO'].str.contains('קרקע')
    df = df.reset_index(drop=True)
    return df


def concat_all_nadlan_deals_from_all_cities_and_save(nadlan_path=work_david/'Nadlan_deals',
                                                     savepath=None, drop_dup=True,
                                                     delete_files=False):
    """concat all nadlan deals for all the cities in nadlan_path"""
    from Migration_main import path_glob
    import numpy as np
    folders = path_glob(nadlan_path, '*/')
    folders = [x for x in folders if any(i.isdigit() for i in x.as_posix())]
    # files = path_glob(nadlan_path, 'Nadlan_deals_city_*_street_*.csv')
    city_codes = [x.as_posix().split('/')[-1] for x in folders]
    city_codes = np.unique(city_codes)
    print('found {} city codes: {}'.format(
        len(city_codes), ', '.join(city_codes)))
    for city_code in sorted(city_codes):
        concat_all_nadlan_deals_from_one_city_and_save(nadlan_path=nadlan_path,
                                                       city_code=int(
                                                           city_code),
                                                       savepath=savepath,
                                                       delete_files=delete_files,
                                                       drop_dup=drop_dup)
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
    # from pathlib import Path
    streets = path_glob(main_path / str(city_code),
                        'Nadlan_deals_city_{}_*street*.csv'.format(city_code))
    # check for the h suffix and filter these files out (already done historic):
    streets = [x for x in streets if '_h' not in x.as_posix().split('/')[-1]]
    print('Found {} streets to check in {}.'.format(len(streets), city_code))
    cnt = 1
    for street in streets:
        filename = street.as_posix().split('.')[0] + '_h.csv'
        # if Path(filename).is_file():
        #     print('{} already exists, skipping...'.format(filename))
        #     cnt += 1
        #     continue
        df = pd.read_csv(street, na_values='None')
        try:
            df = get_historic_nadlan_deals_from_a_street_df(df, unique=True)
        except IndexError as e:
            print('{} in file: {}, deleting'.format(e, street))
            if 'no_streets' and 'nan' in street.as_posix():
                street.unlink()
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
    print('Fetching historic deals on {} street in {} (total {} assets).'.format(
        street, city, total_keys))
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
        print('No historic deals found for {}, skipping...'.format(street))
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
                                                   delete_files=False,
                                                   drop_dup=True):
    """concat all nadlan deals for all streets in a specific city"""
    import pandas as pd
    import numpy as np
    from Migration_main import path_glob
    import click
    city_path = nadlan_path / str(city_code)
    try:
        files = path_glob(
            city_path, 'Nadlan_deals_city_{}_*street*.csv'.format(city_code))
    except FileNotFoundError:
        return pd.DataFrame()
    dfs = [pd.read_csv(file, na_values='None') for file in files]
    df = pd.concat(dfs)
    df['DEALDATETIME'] = pd.to_datetime(df['DEALDATETIME'])
    df.drop(['DEALDATE', 'DISPLAYADRESS'], axis=1, inplace=True)
    print('concated all {} ({}) csv files.'.format(
        df['City'].unique()[0], city_code))
    df = df.sort_index()
    df = df.iloc[:, 2:]
    # first drop records with no full address:
    # df = df[~df['FULLADRESS'].isna()]
    # now go over all the unique GUSHs and fill in the geoloc codes if missing
    # and then drop duplicates
    # good_district = df['District'].unique()[df['District'].unique() != ''][0]
    try:
        good_district = df['District'].value_counts().index[0]
    except IndexError:
        good_district = np.nan
    good_city = df['City'].value_counts().index[0].strip()
    df = df.copy()
    if not pd.isnull(good_district):
        df.loc[:, 'District'].fillna(good_district, inplace=True)
    df.loc[:, 'City'].fillna(good_city, inplace=True)
    df['City'] = df['City'].str.strip()
    df = df[df['City'] == good_city]
    # take care of street_code columns all but duplicates bc of neighhood code/street code:
    if drop_dup:
        df = df.drop_duplicates(
            subset=df.columns.difference(['street_code', 'Street']))
    # lasty, extract Building number from FULLADRESS and is NaN remove record:
    df['Building'] = df['FULLADRESS'].astype(str).str.extract('(\d+)')
    # df = df[~df['Building'].isna()]
    if savepath is not None:
        yrmin = df['DEALDATETIME'].min().year
        yrmax = df['DEALDATETIME'].max().year
        filename = 'Nadlan_deals_city_{}_all_streets_{}-{}.csv'.format(
            city_code, yrmin, yrmax)
        df.to_csv(savepath/filename, na_rep='None')
        print('{} was saved to {}'.format(filename, savepath))
    if delete_files:
        msg = 'Deleting all {} city files, Do you want to continue?'.format(
            city_code)
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
    print('Done scraping {} city ({}) from nadlan.gov.il'.format(
        city_name, city_code))

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
                                         street_code=None,
                                         historic=False, add_extra_cols=True):
    # keep_no_address=False):
    """parse one request of nadlan deals JSON to pandas"""
    import pandas as pd
    if historic:
        df = pd.DataFrame(page)
    else:
        df = pd.DataFrame(page['AllResults'])
    # df.set_index('DEALDATETIME', inplace=True)
    # df.index = pd.to_datetime(df.index)
    df['DEALDATETIME'] = pd.to_datetime(df['DEALDATETIME'])
    df['DEALAMOUNT'] = df['DEALAMOUNT'].str.replace(',', '').astype(int)
    # M^2:
    df['DEALNATURE'] = pd.to_numeric(df['DEALNATURE'])
    # room number:
    df['ASSETROOMNUM'] = pd.to_numeric(df['ASSETROOMNUM'])
    # building year:
    df['BUILDINGYEAR'] = pd.to_numeric(df['BUILDINGYEAR'])
    # number of floors in building:
    df['BUILDINGFLOORS'] = pd.to_numeric(df['BUILDINGFLOORS'])
    # if new from contractor (1st hand):
    df['NEWPROJECTTEXT'] = pd.to_numeric(
        df['NEWPROJECTTEXT']).fillna(0).astype(bool)
    # id TREND_FORMAT != '' , check for historic deals
    if city_code is not None:
        df['city_code'] = city_code
    if street_code is not None:
        df['street_code'] = street_code
    # if keep_no_address:
    #     df = df[df['FULLADRESS']=='']
    if add_extra_cols:
        df = df.reindex(df.columns.tolist() + ['ObjectID', 'DescLayerID', 'X', 'Y', 'District', 'City', 'Neighborhood',
                                               'Street'], axis=1)
        df['DescLayerID'] = df['DescLayerID'].astype(str)
    return df


def produce_nadlan_rest_request(city='רעננה', street='אחוזה',
                                just_neighborhoods=True):
    # full_address=None):
    """produce the body of nadlan deals request, also usfull for geolocations"""
    import requests
    # if full_address is not None:
    # url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetDataByQuery?query={}'.format(full_address)
    # else:
    url = 'https://www.nadlan.gov.il/Nadlan.REST/Main/GetDataByQuery?query={} {}'.format(
        city, street)
    r = requests.get(url)
    body = r.json()
    if r.status_code != 200:
        raise ValueError('couldnt get a response.')
    # if body['ResultType'] != 1 and full_address is None:
    #     raise TypeError(body['ResultLable'])
    if just_neighborhoods:
        if body['DescLayerID'] != 'NEIGHBORHOODS_AREA':
            raise TypeError('result is not a NEIGHBORHOOD!, skipping')
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


def post_nadlan_sviva_rest(body, only_neighborhoods=True):
    """take body from request and post to nadlan.gov.il deals REST API,
    Sviva=other parameters, demographics, etc"""
    import requests
    if only_neighborhoods:
        if body['DescLayerID'] != 'NEIGHBORHOODS_AREA':
            raise ValueError('DescLayerID is {}.. only NEIGHBORHOODS_AREA allowed.'.format(body['DescLayerID']))
    url = 'https://www.nadlan.gov.il/Nadlan.REST//Mwa/GetPreInfo?subjects=127&pointBuffer=1500'
    r = requests.post(url, json=body)
    if r.status_code != 200:
        raise ValueError('couldnt get a response ({}).'.format(r.status_code))
    result = r.json()
    return result


def parse_body_request_to_dataframe(body):
    """parse body request to pandas"""
    import pandas as pd

    def parse_body_navs(body):
        navs = body['Navs']
        if navs is not None:
            nav_dict = {}
            order_dict = {1: 'District', 2: 'City',
                          3: 'Neighborhood', 4: 'Street'}
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


def get_all_no_street_nadlan_deals_from_settelment(city_code=1324,
                                                   pop_warn=5000, savepath=None):
    import os
    city_codes = get_all_city_codes_from_largest_to_smallest()
    city_name = city_codes.loc[city_code]['NameHe']
    if pop_warn is not None:
        pop = city_codes.loc[city_code]['Pop2015']
        if pop >= pop_warn:
            raise ValueError('population of {} is {} (>={})!'.format(
                city_name, pop, pop_warn))
    if not savepath.is_dir():
        print('Folder {} not found...Creating it.'.format(savepath))
        os.mkdir(savepath)
    get_all_no_street_nadlan_deals(
        city=city_name, city_code=city_code, savepath=savepath)
    return


def get_all_no_street_nadlan_deals(city='ארסוף', city_code=1324,
                                   savepath=None, check_for_downloaded_files=True):
    import pandas as pd
    import numpy as np
    import os
    # from json import JSONDecodeError
    # from requests.exceptions import ConnectionError
    from Migration_main import path_glob
    print('Fetching all NO streets nadlan deals in {} ({})'.format(city, city_code))
    if check_for_downloaded_files and savepath is not None:
        try:
            file = path_glob(
                savepath, 'Nadlan_deals_city_{}_no_streets_*.csv'.format(city_code))
            print('{} already found, skipping...'.format(file))
            return pd.DataFrame()
        except FileNotFoundError:
            pass
    try:
        body = produce_nadlan_rest_request(full_address=city)
    except TypeError:
        return None
    if body['DescLayerID'] != 'SETL_MID_POINT':
        print('searched term {} did not result in settelment but something else...'.format(city))
        return None
    try:
        page_files = path_glob(
            savepath/'page_temp', 'Nadlan_deals_city_{}_no_streets_page_*.csv'.format(city_code))
        pages = [x.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
                 for x in page_files]
        pages = sorted([int(x) for x in pages])
        cnt = int(pages[-1])
        print(cnt)
    except FileNotFoundError:
        cnt = 1
        pass
    body['PageNo'] = cnt
    page_dfs = []
    # cnt = 1
    last_page = False
    no_results = False
    while not last_page:
        print('Page : ', cnt)
        try:
            result = post_nadlan_rest(body)
        except TypeError:
            no_results = True
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
        filename = 'Nadlan_deals_city_{}_no_streets_page_{}.csv'.format(
            city_code, cnt)
        df1 = parse_one_json_nadlan_page_to_pandas(
            result, city_code, np.nan, keep_no_address=True)
        if not df1.empty:
            if not (savepath/'page_temp').is_dir():
                os.mkdir(savepath/'page_temp')
                print('{} was created.'.format(savepath/'page_temp'))
            df1.to_csv(savepath/'page_temp'/filename, na_rep='None')
            print('Page {} was saved to {}.'.format(cnt, savepath))
        page_dfs.append(df1)
        cnt += 1
        if result['IsLastPage']:
            last_page = True
        else:
            body['PageNo'] += 1
        no_results = False
    # now after finished, find all temp pages and concat them:
    page_files = path_glob(
        savepath/'page_temp', 'Nadlan_deals_city_{}_no_streets_page_*.csv'.format(city_code))
    page_dfs = [pd.read_csv(x, na_values=np.nan, parse_dates=[
                            'DEALDATETIME']) for x in page_files]
    df = pd.concat(page_dfs)
    df = df.reset_index()
    df = df.sort_index()
    # now some geocoding:
    loc_df = parse_body_request_to_dataframe(body)
    loc_df = loc_df.append([loc_df]*(len(df) - 1), ignore_index=True)
    df = pd.concat([df, loc_df], axis=1)
    # fill in district and city, street:
    df.loc[:, 'District'] = 'None'
    df.loc[:, 'City'] = city
    df.loc[:, 'Street'] = 'None'
    if savepath is not None:
        yrmin = df['DEALDATETIME'].min().year
        yrmax = df['DEALDATETIME'].max().year
        filename = 'Nadlan_deals_city_{}_no_streets_{}-{}.csv'.format(
            city_code, yrmin, yrmax)
        df.to_csv(savepath/filename, na_rep='None')
        print('{} was saved to {}.'.format(filename, savepath))
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
            file = path_glob(
                savepath, 'Nadlan_deals_city_{}_street_{}_*.csv'.format(city_code, street_code))
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
        good_district = df['District'].unique(
        )[df['District'].unique() != ''][0]
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


def read_neighborhood_city_file(path=work_david, file='neighborhood_city_code_counts.csv',
                                add_auto_complete=True):
    import pandas as pd
    import numpy as np
    from cbs_procedures import read_bycode_city_data
    df = pd.read_csv(path/file)
    df.columns = ['city_code', 'Neighborhood', 'to_drop']
    df = df[[x for x in df.columns if 'to_drop' not in x]]
    # filter שכונת:
    df['Neighborhood'] = df['Neighborhood'].str.replace('שכונת', '')
    df['Neighborhood'] = df['Neighborhood'].str.strip()
    if add_auto_complete:
        # add neighborhhod data from autocomplete:
        auto = pd.read_csv(path/'neighborhoods_auto_complete.csv')
        auto = auto.drop('Unnamed: 0', axis=1)
        dfs = []
        for cc in auto['city_code'].unique():
            if cc in df['city_code'].unique():
                nvals = pd.Series(auto[auto['city_code']==cc]['Value'].values)
                df_vals = pd.Series(df[df['city_code']==cc]['Neighborhood'].values)
                merged = pd.concat([nvals, df_vals]).drop_duplicates()
                mdf = merged.to_frame('Neighborhood').reset_index(drop=True)
                mdf['city_code'] = cc
                dfs.append(mdf)
            else:
                nvals = auto[auto['city_code']==cc]['Value'].values
                merged = pd.Series(nvals).drop_duplicates()
                mdf = merged.to_frame('Neighborhood').reset_index(drop=True)
                mdf['city_code'] = cc
                dfs.append(mdf)
        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)
    df = df.drop_duplicates()
    df['neighborhood_code'] = ''
    df['City'] = ''
    bycode = read_bycode_city_data(path)
    df['City'] = df['city_code'].map(bycode['NameHe'].to_dict())
    grps = df.groupby('city_code').groups
    for cc, inds in grps.items():
        n = len(inds)
        df.loc[inds, 'neighborhood_code'] = np.arange(1, n+1)
    df = df[['city_code', 'City', 'neighborhood_code', 'Neighborhood']]
    # fix *:
    df['City'] = df['City'].str.replace('*', '')
    df['City'] = df['City'].str.replace('נוף הגליל', 'נצרת עילית')
    df['Neighborhood'] = df['Neighborhood'].str.replace('א טור', 'א-טור')
    df['Neighborhood'] = df['Neighborhood'].str.replace('א רם', 'א-רם')
    df = df.dropna()
    return df


def neighborhoods_auto_complete_for_all_cities(path=work_david):
    import pandas as pd
    cities = get_all_city_codes_from_largest_to_smallest(path)
    dfs = []
    for city_code in cities.index:
        print('getting {} neighborhoods from autocomplete.'.format(city_code))
        df = auto_complete_neighborhoods_for_one_city(cities, path=path,
                                                      city_code=city_code)
        dfs.append(df)
        sleep_between(0.2, 0.35)
    df = pd.concat(dfs, axis=0)
    return df


def auto_complete_neighborhoods_for_one_city(city_df, path=work_david, city_code=8700):
    import requests
    city = city_df.loc[city_code]['NameHe']
    city = city.replace('-', ' ')
    search_term = '{} ,{}'.format('שכונה', city)
    url = 'https://www.nadlan.gov.il/TldSearch//api/AutoComplete?query={}&ids=16399'.format(search_term)
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError('couldnt get a response ({}).'.format(r.status_code))
    df = parse_autocomplete_neighbors(r.json())
    df['Value'] = df['Value'].str.replace(city, '')
    df['Value'] = df['Value'].str.strip()
    df['City'] = city
    df['city_code'] = city_code
    return df


def parse_autocomplete_neighbors(json):
    import pandas as pd
    try:
        df = pd.DataFrame(json['res']['NEIGHBORHOOD'])
        df = df.drop(['Data', 'Rank'], axis=1)
    except KeyError:
        df = pd.DataFrame(['', '']).T
        df.columns = ['Key', 'Value']
    return df


def process_one_page_from_neighborhoods_search(result_page):
    """take one result page from post_nadlan_rest, specifically
    neighborhoods searchs and process it"""
    import pandas as pd
    df = parse_one_json_nadlan_page_to_pandas(result_page)
    hdfs = []
    for i, row in df.iterrows():
        full_addr = row['FULLADRESS'].strip()
        disp_addr = row['DISPLAYADRESS'].strip()
        keyvalue = row['KEYVALUE']
        has_historic = row['TREND_FORMAT'] != ''
        if full_addr != '' and disp_addr != '':
            body = produce_nadlan_rest_request(*full_addr.split(','),
                                               just_neighborhoods=False)
            sleep_between(0.25, 0.35)
            df_extra = parse_body_request_to_dataframe(body)
            df.loc[i, 'ObjectID':'Street'] = df_extra.T[0]
        else:
            try:
                x, y, parcel_id = get_XY_coords_using_GUSH(row['GUSH'])
                df.at[i, 'X'] = x
                df.at[i, 'Y'] = y
                df.at[i, 'ObjectID'] = parcel_id
                df.at[i, 'DescLayerID'] = 'XY_recovered'
                sleep_between(0.05, 0.1)
            except ValueError:
                print('No X, Y found for {}.'.format(row['GUSH']))
                parcel_id = '0'
                pass
            try:
                df_address = get_address_using_PARCEL_ID(parcel_id)
                df.at[i, 'FULLADRESS'] = df_address['FULLADRESS']
                df.at[i, 'Street'] = df_address['Street']
                sleep_between(0.25, 0.35)
                df.at[i, 'DescLayerID'] = 'ADDR_V1_recovered'
            except ValueError:
                print('No address found for {}.'.format(row['GUSH']))
                pass
        # now get historic deals:
        if has_historic:
            try:
                result = post_nadlan_historic_deals(keyvalue)
                sleep_between(0.25, 0.35)
            except TypeError:
                continue
            df_historic = parse_one_json_nadlan_page_to_pandas(result, historic=True)
            for ii, roww in df_historic.iterrows():
                df_historic.loc[ii, 'ObjectID':'Street'] = df.loc[i, 'ObjectID':'Street']
            hdfs.append(df_historic)
    # if no historic deals at all:
    if hdfs:
        hdf = pd.concat(hdfs, axis=0)
        df = pd.concat([df, hdf], axis=0)
    return df


def process_all_city_nadlan_neighborhood_search(savepath=work_david/'Nadlan_deals_by_neighborhood',
                                                city_code=8700, ncode=None, path=work_david):
    import os
    from Migration_main import path_glob
    import numpy as np
    city_path = savepath / '{}'.format(city_code)
    if not (city_path).is_dir():
        os.mkdir(city_path)
        print('{} was created.'.format(city_path))

    ndf = read_neighborhood_city_file(path)
    city_ndf = ndf[ndf['city_code'] == city_code]
    ns = city_ndf['City'].size
    city = city_ndf['City'].unique().item()
    if ncode is None:
        print('processing city {} with {} neighborhoods.'.format(city, ns))
        try:
            n_files = path_glob(
                city_path, 'Nadlan_deals_city_*_neighborhood_*.csv'.format(city_code))
            ns = [x.as_posix().split('/')[-1].split('.')[0].split('_')[-2]
                  for x in n_files]
            ns = sorted([int(x) for x in ns])
            curr_n = int(ns[-1]) + 1
            print('last neighborhood found is {}, strating at {}.'.format(
                curr_n - 1, curr_n))
            n_codes = np.arange(curr_n, len(city_ndf['neighborhood_code']) + 1)
        except FileNotFoundError:
            n_codes = city_ndf['neighborhood_code']
            pass

        for n in n_codes:
            try:
                # print('neighborhood code: {}'.format(n))
                process_all_pages_for_nadlan_neighborhood_search(
                    city_path, ndf, city_code=city_code, neighborhood_code=n)
            except TypeError as e:
                print(e)
                continue
    else:
        process_all_pages_for_nadlan_neighborhood_search(
            city_path, ndf, city_code=city_code, neighborhood_code=ncode)
    return


def process_all_pages_for_nadlan_neighborhood_search(savepath, city_ndf,
                                                     city_code=8700,
                                                     neighborhood_code=1):
    from Migration_main import path_glob
    import pandas as pd
    import os
    import numpy as np
    city_df = city_ndf[city_ndf['city_code'] == city_code]
    city = city_df['City'].unique()[0]
    neighborhood = city_df[city_df['neighborhood_code']
                           == neighborhood_code]['Neighborhood'].item()
    try:
        page_files = path_glob(
            savepath/'page_temp', 'Nadlan_deals_city_{}_neighborhood_{}_*.csv'.format(city_code, neighborhood_code))
        pages = [x.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
                 for x in page_files]
        pages = sorted([int(x) for x in pages])
        cnt = int(pages[-1]) + 1
        print('last page found is {}, strating at {}.'.format(cnt-1, cnt))
    except FileNotFoundError:
        cnt = 1
        pass
    print('searching for {} neighborhood ({}) in {} ({}).'.format(neighborhood, neighborhood_code, city, city_code))
    body = produce_nadlan_rest_request(
        city, neighborhood, just_neighborhoods=True)
    body['PageNo'] = cnt
    page_dfs = []
    # cnt = 1
    last_page = False
    no_results = False
    # check if cnt -1 is the last page in body:
    b = body.copy()
    b['PageNo'] = cnt - 1
    try:
        result = post_nadlan_rest(b)
        if result['IsLastPage']:
            last_page = True
    except ValueError:
        pass
    while not last_page:
        print('fetching nadlan deals for {}  ({}) in {}, page : {}'.format(neighborhood, neighborhood_code, city, cnt))
        try:
            result = post_nadlan_rest(body)
        except TypeError:
            no_results = True
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
        filename = 'Nadlan_deals_city_{}_neighborhood_{}_page_{}.csv'.format(
            city_code, neighborhood_code, cnt)
        df = process_one_page_from_neighborhoods_search(result)
        if not df['City'].value_counts().empty:
            try:
                vc_city = [x for x in df['City'].value_counts().index if x != ''][0]
            except IndexError:
                vc_city = city
                pass
            try:
                assert vc_city == city
                df.loc[:, 'City'] = city
            except AssertionError:
                df.loc[:, 'City'] = ''
                pass
        if not df['Neighborhood'].value_counts().empty:
            try:
                vc_n = [x for x in df['Neighborhood'].value_counts().index if x != ''][0]
                n_from_search = vc_n.replace('שכונת', '')
                n_from_search = n_from_search.strip()
                n_from_search = n_from_search.replace('ורבורג', 'וורבורג')
            except IndexError:
                n_from_search = neighborhood
            try:
                assert n_from_search == neighborhood
                df.loc[:, 'Neighborhood'] = neighborhood
            except AssertionError:
                df.loc[:, 'Neighborhood'] = ''
        if not df.empty:
            if not (savepath/'page_temp').is_dir():
                os.mkdir(savepath/'page_temp')
                print('{} was created.'.format(savepath/'page_temp'))
            df.to_csv(savepath/'page_temp'/filename, na_rep='None', index=False)
            print('Page {} was saved to {}.'.format(cnt, savepath))
        page_dfs.append(df)
        cnt += 1
        if result['IsLastPage']:
            last_page = True
        else:
            body['PageNo'] += 1
        no_results = False
    # now after finished, find all temp pages and concat them:
    page_files = path_glob(
        savepath/'page_temp', 'Nadlan_deals_city_{}_neighborhood_{}_*.csv'.format(city_code, neighborhood_code))
    page_dfs = [pd.read_csv(x, na_values=np.nan, parse_dates=[
                            'DEALDATETIME']) for x in page_files]
    df = pd.concat(page_dfs)
    df = df.reset_index(drop=True)
    df = df.sort_index()
    if savepath is not None:
        yrmin = df['DEALDATETIME'].min().year
        yrmax = df['DEALDATETIME'].max().year
        filename = 'Nadlan_deals_city_{}_neighborhood_{}_{}-{}.csv'.format(
            city_code, neighborhood_code, yrmin, yrmax)
        df.to_csv(savepath/filename, na_rep='None', index=False)
        print('{} was saved to {}.'.format(filename, savepath))
    return df


def parse_floorno(path=work_david):
    import pandas as pd
    from text_to_num import alpha2digit
    import pandas as pd
    df = pd.read_csv(path/'floors_df_parsing.csv')
    df['en'] = df['en'].str.replace('Watch', 'Second')
    df['en'] = df['en'].str.replace('quarter', 'fourth')
    df['en'] = df['en'].str.replace('Second', '2')
    df['en'] = df['en'].str.replace('second', '2')
    df['en'] = df['en'].str.replace('First', '1')
    df['en'] = df['en'].str.replace('first', '1')
    df['en'] = df['en'].str.replace('land', 'ground')
    df['en'] = df['en'].str.replace('Land', 'ground')
    df['en'] = df['en'].str.replace('Ground', 'ground')
    df['en'] = df['en'].str.replace('foremost', '1')
    df['en'] = df['en'].str.replace('Third', '3')
    df['en'] = df['en'].str.replace('third', '3')
    df['en'] = df['en'].str.replace('floor', '')
    df['en'] = df['en'].str.replace('Floor', '')
    df['en'] = df['en'].str.replace('number', '')
    df['en'] = df['en'].str.replace('A', '1')
    df['en'] = df['en'].str.replace('B', '2')
    df['en'] = df['en'].str.replace('C', '3')
    df['en'] = df['en'].str.replace('D', '4')
    df['en'] = df['en'].str.replace("'", '')
    df['en'] = df['en'].str.replace(".", '')
    df['en'] = df['en'].str.replace("the", '')
    df['en'] = df['en'].str.replace("-", '')
    df['en'] = df['en'].str.replace("God", '5')
    df['en'] = df['en'].str.replace("heads", '1')
    df['en'] = df['en'].str.replace("Twentytwo", '22')
    df['en'] = df['en'].str.replace("Twentythree", '23')
    df['en'] = df['en'].str.replace("Twentyfive", '25')
    df['en'] = df['en'].str.replace('2asement', 'basement')
    df['try1'] = df['en'].apply(alpha2digit, lang='en')
    df['try1'] = df['try1'].str.replace('th', '')
    df['try1'] = df['try1'].str.replace('rd', '')
    df['FLOORNO'] = pd.to_numeric(df[df['try1'].str.isdigit()]['try1'])
    df['More_floors_1'] = df[df['FLOORNO'].isnull()]['try1'].str.extract(r'(\d{1,1})')
    df['More_floors_2'] = df[df['FLOORNO'].isnull()]['try1'].str.extract(r'(\d{2,3})')
    df['Roof'] = df['try1'].str.contains('roof')
    df['Ground'] = df['try1'].str.contains('ground')
    df['Basement'] = df['try1'].str.contains('basement')
    return df
