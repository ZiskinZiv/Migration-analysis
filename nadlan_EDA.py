#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 08:32:19 2021
revi take: plot time series of deal amount for SEI/P2015 clusters (5 or 10) on settelament level
and then on the right a map with corohpleths with mean/median value
for this, i need to prepare the muni shapefile with RC in it.
@author: shlomi
"""
from MA_paths import work_david
from shapely.geometry import *
nadlan_path = work_david / 'Nadlan_deals'
apts = ['דירה', 'דירה בבית קומות']
muni_path = work_david/'gis/muni_il'
dis_dict = {}
dis_dict['ירושלים'] = 1
dis_dict['הצפון'] = 2
dis_dict['חיפה'] = 3
dis_dict['המרכז'] = 4
dis_dict['תל אביב'] = 5
dis_dict['הדרום'] = 6
dis_dict['יו"ש'] = 7
dis_en = {1: 'Jerusalem', 2: 'North', 3: 'Haifa',
          4: 'Center', 5: 'Tel-Aviv', 6: 'South',
          7: 'J&S'}
P2015_2_dict = {1: 1,
                2: 1,
                3: 2,
                4: 2,
                5: 3,
                6: 4,
                7: 4,
                8: 5,
                9: 5,
                10: 5}

P2015_2_name = {1: 'Very Peripheral',
                2: 'Peripheral',
                3: 'In Between',
                4: 'Centralized',
                5: 'Very Centralized'}

def extract_JS_settelments_from_stat_areas(path=work_david, muni_path=muni_path):
    from cbs_procedures import read_statistical_areas_gis_file
    from cbs_procedures import read_bycode_city_data
    import pandas as pd
    import geopandas as gpd
    st = read_statistical_areas_gis_file(path)
    print('extrcting JS big settelments...')
    # J&S city codes from nadlan database:
    js_cc = [3780, 3616, 3730, 3797, 3760, 3570, 3769, 3640, 3720,
             3778]
    # js_st = st[st['city_code'].isin(js_cc)]
    ccs = st[st['city_code'].isin(js_cc)]['city_code'].unique()
    js = st[st['city_code'].isin(ccs)]
    sers = []
    for cc in ccs:
        cols = js[js['city_code']==cc].loc[:, ['city_code', 'NameHe', 'NameEn']]
        ser = gpd.GeoSeries(js[js['city_code']==cc]['geometry'].unary_union)
        ser['city_code'] = cols['city_code'].unique()[0]
        ser['NameHe'] = cols['NameHe'].unique()[0]
        ser['NameEn'] = cols['NameEn'].unique()[0]
        ser = ser.rename({0: 'geometry'})
        sers.append(ser)
    gdf = gpd.GeoDataFrame(sers, geometry='geometry')
    geos_to_complete = [1, 2, 4, 8, 9]
    city_codes_to_complete = [ccs[x] for x in geos_to_complete]
    bycode = read_bycode_city_data(path)
    names = [bycode.loc[x]['NameHe'] for x in city_codes_to_complete]
    js = gpd.read_file(muni_path/'JS_plans.shp')
    geos = [js[js['P_NAME']==x].geometry.unary_union for x in names]
    sers = []
    for i, geo in zip(geos_to_complete, geos):
        ser = gpd.GeoSeries(geo)
        ser['city_code'] = gdf.iloc[i]['city_code']
        ser['NameHe'] = gdf.iloc[i]['NameHe']
        ser['NameEn'] = gdf.iloc[i]['NameEn']
        ser = ser.rename({0: 'geometry'})
        sers.append(ser)
    gdf = gdf.drop(geos_to_complete, axis=0)
    gdf1 = gpd.GeoDataFrame(sers, geometry='geometry')
    gdf = pd.concat([gdf, gdf1], axis=0)
    gdf['district'] = 'יו"ש'
    return gdf


def prepare_just_city_codes_gis_areas(path=work_david, muni_path=muni_path):
    import geopandas as gpd
    import pandas as pd
    from cbs_procedures import read_statistical_areas_gis_file
    js = extract_JS_settelments_from_stat_areas(path, muni_path)
    js = js.drop('district', axis=1)
    js_ccs = js['city_code'].unique()
    st = read_statistical_areas_gis_file(path)
    ccs = st['city_code'].unique()
    sers = []
    ccs = [x for x in ccs if x not in js_ccs]
    for cc in ccs:
        geo = st[st['city_code'] == cc]['geometry'].unary_union
        geo = remove_third_dimension(geo)
        ser = gpd.GeoSeries(geo)
        ser['city_code'] = cc
        ser['NameHe'] = st[st['city_code'] == cc]['NameHe'].unique()[0]
        ser['NameEn'] = st[st['city_code'] == cc]['NameEn'].unique()[0]
        ser = ser.rename({0: 'geometry'})
        sers.append(ser)
    gdf_cc = gpd.GeoDataFrame(sers, geometry='geometry')
    gdf = pd.concat([gdf_cc, js], axis=0)
    gdf = gdf.set_index('city_code')
    filename = 'Municipal+J&S+city_code_level.shp'
    gdf.to_file(muni_path/filename, encoding='cp1255', index=True, na_rep='None')
    print('{} was saved to {}.'.format(filename, muni_path))
    return gdf


def prepare_municiapal_level_and_RC_gis_areas(path=work_david, muni_path=muni_path):
    import geopandas as gpd
    import pandas as pd
    js = extract_JS_settelments_from_stat_areas(path, muni_path)
    muni = gpd.read_file(path/'gis/muni_il/muni_il.shp')
    muni['city_code'] = pd.to_numeric(muni['CR_LAMAS'])
    muni['Machoz'] = muni['Machoz'].str.replace('צפון', 'הצפון')
    muni['Machoz'] = muni['Machoz'].str.replace('דרום', 'הדרום')
    muni['Machoz'] = muni['Machoz'].str.replace('מרכז', 'המרכז')
    muni_type_dict = {}
    muni_type_dict['עירייה'] = 'City'
    muni_type_dict['מועצה מקומית'] = 'LC'
    muni_type_dict['מועצה אזורית'] = 'RC'
    muni_type_dict['ללא שיפוט'] = 'NA'
    muni_type_dict['מועצה מקומית תעשייתית'] = 'ILC'
    muni['muni_type'] = muni['Sug_Muni'].map(muni_type_dict)
    muni['rc_code'] = muni[muni['muni_type'] ==
                           'RC']['CR_PNIM'].str[2:4].astype(int)
    print('aggragating polygons to city/rc level...')
    rc = muni[muni['muni_type'] == 'RC']
    non_rc = muni[muni['muni_type'] != 'RC']
    sers = []
    for nrc in rc['rc_code'].unique():
        geo = rc[rc['rc_code'] == nrc]['geometry'].unary_union
        geo = remove_third_dimension(geo)
        ser = gpd.GeoSeries(geo)
        ser['rc_code'] = nrc
        ser['NameHe'] = rc[rc['rc_code'] == nrc]['Muni_Heb'].unique()[0]
        ser['NameEn'] = rc[rc['rc_code'] == nrc]['Muni_Eng'].unique()[0]
        ser['district'] = rc[rc['rc_code'] == nrc]['Machoz'].unique()[0]
        ser = ser.rename({0: 'geometry'})
        sers.append(ser)
    gdf_rc = gpd.GeoDataFrame(sers, geometry='geometry')
    sers = []
    ccs = non_rc[~non_rc['city_code'].isnull()]['city_code'].unique()
    for cc in ccs:
        # print(cc)
        geo = non_rc[non_rc['city_code'] == cc]['geometry'].unary_union
        geo = remove_third_dimension(geo)
        ser = gpd.GeoSeries(geo)
        ser['city_code'] = cc
        ser['NameHe'] = non_rc[non_rc['city_code'] == cc]['Muni_Heb'].unique()[
            0]
        ser['NameEn'] = non_rc[non_rc['city_code'] == cc]['Muni_Eng'].unique()[
            0]
        ser['district'] = non_rc[non_rc['city_code'] == cc]['Machoz'].unique()[
            0]
        ser = ser.rename({0: 'geometry'})
        sers.append(ser)
    gdf_nonrc = gpd.GeoDataFrame(sers, geometry='geometry')
    gdf = pd.concat([gdf_rc, gdf_nonrc, js], axis=0)
    gdf.geometry = gdf.geometry.simplify(10)
    gdf = gdf.reset_index(drop=True)
    filename = 'Municipal+J&S+Regional.shp'
    gdf.to_file(muni_path/filename, encoding='cp1255')
    print('{} was saved to {}.'.format(filename, muni_path))
    return gdf


def remove_third_dimension(geom):
    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        exterior = geom.exterior
        new_exterior = remove_third_dimension(exterior)

        interiors = geom.interiors
        new_interiors = []
        for int in interiors:
            new_interiors.append(remove_third_dimension(int))

        return Polygon(new_exterior, new_interiors)

    elif isinstance(geom, LinearRing):
        return LinearRing([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, LineString):
        return LineString([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, Point):
        return Point([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, MultiPoint):
        points = list(geom.geoms)
        new_points = []
        for point in points:
            new_points.append(remove_third_dimension(point))

        return MultiPoint(new_points)

    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        new_lines = []
        for line in lines:
            new_lines.append(remove_third_dimension(line))

        return MultiLineString(new_lines)

    elif isinstance(geom, MultiPolygon):
        pols = list(geom.geoms)

        new_pols = []
        for pol in pols:
            new_pols.append(remove_third_dimension(pol))

        return MultiPolygon(new_pols)

    elif isinstance(geom, GeometryCollection):
        geoms = list(geom.geoms)

        new_geoms = []
        for geom in geoms:
            new_geoms.append(remove_third_dimension(geom))

        return GeometryCollection(new_geoms)

    else:
        raise RuntimeError("Currently this type of geometry is not supported: {}".format(type(geom)))


def create_israel_districts(path=muni_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import MultiPolygon, Polygon, LineString
    from shapely.ops import cascaded_union
    muni = gpd.read_file(path/'muni_il.shp')
    muni['Machoz'] = muni['Machoz'].str.replace('צפון', 'הצפון')
    muni['Machoz'] = muni['Machoz'].str.replace('דרום', 'הדרום')
    muni['Machoz'] = muni['Machoz'].str.replace('מרכז', 'המרכז')

    dists = muni['Machoz'].unique()
    sers = []
    for dis in dists:
        print(dis)
        # print(dis)
        geo = muni[muni['Machoz'] == dis].geometry.unary_union
        if isinstance(geo, MultiPolygon):
            eps = 0.01
            omega = cascaded_union([
                Polygon(component.exterior).buffer(eps).buffer(-eps) for component in geo
            ])
            geo = omega[0]
        geo = remove_third_dimension(geo)
        ser = gpd.GeoSeries(geo)
        ser = ser.rename({0: 'geometry'})
        # print(type(ser))
        ser['district'] = dis
        ser['district_EN'] = dis_en[dis_dict[dis]]
        ser['district_code'] = dis_dict[dis]
        bound = ser['geometry'].boundary
        if not isinstance(bound, LineString):
            ser['geometry'] = Polygon(bound[0])
        # ser['geometry'] = ser['geometry'].simplify(0.1)
        # ser.crs = muni.crs
        sers.append(ser)
    # now add J&S:
    js = gpd.read_file(path/'J&S_matakim.geojson')
    js = js.to_crs(2039)
    js1 = gpd.GeoSeries(js.geometry.unary_union)
    js1 = js1.rename({0: 'geometry'})
    js1['district'] = 'יו"ש'
    js1['district_EN'] = 'J&S'
    js1['district_code'] = 7
    js1 = gpd.GeoDataFrame([js1])
    b = js1.geometry.boundary.values[0]
    js1['geometry'] = Polygon(b[0])
    js1.index = [6]
    # sers.append(js1)
    dgf = gpd.GeoDataFrame(sers, geometry='geometry', crs=muni.crs)
    dgf = pd.concat([dgf, js1], axis=0)
    dgf = dgf.rename(
        {'district': 'NameHe', 'district_EN': 'NameEn', 'district_code': 'Code'}, axis=1)
    dgf.geometry = dgf.geometry.simplify(10)
    filename = 'Israel_districts_incl_J&S.shp'
    dgf.to_file(path/filename)
    print('{} was saved to {}.'.format(filename, path))
    return dgf


def create_higher_group_category(df, existing_col='SEI_cluster', n_groups=2,
                                 new_col='SEI2_cluster', names=None):
    import pandas as pd
    lower_group = sorted(df[existing_col].dropna().unique())
    new_group = [lower_group[i:i+n_groups+1]
                 for i in range(0, len(lower_group), n_groups+1)]
    new_dict = {}
    if names is not None:
        assert len(names) == len(new_group)
    for i, item in enumerate(new_group):
        if names is not None:
            new_dict[names[i]] = new_group[i]
        else:
            new_dict[i+1] = new_group[i]
    m = pd.Series(new_dict).explode().sort_values()
    d = {x: y for (x, y) in zip(m.values, m.index)}
    df[new_col] = df[existing_col].map(d)
    return df


def geolocate_nadlan_deals_within_city_or_rc(df, muni_path=muni_path,
                                             savepath=work_david):
    import geopandas as gpd
    import pandas as pd
    # run load_nadlan_combined_deal with return_XY and without add_geo_layers:
    gdf = gpd.read_file(muni_path/'Municipal+J&S+Regional.shp')
    print('geolocating nadlan deals within city or RC...')
    total = gdf.index.size
    keys = []
    for i, row in gdf.iterrows():
        print('index: {} / {}'.format(i, total))
        within = df.geometry.within(row['geometry'])
        if within.sum() == 0:
            print('empty..')
            continue
        inds = df.loc[within].index
        dff = pd.DataFrame(df.loc[inds, 'KEYVALUE'])
        dff['muni_gdf_index'] = i
        keys.append(dff)
    filename = 'Muni_gdf_KEYVALUE_index.csv'
    dff = pd.concat(keys, axis=0)
    dff.to_csv(savepath/filename, na_rep='None')
    print('Done!')
    return dff


def load_nadlan_combined_deal(path=work_david, times=['1998Q1', '2021Q1'],
                              dealamount_iqr=2, return_XY=False, add_bgr='Total',
                              add_geo_layers=True, add_mean_salaries=True,
                              add_neighborhood_uniq_code=True):
    import pandas as pd
    from Migration_main import path_glob
    import geopandas as gpd
    from cbs_procedures import read_statistical_areas_gis_file
    import numpy as np
    from cbs_procedures import read_mean_salary

    def add_bgr_func(grp, bgr, rooms='Total'):
        import numpy as np
        cc_as_str = str(grp['city_code'].unique()[0])
        try:
            gr = bgr.loc[cc_as_str][rooms]
        except KeyError:
            gr = np.nan
        grp['Building_Growth_Rate'] = gr
        return grp

    def add_stat_area_func(grp, stat_gdf):
        city_code11 = grp['city_stat_code'].unique()[0]
        geo = stat_gdf[stat_gdf['city_stat11']==city_code11].geometry.item()
        grp['stat_geo'] = [geo]*len(grp)
        return grp

    def add_district_area_func(grp, dis_df):
        district_code = grp['district_code'].unique()[0]
        geo = dis_df[dis_df['Code']==district_code].geometry.item()
        grp['district_geo'] = [geo]*len(grp)
        return grp

    def add_mean_salary_func(grp, sal):
        year = grp['year'].unique()[0]
        salary = sal[sal['year']==year]['mean_salary'].item()
        grp['mean_salary'] = [salary]*len(grp)
        return grp


    file = path_glob(
        path, 'Nadlan_deals_neighborhood_combined_processed_*.csv')
    dtypes = {'FULLADRESS': 'object', 'Street': 'object', 'FLOORNO': float,
              'NEWPROJECTTEXT': bool, 'PROJECTNAME': 'object', 'DEALAMOUNT': float}
    df = pd.read_csv(file[0], na_values='None', parse_dates=['DEALDATETIME'],
                     dtype=dtypes)
    # filter nans:
    df = df[~df['district'].isnull()]
    if times is not None:
        print('Slicing to times {} to {}.'.format(*times))
        # df = df[df['year'].isin(np.arange(years[0], years[1] + 1))]
        df = df.set_index('DEALDATETIME')
        df = df.loc[times[0]:times[1]]
        df = df.reset_index()
    if dealamount_iqr is not None:
        print('Filtering DEALAMOUNT with IQR of  {}.'.format(dealamount_iqr))
        df = df[~df.groupby('year')['DEALAMOUNT'].apply(
            is_outlier, method='iqr', k=dealamount_iqr)]
    df = df.reset_index(drop=True)
    print('loading gdf muni index...')
    df['P2015_cluster2'] = df['P2015_cluster'].map(P2015_2_dict)
    gdf_index = pd.read_csv(path/'Muni_gdf_KEYVALUE_index.csv', na_values='None')
    gdf_index = gdf_index.set_index('KEYVALUE')
    di = gdf_index['muni_gdf_index'].to_dict()
    df['gdf_muni_index'] = df['KEYVALUE'].map(di)
    if return_XY:
        inds = df[df['X'] == 0].index
        df.loc[inds, 'X'] = np.nan
        inds = df[df['Y'] == 0].index
        df.loc[inds, 'Y'] = np.nan
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['X'], df['Y']))
    if add_mean_salaries:
        sal = read_mean_salary()
        df = df.groupby('year').apply(add_mean_salary_func, sal)
        df['MSAL_per_ASSET'] = (df['DEALAMOUNT'] / df['mean_salary']).round()
    if add_bgr is not None:
        print('Adding Building Growth rate.')
        file = path_glob(path, 'Building_*_growth_rate_*.csv')[0]
        bgr = pd.read_csv(file, na_values='None', index_col='ID')
        df = df.groupby('city_code').apply(add_bgr_func, bgr, rooms=add_bgr)
        df.loc[df['Building_Growth_Rate'] == 0] = np.nan
    if add_geo_layers:
        print('adding statistical area geometry')
        stat_gdf = read_statistical_areas_gis_file(path)
        df = df.groupby('city_stat_code').apply(add_stat_area_func, stat_gdf)
        print('adding district area geometry')
        dis_df = gpd.read_file(path/'gis/muni_il/Israel_districts_incl_J&S.shp')
        df['district_code'] = df['district'].map(dis_dict)
        df = df.groupby('district_code').apply(add_district_area_func, dis_df)
    if add_neighborhood_uniq_code:
        print('adding neighborhood uniqid...')
        sviva = pd.read_csv(
            work_david/'Nadlan_neighborhoods_sviva_data.csv', na_values='None')
        for i, row in sviva.iterrows():
            name = row['NEIG_NAMEHE']
            uniq = row['NEIG_UNIQ_ID']
            inds = df[df['Neighborhood']==name].index
            df.loc[inds, 'Neighborhood_code'] = uniq
        df = df.rename({'Neighborhood_code': 'Neighborhood_uniqid'}, axis=1)
    return df


def load_nadlan_deals(path=work_david, csv=True,
                      times=['1998Q1', '2021Q1'], dealamount_iqr=2,
                      fix_new_status=True, add_SEI2_cluster=True,
                      add_peripheri_data=True, add_bycode_data=True
                      ):
    import pandas as pd
    import numpy as np
    from Migration_main import path_glob
    from cbs_procedures import read_periphery_index
    from cbs_procedures import read_bycode_city_data
    if csv:
        file = path_glob(path, 'Nadlan_deals_processed_*.csv')
        dtypes = {'FULLADRESS': 'object', 'Street': 'object', 'FLOORNO': 'object',
                  'NEWPROJECTTEXT': 'object', 'PROJECTNAME': 'object', 'DEALAMOUNT': float}
        df = pd.read_csv(file[0], na_values='None', parse_dates=['DEALDATETIME'],
                         dtype=dtypes)
    else:
        file = path_glob(path, 'Nadlan_deals_processed_*.hdf')
        df = pd.read_hdf(file)
    df['year'] = df['DEALDATETIME'].dt.year
    df['month'] = df['DEALDATETIME'].dt.month
    df['quarter'] = df['DEALDATETIME'].dt.quarter
    df['YQ'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    if times is not None:
        print('Slicing to times {} to {}.'.format(*times))
        # df = df[df['year'].isin(np.arange(years[0], years[1] + 1))]
        df = df.set_index('DEALDATETIME')
        df = df.loc[times[0]:times[1]]
        df = df.reset_index()
    if dealamount_iqr is not None:
        print('Filtering DEALAMOUNT with IQR of  {}.'.format(dealamount_iqr))
        df = df[~df.groupby('year')['DEALAMOUNT'].apply(
            is_outlier, method='iqr', k=dealamount_iqr)]
    if fix_new_status:
        inds = df.loc[(df['Age'] < 0) & (df['Age'] > -5)].index
        df.loc[inds, 'New'] = True
        df['NEWPROJECTTEXT'] = pd.to_numeric(df['NEWPROJECTTEXT']).fillna(0)
        df['NEWPROJECTTEXT'] = df['NEWPROJECTTEXT'].astype(bool)
    if add_SEI2_cluster:
        SEI_cluster = [x+1 for x in range(10)]
        new = [SEI_cluster[i:i+2] for i in range(0, len(SEI_cluster), 2)]
        SEI2 = {}
        for i, item in enumerate(new):
            SEI2[i+1] = new[i]
        m = pd.Series(SEI2).explode().sort_values()
        d = {x: y for (x, y) in zip(m.values, m.index)}
        df['SEI2_cluster'] = df['SEI_cluster'].map(d)
    if add_peripheri_data:
        pdf = read_periphery_index()
        cols = ['TLV_proximity_value', 'TLV_proximity_rank', 'PAI_value',
                'PAI_rank', 'P2015_value', 'P2015_rank', 'P2015_cluster']
        dicts = [pdf[x].to_dict() for x in cols]
        series = [df['city_code'].map(x) for x in dicts]
        pdf1 = pd.concat(series, axis=1)
        pdf1.columns = cols
        df = pd.concat([df, pdf1], axis=1)
    if add_bycode_data:
        bdf = read_bycode_city_data()
        cols = ['district', 'district_EN', 'region', 'natural_area']
        dicts = [bdf[x].to_dict() for x in cols]
        series = [df['city_code'].map(x) for x in dicts]
        bdf1 = pd.concat(series, axis=1)
        bdf1.columns = cols
        df = pd.concat([df, bdf1], axis=1)
    return df


def prepare_periphery_sei_index_map(path=work_david, muni_path=muni_path):
    from cbs_procedures import read_periphery_index
    from cbs_procedures import read_social_economic_index
    import geopandas as gpd
    df = read_periphery_index(path)
    sei = read_social_economic_index(path, return_stat=False)
    muni_gdf = gpd.read_file(muni_path/'Municipal+J&S+Regional.shp')
    # first put RC P2015 cluster, rank and value:
    rc = df[df['Type'] == 'RC']
    sei_rc = sei[sei['Type'] == 'RC']
    for rc_n in rc['municipal_status'].unique():
        ind = muni_gdf[muni_gdf['rc_code'] == rc_n].index
        muni_gdf.loc[ind, 'P2015_value'] = rc[rc['municipal_status']
                                              == rc_n]['P2015_value'].mean()
        muni_gdf.loc[ind, 'P2015_cluster'] = rc[rc['municipal_status']
                                                == rc_n]['RC_P2015_cluster'].unique()[0]
        muni_gdf.loc[ind, 'P2015_cluster2'] = rc[rc['municipal_status']
                                             == rc_n]['RC_P2_cluster'].unique()[0]
        muni_gdf.loc[ind, 'SEI_value'] = sei_rc[sei_rc['muni_state']
                                              == rc_n]['index2017'].mean()
        muni_gdf.loc[ind, 'SEI_cluster'] = sei_rc[sei_rc['muni_state']
                                              == rc_n]['RC_cluster2017'].unique()[0]
        muni_gdf.loc[ind, 'SEI2_cluster'] = sei_rc[sei_rc['muni_state']
                                              == rc_n]['RC_SEI2_cluster'].unique()[0]

    city = df[df['Type'] == 'City/LC']
    sei_city = sei[sei['Type'] == 'City/LC']
    for cc in city.index:
        ind = muni_gdf[muni_gdf['city_code'] == cc].index
        muni_gdf.loc[ind, 'P2015_value'] = city.loc[cc, 'P2015_value']
        muni_gdf.loc[ind, 'P2015_cluster'] = city.loc[cc, 'P2015_cluster']
        muni_gdf.loc[ind, 'P2015_cluster2'] = city.loc[cc, 'P2_cluster']
        muni_gdf.loc[ind, 'SEI_value'] = sei_city.loc[cc, 'index2017']
        muni_gdf.loc[ind, 'SEI_cluster'] = sei_city.loc[cc, 'cluster2017']
        muni_gdf.loc[ind, 'SEI2_cluster'] = sei_city.loc[cc, 'SEI2_cluster']
    return muni_gdf


def plot_mean_salary_per_asset(df, year=2000, rooms=[3, 4]):
    import geopandas as gpd
    import seaborn as sns
    from pysal.viz.splot.mapping import vba_choropleth
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.5)
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    print('picked {} only.'.format(apts))
    df = df[df['ASSETROOMNUM'].isin(rooms)]
    print('picked {} rooms only.'.format(rooms))
    gdf = gpd.GeoDataFrame(df, geometry='district_geo')
    print('picked year {}.'.format(year))
    gdf = gdf[gdf['year'] == year]
    # return gdf
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    gdf['district_counts'] = gdf.groupby('district')['DEALAMOUNT'].transform('count')
    x = gdf['MSAL_per_ASSET'].values
    y = gdf['district_counts'].values
    gdf.plot(ax=axs[0],
             column="MSAL_per_ASSET",
             legend=True,
             scheme='quantiles',
             cmap='Blues')
    # vba_choropleth(x, y, gdf, rgb_mapclassify=dict(classifier='quantiles'),
                   # alpha_mapclassify=dict(classifier='quantiles'),
                   # cmap='RdBu', ax=axs[1])
    fig.tight_layout()
    return fig


def calculate_pct_change_by_yearly_periods_and_grps(df,
                                                    period1=[1999, 2007],
                                                    period2=[2017, 2019],
                                                    col='DEALAMOUNT',
                                                    agg='median',
                                                    grp='gdf_muni_index',
                                                    min_p1_deals=50):
    print('calculating pct change for {} col using {} grouping and {} statistic.'.format(col, grp, agg))
    print('periods are: {}-{} compared to {}-{}'.format(period2[0], period2[1], period1[0], period1[1]))
    df1 = df.loc[(df['year']>=period1[0]) & (df['year']<=period1[1])]
    df2 = df.loc[(df['year']>=period2[0]) & (df['year']<=period2[1])]
    df1_agg = df1.groupby(grp).agg(agg)
    df1_cnt = df1.groupby(grp)[col].agg('count')
    df2_agg = df2.groupby(grp).agg(agg)
    df2_cnt = df2.groupby(grp)[col].agg('count')
    df_col = df2_agg[col] - df1_agg[col]
    df_col /= df1_agg[col]
    df_col *= 100
    df_col = df_col.round()
    df_col = df_col.to_frame('pct_change')
    df_col['period1_cnt'] = df1_cnt
    df_col['period2_cnt'] = df2_cnt
    if min_p1_deals is not None:
        print('filtering minimum deals of {} for {}-{} period.'.format(min_p1_deals, period1[0], period1[1]))
        df_col = df_col[df_col['period1_cnt']>=min_p1_deals]
    return df_col


def plot_choropleth_muni_level(df, rooms=[3, 4], muni_path=muni_path,
                               hue='SEI2_cluster',
                               col='NIS_per_M2'):
    # import geopandas as gpd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.dates as mdates
    import contextily as ctx

    sns.set_theme(style='ticks', font_scale=1.5)
    cmap = sns.dark_palette((260, 75, 60), input="husl", n_colors=5, as_cmap=True)
    cmap = sns.cubehelix_palette(5, gamma = 1, as_cmap=True)
    cmap = sns.cubehelix_palette(5, start = .5, rot = -.75, as_cmap=True)
    # cmap = sns.color_palette("Greens", 5, as_cmap=True)
    # colors = sns.color_palette("RdPu", 10)[5:]
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    df = df.loc[(df['year'] >= 1999) & (df['year'] <= 2019)]
    print('picked {} only.'.format(apts))
    if rooms is not None:
        df = df[(df['ASSETROOMNUM'] >= rooms[0]) &
                (df['ASSETROOMNUM'] <= rooms[1])]
        # df = df[df['ASSETROOMNUM'].isin(rooms)]
        print('picked {} rooms only.'.format(rooms))
    if col == 'NIS_per_M2':
        ylabel = r'Median price per M$^2$ [NIS]'
    if hue == 'SEI2_cluster':
        leg_title = 'Social-Economic cluster'
    elif hue == 'P2015_cluster2':
        leg_title = 'Periphery cluster'
    fig, ax = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(20, 10))
    # df['P2015_cluster2'] = df['P2015_cluster2'].map(P2015_2_name)
    # df = df.rename({'P2015_cluster2': 'Centrality level'}, axis=1)
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    sns.lineplot(data=df, x='year', y=col, hue=hue, n_boot=100,
                 palette=cmap, estimator="mean", ax=ax[0], ci=99,
                 style=hue, lw=2, seed=1)
    ax[0].grid(True)
    ax[0].set_ylabel(ylabel)
    gdf = prepare_periphery_sei_index_map(work_david, muni_path)
    gdf.crs = 2039
    gdf = gdf.to_crs(3857)
    gdf.plot(column=hue, categorical=True, legend=False,
             cmap=cmap, ax=ax[1], edgecolor='k', linewidth=0.25, alpha=0.9)
    handles, labels = ax[0].get_legend_handles_labels()
    labels = [int(float(x)) for x in labels]
    labels = ['{:d}'.format(x) for x in labels]
    ax[0].legend(handles=handles, labels=labels, title=leg_title)
    # leg.set_bbox_to_anchor((0.0, 1.0, 0.0, 0.0))
    # ax[1].set_axis_off()
    ax[0].set_xlabel('')
    ax[1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[0].xaxis.set_major_locator(mdates.YearLocator(2))
    ax[0].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.subplots_adjust(top=0.942,
                        bottom=0.089,
                        left=0.081,
                        right=0.987,
                        hspace=0.02,
                        wspace=0.0)
    ctx.add_basemap(ax[1], url=ctx.providers.Stamen.TerrainBackground)
    # for axis in ['top','bottom','left','right']:
    #     ax[1].spines[axis].set_linewidth(0.5)
    return fig


# def plot_choropleth_muni_level(df, rooms=[3, 4], muni_path=muni_path,
#                                muni_type='gdf_muni_index', min_p1=50,
#                                agg='median', col='NIS_per_M2'):
#     import geopandas as gpd
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # def add_muni_geo_func(grp, muni_gdf):
#     #     inds = grp['gdf_muni_index'].unique()[0]
#     #     geo = muni_gdf.loc[inds].geometry
#     #     grp['muni_geo'] = [geo]*len(grp)
#     #     return grp

#     sns.set_theme(style='ticks', font_scale=1.5)
#     df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
#     df = df.loc[(df['year']>=1999) & (df['year']<=2019)]
#     print('picked {} only.'.format(apts))
#     if rooms is not None:
#         df = df[(df['ASSETROOMNUM']>=rooms[0]) & (df['ASSETROOMNUM']<=rooms[1])]
#         # df = df[df['ASSETROOMNUM'].isin(rooms)]
#         print('picked {} rooms only.'.format(rooms))
#     df_pct = calculate_pct_change_by_yearly_periods_and_grps(df,
#                                                              grp=muni_type,
#                                                              min_p1_deals=min_p1,
#                                                              col=col,
#                                                              agg=agg)
#     fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(20, 10))
#     df['P2015_cluster2'] = df['P2015_cluster2'].map(P2015_2_name)
#     df = df.rename({'P2015_cluster2': 'Centrality level'}, axis=1)
#     sns.lineplot(data=df, x='year', y=col, hue='Centrality level', n_boot=100,
#                  palette='Set1',estimator=np.median, ax=ax[0],
#                  hue_order=[x for x in reversed(P2015_2_name.values())])
#     ax[0].grid(True)
#     if muni_type=='gdf_muni_index':
#         muni_gdf = gpd.read_file(muni_path/'Municipal+J&S+Regional.shp')
#     elif muni_type=='city_code':
#         muni_gdf = gpd.read_file(muni_path/'Municipal+J&S+city_code_level.shp')
#         muni_gdf = muni_gdf.set_index('city_code')
#     # df = df.groupby('gdf_muni_index').apply(add_muni_geo_func, muni_gdf)
#     # gdf = gpd.GeoDataFrame(df, geometry='muni_geo')
#     # inds = muni_gdf[muni_gdf.index.isin(df.index)].index
#     df_pct.loc[:, 'geometry'] = muni_gdf.loc[df_pct.index]['geometry']
#     gdf = gpd.GeoDataFrame(df_pct, geometry='geometry')
#     gdf[gdf['pct_change'] >= 0].plot('pct_change', legend=True, scheme="User_Defined",
#                                      k=5, cmap='viridis', classification_kwds=dict(bins=[50, 100, 150, 200, 250]),
#                                      ax=ax[1])
#     return gdf


def add_city_polygons_to_nadlan_df(df, muni_path=muni_path):
    import geopandas as gpd
    muni = load_muni_il(path=muni_path)
    ccs = list(set(df['city_code']).intersection(set(muni['CR_LAMAS'])))
    muni = muni[muni['CR_LAMAS'].isin(ccs)]
    muni = muni.reset_index(drop=True)
    df = df[df['city_code'].isin(ccs)]
    df = df.drop('geometry', axis=1)
    df = df.reset_index(drop=True)
    # TODO: fix this:
    for i, row in muni.iterrows():
        cc = row['CR_LAMAS']
        geo = row['geometry']
        inds = df[df['city_code']==cc].index
        df.loc[inds, 'geometry'] = [geo]*len(inds)
    df = gpd.GeoDataFrame(df,geometry='geometry')
    return df

def is_outlier(s, k=3, method='std'):
    # add IQR
    if method == 'std':
        lower_limit = s.mean() - (s.std() * k)
        upper_limit = s.mean() + (s.std() * k)
    elif method == 'iqr':
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3-q1  # Interquartile range
        lower_limit = q1 - k * iqr
        upper_limit = q3 + k * iqr
    return ~s.between(lower_limit, upper_limit)

# df1 = df[~df.groupby('year')['DEALAMOUNT'].apply(is_outlier)]


def keep_only_historic_changed_assets(df):
    df = df.reset_index(drop=True)
    grps = df.groupby('GUSH').groups
    inds = []
    for gush, ind in grps.items():
        if len(ind) > 1:
            inds.append([x for x in ind])
    flat_list = [item for sublist in inds for item in sublist]
    df_slice = df.loc[flat_list]
    return df_slice


def create_neighborhood_polygons(gdf):
    """ an attempt to muild neighborhoods polygons from asset points"""
    import numpy as np
    gdf = gdf.reset_index()
    neis = gdf['Neighborhood'].unique()
    gdf['neighborhood_shape'] = gdf.geometry
    # Must be a geodataframe:
    for nei in neis:
        gdf1 = gdf[gdf['Neighborhood'] == nei]
        inds = gdf1.index
        polygon = gdf1.geometry.unary_union.convex_hull
        # gdf.loc[inds, 'neighborhood_shape'] = [polygon for x in range(len(inds))]
        gdf.loc[inds, 'neighborhood_shape'] = polygon
    return gdf


def convert_da_to_long_form_df(da, var_name=None, value_name=None):
    """ convert xarray dataarray to long form pandas df
    to use with seaborn"""
    import xarray as xr
    if var_name is None:
        var_name = 'var'
    if value_name is None:
        value_name = 'value'
    dims = [x for x in da.dims]
    if isinstance(da, xr.Dataset):
        value_vars = [x for x in da]
    elif isinstance(da, xr.DataArray):
        value_vars = [da.name]
    df = da.to_dataframe()
    for i, dim in enumerate(da.dims):
        df[dim] = df.index.get_level_values(i)
    df = df.melt(value_vars=value_vars, value_name=value_name,
                 id_vars=dims, var_name=var_name)
    return df


def calculate_recurrent_times_and_pct_change(df, plot=True):
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    df = df.sort_values('DEALDATETIME')
    dff = df.groupby('GUSH')['DEALAMOUNT'].count()
    dff2 = df[df['GUSH'].isin(dff[dff > 1].index)]
    seconds_between_deals = dff2.groupby(
        'GUSH')['DEALDATETIME'].diff().dt.total_seconds()
    deals_pct_change = dff2.groupby('GUSH')['DEALAMOUNT'].pct_change()
    df['years_between_deals'] = seconds_between_deals / 60 / 60 / 24 / 365.25
    df['mean_years_between_deals'] = df.groupby(
        'GUSH')['years_between_deals'].transform('mean')
    df['deals_pct_change'] = deals_pct_change * 100
    df['mean_deals_pct_change'] = df.groupby(
        'GUSH')['deals_pct_change'].transform('mean')
    # drop duplicated dt's:
    deals_inds_to_drop = deals_pct_change[deals_pct_change == 0].index
    seconds_inds_to_drop = seconds_between_deals[seconds_between_deals == 0].index
    inds_to_drop = deals_inds_to_drop.union(seconds_inds_to_drop)
    df = df.drop(inds_to_drop, axis=0)
    print('Dropped {} deals'.format(len(inds_to_drop)))
    if plot:
        g = sns.JointGrid(data=df, x='years_between_deals',
                          y='deals_pct_change', height=7.5)
        g.plot_joint(sns.kdeplot, fill=True, cut=1, gridsize=100)
        g.plot_marginals(sns.histplot)
        g.ax_joint.grid(True)
        g.ax_joint.set_xlim(-1, 21)
        g.ax_joint.set_ylim(-100, 260)
        g.ax_joint.set_ylabel('Change in recurrent deals [%]')
        g.ax_joint.set_xlabel('Years between recurrent deals')
        g.fig.tight_layout()
        return g
    else:
        return df


def plot_recurrent_deals(df, max_number_of_sells=6, rooms=[2, 3, 4, 5]):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    sns.set_theme(style='ticks', font_scale=1.5)
    if rooms is not None:
        df = df[df['ASSETROOMNUM'].isin(rooms)]
        df = df.rename({'ASSETROOMNUM': 'Number of rooms'}, axis=1)
        df['Number of rooms'] = df['Number of rooms'].astype(int)
        df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
        dff = df.groupby(['GUSH', 'Number of rooms'])['DEALAMOUNT'].count()
        dff = dff[dff <= 6]
        df1 = dff.groupby('Number of rooms').value_counts()
        f, ax = plt.subplots(figsize=(7, 7))
        sns.lineplot(x='DEALAMOUNT', y=df1, data=df1, hue='Number of rooms',
                     style='Number of rooms', palette='Set1', ax=ax, markers=True,
                     markersize=15)
        ax.set(xscale="linear", yscale="log")
        ax.grid(True)
        ax.set_xlabel('Number of times an apartment is sold')
        ax.set_ylabel('Total deals')
    else:
        df1 = df[df['DEALNATUREDESCRIPTION'].isin(apts)].groupby('GUSH')[
            'DEALAMOUNT'].count()
        v = np.arange(1, max_number_of_sells + 1)
        n = [len(df1[df1 == x]) for x in v]
        dfn = pd.DataFrame(n, index=v)
        dfn.columns = ['Number of Deals']
        f, ax = plt.subplots(figsize=(7, 7))
        ax = sns.scatterplot(x=dfn.index, y='Number of Deals',
                             data=dfn, ax=ax, s=50)
        p = np.polyfit(v, np.log(n), 1)
        fit = np.exp(np.polyval(p, v))
        print(fit)
        dfn['Fit'] = fit
        ax = sns.lineplot(x=dfn.index, y='Fit', data=dfn, ax=ax, color='r')
        ax.set(xscale="linear", yscale="log")
        ax.grid(True)
        ax.set_xlabel('Number of times an apartment is sold')
    return f


def plot_deal_amount_room_number(df, rooms=[2, 3, 4, 5],
                                 path=nadlan_path, yrmin='2000', yrmax='2020',
                                 just_with_historic_change=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from cbs_procedures import read_bycode_city_data
    import numpy as np
    sns.set_theme(style='ticks', font_scale=1.5)
    # df = df.loc[(df['ASSETROOMNUM'] >= room_min) &
    #             (df['ASSETROOMNUM'] <= room_max)]
    df = df[df['ASSETROOMNUM'].isin(rooms)]
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
    city_name = bycode[bycode['city_code'] == city_code]['NameEn'].values[0]
    fig.suptitle('Real-Estate prices in {}'.format(city_name))
    return ax


def plot_groupby_m2_price_time_series(df, grps=['City', 'Neighborhood'],
                                      col='NIS_per_M2'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    groups = grps.copy()
    groups.append('year')
    dfn = df.groupby(groups, as_index=False)[col].mean().groupby('year').mean()
    std = df.groupby(groups, as_index=False)[
        col].mean().groupby('year').std()[col].values
    dfn['plus_std'] = dfn[col] + std
    dfn['minus_std'] = dfn[col] - std
    fig, ax = plt.subplots(figsize=(14, 5))
    dfn[col].plot(ax=ax)
    ax.fill_between(dfn.index, dfn['minus_std'], dfn['plus_std'], alpha=0.3)
    ax.set_ylabel(r'NIS per M$^2$')
    fig.tight_layout()
    ax.grid(True)
    return fig


def plot_room_number_deals(df, rooms_range=[2, 6]):
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    if rooms_range is not None:
        df = df.loc[(df['ASSETROOMNUM'] >= rooms_range[0]) &
                    (df['ASSETROOMNUM'] <= rooms_range[1])]
    dff = df.groupby(['ASSETROOMNUM', 'year'])['DEALAMOUNT'].count()
    da = dff.to_xarray()
    dff = convert_da_to_long_form_df(da, value_name='Deals')
    ax = sns.barplot(data=dff, x='year', y='Deals',
                     hue='ASSETROOMNUM', palette='Set1')
    ax.grid(True)
    return dff


def plot_price_per_m2(df, n_boot=100, hue='SEI2_cluster', y='NIS_per_M2'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.5)
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    df['Q'] = pd.to_datetime(df['YQ'])
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(data=df, x='Q', y=y, hue=hue,
                 ci=95, n_boot=n_boot, palette='Set1', ax=ax)
    ax.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel(r'Apartment price per m$^2$')
    fig.tight_layout()
    return


def compare_kiryat_gat_israel_dealamount(df_kg, df_isr):
    # TODO: complete this
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    df_kg = df_kg.groupby(['rooms', 'YQ'])[
        'DEALAMOUNT'].mean().to_frame().unstack().T.droplevel(0)
    df_isr = df_isr.groupby(['rooms', 'YQ'])[
        'DEALAMOUNT'].mean().to_frame().unstack().T.droplevel(0)
    # df_kg['YQ'] = df_kg.index
    # df_isr['YQ'] = df_isr.index
    # df_kg = df_kg.melt(id_vars='YQ', value_name='price_in_kg')
    # df_isr = df_isr.melt(id_vars='YQ', value_name='price_in_israel')
    # df = pd.concat([df_kg, df_isr], axis=1)
    df = df_kg / df_isr
    # df['price_diff'] = df['price_in_kg'] - df['price_in_israel']
    fig, ax = plt.subplots(figsize=(15.5, 6))
    df1 = df  # / 1e6
    df1.index = pd.to_datetime(df1.index)
    df1.plot(ax=ax, cmap=sns.color_palette("tab10", as_cmap=True))
    ax.set_ylabel('Price difference [million NIS]')
    df2 = df1.rolling(4, center=True).mean()
    df2.columns = ['{} rolling mean'.format(x) for x in df2.columns]
    df2.plot(ax=ax, cmap=sns.color_palette("tab10", as_cmap=True), ls='--')
    ax.axvline(pd.to_datetime('2008-07-01'), color='g')
    ax.axvline(pd.to_datetime('2006-01-01'), color='r')
    ax.grid(True)
    ax.set_xlabel('')
    fig.suptitle('Kiryat Gat - Israel mean apartment prices')
    fig.tight_layout()
    return df


def plot_bootstrapped_dff_rooms(dff, time='year'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(15, 5))
    dff['price'] = dff['DEALAMOUNT'] / 1e6
    dff[time] = pd.to_datetime(dff[time])
    sns.lineplot(data=dff, x=time, y='price', ci='sd', hue='rooms',
                 ax=ax, style='rooms')
    ax.grid(True)
    ax.set_xlabel('')

    # ax.set_xticks(np.arange(1998, 2021, 2))
    ax.set_ylabel('Apartment price [millions NIS]')
    fig.suptitle('Mean apartment prices in Israel')
    fig.tight_layout()
    return fig


def prepare_bootstrapped_dfs_by_year_and_apts(df, nr=1000, frac=0.05,
                                              city_code=None, grp='year'):
    # for all Israel i did nr=5000, frac=0.35
    import pandas as pd
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    if city_code is not None:
        df = df[df['city_code'] == city_code]
        print('{} city selected ({})'.format(
            df['City'].unique()[0], city_code))
    groups = [[2, 3], [3, 4], [4, 5]]
    grp_dfs = []
    for apt_group in groups:
        print('{} rooms selected.'.format(
            ','.join([str(x) for x in apt_group])))
        df1 = df[df['ASSETROOMNUM'].isin(apt_group)]
        dff = bootstrap_df_by_year(
            df1, col='DEALAMOUNT', n_replicas=nr, frac_deals=frac,
            grp=grp)
        dff['rooms'] = '-'.join([str(x) for x in apt_group])
        grp_dfs.append(dff)
    return pd.concat(grp_dfs, axis=0)


def bootstrap_df_by_year(df, grp='year', frac_deals=0.1, col='NIS_per_M2',
                         n_replicas=1000, plot=False):
    import pandas as pd
    import seaborn as sns
    count = df.groupby(grp)[col].count()
    dffs = []
    for year in count.index:
        samples = count.loc[year] * frac_deals
        print('bootstrapping year {} with {} samples.'.format(year, samples))
        df1 = df[df[grp] == year][col].dropna()
        # print(len(df1))
        # dff = bootstrap_df(df1, n_rep=n_replicas,
        # frac=frac_deals, n_sam=None, col=col, grp=None)
        dff = pd.Series([df1.sample(n=None, frac=frac_deals, replace=True,
                        random_state=None).mean() for i in range(n_replicas)])
        dff = dff.to_frame(year).reset_index(drop=True)
        dffs.append(dff)
    stats = pd.concat(dffs, axis=1).melt(var_name=grp, value_name=col)
    if plot:
        sns.lineplot(data=stats, x=grp, y=col, ci='sd')
    return stats


# def bootstrap_df(df, n_rep=1000, n_sam=1000, frac=None, grp='year', stat='mean',
#                  col='NIS_per_M2'):
#     import pandas as pd
#     print('bootstapping the {} from {} replicas of {} samples from {} in df.'.format(
#         stat, n_rep, n_sam, col))
#     if grp is not None:
#         print('{} groupby chosen.'.format(grp))
#         stats = pd.DataFrame([df.groupby(grp).sample(
#             n=n_sam, frac=frac, replace=True, random_state=None).groupby(grp)[col].agg(stat) for i in range(n_rep)])
#         stats = stats.melt(value_name=col)
#     else:
#         stats = pd.Series([df[col].sample(n=n_sam, frac=frac, replace=True, random_state=None).agg(stat) for i in range(n_rep)])

#     return stats


def load_muni_il(path=work_david/'gis/muni_il', union=True):
    import geopandas as gpd
    import pandas as pd
    muni = gpd.read_file(path/'muni_il.shp')
    muni['CR_LAMAS'] = pd.to_numeric(muni['CR_LAMAS']).astype('Int64')
    muni['CR_PNIM'] = pd.to_numeric(muni['CR_PNIM']).astype('Int64')
    groups = muni.groupby('CR_LAMAS').groups
    geo = []
    ser = []
    for i, (cc, inds) in enumerate(groups.items()):
        ser.append(muni.loc[inds].iloc[0])
        geo.append(muni.loc[inds].geometry.unary_union)
    gdf = gpd.GeoDataFrame(ser, geometry=geo)
    gdf['geometry'] = geo
    return gdf


def run_lag_analysis_boi_interest_nadlan(ndf, idf, i_col='effective', months=48):
    ndf = ndf.set_index('DEALDATETIME')
    ndf_mean = ndf['DEALAMOUNT'].resample('M').mean()
    idf = idf[i_col].to_frame()
    idf = idf.rolling(6, center=True).mean()
    idf['apt_prices'] = ndf_mean.rolling(6, center=True).mean()
    for i in range(months):
        idf['{}_{}'.format(i_col, i+1)] = idf[i_col].shift(-i-1)
    return idf
