#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:03:54 2020
Directions for network proccessing:
    1) run read_and_write_as_hdf(path=path to your Place-to-place migration-IL.xlsb)
    2) use df returned from 1) or run df=pd.read_hdf(path from 1 /
#                                                   'Migration_data_IL.hdf')
    3) run G = build_directed_graph(df, year, level, return_json)
    year can be any number from 2000 to 2017 or a list with two values (e.g., [2000, 2005])
    level can be :'district', 'county' or 'city',
    you can also use return_json=True in order to export the G networkx object
    into JSON.
    4) if you want, use cent_df= centrality_analysis(G, weight_col) to 
    create centrality indicies with ot without weights (e.g., weight='Percent-migrants')
    5) there is a plot_network function, but it is for the district, county level
    (the city level works though the outcome is not that nice)
@author: shlomi
"""
# TODO: add weight selection with what david wanted
# TODO: add docstrings
from MA_paths import work_david
from MA_paths import savefig_path
import matplotlib.ticker as ticker
gis_path = work_david / 'gis'



level_dict = {
    'district': {'source': 'OutDistrict', 'target': 'InDistrict'},
    'county': {'source': 'OutCounty', 'target': 'InCounty'},
    'city': {'source': 'OutID', 'target': 'InID'}
}


def linear_map(arr, lb=0, ub=1):
    import numpy as np
    dh = np.max(arr)
    dl = np.min(arr)
    # print dl
    arr = (((arr - dl) * (ub - lb)) / (dh - dl)) + lb
    return arr


def create_G_with_df(df, level='city', graph_type='directed'):
    import networkx as nx

    def feed_edges_attrs_to_G(dfs, degree='in'):
        # first add edges from out to in:
        if degree == 'out':
            s = [x for x in dfs[source]]
            t = [x for x in dfs[target]]
        elif degree == 'in':
            s = [x for x in dfs[target]]
            t = [x for x in dfs[source]]
        # outpop = [x for x in out_df['OutPop']]
        number = [x for x in dfs['Number']]
        # total = [x for x in dfs['Total']]
        pmigrants = [x for x in dfs['Percent-migrants']]
        outpercent = [x for x in dfs['Outpercent']]
        attrs = []
        for num, pmig, outp in zip(number, pmigrants, outpercent):
            attrs.append(dict(number=num, percent_migrants=pmig,
                              outpercent=outp))
        for sour, targ, attr in zip(s, t, attrs):
            G.add_edge(sour, targ, **attr)
        return G

    # build graph with graph type:
    if graph_type == 'directed':
        G = nx.DiGraph()
    elif graph_type == 'multi-directed':
        G = nx.MultiDiGraph()
    else:
        raise("Only 'directed' and 'multi-directed' graph_type are allowed!")
    # get hirarchy:
    source = level_dict.get(level)['source']
    target = level_dict.get(level)['target']
    # select only inflow:
    in_df = df[df['Direction'] == 'inflow']
    # get nodes:
    nodes = [x for x in in_df[target].unique()] + \
        [x for x in in_df[source].unique()]
    nodes = list(set(nodes))
    G.add_nodes_from(nodes)
    # seperate in/out dfs:
    # out_df = df[df['Direction'] == 'outflow']
    # G = feed_edges_attrs_to_G(out_df, degree='out')
    G = feed_edges_attrs_to_G(in_df, degree='in')
    return G


def array_scale(arr, lower=0.2, upper=10):
    """scales arr to be between lower and upper"""
    import numpy as np
    arr = np.array(arr)
    maxi = arr.max()
    mini = arr.min()
    data = (((arr-mini)*(upper-lower))/(maxi-mini))+lower
    return data
#
#
#def bin_data(arr, bins=None):
#    from scipy import stats
#    import numpy as np
#    if bins is None:
#        bins = np.linspace(10,100,10)
#    ranked = stats.rankdata(np.array(arr))
#    data_percentile = ranked / len(arr) * 100
#    data_binned_indices = np.digitize(data_percentile, bins, right=True)
#    return data_binned_indices


def get_out_id_greater_than_1(dfs_in, field='id', year=2000, savepath=None):
    """ask david about how to fix the lines,
    This procedure gives the problamtic interaction of inid and out id"""
    import pandas as pd
#    ids_list = []
    if field == 'id':
        in_f = 'InID'
        out_f = 'OutID'
    elif field == 'name':
        in_f = 'InEN'
        out_f = 'OutEN'
    df_out = []
    for in_id in dfs_in[in_f]:
        vc = dfs_in[dfs_in[in_f] == in_id][out_f].value_counts()
        out_ids_ser = vc[vc > 1]
        if out_ids_ser.any():
            df_out.append(dfs_in[dfs_in[in_f] == in_id][dfs_in[out_f].isin(out_ids_ser.index.to_list())])
#            ids_list += out_ids_ser.index.tolist()
#    return list(set(ids_list))
    try:
        df = pd.concat(df_out)
    except ValueError:
        print('duplicates in {} not found, skipping...'.format(year))
        return
    df = df.drop_duplicates()
    df.index.name = 'line_number'
    if savepath is not None:
        print('saving year {}'.format(year))
        df.to_csv(savepath / 'duplicate_migration_IL_{}_ids.csv'.format(year))
    return df


def read_geo_name_cities(path=work_david):
    import pandas as pd
    geo = pd.read_excel(
        work_david /
        'Place-to-place migration-IL.xlsb',
        engine='pyxlsb', sheet_name='geo')
    geo = geo.loc[:, 'ID':'LON']
    geo['Rank'] = geo['Rank'].fillna(0)
    geo['ITM-X'] = geo['ITM-X'].fillna(0)
    geo['ITM-Y'] = geo['ITM-Y'].fillna(0)
    geo = geo.set_index('ID2').dropna()
    geo = geo.loc[~geo.index.duplicated(keep='first')]
    geo['ID'] = geo['ID'].astype(int)
    return geo


def read_and_write_as_hdf(path=work_david):
    """reads David's original binary excell file and writes a pandas HDF file,
    path is the same for reading and writing"""
    import pandas as pd
    df = pd.read_excel(
            work_david /
            'Place-to-place migration-IL.xlsb',
            engine='pyxlsb', sheet_name='raw')
    print('found naming error...fixing.')
    df = df.rename({'OuLAT': 'OutLAT'}, axis=1)
    filename = 'Migration_data_IL.hdf'
    print('saving {} as HDF file to {}'.format(filename, work_david))
    df.to_hdf(
        work_david /
        filename,
        complevel=9,
        mode='w',
        key='migration')
    return df


def load_migration_df(path=work_david, direction='inflow'):
    import pandas as pd
    df = pd.read_hdf(path / 'Migration_data_IL.hdf')
    print('{} direction selected.'.format(direction))
    df = df[df['Direction'] == direction]
    return df


def choose_year(df, year=2000, dropna=True, verbose=True):
    """return a pandas DataFrame with the selected years.
    Input:  df : original big pandas DataFrame,
            year : either 1 values or two values (start, end)
            dropna : drop NaN's
            verbose : verbosity
    Output: sliced df"""
    import numpy as np
    if dropna:
        df = df.dropna()
    df = df[df['Year'] != 'ALL']
    if isinstance(year, int) or isinstance(year, str) or isinstance(year, np.int64):
        sliced_df = df.query('Year=={}'.format(year), engine='python')
    else:
        if len(year) == 2:
            cyear_min = int(year[0])
            cyear_max = int(year[1])
            if cyear_min > cyear_max:
                raise('chosen minimum year is later than maximum year!')
            sliced_df = df.query('Year>={} & Year<={}'.format(cyear_min, cyear_max), engine='python')
        elif len(year) == 1:
            year = year[0]
            sliced_df = df.query('Year=={}'.format(year), engine='python')
        else:
            raise('if year is list it should be with length 2!')
    minyear = sliced_df['Year'].unique().astype(int).min()
    maxyear = sliced_df['Year'].unique().astype(int).max()
    if minyear == maxyear:
        if verbose:
            print('chosen year {}'.format(minyear))
    else:
        if verbose:
            print('chosen years {}-{}'.format(minyear, maxyear))
    return sliced_df


def build_directed_graph(df, path=work_david, year=2000, level='district',
                         graph_type='directed', return_json=False):
    import networkx as nx
    import numpy as np
    from networkx import NetworkXNotImplemented
    """Build a directed graph with a specific level hierarchy and year/s.
    Input:  df: original pandas DataFrame
            year: selected year/s
            level: district, county or city
            return_json: convert networkx DiGraph object toJSON object and
            return it.
    Output: G: networkx DiGraph object or JSON object"""
    print('Building {} graph with {} hierarchy level'.format(graph_type, level))
#    source = level_dict.get(level)['source']
#    target = level_dict.get(level)['target']
    df_sliced = choose_year(df, year=year, dropna=True)
    node_sizes = node_sizes_source_target(df, year=year, level=level)
#    node_geo = get_lat_lon_from_df_per_year(df, year=year, level=level)
#    if weight_col is not None:
#        df['weights'] = normalize(df[weight_col], 1, 10)
#    else:
#        df['weights'] = np.ones(len(df))
    # df = df[df['Percent-migrants'] != 0]
    G = create_G_with_df(df_sliced, level=level, graph_type=graph_type)
#    G = nx.from_pandas_edgelist(
#        df_sliced,
#        source=source,
#        target=target,
#        edge_attr=[
#            source,
#            'Percent-migrants',
#            'Direction',
#            'Number',
#            'Total',
#            'Distance',
#            'Angle'],
#        create_using=Graph)
    # enter geographical coords as node attrs:
    geo = read_geo_name_cities(path=path)
    for col in geo.columns:
        dict_like = dict(zip([x for x in G.nodes()], [
                         geo.loc[x, col] for x in G.nodes()]))
        nx.set_node_attributes(G, dict_like, name=col)
    # slice df for just inflow:
    df_in = df_sliced[df_sliced['Direction'] == 'inflow']
    # calculate popularity index:
    pi_dict = calculate_poplarity_index_for_InID(df_in)
    # get the total number of immgrants to city:
    totals = []
    for node in [x for x in G.nodes()]:
        valc = df_in[df_in['InID']==node]['Total'].value_counts()
        if valc.empty:
            totals.append(0)
        else:
            totals.append(valc.index[0])
    total_dict = dict(zip([x for x in G.nodes()], totals))
    # set some node attrs:
    nx.set_node_attributes(G, total_dict, 'total_in')
    nx.set_node_attributes(G, pi_dict, 'popularity')
    nx.set_node_attributes(G, node_sizes, 'size')
#    nx.set_node_attributes(G, node_geo, 'coords_lat_lon')
    G.name = 'Israeli migration network'
    G.graph['level'] = level
    G.graph['year'] = year
    G.graph['density'] = nx.density(G)
    try:
        G.graph['triadic_closure'] = nx.transitivity(G)
    except NetworkXNotImplemented as e:
        print('nx.transitivity {}'.format(e))
    # G.graph['global_reaching_centrality'] = nx.global_reaching_centrality(G, weight=weight_col)
    # G.graph['average_clustering'] = nx.average_clustering(G, weight=weight_col)
#    if weight_col is not None:
#        print('adding {} as weights'.format(weight_col))
#        # add weights:
#        edgelist = [x for x in nx.to_edgelist(G)]
#        weighted_edges = [
#            (edgelist[x][0],
#             edgelist[x][1],
#             edgelist[x][2][weight_col]) for x in range(
#                len(edgelist))]
#        G.add_weighted_edges_from(weighted_edges)
    print(nx.info(G))
    for key, val in G.graph.items():
        if isinstance(val, float):
            print(key + ' : {:.2f}'.format(val))
        else:
            print(key + ' :', val)
    # G, metdf = calculate_metrics(G, weight_col=weight_col)
    if return_json:
        return nx.node_link_data(G)
    else:
        return G


def calculate_poplarity_index_for_InID(df_in):
    all_migrants = df_in['Number'].sum()
    pi_key = []
    pi_val = []
    for inid in df_in['InID'].unique():
        all_inid = df_in[df_in['InID'] == inid]['Number'].sum()
        pi = (all_migrants - all_inid) / all_inid
        pi_key.append(inid)
        pi_val.append(pi)
    pi_dict = dict(zip(pi_key, pi_val))
    return pi_dict


def calculate_centrality_to_dataframe(G, centrality='in_degree',
                                      weight_col='Percent-migrants'):
    """create centrality indicies for directed graph G.
    Input: G: networkx DiGraph object
           centrality: type of centrality test to preform
           weight_col: apply weights that should exist in G.edges attributes
           choose None to run without weights
    Output: pandas DataFrame with centrality as columns and nodes as index."""
    import pandas as pd
    import networkx as nx
    print('preforming {} centrality analysis'.format(centrality))
    if centrality == 'in_degree':
        d = dict(G.in_degree(G.nodes(), weight=weight_col))
    elif centrality == 'out_degree':
        d = dict(G.out_degree(G.nodes(), weight=weight_col))
    elif centrality == 'degree':
        d = dict(G.degree(G.nodes(), weight=weight_col))
    elif centrality == 'eigenvector':
        d = dict(nx.eigenvector_centrality(G, weight=weight_col))
    # beware, katz has convergence issues:
    elif centrality == 'katz':
        d = dict(nx.katz_centrality(G, weight=weight_col))
    elif centrality == 'closeness':
        d = dict(nx.closeness_centrality(G, distance='distance'))
    elif centrality == 'betweenness':
        d = dict(nx.betweenness_centrality(G, weight=weight_col))
    elif centrality == 'load':
        d = dict(nx.load_centrality(G, weight=weight_col))
    elif centrality == 'harmonic':
        d = dict(nx.harmonic_centrality(G, distance='distance'))
    # now, non-centrality methods such as clustering, etc...
    elif centrality == 'clustering':
        d = dict(nx.clustering(G, weight=weight_col))    
    nx.set_node_attributes(G, d, centrality)
    df = pd.DataFrame([x for x in d.values()], index=[
                      x for x in d.keys()], columns=[centrality])
    df = df.sort_values(centrality, ascending=False)
    return df


def centrality_analysis(G, weight_col='percent_migrants'):
    """main function to run centrality analysis.
    Input: G: networkx DiGraph object
           weight_col: apply weights that should exist in G.edges attributes
           choose None to run without weights
    Output: pandas DataFrame with centrality as columns and nodes as index."""
    from networkx import NetworkXNotImplemented
    print('Weight chosen: {}'.format(weight_col))
    df = calculate_centrality_to_dataframe(G, 'in_degree', weight_col=weight_col)
    df['out_degree'] = calculate_centrality_to_dataframe(
        G, 'out_degree', weight_col=weight_col)
    df['degree'] = calculate_centrality_to_dataframe(
        G, 'degree', weight_col=weight_col)
    try:
        df['eigenvector'] = calculate_centrality_to_dataframe(
                G, 'eigenvector', weight_col=weight_col)
    except NetworkXNotImplemented as e:
        print('nx.eigenvector_centrality {}'.format(e))
    df['closeness'] = calculate_centrality_to_dataframe(
        G, 'closeness', weight_col=weight_col)
    try:
        df['betweenness'] = calculate_centrality_to_dataframe(
                G, 'betweenness', weight_col=weight_col)
    except NetworkXNotImplemented as e:
        print('nx.betweenness_centrality {}'.format(e))
    df['load'] = calculate_centrality_to_dataframe(G, 'load', weight_col=weight_col)
    df['harmonic'] = calculate_centrality_to_dataframe(
        G, 'harmonic', weight_col=weight_col)
    try:
        df['clustering'] = calculate_centrality_to_dataframe(
                G, 'clustering', weight_col=weight_col)
    except NetworkXNotImplemented as e:
        print('nx.clustering {}'.format(e))
    return df


def power_law_fit_and_plot(cent_df, col='in_degree'):
    import powerlaw
    data = cent_df[col].dropna() + 1
    fit = powerlaw.Fit(data, discrete=True, xmin=1)
    ####
    figCCDF = fit.plot_pdf(color='b', linewidth=0, marker='o')
    # ax = powerlaw.plot_pdf(data, color='g', linewidth=0, marker='o')
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=figCCDF)
    fit.plot_ccdf(color='r', linewidth=0, marker='x', ax=figCCDF)
    fit.power_law.plot_ccdf(color='r', linestyle='--', ax=figCCDF)
    ####
    print(fit.alpha)
    figCCDF.set_ylabel(u"p(X),  p(X≥x)")
    figCCDF.set_xlabel(r"{} Frequency".format(col))
    return

#def calculate_metrics(G, sort='degree', weight_col='Percent-migrants'):
#    import networkx as nx
#    import pandas as pd
#    metrics = {'density': nx.density(G),
#               'triadic_closure': nx.transitivity(G)}
#    degree_dict = dict(G.degree(G.nodes(), weight=weight_col))
#    nx.set_node_attributes(G, degree_dict, 'degree')
#    betweenness_dict = nx.betweenness_centrality(
#        G, weight=weight_col)  # Run betweenness centrality
#    eigenvector_dict = nx.eigenvector_centrality(
#        G, weight=weight_col)  # Run eigenvector centrality
#    # Assign each to an attribute in your network
#    nx.set_node_attributes(G, betweenness_dict, 'betweenness')
#    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
#    G.graph.update(metrics)
#    nodes = [x for x in G.nodes]
#    metdf = pd.DataFrame([x for x in betweenness_dict.values()], index=nodes,
#                          columns=['betweenness'])
#    metdf['eigenvector'] = [x for x in eigenvector_dict.values()]
#    metdf['degree'] = [x for x in degree_dict.values()]
#    if sort is not None:
#        metdf = metdf.sort_values(sort, ascending=False)
#    return G, metdf


#def show_metrics(G, metric='degree', top=20):
#    from operator import itemgetter
#    nodes = [x for x in G.nodes]
#    metrics = [G.nodes[x][metric] for x in nodes]
#    met_dict = dict(zip(nodes, metrics))
#    sorted_metrics = sorted(met_dict.items(), key=itemgetter(1), reverse=True)
@ticker.FuncFormatter
def lon_formatter(x, pos):
    if x < 0:
        return r'{0:.1f}$\degree$W'.format(abs(x))
    elif x > 0:
        return r'{0:.1f}$\degree$E'.format(abs(x))
    elif x == 0:
        return r'0$\degree$'


@ticker.FuncFormatter
def lat_formatter(x, pos):
    if x < 0:
        return r'{0:.1f}$\degree$S'.format(abs(x))
    elif x > 0:
        return r'{0:.1f}$\degree$N'.format(abs(x))
    elif x == 0:
        return r'0$\degree$'

def plot_network_on_israel_map(G=None, gis_path=gis_path, fontsize=18,
                               save=True):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.io.shapereader import Reader
    import networkx as nx
    import numpy as np
    import matplotlib.ticker as mticker
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,LatitudeLocator)
    import geopandas as gpd
#    isr_with_yosh = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')

    fname = gis_path / 'Israel_and_Yosh.shp'
    east = 36
    west = 34
    north = 34
    south = 29
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
#    ax.add_geometries(Reader(fname).geometries(),
#                      ccrs.PlateCarree(), zorder=0,
#                      edgecolor='k', facecolor='w')
    # Put a background image on for nice sea rendering.

    ax.add_feature(cfeature.LAND.with_scale('10m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    ax.add_feature(cfeature.LAKES.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
#    fig, ax = plt.subplots(figsize=(7, 20))
#    ax = isr_with_yosh.plot(ax=ax)
#    ax.xaxis.set_major_locator(mticker.MaxNLocator(2))
#    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
#    ax.yaxis.set_major_formatter(lat_formatter)
#    ax.xaxis.set_major_formatter(lon_formatter)
#    ax.tick_params(top=True, bottom=True, left=True, right=True,
#                   direction='out', labelsize=ticklabelsize)

    deg = nx.degree(G)
    labels = {node: node if deg[node] >= 200 else ''
              for node in G.nodes}
    pos = {key:(value['LON'], value['LAT']) for (key,value) in G.nodes().items()}
    pop_sizes = np.array([G.nodes()[x]['size'] for x in G.nodes()])
    pop_sizes = linear_map(pop_sizes, 10, 50)
    popularity = np.array([G.nodes()[x].get('popularity', 0) for x in G.nodes()])
    popularity = linear_map(popularity, 0, 100)
    total = np.array([G.nodes()[x].get('total_in', 0) for x in G.nodes()])
    total_width = linear_map(total, 0.1, 5)
    number = np.array([G.edges()[x].get('number', 0) for x in G.edges()])
    bins = [0, 250, 500, 1000, 2500]
    subgraphs = get_subgraph_list(G, attr={'edge': 'number'}, bins=bins)
    evenly_spaced_interval = np.linspace(0, 1, len(bins))
    colors = [plt.cm.gist_rainbow(x) for x in evenly_spaced_interval]
    widths = np.linspace(0.045, 4, len(bins))
    for i, subgraph in enumerate(subgraphs):
        pos = {key:(value['LON'], value['LAT']) for (key,value) in subgraph.nodes().items()}
        nx.draw_networkx_edges(subgraph, ax=ax, pos=pos, alpha=1.0, width=widths[i],
                               edge_color=colors[i], arrows=False, edge_cmap=None)
    leg_labels = get_bins_legend(bins)
    leg = plt.legend(leg_labels, loc='upper left', fontsize=fontsize)
    leg.set_title('Number of immegrants',prop={'size': fontsize-2})
    # ec = nx.draw_networkx_edges(G, ax=ax, pos=pos, alpha=0.5, width=total_width,
    #                             edge_color='k', arrows=False, edge_cmap=plt.cm.plasma)
    # nc = nx.draw_networkx_nodes(G, ax=ax, pos=pos, nodelist=G.nodes(),
    #                             node_size=pop_sizes, node_color=total,
    #                         cmap=plt.cm.autumn)
    # nc.set_zorder(2)
    # ec.set_zorder(1)
    # plt.colorbar(nc)
#    nx.draw_networkx(G, ax=ax,
#                     font_size=10,
#                     alpha=.5,
#                     width=.045,
#                     edge_color='r',
#                     node_size=pop_sizes,
#                     labels=labels,
#                     pos=pos,
#                     node_color=popularity,
#                     cmap=plt.cm.autumn)
    year = G.graph['year']
    name = G.graph['name']
    fig.suptitle('{} for year {}'.format(name, year), fontsize=fontsize)
#    ax.coastlines(resolution='50m')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels=False
    gl.right_labels=False
    # gl.left_labels=False
    # gl.bottom_labels=False
    gl.ylocator = LatitudeLocator()
    gl.xlocator = mticker.MaxNLocator(2)  
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    fig.canvas.draw()
    ysegs = gl.yline_artists[0].get_segments()
    yticks = [yseg[0,1] for yseg in ysegs]
    xsegs = gl.xline_artists[0].get_segments()
    xticks = [xseg[0,0] for xseg in xsegs]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    
    # ax.xaxis.set_major_locator(mticker.MaxNLocator(2))
    # ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
    # ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
#    ax.set_yticks([29, 30, 31, 32, 33], crs=ccrs.PlateCarree())
#    lon_formatter = LongitudeFormatter(zero_direction_label=True)
#    lat_formatter = LatitudeFormatter()
#    ax.xaxis.set_major_formatter(lon_formatter)
#    ax.yaxis.set_major_formatter(lat_formatter)
#    ax.xaxis.set_major_locator(ticker.MaxNLocator(2))
#    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
#    ax.yaxis.set_major_formatter(lat_formatter)
#    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(top=True, bottom=True, left=True, right=False,
                   direction='out', labelbottom=False, labelleft=False, labelsize=fontsize)

#    ax.figure.tight_layout()

    fig.tight_layout()
    fig.subplots_adjust(top=0.943,
                        bottom=0.038,
                        left=0.008,
                        right=0.885,
                        hspace=0.2,
                        wspace=0.2)
    if save:
        filename = 'Migration_israel_map_{}.png'.format(year)
        plt.savefig(savefig_path / filename, bbox_inches='tight', orientation='portrait')
    return gl


def plot_network(G, edge_width='percent_migrants'):
    import matplotlib.pyplot as plt
    import networkx as nx
    # nodedict = node_sizes_source_target(df, year=year, level=level)
    node_sizes = [G.nodes[x]['size'] / 70 for x in G.nodes()]
    full_nodes = [x for x in G.nodes()]
    assert len(node_sizes) == len(full_nodes)
    # nodedict = calculate_node_size_per_year(df, year=year, level=level)
    fig, ax = plt.subplots(figsize=(20, 20))
    node_cmap = plt.get_cmap('tab20', len(full_nodes))
    node_colors = [node_cmap(x) for x in range(len(full_nodes))]
    nodecolors_dict = dict(zip(full_nodes, node_colors))
    source = level_dict.get(G.graph['level'])['source']
#    edges_data = [G[u][v][source] for u, v in G.edges]
    nodes_source = list(set([(u) for u, v in G.edges]))
    edges_colors = [nodecolors_dict.get(x) for x in nodes_source]
    weights = [G[u][v][edge_width] for u, v in G.edges]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, nodelist=full_nodes,
            node_size=node_sizes,
            with_labels=True, edge_cmap=None,
            edges=G.edges,
            edge_color=edges_colors,
            node_color=node_colors,
            node_cmap=None, width=array_scale(weights))
    fig.suptitle('Migration network for year {}'.format(G.graph['year']))
    return ax


def get_subgraph_list(G, attr={'edge': 'number'}, bins=[100, 500, 10000]):
    import numpy as np
    import pandas as pd
    # TODO: get subgraph depending on the attr, use digitize:
    glist = []
    if 'edge' in attr.keys():
        name = attr['edge']
        edges = [x for x in G.edges()]
        attr_val=[G.edges()[x].get(name, 0) for x in G.edges()]
        attr_df = pd.DataFrame({name: attr_val})
        attr_df[['from','to']] = np.array(edges)
        # digi = np.digitize(attr_val, bins=bins)
        labels = np.arange(0, len(bins) - 1)
        attr_df['{}_bins'.format(name)] = pd.cut(attr_df['number'], bins=bins,labels=labels)
        # bin_items = list(set(digi))
        # for bini in bin_items:
        #     edges_slice = edges[np.where(digi==bini)]
        #     t2=[tuple(x) for x in edges_slice.tolist()]
        #     edge_subgraph = G.edge_subgraph(t2)
        #     glist.append(edge_subgraph)
        for bin_ind in labels:
            sliced = attr_df[attr_df['{}_bins'.format(name)]==bin_ind][['from','to']].to_numpy()
            t2=[tuple(x) for x in sliced.tolist()]
            edge_subgraph = G.edge_subgraph(t2)
            glist.append(edge_subgraph)
    elif 'node' in attr.keys():
        attr_val=np.array([G.nodes()[x].get(attr['node'], 0) for x in G.nodes()])
    
    return glist


def get_bins_legend(bins=[100, 500, 10000]):
    labels = []
    for i in range(len(bins) - 1):
        label = '{} - {}'.format(bins[i], bins[i+1])
        labels.append(label)
    return labels
    
def node_sizes_source_target(df, year=2000, level='district'):
    size_in = calculate_node_size_per_year(
        df, year=year, level=level, direction='inflow')
    size_out = calculate_node_size_per_year(
        df, year=year, level=level, direction='outflow')
    size_node = {**size_in, **size_out}
    return size_node


def calculate_node_size_per_year(df, year=2000, level='district',
                                 direction='outflow'):
    df = choose_year(df, year=year, verbose=False)
    # df = df[df['Direction'] == direction]
    if direction == 'outflow':
        prefix = level_dict.get(level)['source']
    elif direction == 'inflow':
        prefix = level_dict.get(level)['target']
    nodes = df[prefix].unique()
    node_list = []
    for node in nodes:
        dfc = df[df[prefix] == node]
        # dfc = dfc[dfc['Percent-migrants'] != 0]
        if direction == 'outflow':
            size = sum(dfc['OutPop'].value_counts().index)
        elif direction == 'inflow':
            size = sum(dfc['InPop'].value_counts().index)
        #size =  (dfc['Number'].div((dfc['Percent-migrants']))).sum()
        node_list.append(size)
    size_dict = dict(zip(nodes, node_list))
    return size_dict


def get_lat_lon_from_df_per_year(df, year=2000, level='district'):
    geo_in = get_lat_lon_per_year(
        df, year=year, level=level, direction='inflow')
    geo_out = get_lat_lon_per_year(
        df, year=year, level=level, direction='outflow')
    geo_node = {**geo_in, **geo_out}
    return geo_node


def get_lat_lon_per_year(df, year=2000, level='district',
                         direction='outflow'):
    import numpy as np
    df = choose_year(df, year=year, verbose=False)
    # df = df[df['Direction'] == direction]
    if direction == 'outflow':
        prefix = level_dict.get(level)['source']
    elif direction == 'inflow':
        prefix = level_dict.get(level)['target']
    nodes = df[prefix].unique()
    node_list = []
    for node in nodes:
        dfc = df[df[prefix] == node]
        # dfc = dfc[dfc['Percent-migrants'] != 0]
        if direction == 'outflow':
            lon = np.mean(dfc['OutLON'].value_counts().index)
            lat = np.mean(dfc['OutLAT'].value_counts().index)
        elif direction == 'inflow':
            lon = np.mean(dfc['InLON'].value_counts().index)
            lat = np.mean(dfc['InLAT'].value_counts().index)
        #size =  (dfc['Number'].div((dfc['Percent-migrants']))).sum()
        node_list.append([lat, lon])
    geo_dict = dict(zip(nodes, node_list))
    return geo_dict


#df = pd.read_hdf(work_david /
#                 'Migration_data_IL.hdf')
#G = build_directed_graph(df, year=2000, level='county')




