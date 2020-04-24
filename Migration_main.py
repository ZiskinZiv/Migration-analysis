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

level_dict = {
    'district': {'source': 'OutDistrict', 'target': 'InDistrict'},
    'county': {'source': 'OutCounty', 'target': 'InCounty'},
    'city': {'source': 'OutEN', 'target': 'InEN'}
}


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


def read_and_write_as_hdf(path=work_david):
    """reads David's original binary excell file and writes a pandas HDF file,
    path is the same for reading and writing"""
    import pandas as pd
    df = pd.read_excel(
            work_david /
            'Place-to-place migration-IL.xlsb',
            engine='pyxlsb')
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


def choose_year(df, year=2000, dropna=True, verbose=True):
    """return a pandas DataFrame with the selected years.
    Input:  df : original big pandas DataFrame,
            year : either 1 values or two values (start, end)
            dropna : drop NaN's
            verbose : verbosity
    Output: sliced df"""
    if dropna:
        df = df.dropna()
    df = df[df['Year'] != 'ALL']
    if isinstance(year, int) or isinstance(year, str):
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


def build_directed_graph(df, year=2000, level='district',
                         graph_type='multi-directed',return_json=False):
    import networkx as nx
    from networkx import NetworkXNotImplemented
    """Build a directed graph with a specific level hierarchy and year/s.
    Input:  df: original pandas DataFrame
            year: selected year/s
            level: district, county or city
            return_json: convert networkx DiGraph object toJSON object and
            return it.
    Output: G: networkx DiGraph object or JSON object"""
    if graph_type == 'directed':
        Graph = nx.DiGraph()
    elif graph_type == 'multi-directed':
        Graph = nx.MultiDiGraph()
    else:
        raise("Only 'directed' and 'multi-directed' graph_type are allowed!")
    print('Building {} graph with {} hierarchy level'.format(graph_type, level))
    source = level_dict.get(level)['source']
    target = level_dict.get(level)['target']
    df_sliced = choose_year(df, year=year, dropna=True)
    node_sizes = node_sizes_source_target(df, year=year, level=level)
    node_geo = get_lat_lon_from_df_per_year(df, year=year, level=level)
#    if weight_col is not None:
#        df['weights'] = normalize(df[weight_col], 1, 10)
#    else:
#        df['weights'] = np.ones(len(df))
    # df = df[df['Percent-migrants'] != 0]
    G = nx.from_pandas_edgelist(
        df_sliced,
        source=source,
        target=target,
        edge_attr=[
            source,
            'Percent-migrants',
            'Direction',
            'Number',
            'Total',
            'Distance',
            'Angle'],
        create_using=Graph)
    nx.set_node_attributes(G, node_sizes, 'size')
    nx.set_node_attributes(G, node_geo, 'coords_lat_lon')
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
        print(key + ' :', val)
    # G, metdf = calculate_metrics(G, weight_col=weight_col)
    if return_json:
        return nx.node_link_data(G)
    else:
        return G


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


def centrality_analysis(G, weight_col='Percent-migrants'):
    """main function to run centrality analysis.
    Input: G: networkx DiGraph object
           weight_col: apply weights that should exist in G.edges attributes
           choose None to run without weights
    Output: pandas DataFrame with centrality as columns and nodes as index."""
    from networkx import NetworkXNotImplemented
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


def plot_network(G, edge_width='Percent-migrants'):
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
    edges_data = [G[u][v][source] for u, v in G.edges]
    edges_colors = [nodecolors_dict.get(x) for x in edges_data]
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




