#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:03:54 2020

@author: shlomi
"""

from MA_paths import work_david
import pandas as pd
import matplotlib.pyplot as plt

level_dict = {
    'district': {'source': 'OutDistrict', 'target': 'InDistrict'},
    'county': {'source': 'OutCounty', 'target': 'InCounty'},
    'city': {'source': 'OutEN', 'target': 'InEN'}
}


def normalize(arr, lower=0.2, upper=10):
    import numpy as np
    arr = np.array(arr)
    maxi = arr.max()
    mini = arr.min()
    data = (((arr-mini)*(upper-lower))/(maxi-mini))+lower
    return data


def bin_data(arr, bins=None):
    from scipy import stats
    import numpy as np
    if bins is None:
        bins = np.linspace(10,100,10)
    ranked = stats.rankdata(np.array(arr))
    data_percentile = ranked / len(arr) * 100
    data_binned_indices = np.digitize(data_percentile, bins, right=True)
    return data_binned_indices


def read_and_write_as_hdf(path=work_david):
    import pandas as pd
    df = pd.read_excel(
            work_david /
            'Place-to-place migration-IL.xlsb',
            engine='pyxlsb')
    df.to_hdf(
        work_david /
        'Migration_data_IL.hdf',
        complevel=9,
        mode='w',
        key='migration')
    return df


def build_directed_graph(df, year=2000, level='district',
                         weight_col='Percent-migrants', plot=True):
    import networkx as nx
    import numpy as np
    print('Building directed graph with {} hirerchy level'.format(level))
    source = level_dict.get(level)['source']
    target = level_dict.get(level)['target']
    df = df[df['Year'] == year]
    df = df.dropna()
    node_sizes = node_sizes_source_target(df, year=year, level=level)
#    if weight_col is not None:
#        df['weights'] = normalize(df[weight_col], 1, 10)
#    else:
#        df['weights'] = np.ones(len(df))
    # df = df[df['Percent-migrants'] != 0]
    G = nx.from_pandas_edgelist(
        df,
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
        create_using=nx.DiGraph())
    nx.set_node_attributes(G, node_sizes, 'size')
    G.name = 'Israeli migration network'
    G.graph['level'] = level
    G.graph['year'] = year
    G.graph['density'] = nx.density(G)
    G.graph['triadic_closure'] = nx.transitivity(G)
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
    if plot:
        plot_network(G, edge_width=weight_col)
    return G


def convert_centrality_to_series(G, centrality='in_degree',
                                 weight_col='Percent-migrants'):
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
    df = convert_centrality_to_series(G, 'in_degree', weight_col=weight_col)
    df['out_degree'] = convert_centrality_to_series(
        G, 'out_degree', weight_col=weight_col)
    df['degree'] = convert_centrality_to_series(
        G, 'degree', weight_col=weight_col)
    df['eigenvector'] = convert_centrality_to_series(
        G, 'eigenvector', weight_col=weight_col)
    df['closeness'] = convert_centrality_to_series(
        G, 'closeness', weight_col=weight_col)
    df['betweenness'] = convert_centrality_to_series(
        G, 'betweenness', weight_col=weight_col)
    df['load'] = convert_centrality_to_series(G, 'load', weight_col=weight_col)
    df['harmonic'] = convert_centrality_to_series(
        G, 'harmonic', weight_col=weight_col)
    df['clustering'] = convert_centrality_to_series(
        G, 'clustering', weight_col=weight_col)
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
            node_cmap=None, width=normalize(weights))
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
    df = df[df['Year'] == year].dropna()
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


#df = pd.read_hdf(work_david /
#                 'Migration_data_IL.hdf')
#G = build_directed_graph(df, year=2000, level='county')


