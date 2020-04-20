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


def build_directed_graph(df, year=2000, level='district', plot=True):
    import networkx as nx
    source = level_dict.get(level)['source']
    target = level_dict.get(level)['target']
    df = df[df['Year'] == year]
    # df = df[df['Percent-migrants'] != 0]
    G = nx.from_pandas_edgelist(
        df.dropna(),
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
    G.graph['year'] = year
    if plot:
        plot_network(G, df, year, level)
    return G


def plot_network(G, df, year=2000, level='district'):
    import matplotlib.pyplot as plt
    import networkx as nx
    nodedict = node_sizes_source_target(df, year=year, level=level)
    # nodedict = calculate_node_size_per_year(df, year=year, level=level)
    fig, ax = plt.subplots(figsize=(20, 20))
    node_cmap = plt.get_cmap('tab20', len(nodedict))
    node_colors = [node_cmap(x) for x in range(len(nodedict))]
    nodecolors_dict = dict(zip(nodedict.keys(), node_colors))
    source = level_dict.get(level)['source']
    edges_data = [G[u][v][source] for u, v in G.edges]
    edges_colors = [nodecolors_dict.get(x) for x in edges_data]
    weights = [G[u][v]['Percent-migrants'] for u, v in G.edges]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, nodelist=[x for x in nodedict.keys()],
            node_size=[v / 70 for v in nodedict.values()],
            with_labels=True, edge_cmap=None,
            edges=G.edges,
            edge_color=edges_colors,
            node_color=node_colors,
            node_cmap=None, width=normalize(weights))
    fig.suptitle('Migration network for year {}'.format(year))
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


df=pd.read_hdf(work_david /
        'Migration_data_IL.hdf')
G = build_directed_graph(df, year=2000, level='county')

#G = nx.from_pandas_edgelist(
#    df,
#    source='OutEN',
#    target='InEN',
#    edge_attr=[
#        'Year',
#        'Direction',
#        'Number',
#        'Distance',
#        'Angle'])
