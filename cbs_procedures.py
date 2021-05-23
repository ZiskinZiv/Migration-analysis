#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:44:45 2021

@author: ziskin
"""

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
    # root = ET.XML(r.content)
    # data = []
    # cols = []
    # for i, child in enumerate(root):
    #     data.append([subchild.text for subchild in child])
    #     cols.append(child.tag)
    # df = pd.DataFrame(data).T  # Write in DF and transpose it
    # df.columns = cols
    return df
