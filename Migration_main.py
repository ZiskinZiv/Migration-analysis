#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:03:54 2020

@author: shlomi
"""

from MA_paths import work_david


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
