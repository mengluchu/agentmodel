#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:24:48 2021

@author: menglu
"""

import pandas as pd
import numpy as np  
import os
import math
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry import LineString

import scipy.stats
import scipy
import shapely.speedups
from shapely.ops import nearest_points 
import networkx as nx
import osmnx as ox
import pyproj
from shapely.ops import transform

aplong_ulr = 'https://raw.githubusercontent.com/mengluchu/mobiair/master/mapping_data/DENL17_hr.csv'
spreadurl = 'https://raw.githubusercontent.com/mengluchu/mobiair/master/mapping_data/DENL17_hr_spread.csv'
res = 100 
ap = pd.read_csv(spreadurl)
coords = ap[['Latitude','Longitude']] # social scientist prefer lat lon
#get coordinate pairs form df 
def df2tupple(i):
     return list(coords.to_records(index = False))[i]
i = 49
dis = 200
df2tupple(i)
G = ox.graph_from_point(df2tupple(i), dist=dis, network_type='drive')
ox.plot_graph(G)
ox.basic_stats(G, area = dis*dis)

