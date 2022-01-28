#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:33:12 2021

@author: menglu
"""
import osmnx as ox
import networkx

# do this for car, walk, driveways, and eventually the whole netherlands. 
cls = "walk"
G = create_graph('Utrecht', 5000, cls_)# walk
ox.plot_graph(G)
G = ox.add_edge_speeds(G) #Impute but only for "car"
G = ox.add_edge_travel_times(G) #Travel time
ox.io.save_graphml(G, filepath=filedir+"utgraph.graphml")
