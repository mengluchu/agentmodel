#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:01:07 2022

@author: meng
"""
import geopandas as gpd
 
gdf_prov = gpd.read_file("/home/meng/Documents/GitHub/mobiair/NLD_adm_shp/NL_poly.shp")
    #rd new  projection
Utrecht = gdf_prov[gdf_prov.PROV_NAAM == "Utrecht"]

Utrecht.to_file("~/Documents/utrecht.gpkg", driver="GPKG")
