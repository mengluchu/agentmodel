#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:21:40 2021

@author: menglu
"""
#os.chdir("../meng/Documents/GitHub/agent/agentmodel/activity_modeling/core/")

###########################################################################################
# please mind the reference system to use, in gethw() and the parameters in getcon()    
#free-time activity garden: 100m (0.001 degree), take a walk (2000 m) (0.02 degree)
##############################################################################################
import pandas as pd 
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from rasterstats import zonal_stats, point_query
import pyproj
from shapely.ops import transform
import modelfun as m
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show_hist
from math import modf
import osmnx as ox
from scipy import signal 
from IPython import get_ipython
import statistics
import random

random.seed(10)

scriptdir = "/home/meng/Documents/GitHub/agent/agentmodel/activity_modeling/core"
os.chdir(scriptdir)

#get_ipython().run_line_magic('matplotlib', 'qt')
#get_ipython().run_line_magic('matplotlib', 'inline')

#wuhan = ox.geocode_to_gdf('武汉, China')
#utrecht = ox.geocode_to_gdf('Utrecht province') 
#wuhan.plot() 
filedir = "/home/meng/Documents/GitHub/mobiair/"
preddir =f"/home/meng/Downloads/hr_pred/"
savedir = "/home/meng/Documents/mobiresults/Uni/" # each profile a savedir. 
savedir2 = "/home/meng/Documents/mobiresults/Uni_2/" # each profile a savedir. 
                      
def wgs2laea (p):
     
    rd= pyproj.CRS('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    project = pyproj.Transformer.from_crs("EPSG:4326", rd, always_xy=True)
    p=transform(project.transform, p)
    return (p)  

def plot_raster ():
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(55,15)) 
                            
    for i, ax in enumerate(axs.flat):
        src = rasterio.open(f'{preddir}hr_pred_utrecht_X{i}_.tif')

        ax.set_axis_off()
        a=ax.imshow(src.read(1), cmap='pink')
        ax.set_title(f' {i:02d}:00')
         
    cbar = fig.colorbar(a, ax=axs.ravel().tolist())
    cbar.set_label(r'$NO_2$', rotation = 270)

    plt.show()

#plt.savefig(savedir+"prediUt.png")

#show_hist(src, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

#plt.show()

#ls = os.listdir(preddir)

def gethw (df):
    ph=Point(float(df.home_lon) , float(df.home_lat) )
    #ph = wgs2laea(ph)
    pw=Point(float(df.work_lon) , float(df.work_lat) )
    #pw = wgs2laea(pw)
    return(ph, pw)

def buffermean(p, ext , rasterfile):
   
    pbuf=p.buffer(ext)
    z= zonal_stats(pbuf, rasterfile, stats = 'mean')[0]['mean']
    
    return z  


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


#plt.imshow(gkern(2000,500))
    
#extract pollution map in the buffer and convolve with gaussian kernel.  
def gaussKernconv(p, ext , rasterfile, sd = 10):
   
    pbuf=p.buffer(ext)
    getras= zonal_stats(pbuf,rasterfile ,stats="count", raster_out=True)[0]['mini_raster_array']
    #plt.imshow(getras)
    #plt.colorbar()
 
    ag = gkern (getras.data.shape[0], sd)
    ag= ag/ np.sum(ag)
    plt.imshow(ag)
    z = signal.convolve2d(getras.data,ag, mode = 'valid') # valid does not pad. 

    #np.sum(ag)      
    return z  


''' for testing
j =
homework.loc[j]
ph, pw = gethw(homework.loc[j])
rasterfile =f'{preddir}hr_pred_utrecht_X{1}_.tif'
point_query(pw, rasterfile ).pop() 
buffermean(ph, 0.001, rasterfile)
p = ph
extgaus = 1
gaussKernconv(ph, extgaus, rasterfile, sd = 0.1)
''' 


#for j in range (35):
 
#    ph, pw = gethw(homework.loc[9])
#    rasterfile =f'{preddir}hr_pred_utrecht_X{1}_.tif'
#    print(point_query(pw, rasterfile ).pop(), point_query(ph, rasterfile ).pop() )


def value_extract (p, rasterfile):
    if point_query(p, rasterfile ).pop() is not None:
        return point_query(p, rasterfile ).pop()
    else: return -99
#value_extract(ph, rasterfile)

def getcon(act_num, rasterfile, df, routegeom, ext = 0.001, extgaus = 2, sd=0.1, indoor_ratio = 0.7):
    ph, pw = gethw(df)
    if point_query(ph, rasterfile ).pop() is not None and point_query(ph, rasterfile ).pop() is not None:
    
     #   return {
     #          1: indoor_ratio * value_extract(ph, rasterfile ), #home
     #          2: np.nan_to_num(np.mean(point_query(routegeom, rasterfile)),0),# route #can easily make buffer out of it as well, here the ap already 100m so not needed.
     #          3: indoor_ratio *value_extract(pw, rasterfile ), # work_indoor
     #          4: buffermean(ph, ext, rasterfile), # freetime 1 will change into to sport (second route)
     #          5: gaussKernconv(ph, extgaus, rasterfile, sd = 0.1), # freetime 2, distance decay, outdoor. 
     #          6: buffermean(ph, ext, rasterfile)  # freetime 3, in garden or terras  
     #          } [act_num]
        if act_num == 1 :
            con = indoor_ratio * value_extract(ph, rasterfile )
        elif act_num ==2:
            if point_query(routegeom, rasterfile)[0] is not None:
              #  print(point_query(routegeom, rasterfile))
                con = np.nan_to_num(np.nanmean(point_query(routegeom, rasterfile)),0)
            else:
                con = -99
        elif act_num == 3: 
            con = indoor_ratio *value_extract(pw, rasterfile ) # work_indoor
        elif act_num ==4:
            con = buffermean(ph, ext, rasterfile)  # freetime 1 will change into to sport (second route)
        elif act_num==5:
            con = gaussKernconv(ph, extgaus, rasterfile, sd = 0.1) 
        elif act_num==6:
            # freetime 2, distance decay, outdoor. 
            con = buffermean(ph, ext, rasterfile)  # freetime 3, in garden or terras  
        return con

                                              
    else: 
        return {1:-99,
                2:-99,
                3:-99,
                4:-99,
                5:-99,
                6:-99
            }[act_num]
#schefile = os.listdir(schedir)
 

 
# rasterfile = f'{preddir}NL100_t{i}.tif'

def remove_none(lst):
    lst = [i for i in lst if i is not None]
   
    
    return lst
'''  
#test
for i in range(1,7):
    getconcen(act_num= i,
              rasterfile=f'{preddir}NL100_t{i}.tif', 
              df = Uni_ut_homework.loc[j], 
              routegeom=route.loc[j]['geometry'])
'''
#still doing hourly
ext = 0.001 # 300 m 
iteration = 28

ODdir = savedir +"genloc/"
ODfile =f'h2w_{iteration}.csv' 
homework =gpd.read_file(ODdir+ODfile) # for comparison #gpd can read csv as well, just geom as None.

#real: use real locations (i.e. no iteration)
def cal_exp(filedir, savedir, iteration, real = False, ext = 0.001, extgaus=2, gaussd = 0.1,  save_csv = True):
    ODdir = savedir +"genloc/"
    if real:
        ODfile =f'h2w_real.csv'
    else:
        ODfile =f'h2w_{iteration}.csv' 
    homework =gpd.read_file(ODdir+ODfile) # for comparison #gpd can read csv as well, just geom as None.
    routedir = savedir+'genroute/'
    routefile = f'route_{iteration}.gpkg' # get route file for all people, only one route file,geodataframe
    route= gpd.read_file(routedir+routefile)
    #route = route.to_crs('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    
    schedir = savedir+'gensche/'
    # exp is concentration weighted by time duration
    exp_each_act = []
    exp_each_person= []
    n = len(homework)
    for j in range(n): #iterate over each person
    
        sched = pd.read_csv(f'{schedir}ws_iter_{iteration}_id_{j}.csv') #each person has a schedule, only schedule is file per person.
        start = sched['start_time']
        end =sched['end_time']
        start_int=np.floor(start).astype(int)
        end_int = np.floor(end).astype(int) # for using range,this value should plus 1 as range always omit the last value.
        act_num = sched['activity_code']
        
        for k in range(sched.shape[0]): # iterate over schedule
            conh = 0 # hourly concentration for each activity
            missingtimeh = 0 
            missingtime = 0
            #if start_int[k] == end_int[k]-1: # less than one hour
            if end[k] - start[k] < 1: # less than 1 hour trip, will just use the concentration hour or the starttime (start_int)
                if start_int[k]== end_int[k]: # if in the same hour
                    con = getcon(act_num[k], f'{preddir}hr_pred_utrecht_X{start_int[k]}_.tif',  homework.loc[j], route.loc[j]['geometry'], ext = ext, extgaus = extgaus, sd = gaussd)                    
                    if con is not None or con is not np.nan:
                        #conh = con * (end[k]-start[k]) # start percentage multiply by concentration of the hour, next hour will get the rest of the percentage
                        conh = con * (end[k]-start[k]) #  percentage multiply by concentration of the hour 

                        missingtimeh=0
                    else: 
                        conh = 0
                        missingtimeh = end[k]-start[k]                        
                else:           # if not in the same hour 
                 
                    constart = getcon(act_num[k], f'{preddir}hr_pred_utrecht_X{start_int[k]}_.tif',  homework.loc[j], route.loc[j]['geometry'], ext = ext, extgaus = extgaus, sd = gaussd)
                    conend = getcon(act_num[k], f'{preddir}hr_pred_utrecht_X{end_int[k]}_.tif',  homework.loc[j], route.loc[j]['geometry'], ext = ext, extgaus = extgaus, sd = gaussd)
                    
                    if con is not None or con is not np.nan:
                        #conh = con * (end[k]-start[k]) # start percentage multiply by concentration of the hour, next hour will get the rest of the percentage
                        conh = constart * (start[k]-start_int[k]) + conend* (end[k]-end_int[k]) # same as using modf, start percentage multiply by concentration of the hour, end will get the rest of the percentage
                        missingtimeh=0
                    else: 
                        conh = 0
                        missingtimeh = end[k]-start[k]
                 
            else: # more than one hour
                for i in range(start_int[k],end_int[k]): # iterate over raster
                    con = getcon(act_num[k], f'{preddir}hr_pred_utrecht_X{i}_.tif', homework.loc[j], route.loc[j]['geometry'], ext = ext, extgaus = extgaus, sd = gaussd)
                    #control at the beginning and in the end.
                    if i ==start_int[k]: # first hour may be from e.g. 7:20 instead of 7:00
                        if con is not None or con is not np.nan:
                            cons = con * (1- modf(start[k])[0]) # start percentage multiply by concentration of the hour, next hour will get the rest of the percentage
                            missingtime=0
                        else: 
                            cons = 0
                            missingtime = modf(start[k])[0]
                    elif i == end_int[k]: # last hour may be to e.g. 9:20 instead of 9:00
                        if con is not None or con is not np.nan:
                            cons = con * modf(end[k])[0] # end percentage
                            missingtime=0
                        else: # for none values or nan, assign valye 0 and note missing times
                            cons = 0
                            missingtime = modf(end[k])[0]               
                    else:
                        if con is not None or con is not np.nan:
                            cons = con # middle times
                            missingtime=0
                        else: # for none values or nan, assign valye 0 and note missing times
                            cons = 0
                            missingtime = end_int - start_int  -1          
    
                    
                # summing exposures
                    conh= conh +cons
                    #exp_each_hour.append(cons)
                    missingtimeh = missingtimeh + missingtime 
                    
            exp = conh/(end[k]-start[k]-missingtimeh+0.01) # average exp per activity
            if not np.isscalar(exp):
                exp.item()
            exp_each_act.append(exp)
             
            #con_each_person.append(np.nanmean(remove_none(con_each_act[k*j : (k+1)*j ]) ))      
        # meanactexp = np.nanmean(remove_none(exp_each_act[j*sched.shape[0]:(j+1)*sched.shape[0]]), keepdims =False)
        meanactexp = statistics.mean(remove_none(exp_each_act[j*sched.shape[0]:(j+1)*sched.shape[0]]))
        meanactexp = np.where(np.isscalar(meanactexp), meanactexp, meanactexp.item()).item()
        #add this because sometimes it returns a nested array like [[1]], a strange behaviour of remove_none(nparray): 
        exp_each_person.append(meanactexp)
        print(j)
        #print (exp_each_person)
    
    if save_csv:
        exposuredir =f"{savedir}exposure/" 
        m.makefolder(exposuredir)    
        pd.DataFrame(exp_each_act).to_csv(f'{exposuredir}iter_{iteration}_act.csv')
        pd.DataFrame(exp_each_person).to_csv(f'{exposuredir}iter_{iteration}_person.csv')
    return (exp_each_act, exp_each_person)    
      
#act, person = cal_exp(filedir, savedir, iteration, save_csv = True)
#for multiple iterations


# plot
def formattime(timeinput): 
  
    minute, hour = modf(timeinput)
    minute = np.floor(minute *60)     
    return "%02d:%02d" % (hour, minute) 

#act activity exposure
def plotact(sub1, sub2,  act, savename="1", simplify = True, select = 0):
    
    schedir = savedir+'gensche/'

    fig, ax = plt.subplots(sub1,sub2,figsize=(18, 56), sharey=True )
    axs = ax.flatten()
    for i1 in range (sub1*sub2):
            i = i1 + select
            sch = pd.read_csv(f'{schedir}ws_iter_{str(iteration)}_id_{i}.csv')
            st = sch['start_time']
            et = sch['end_time']
            axs[i1].plot(list(st),act[i*7:(i+1)*7], "ko-") 
            #st= np.round(st,1)
            ind = np.where(np.diff(st)<1.5)[0]
            if simplify:
                xlabels = list(sch['activity'])
            else:
                xlabels = [f"{x3}: {x1} to {x2}" for x1, x2, x3, in zip(map(formattime,list(st)),map(formattime,list(et)),list(sch['activity']))]
            
            for j in range(7):
                x, y = st[j],list(act[i*7:(i+1)*7])[j] 
                #t = axs[i1].text(x, y+2, xlabels[j] )
            #list(sch['activity'])[j] 
            et = et.drop(ind)
            st=st.drop(ind)
            xlabels = [f"{x1} to \n{x2}" for x1, x2, in zip(map(formattime,st), map(formattime,et))]
            axs[i1].set_title(f'person ID: {i}')
            axs[i1].set_xlabel('hour')
            axs[i1].set_xticks(st) 
            axs[i1].set_xticklabels(map(formattime,st))
            axs[i1].tick_params(axis='x',labelrotation =45,bottom=True,length=5)
            axs[i1].set_ylabel("Exposure: " r'$ \mathrm{NO}_2$', fontsize=10)
    #fig.supxlabel("hour")
    #fig.supylabel("Exposure: " r'$ \mathrm{NO}_2$', fontsize=10)
    fig.tight_layout()      
    fig.savefig(f'{savedir}exposure_act{savename}.png') 

# known locations
#for iteration in range (29, 39): 
#    cal_exp(filedir, savedir2, iteration, real = True, save_csv = True)


# simulated locations
for iteration in range (29, 39):
    cal_exp(filedir, savedir, iteration, save_csv = True)

lat = np.array(homework.home_lat).astype(float)
lon = np.array(homework.home_lon).astype(float)

act = pd.read_csv(f"{savedir}exposure/iter_{iteration}_act.csv").iloc[:,1]
person = pd.read_csv(f"{savedir}exposure/iter_{iteration}_person.csv").iloc[:,1]
  
df1 = [person,lat, lon]

 
df2 = pd.DataFrame(data=df1).T
df2 = df2.rename (columns = {'0':"personal_exposure", "Unnamed 0": "lat", "Unnamed 1": "lon" })
exp_gdf = gpd.GeoDataFrame(df2["personal_exposure"], crs={'init': 'epsg:4326'},
                                     geometry=[Point(xy) for xy in zip(df2.lon, df2.lat)])

exp_gdf.to_file(f'{savedir}person_iter{iteration}.gpkg')

#visualise
fig, ax = plt.subplots()
ax.set_aspect('equal')
exp_gdf.plot(ax=ax, column = 'personal_exposure',legend=True)
#
i=1
src = rasterio.open(f'{preddir}hr_pred_utrecht_X{i}_.tif')
ax.imshow(src.read().squeeze())
ax.set_axis_off()

#plot activity

plotact(sub1 = 2, sub2 =4, act = act, savename="more", simplify=True, select = 2)    

plt.show()
plt.close('all')


#
import glob
files = glob.glob(f"{savedir}exposure/*person*")
 
files = glob.glob(f"{savedir2}exposure/*person*")

files
df_from_each_file = (pd.read_csv(f, sep=',').iloc[:,1] for f in  files)
df_merged   = pd.concat(df_from_each_file, ignore_index=True, axis =1)
df_merged["lat"] = lat
df_merged["lon"] = lon 
df_merged.describe()
df_merged.head()
df_merged.to_csv( f"{savedir}/allperson.csv")
 
  
exp_gdf = gpd.GeoDataFrame(df_merged, crs= 4326,
                                     geometry=[Point(xy) for xy in zip(df_merged.lon, df_merged.lat)])
repr(exp_gdf).encode('utf-8')
exp_gdf.to_file(f'{savedir}person_all.gpkg')
exp_gdf.plot() 
