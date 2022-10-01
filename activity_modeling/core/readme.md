### The activity simulation and exposure models:

1. streamline_osmnx.py runs the activity model.
2. calc_exposure.py then calculates the expsosure. 

### Note:

1. modelfun.py consists of most of the functions used in the activity model, the steamline_osmnx.py imports all the functions from modelfun.py.
2. The activity model requires road network data in the format of the road network graphs in OSMnx. download_osmnx.py could be used to download the road networks to be used in the activity model. For newest update, please refer to https://osmnx.readthedocs.io/en/stable/.
