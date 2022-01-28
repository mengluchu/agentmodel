library(osmdata)
library(raster)
library(sf) 
#allbuilding = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "building") %>% osmdata_sf() 
traffic = opq(bbox="utrecht netherlands" )%>%add_osm_feature(key = "traffic") %>% osmdata_sf() 

indoorsport = opq(bbox="utrecht,netherlands" )%>%add_osm_feature(key = "leisure", value =c("sports_hall", "sports_centre", "fitness_centre")) %>% osmdata_sf() 
Univ_col = opq(bbox="utrecht,netherlands" )%>%add_osm_feature(key = "amenity", value =c("university", "college")) %>% osmdata_sf() 
school = opq(bbox="utrecht,netherlands" )%>%add_osm_feature(key = "amenity", value =c("school")) %>% osmdata_sf() 

 
plot(Univ_col$osm_polygon$geometry)
Univ_col = Univ_col$osm_polygons%>%st_centroid()  
plot(Univ_col$geometry)

st_write(Univ_col, "/home/meng/Documents/GitHub/mobiair/mobi_model_data/locationdata/Ut_Uni_coll.csv", layer_options = "GEOMETRY=AS_XY")
st_write(Univ_col, "/home/meng/Documents/GitHub/mobiair/mobi_model_data/locationdata/Ut_Uni_coll.gpkg")

# all the university and colleage centroids. 
