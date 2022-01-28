spa= read.csv("~/Documents/GitHub/mobiair/spadia/MOBIAIR_ABM_SAPALDIA_V1.csv")
head(spa)
geocode =read.csv("~/Documents/GitHub/mobiair/spadia/MOBIAIR_ABM_SAPALDIA_geocode_V1.csv")
names(geocode)
names(spa)

spa= merge(spa, geocode)
summary(as.Date(strftime(spa$doi_s3, format = "%Y-%m-%d")))
summary(as.Date(strftime(spa$doi_s4, format = "%Y-%m-%d")))
table(spa$mobi_id)
plot( spa[,3:15])
library(dplyr)
spa%>%filter(mobi_id=="mobi_0005")
spa$dist_busstop_s3%>%hist
#
create_report(spa)
library(sf)

library(ggplot2)
library(tidyr)
#install.packages("OpenStreetMap")
library(OpenStreetMap)

ggplot(spa, aes(x, y))+
  geom_path(aes(group = mobi_id), arrow = arrow(),color = rainbow(10211))+
  scale_y_reverse()+
  scale_x_reverse()+
  coord_fixed()

spa = spa%>%drop_na("x", "y")

a1 = st_as_sf(spa,coords = c("x", "y"), na.fail = F, crs = 21781)
#EPSG:21781 - Swiss CH1903 / LV03
plot(a1["mobi_id"])
a1=st_transform(a1, crs =4326, na.fail=F)

#st_coordinates(a1) convert geometry to df!!!!!
a1 = cbind(a1, st_coordinates(a1))

sf2df=function(sfo,addcoord = F){
  if (addcoord)
    sfo = cbind(sfo, st_coordinates(sfo))
  
  st_drop_geometry(sfo)
}
# reproject onto WGS84

Edist = function(x1,y1, x2,y2){
  sqrt((x1-x2)^2 + (y1-y2)^2)}
3^2
a1=a1%>%filter(source=="SAPALDIA 3 address"| source =="work address SAP3")
nrow(a1)
table(a1$source)
 
work = group_split(a1, source)[[2]]%>% distinct(mobi_id, .keep_all = TRUE) %>%sf2df%>%select(X, Y, source, mobi_id)
 #2367, 2083
home = group_split(a1, source)[[1]]%>% distinct(mobi_id, .keep_all = TRUE)  %>%sf2df # 6022
hws3 = merge (home, work, by ="mobi_id", left =T)
nrow(hws3)
hws3dis= hws3%>%mutate(dist =  Edist(X.x, Y.x, X.y, Y.y))
hist(hws3dis$dist , breaks = 100, xlim = c(0.03,2)) # 1: 100 km
summary(hws3dis$dist) 

# other 'type' options are "osm", "maptoolkit-topo", "bing", "stamen-toner",
# "stamen-watercolor", "esri", "esri-topo", "nps", "apple-iphoto", "skobbler";

sa_map <- openmap(c(48.041920, 6.013242), c(46.060519, 10.447348), zoom = NULL,
                  type = "esri-topo", mergeTiles = TRUE)
 
sa_map2 <- openproj(sa_map)
names(a1)
OpenStreetMap::autoplot.OpenStreetMap(sa_map2) +geom_point(data= a1, aes(x=X, y=Y))+
  geom_path(data = a1, aes(group = mobi_id, x = X, y =Y,color = as.numeric(age_s3)))

a1$age_s3

 
ggplot(a1) +  geom_sf()
   
library("mapview")
mapview(a1)+
  geom_path(aes(group = mobi_id), arrow = arrow(),color = rainbow(10211)) 

OpenStreetMap::autoplot.OpenStreetMap(sa_map) +  
  geom_sf(data = a1)
  

plot(sa_map2_plt)