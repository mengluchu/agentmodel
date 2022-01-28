library(data.table)
library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)
library(ranger)
library(naniar)

library(mltools)
library(data.table)
library(nnet)
#
person = fread("/Volumes/Meng_Mac/mobiair/mobility_data/CH/Person_2015.csv")
head(person)
nrow(person)
persons=person%>%rename("age" = alter, "gender"= gesl, 
                "civil" = zivil, "employment" = AMSTAT,
                "education" = HAUSB, "employ_time" = ERWERB, 
                "work_X"=A_X, "work_Y"= A_Y, "workloc" = A_stat_stadt_2012,
                "edu_X" = AU_X, "edu_Y"=AU_Y, "eduloc"=AU_stat_stadt_2012,
                "home_X" = W_X, "home_Y"=W_Y, "homeloc" = W_stat_stadt_2012, "income" = F20601 )%>%
  select(c("HHNR", "age" , "gender", "civil" , "employment",  "education" , "employ_time" , 
           "work_X", "work_Y", "workloc" ,
           "edu_X"  , "edu_Y", "eduloc" ,
           "home_X" , "home_Y" , "homeloc" , "income" ) )

#
#1:3 ~ obligatory education or less 
#4:12 ~ secondary education (school or apprenticeship etc) 
#13:19 ~ tertiary education (university, vocational training with diploma etc) 
persons= persons %>%mutate(edu= case_when(education<3~ "obi_less", 
                                         education %in% c(4:12)~"secondary_edu",
                                         education %in% c(13:19) ~"tertiary_edu")) #passager, driver


mobility = fread("/Volumes/Meng_Mac/mobiair/mobility_data/CH/Mobility_2015.csv")
 
mobilities = mobility %>% rename("travel_mean"=f51300,  "purpose"=f52900, "deptime" = "f51100")%>% select ("travel_mean", "purpose", "e_dauer", "schaetzdist", "deptime", rdist, "HHNR", "WEGNR")
mobilities= mobilities %>%mutate(travel_mean_re= case_when(travel_mean%in%c(11,12,9)~ "pub_trans", #tram&metro, train,bus
  travel_mean%in%c(1)~ "foot",
  travel_mean%in%c(2)~ "bike", 
  travel_mean%in%c(7,8)~ "car")) #passager, driver
 
trips = fread("/Volumes/Meng_Mac/mobiair/mobility_data/CH/Trips_2015.csv")
trips%>%head()
wm = mobilities%>%filter(purpose == 3)%>%select(-purpose)# to education
pw = merge(wm, persons)
table(pw$edu)
table(pw$travel_mean_re)

wm = mobilities%>%filter(purpose == 2)%>%select(-purpose)# to work
pw = merge(wm, persons)
table(pw$travel_mean_re)
barplot(table(pw$travel_mean_re))
head(pw)

xy= pw%>%dplyr::select(travel_mean_re, rdist, deptime , employment , edu)
vis_miss(xy)

xy$employment=as.factor(xy$employment)
xy$edu =as.factor(xy$edu)
xy$travel_mean_re=as.factor(xy$travel_mean_re)
#xyh <- one_hot(as.data.table(xy))
names(xy) 
#xyh = xyh%>%select(-employment_2,-employment_3, -employment_1, -edu_less_2y, -edu_non_uni)%>%unique()
ra= ranger(travel_mean_re~., data = na.omit(xy), importance = "permutation")
 
ra
barplot(sort(ra$variable.importance, decreasing = F), horiz = T)

 #str(xyh)
# baseline model "by foot"
 
# multimnom from nnet
xy$travel_mean_re<- relevel(xy$travel_mean_re, ref="car")
test = multinom(travel_mean_re~., data = na.omit(xy))
test
pre = predict(test, newdata = xy, "class")
summary(test)
exp(coef(test))
ctable <- table(pw$travel_mean_re, pre)
ctable
# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable))/sum(ctable))*100,2)

test
exp(coef(test))

z <- summary(test)$coefficients/summary(test)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2
 