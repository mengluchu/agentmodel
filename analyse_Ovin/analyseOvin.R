# from a persons social occupation, e.g. full time working or not, and the purpose, e.g. go to work, and going to do hobby, we can get a distribution of the distance. We sample from this distribution, calculate a buffer of possible locations to go.
library(data.table)
library(dplyr)
library(ggplot)
library(purrr)
library(tidyr)
a = fread("~/Documents/GitHub/mobiair/human_data/dutch_activity/Ovin.csv")
table(a$Verpl)

#a%>%select(KAf_mean, Rvm, MaatsPart, Doel, age_lb, dep_lat, dep_lon,  arr_lat, arr_lon)
names(a)
#summary(as.numeric(a$VertPC))
#summary(lm(KAf_high~age_lb+Sted, data =a ))
work = a%>%filter(Doel == "Werken")
shopping = a%>%filter(Doel == "Winkelen/boodschappen doen")

#profile
schoolstu = a%>%filter(MaatsPart=="Scholier/student"&age_lb<18) #22,485/30,218
uni = a%>%filter(MaatsPart=="Scholier/student"&age_lb>18)#22,485/30,218
fulltime = a%>%filter(MaatsPart=="Werkzaam >= 30 uur per week")#22,485/30,218
parttime = a%>%filter(MaatsPart== "Werkzaam 12-30 uur per week" )#22,485/30,218

#purpose
studentshopping = a%>%filter(Doel == "Winkelen/boodschappen doen"& MaatsPart=="Scholier/student")
single = a%>%filter(Doel == "Werken" & MaatsPart=="Eigen huishouding")
student = a%>%filter(Doel == "Onderwijs/cursus volgen" & MaatsPart=="Scholier/student")
halftime = a%>%filter(Doel == "Werken" & MaatsPart=="Werkzaam 12-30 uur per week")
 
summary(lm(KAf_mean~Rvm+Maat, data =a ))

summary(lm(KAf_mean~Rvm, data =a))

summary(lm(KAf_mean~age_lb, data =a ))

summary(lm(KAf_mean~age_lb, data =a ))

names(a)
summary( aov(KAf_mean~MaatsPart, data = work)) # occupation is associated with travel distance
summary( aov(age_lb~Rvm, data = work)) # age is associated with travel mean
summary( aov(age_lb~Rvm+KVertTijd, data = work)) # age is associated with travel mean and time to depart
install.packages("ggpubr")
library("ggpubr")
ggboxplot(work, x = "Rvm", y = "age_lb",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))

ggboxplot(uni, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))


ggboxplot(schoolstu, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))

ggbarplot(ss,x= "Rvm", y = "freq")+theme(axis.text.x = element_text(angle = 90))

train = ss%>%filter(Rvm =="Trein")%>%select(freq)
walk = ss%>%filter(Rvm =="Te voet")%>%select(freq)

bike = ss%>%filter(Rvm =="Fiets (elektrisch en/of niet-elektrisch)")%>%select(freq)
auto = 1-train - walk - bike



ggboxplot(halftime, x = "Rvm", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))

ggboxplot(work, x = "MaatsPart", y = "KAf_mean",
          palette = c("#00AFBB", "#E7B800"))+theme(axis.text.x = element_text(angle = 90))

#Generate distribution of travel distance to work according to social occupationalStatus, then based on this we can choose work locations
socialpartition = unique(a$MaatsPart)
i = 1
mu1 = c()
sd1 = c()
for ( i in 1: 10){
socialpartition[i]
mmax = a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[i])%>%select(KAf_mean)%>%max
mu =  a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[i])%>%select(KAf_mean)%>% apply(2, function(x) x+0.1) %>% apply(2, log)%>%apply(2, mean)
sd =  a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[i])%>%select(KAf_mean)%>% apply(2, function(x) x+0.1)  %>% apply(2, log)%>%apply(2, sd)
mu1 [i] = mu
sd1[i] = sd
}
df = shopping

getmean_sd = function (df)
{
mu =  df%>%select(KAf_mean) %>%apply(2, function(x) x+1) %>% apply(2, log)%>% apply(2, mean)
sd =  df%>%select(KAf_mean) %>%apply(2, function(x) x+1) %>% apply(2, log)%>% apply(2, sd)
return (list(mu, sd))
}
ss= student%>%select(KAf_mean)
ss2= student%>%select(KAf_mean)%>%apply(2, function(x) x+1)
dfss = data.frame(ss)
dfss["log(distance)"] = (log(ss2)-1)
names(dfss)[1]= "distance"
dfss = melt(dfss)

ggplot(dfss, aes(x=value)) +
  geom_histogram(color="black",aes(y = stat(count) / sum(count)), fill="white", bins = 30)+geom_density(alpha=.2, fill="lightblue") +facet_grid(variable~.)+
  ylab("frequency")+xlab("distance to school")+theme_bw()+
  theme(
    panel.border = element_blank(), # frame or not
    strip.background = element_rect(
      color="#FFFFFF", fill="#FFFFFF", size=0.5, linetype = NULL),

    strip.text.y = element_text(
      size = 12, color = "black", face = "bold"
    ),
    panel.grid.major = element_blank(),panel.grid.minor = element_blank()
  )

ggsave("~/Documents/GitHub/mobiair/ditance_to_school.png")

hist(log(ss))
getmean_sd (student)

getmean_sd (shopping)
getmean_sd (work)
getmean_sd (studentshopping)
getmean_sd (student)

mu1 [i] = mu
sd1[i] = sd
#

t1 = rnorm(1000, mean = 20, sd = 1)
tlog = log (t1)
t2 = rlnorm(1000, meanlog = mean(tlog), sdlog= sd(tlog))

hist(t1)
hist(t2)
mean(t1)
mean(t2)
sd(t1)
sd(t2)
              #exp(mean(t1) + 0.5*(sd(t1)^2))
hist(t1)
hist(exp(t2))

plot(sd1)
plot(mu1, col = "red")
expn = exp(rnorm (20, mu, sd))
expn[expn<mmax]

par(mfrow =c(1,2))
hist(expn[expn<50])
hist(a%>%filter(Doel == "Werken" & MaatsPart==socialpartition[1])%>%select(KAf_mean)%>%unlist)
   #apply(2, function(x){x+1})
exp(log(10))
 dlnorm(x, meanlog = 0, sdlog = 1, log = FALSE)
