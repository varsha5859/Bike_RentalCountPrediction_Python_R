rm(list=ls(all=T))

setwd("C:/Users/BITTU/Desktop/Project 1")
getwd()
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "unbalanced", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
#install.packages("rpart.plot")
library(Hmisc)
library(car)
library(DMwR)
library(Metrics)
library(rpart.plot)
lapply(x, require, character.only = TRUE)
rm(x)
#install.packages("sqldf")
library(sqldf)
library('kableExtra')
library(MASS)
library(psych)

#read the data 

day=read.csv("day.csv")

head(day[, 1:10]) %>% kable(caption = "Bike Rental Count (Columns: 1-10)",
                            booktabs = TRUE, longtable = TRUE)
head(day[, 11:16]) %>% kable(caption = "Bike Rental Count (Columns: 11-16)",
                            booktabs = TRUE, longtable = TRUE)
var <- colnames(day)[-ncol(day)]
num <- 1:length(var)
df = data.frame(S.No. = num, Predictor = var)
kable(df, caption = "Predictor Variables", booktabs = TRUE,
      longtable = TRUE)
#converting the data into required format
day$season=as.factor(as.character(day$season))
day$yr=as.factor(as.character(day$yr))
day$holiday=as.factor(as.character(day$holiday))
day$workingday=as.factor(as.character(day$workingday))
day$mnth=as.factor(as.character(day$mnth))
day$weekday=as.factor(as.character(day$weekday))
day$weathersit=as.factor(as.character(day$weathersit))

str(day)
#missing value analysis
missing_val=data.frame(apply(day,2,function(x){sum(is.na(x))}))
missing_val
###no missing values  found in the data

#histograms
hist(day$cnt, breaks = 25, 
     ylab = 'Frequency of Rental', xlab = 'Total Bike Rental Count',
     main = 'Distribution of Total Bike Rental Count', col = 'blue' )
hist(day$windspeed, main="Histogram for Wind Speed", 
     xlab="wind speed", col = "red")
hist(day$temp, main="Histogram for Temperature", 
     xlab="temperature", col = "green")
hist(day$hum, main="Histogram for Humidity", 
     xlab="temperature", col = "grey")
#plots
plot(day$temp, day$cnt ,
     type = 'h', col= 'red', xlab = 'Temperature', ylab = 'Total Bike Rentals')
plot(day$atemp, day$cnt ,
     type = 'h', col= 'blue', xlab = 'Feel Temperature', ylab = 'Total Bike Rentals')
plot(day$windspeed, day$cnt ,
     type = 'h', col= 'green', xlab = 'Windspeed', ylab = 'Total Bike Rentals')
plot(day$hum, day$cnt ,
     type = 'h', col= 'black', xlab = 'Humidity', ylab = 'Total Bike Rentals')
ggplot (day, aes( x= temp, y = cnt, colour = cnt))+
  geom_point()+geom_smooth()+xlab("Temperature") + 
  ylab ("Total Count")+
  ggtitle("Total Count of Bikes used depending on Temperature")

#Boxplot analysis of the variables
boxplot(day$cnt ~ day$season,
        data = day,
        main = "Total Bike Rentals Vs Season",
        xlab = "Season",
        ylab = "Total Bike Rentals",
        col = c("red", "red1", "red2", "red3")) 

boxplot(day$cnt ~ day$holiday,
        data = day,
        main = "Total Bike Rentals Vs Holiday/Working Day",
        xlab = "Holiday/Working Day",
        ylab = "Total Bike Rentals",
        col = c("blue", "blue1", "blue2", "blue3")) 
boxplot(day$cnt ~ day$weathersit,
        data = day,
        main = "Total Bike Rentals Vs Weather Situation",
        xlab = "Weather Situation",
        ylab = "Total Bike Rentals",
        col = c("green", "green1", "green2", "green3")) 
boxplot(day$cnt ~ day$mnth,
        data = day,
        main = "Total Bike Rentals Vs Month",
        xlab = "Month",
        ylab = "Total Bike Rentals",
        col = c("yellow")) 
boxplot(day$cnt ~ day$weekday,
        data = day,
        main = "Total Bike Rentals Vs Day of Week",
        xlab = "Day of Week",
        ylab = "Total Bike Rentals",
        col = c("black")) 

num_index=sapply(day,is.numeric)
num_data=day[,num_index]
cnames=colnames(num_data)
for (i in 1:length(cnames))
 {
   assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = 'cnt'), data = subset(day))+ 
            stat_boxplot(geom = "errorbar", width = 0.5) +
            geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                         outlier.size=1, notch=FALSE) +
            theme(legend.position="bottom")+
            labs(y=cnames[i],x="cnt")+
            ggtitle(paste("Box plot of count for",cnames[i])))
 }
### Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,gn3,gn4,ncol=5)

##scaterplots to check variables which are negatively correlated
scatterplotMatrix(formula = ~ day$windspeed + day$cnt, cex=0.6,
                  data=day, main = "Effect of Windspeed on Bike Rentals" )
scatterplotMatrix(formula = ~ day$hum + day$cnt, cex=0.6,
                  data=day, main = "Effect of Humidity on Bike Rentals" )

#plotting corrgram 
res2 = rcorr(as.matrix(num_data))
res2
num_index=sapply(day,is.numeric)
num_data=day[,num_index]
cnames=colnames(num_data)
corrgram(day[,num_index], order = F,upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
str(num_data)

##anova  test for independence
factor_data=subset(day,select=c(season,holiday,weekday,workingday,weathersit))
names(factor_data)
season_anv=aov(cnt~season,data=day)
summary(season_anv)
holiday_anv=aov(cnt~holiday,data=day)
summary(holiday_anv)

workingday_anv=aov(cnt~workingday,data=day)
summary(workingday_anv)

weathersit_anv=aov(cnt~weathersit,data=day)
summary(weathersit_anv)


#divide data into train and test data
train_ind=sample(1:nrow(day),0.8*nrow(day))
train_data=day[train_ind,]
test_data=day[-train_ind,]


# decission tree regression
fit = rpart(cnt ~ season + workingday+ weathersit + temp, data = train_data, method = "anova")
pr=predict(fit,test_data[,-16])
rpart.plot(fit,box.palette ="RdBu",shadow.col="gray",nn=TRUE)
mapee(test_data[,16],pr)
 


#error metrics
#calculate mape
actuals_preds = data.frame(cbind(actuals=test_data$cnt, predicteds=predictions))
mapee = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}

#linear regression model
lin.mod =lm(cnt ~workingday+season+ weathersit + temp, data =train_data)
predictions=predict(lin.mod,test_data[,-16])
mapee(test_data[,16],predictions)


#mape value for linear regression model is 28.011
#mape value for Decision tree regression model is 26.85