
library(dplyr)

#Read the data
clock <- read.table("http://www.statsci.org/data/general/auction.txt", header = T)

#Abline
plot(clock$Age, clock$Price)
simple.reg1<-lm(Price ~ Age, data=clock)
abline(simple.reg1, col='red', lwd=2)

plot(clock$Bidders, clock$Price)
simple.reg2<-lm(Price ~ Bidders, data=clock)
abline(simple.reg2, col='red', lw=2)


#Descriptive graph for the variables
hist(clock$Price, col = "gray", 
     main = paste("Histogram of Selling Price"), 
     xlab = "Pounds sterling")

hist(clock$Bidders, col = "gray", 
     main = paste("Histogram of Bidders"), 
     xlab = "Amount of Bidders")

hist(clock$Age, col = "gray", 
     main = paste("Histogram of Age of the clock"), 
     xlab = "Years")

#Test the mean of the residuals is zero
mod <- lm(Price ~ Bidders + Age, data=clock)
mean(mod$residuals)

#Test homocedasticity of residuals and normal distributio of residuals
par(mfrow=c(2,2))
mod_1 <- lm(Price ~ Bidders + Age, data=clock)
plot(mod_1)

#Test for autocorrelation
lmtest::dwtest(mod_1)

#Assumption: the x variables and residuals are uncorrelated
cor.test(clock$Bidders, mod_1$residuals)
cor.test(clock$Age, mod_1$residuals)

#No perfect multicolinearity
library(car)
vif(mod_1)

#Linear regression model
mod_1 <- lm(Price ~ Bidders + Age, data=clock)
summary(mod_1)


