###
### $B.R
###


library(plyr) # for ddply
library(MASS) # for fitdistr
library(nortest) # for ad.test
library(reshape2) # for dcast, melt
library(afex) # for aov_ez
library(performance) # for check_homogeneity
library(glmmTMB) # for glmmTMB
library(lme4) # for lmer, glmer
library(lmerTest)
library(car) # for Anova
library(multcomp) # for glht
library(emmeans) # for emm, emmeans



##
## 1. user_dependent.csv
##
df <- read.csv("user_dependent.csv")
df$pID = as.factor(df$pID)
df$algoType = as.factor(df$algoType)
df$ptype = as.factor(df$ptype)

contrasts(df$algoType) <- "contr.sum"
contrasts(df$ptype) <- "contr.sum"

View(df)


# EDA
ddply(df, ~ algoType + ptype, function(data) c(
  "Nrows"=nrow(data),
  "Min"=min(data$test_mean),
  "Mean"=mean(data$test_mean), 
  "SD"=sd(data$test_mean),
  "Var"=var(data$test_mean),
  "Median"=median(data$test_mean),
  "IQR"=IQR(data$test_mean),
  "Max"=max(data$test_mean)
))

boxplot(test_mean ~ algoType, 
        xlab="Algorithm Type",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Algorithm",
        col=c("lightblue","lightgreen","lightyellow"),
        data=df)

m <- ddply(df, ~ algoType, function(data) c(
  "Mean"=mean(data$test_mean),  
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ algoType,  # b stores X midpoint of each bar
             xlab="Algorithm Type",
             ylab="Recognition Error Rate",
             ylim=c(0,.5),
             main="Recognition Error Rate by Algorithm",
             col=c("lightblue","lightgreen","lightyellow"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen","goldenrod"))


boxplot(test_mean ~ ptype, 
        xlab="Disability",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Disability",
        col=c("lightblue","lightgreen"),
        data=df)

m <- ddply(df, ~ ptype, function(data) c(
  "Mean"=mean(data$test_mean), 
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ ptype,  # b stores X midpoint of each bar
             xlab="Disability",
             ylab="Recognition Error Rate",
             ylim=c(0,.4),
             main="Recognition Error Rate by Disability",
             col=c("lightblue","lightgreen"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen"))

with(df, interaction.plot(
  algoType, ptype, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  trace.label="Disability",
  ylim=c(0,.4),
  main="Recognition Error Rate by Algorithm, Disability", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen")
))
m <- ddply(df, ~ algoType + ptype, function(data) c(
  "Mean"=mean(data$test_mean),
  "SD"=sd(data$test_mean)
))
dx = 0.005  # nudge
arrows(x0=1-dx, y0=m[1,]$Mean - m[1,]$SD, x1=1-dx, y1=m[1,]$Mean + m[1,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=1+dx, y0=m[2,]$Mean - m[2,]$SD, x1=1+dx, y1=m[2,]$Mean + m[2,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=2-dx, y0=m[3,]$Mean - m[3,]$SD, x1=2-dx, y1=m[3,]$Mean + m[3,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=2+dx, y0=m[4,]$Mean - m[4,]$SD, x1=2+dx, y1=m[4,]$Mean + m[4,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=3-dx, y0=m[5,]$Mean - m[5,]$SD, x1=3-dx, y1=m[5,]$Mean + m[5,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=3+dx, y0=m[6,]$Mean - m[6,]$SD, x1=3+dx, y1=m[6,]$Mean + m[6,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")


## goodness-of-fit histograms

hist(df$test_mean, main="Histogram of Recognition Error Rate", freq=TRUE, xlab="Error Rate", xlim=c(0,1)) # frequency (counts)
hist(df$test_mean, main="Histogram of Recognition Error Rate", freq=FALSE, xlab="Error Rate", xlim=c(0,1)) # density (area sums to 1.00)

f = fitdistr(df$test_mean, "normal")$estimate  # normal
curve(dnorm(x, mean=f[1], sd=f[2]), col="blue", lty=1, lwd=3, add=TRUE) # add normal curve
ks.test(df$test_mean, "pnorm", mean=f[1], sd=f[2]) # p-value = 3.234e-12 # p = 1.935e-12

f = fitdistr(df$test_mean, "lognormal")$estimate  # lognormal
curve(dlnorm(x, meanlog=f[1], sdlog=f[2]), col="purple", lty=1, lwd=3, add=TRUE) # add lognormal curve
ks.test(df$test_mean, "plnorm", meanlog=f[1], sdlog=f[2]) #p-value < 2.2e-16 # p < 2.2e-16

f = fitdistr(round(df$test_mean,0), "Poisson")$estimate  # Poisson
curve(dpois(round(x,0), lambda=f[1]), col="gold", lty=1, lwd=3, add=TRUE) # add Poisson curve
ks.test(df$test_mean, "ppois", lambda=f[1]) # p-value < 2.2e-16 # p < 2.2e-16

f = fitdistr(round(df$test_mean,0), "negative binomial", lower=1e-6)$estimate  # negative binomial
curve(dnbinom(round(x,0), size=f[1], mu=f[2]), col="darkgray", lty=1, lwd=3, add=TRUE) # add negative binomial curve
ks.test(df$test_mean, "pnbinom", size=f[1], mu=f[2]) # p-value = NA # p < 2.2e-16

f = fitdistr(df$test_mean, "exponential")$estimate  # exponential
curve(dexp(x, rate=f[1]), col="darkred", lty=1, lwd=3, add=TRUE) # add exponential curve
ks.test(df$test_mean, "pexp", rate=f[1]) # p-value = 0.0001412 # p = 0.000166

df$test_mean_1 = df$test_mean + 1 ## define new column
hist(df$test_mean_1, main="Histogram of Recognition Error Rate", freq=FALSE, xlab="Error Rate", xlim=c(1,2)) # density (area sums to 1.00)
f = fitdistr(df$test_mean_1, "gamma")$estimate  # Gamma
curve(dgamma(x, shape=f[1], rate=f[2]), col="darkgreen", lty=1, lwd=3, add=TRUE) # add Gamma curve
ks.test(df$test_mean_1, "pgamma", shape=f[1], rate=f[2]) # p-value = 7.656e-10# p = 1.381e-09


## ANOVA assumptions
m = aov_ez(dv="test_mean", within=c("algoType","nTemplates"), between="ptype", id="pID", type=3, data=df)
r = residuals(m$lm)
length(r) # 918
mean(r); sum(r) # should be ~0
plot(r[1:length(r)]); abline(h=0) # should be random
hist(r) # should be normal

hist(r, freq=FALSE)
f = fitdistr(r, "normal")$estimate  # normal
curve(dnorm(x, mean=f[1], sd=f[2]), col="blue", lty=1, lwd=3, add=TRUE) # add normal curve
ks.test(r, "pnorm", mean=f[1], sd=f[2]) # p-value = 0.0001343 # p = 0.0002092

rg = as.numeric(unlist(r + abs(min(r)) + 1))
hist(rg, freq=FALSE)
f = fitdistr(rg, "gamma")$estimate  # Gamma
curve(dgamma(x, shape=f[1], rate=f[2]), col="darkgreen", lty=1, lwd=3, add=TRUE) # add Gamma curve
ks.test(rg, "pgamma", shape=f[1], rate=f[2]) # p-value = 0.001177 # p = 0.003216

qqnorm(r); qqline(r) # Q-Q plot
shapiro.test(r)  # p-value = 7.946e-10 # p < 2.2e-16
ad.test(r)  # p-value = 2.865e-15 # p < 2.2e-16

print(check_homogeneity(m))# p = 0.363 # Levene's Test: p = 0.308
print(check_normality(m))   # p < .001 # p < .001


## Gamma regression

# no interactions
m0 = glmer(test_mean_1 ~ algoType + ptype + nTemplates + (1|pID), data=df, family=Gamma(link="inverse"))
summary(m0)
Anova(m0, type=3)


# full factorial model

#df$nTemplates = as.numeric(df$nTemplates) # templates as continuous
m1 = glmer(test_mean_1 ~ algoType * nTemplates * ptype + (1|pID), data=df, family=Gamma(link="inverse"))
summary(m1)
Anova(m1, type=3) ### preferred model

# Response: test_mean_1
#                               Chisq Df Pr(>Chisq)    
# (Intercept)               2846.2355  1  < 2.2e-16 ***
# algoType                   170.7059  2  < 2.2e-16 ***
# nTemplates                 882.3611  1  < 2.2e-16 ***
# ptype                        0.5058  1     0.4769    
# algoType:nTemplates         69.0748  2  1.001e-15 ***
# algoType:ptype               0.7758  2     0.6785    
# nTemplates:ptype             1.5299  1     0.2161    
# algoType:nTemplates:ptype    2.0409  2     0.3604  


#Response: test_mean_1 OLD
#                              Chisq Df Pr(>Chisq)    
#(Intercept)               2362.8503  1    < 2e-16 ***
#algoType                   139.5921  2    < 2e-16 ***
#nTemplates                 439.3849  1    < 2e-16 ***
#ptype                        1.1685  1    0.27971    
#algoType:nTemplates          6.5383  2    0.03804 *  
#algoType:ptype               0.0595  2    0.97071    
#nTemplates:ptype             3.5157  1    0.06079 .  
#algoType:nTemplates:ptype    1.6790  2    0.43193

with(df, interaction.plot(
  algoType, nTemplates, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  ylim=c(0,0.5),
  main="Recognition Error Rate by Algorithm, No. Templates", 
  lwd=3, 
  lty=1,
  col=rainbow(9)
))

# use Holm's sequential Bonferroni procedure to correct for multiple comparisons
summary(glht(m1, emm(pairwise ~ algoType)), test=adjusted(type="holm"))
summary(glht(m1, emm(pairwise ~ nTemplates*algoType)), test=adjusted(type="holm"))


# Post hoc pairwise comparisons, corrected with Holm's sequential Bonferroni procedure, 
# indicated that all three algorithms are significantly different in terms of recognition 
# error rate (p < .0001).







##
## 2. gesture_articulation.csv
##
df <- read.csv("gesture_articulation.csv")
df$pID = as.factor(df$pID)
df$algoType = as.factor(df$algoType)
df$gestureType = as.factor(df$gestureType)
df$ptype = as.factor(df$ptype)

contrasts(df$algoType) <- "contr.sum"
contrasts(df$gestureType) <- "contr.sum"
contrasts(df$ptype) <- "contr.sum"

View(df)


# EDA
ddply(df, ~ algoType + gestureType + ptype, function(data) c(
  "Nrows"=nrow(data),
  "Min"=min(data$test_mean),
  "Mean"=mean(data$test_mean), 
  "SD"=sd(data$test_mean),
  "Var"=var(data$test_mean),
  "Median"=median(data$test_mean),
  "IQR"=IQR(data$test_mean),
  "Max"=max(data$test_mean)
))

## algorithm
boxplot(test_mean ~ algoType, 
        xlab="Algorithm Type",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Algorithm",
        col=c("lightblue","lightgreen","lightyellow"),
        data=df)

m <- ddply(df, ~ algoType, function(data) c(
  "Mean"=mean(data$test_mean),  
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ algoType,  # b stores X midpoint of each bar
             xlab="Algorithm Type",
             ylab="Recognition Error Rate",
             ylim=c(0,1),
             main="Recognition Error Rate by Algorithm",
             col=c("lightblue","lightgreen","lightyellow"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen","goldenrod"))


## gesture type
boxplot(test_mean ~ gestureType, 
        xlab="Gesture Type",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Gesture Type",
        col=c("lightblue","lightgreen","lightyellow"),
        data=df)

m <- ddply(df, ~ gestureType, function(data) c(
  "Mean"=mean(data$test_mean),  
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ gestureType,  # b stores X midpoint of each bar
             xlab="Gesture Type",
             ylab="Recognition Error Rate",
             ylim=c(0,1),
             main="Recognition Error Rate by Gesture Type",
             col=c("lightblue","lightgreen","lightyellow"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen","goldenrod"))


## disability
boxplot(test_mean ~ ptype, 
        xlab="Disability",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Disability",
        col=c("lightblue","lightgreen"),
        data=df)

m <- ddply(df, ~ ptype, function(data) c(
  "Mean"=mean(data$test_mean), 
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ ptype,  # b stores X midpoint of each bar
             xlab="Disability",
             ylab="Recognition Error Rate",
             ylim=c(0,1),
             main="Recognition Error Rate by Disability",
             col=c("lightblue","lightgreen"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen"))


# Algorithm x Disability
with(df, interaction.plot(
  algoType, ptype, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  trace.label="Disability",
  ylim=c(0,1),
  main="Recognition Error Rate by Algorithm, Disability", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen")
))
m <- ddply(df, ~ algoType + ptype, function(data) c(
  "Mean"=mean(data$test_mean),
  "SD"=sd(data$test_mean)
))
dx = 0.005  # nudge
arrows(x0=1-dx, y0=m[1,]$Mean - m[1,]$SD, x1=1-dx, y1=m[1,]$Mean + m[1,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=1+dx, y0=m[2,]$Mean - m[2,]$SD, x1=1+dx, y1=m[2,]$Mean + m[2,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=2-dx, y0=m[3,]$Mean - m[3,]$SD, x1=2-dx, y1=m[3,]$Mean + m[3,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=2+dx, y0=m[4,]$Mean - m[4,]$SD, x1=2+dx, y1=m[4,]$Mean + m[4,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=3-dx, y0=m[5,]$Mean - m[5,]$SD, x1=3-dx, y1=m[5,]$Mean + m[5,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=3+dx, y0=m[6,]$Mean - m[6,]$SD, x1=3+dx, y1=m[6,]$Mean + m[6,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")


# Algorithm x Gesture Type
with(df, interaction.plot(
  algoType, gestureType, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  ylim=c(0,1),
  main="Recognition Error Rate by Algorithm, Gesture Type", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen","goldenrod")
))
m <- ddply(df, ~ algoType + gestureType, function(data) c(
  "Mean"=mean(data$test_mean),
  "SD"=sd(data$test_mean)
))
dx = 0.015  # nudge
arrows(x0=1-dx, y0=m[1,]$Mean - m[1,]$SD, x1=1-dx, y1=m[1,]$Mean + m[1,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=1+00, y0=m[2,]$Mean - m[2,]$SD, x1=1+00, y1=m[2,]$Mean + m[2,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")
arrows(x0=1+dx, y0=m[3,]$Mean - m[3,]$SD, x1=1+dx, y1=m[3,]$Mean + m[3,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="goldenrod")

arrows(x0=2-dx, y0=m[4,]$Mean - m[4,]$SD, x1=2-dx, y1=m[4,]$Mean + m[4,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=2+00, y0=m[5,]$Mean - m[5,]$SD, x1=2+00, y1=m[5,]$Mean + m[5,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")
arrows(x0=2+dx, y0=m[6,]$Mean - m[6,]$SD, x1=2+dx, y1=m[6,]$Mean + m[6,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="goldenrod")

arrows(x0=3-dx, y0=m[7,]$Mean - m[7,]$SD, x1=3-dx, y1=m[7,]$Mean + m[7,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=3+00, y0=m[8,]$Mean - m[8,]$SD, x1=3+00, y1=m[8,]$Mean + m[8,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")
arrows(x0=3+dx, y0=m[9,]$Mean - m[9,]$SD, x1=3+dx, y1=m[9,]$Mean + m[9,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="goldenrod")


# Gesture Type x Disability
with(df, interaction.plot(
  gestureType, ptype, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  trace.label="Disability",
  ylim=c(0,1),
  main="Recognition Error Rate by Gesture Type, Disability", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen")
))
m <- ddply(df, ~ gestureType + ptype, function(data) c(
  "Mean"=mean(data$test_mean),
  "SD"=sd(data$test_mean)
))
dx = 0.005  # nudge
arrows(x0=1-dx, y0=m[1,]$Mean - m[1,]$SD, x1=1-dx, y1=m[1,]$Mean + m[1,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=1+dx, y0=m[2,]$Mean - m[2,]$SD, x1=1+dx, y1=m[2,]$Mean + m[2,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=2-dx, y0=m[3,]$Mean - m[3,]$SD, x1=2-dx, y1=m[3,]$Mean + m[3,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=2+dx, y0=m[4,]$Mean - m[4,]$SD, x1=2+dx, y1=m[4,]$Mean + m[4,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=3-dx, y0=m[5,]$Mean - m[5,]$SD, x1=3-dx, y1=m[5,]$Mean + m[5,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=3+dx, y0=m[6,]$Mean - m[6,]$SD, x1=3+dx, y1=m[6,]$Mean + m[6,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")


## goodness-of-fit histograms

hist(df$test_mean, main="Histogram of Recognition Error Rate", freq=TRUE, xlab="Error Rate", xlim=c(0,1)) # frequency (counts)
hist(df$test_mean, main="Histogram of Recognition Error Rate", freq=FALSE, xlab="Error Rate", xlim=c(0,1)) # density (area sums to 1.00)

f = fitdistr(df$test_mean, "normal")$estimate  # normal
curve(dnorm(x, mean=f[1], sd=f[2]), col="blue", lty=1, lwd=3, add=TRUE) # add normal curve
ks.test(df$test_mean, "pnorm", mean=f[1], sd=f[2]) # p< 2.2e-16

f = fitdistr(df$test_mean, "lognormal")$estimate  # lognormal
curve(dlnorm(x, meanlog=f[1], sdlog=f[2]), col="purple", lty=1, lwd=3, add=TRUE) # add lognormal curve
ks.test(df$test_mean, "plnorm", meanlog=f[1], sdlog=f[2]) # p < 2.2e-16

f = fitdistr(round(df$test_mean,0), "Poisson")$estimate  # Poisson
curve(dpois(round(x,0), lambda=f[1]), col="gold", lty=1, lwd=3, add=TRUE) # add Poisson curve
ks.test(df$test_mean, "ppois", lambda=f[1]) # p < 2.2e-16

f = fitdistr(round(df$test_mean,0), "negative binomial", lower=1e-6)$estimate  # negative binomial
curve(dnbinom(round(x,0), size=f[1], mu=f[2]), col="darkgray", lty=1, lwd=3, add=TRUE) # add negative binomial curve
ks.test(df$test_mean, "pnbinom", size=f[1], mu=f[2]) # p < 2.2e-16

f = fitdistr(df$test_mean, "exponential")$estimate  # exponential
curve(dexp(x, rate=f[1]), col="darkred", lty=1, lwd=3, add=TRUE) # add exponential curve
ks.test(df$test_mean, "pexp", rate=f[1]) # p < 2.2e-16

df$test_mean_1 = df$test_mean + 1 ## add a new column
hist(df$test_mean_1, main="Histogram of Recognition Error Rate", freq=FALSE, xlab="Error Rate", xlim=c(1,2)) # density (area sums to 1.00)
f = fitdistr(df$test_mean_1, "gamma")$estimate  # Gamma
curve(dgamma(x, shape=f[1], rate=f[2]), col="darkgreen", lty=1, lwd=3, add=TRUE) # add Gamma curve
ks.test(df$test_mean_1, "pgamma", shape=f[1], rate=f[2]) # NaNs!

df$test_mean_100 = round(df$test_mean * 100, 0)
View(df)

hist(df$test_mean_100, main="Histogram of Recognition Errors per 100", freq=FALSE, xlab="Errors", xlim=c(0,100)) # density (area sums to 1.00)

f = fitdistr(round(df$test_mean_100,0), "Poisson")$estimate  # Poisson
curve(dpois(round(x,0), lambda=f[1]), col="gold", lty=1, lwd=3, add=TRUE) # add Poisson curve
ks.test(df$test_mean_100, "ppois", lambda=f[1]) # p < 2.2e-16

f = fitdistr(round(df$test_mean_100,0), "negative binomial", lower=1e-6)$estimate  # negative binomial
curve(dnbinom(round(x,0), size=f[1], mu=f[2]), col="darkgray", lty=1, lwd=3, add=TRUE) # add negative binomial curve
ks.test(df$test_mean_100, "pnbinom", size=f[1], mu=f[2]) # p < 2.2e-16


## ANOVA assumptions
m = aov_ez(dv="test_mean", within=c("algoType","gestureType","nTemplates"), between="ptype", id="pID", type=3, data=df)
r = residuals(m$lm)
length(r) # 2673
mean(r); sum(r) # should be ~0
plot(r[1:length(r)]); abline(h=0) # should be random
hist(r) # should be normal

hist(r, freq=FALSE)
f = fitdistr(r, "normal")$estimate  # normal
curve(dnorm(x, mean=f[1], sd=f[2]), col="blue", lty=1, lwd=3, add=TRUE) # add normal curve
ks.test(r, "pnorm", mean=f[1], sd=f[2]) # p-value = 8.57e-08 # p = 6.993e-05

rg = as.numeric(unlist(r + abs(min(r)) + 1))
hist(rg, freq=FALSE)
f = fitdistr(rg, "gamma")$estimate  # Gamma
curve(dgamma(x, shape=f[1], rate=f[2]), col="darkgreen", lty=1, lwd=3, add=TRUE) # add Gamma curve
ks.test(rg, "pgamma", shape=f[1], rate=f[2]) # p-value = 0.001616 # 0.0002077

qqnorm(r); qqline(r) # Q-Q plot
shapiro.test(r)  # p-value = 1.469e-13 # p < 2.2e-16
ad.test(r) # p-value < 2.2e-16 # p < 2.2e-16

print(check_homogeneity(m)) # Levene's Test: p = 0.896
print(check_normality(m))   # p < .001


## fit a linear mixed model (LMM)
m = lmer(test_mean ~ algoType * gestureType * ptype * nTemplates + (1|pID), data=df)
summary(m)
Anova(m, type=3)

#Response: test_mean
#                                         Chisq Df Pr(>Chisq)    
#(Intercept)                           330.6834  1  < 2.2e-16 ***
#algoType                              208.2876  2  < 2.2e-16 ***
#gestureType                            16.7365  2  0.0002321 ***
#ptype                                   3.2844  1  0.0699410 .  
#nTemplates                             31.0228  1   2.55e-08 ***
#algoType:gestureType                    4.1388  4  0.3875519    
#algoType:ptype                         16.2449  2  0.0002968 ***
#gestureType:ptype                       4.2837  2  0.1174394    
#algoType:nTemplates                     3.4965  2  0.1740748    
#gestureType:nTemplates                  0.4043  2  0.8169636    
#ptype:nTemplates                        0.1041  1  0.7469081    
#algoType:gestureType:ptype              8.2184  4  0.0838990 .  
#algoType:gestureType:nTemplates         1.5020  4  0.8262889    
#algoType:ptype:nTemplates               1.8762  2  0.3913641    
#gestureType:ptype:nTemplates            0.1990  2  0.9052809    
#algoType:gestureType:ptype:nTemplates   1.1438  4  0.8872568


## fit count models and ZI count models

#Poisson
m0 = glmer(test_mean_100 ~ algoType * gestureType * ptype * nTemplates + (1|pID), data=df, family=poisson)
summary(m0)
Anova(m0, type=3)
check_zeroinflation(m0)

m1 = glmmTMB(test_mean_100 ~ algoType * gestureType * ptype * nTemplates + (1|pID), 
             data=df, 
             family=poisson, 
             ziformula=~algoType*gestureType*ptype*nTemplates
)


# Negative Binomial
m2 = glmer.nb(test_mean_100 ~ algoType * gestureType * ptype * nTemplates + (1|pID), data=df)
summary(m2)
Anova(m2, type=3)

#Response: test_mean_100
#                                          Chisq Df Pr(>Chisq)    
#(Intercept)                           1146.6396  1  < 2.2e-16 ***
#algoType                               202.9578  2  < 2.2e-16 ***
#gestureType                             12.8030  2   0.001659 ** 
#ptype                                    3.5363  1   0.060038 .  
#nTemplates                              40.0428  1  2.485e-10 ***
#algoType:gestureType                     6.0873  4   0.192725    
#algoType:ptype                          30.2459  2  2.705e-07 ***
#gestureType:ptype                        3.1013  2   0.212111    
#algoType:nTemplates                      8.6248  2   0.013401 *  
#gestureType:nTemplates                   0.6393  2   0.726401    
#ptype:nTemplates                         4.9367  1   0.026292 *  
#algoType:gestureType:ptype               2.3331  4   0.674749    
#algoType:gestureType:nTemplates          3.7346  4   0.443115    
#algoType:ptype:nTemplates                0.5733  2   0.750774    
#gestureType:ptype:nTemplates             0.1496  2   0.927921    
#algoType:gestureType:ptype:nTemplates    5.4997  4   0.239754

boxplot(test_mean_100 ~ nTemplates, 
        xlab="Disability",
        ylab="Recognition Errors per 100",
        ylim=c(0,100),
        main="Recognition Errors per 100 by No. Templates",
        col=rainbow(9),
        data=df)

with(df, interaction.plot(
  algoType, ptype, test_mean_100, 
  xlab="Algorithm", 
  ylab="Recognition Errors per 100", 
  trace.label="Disability",
  ylim=c(0,100),
  main="Recognition Errors per 100 by Algorithm, Disability", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen")
))

with(df, interaction.plot(
  algoType, nTemplates, test_mean_100, 
  xlab="Algorithm", 
  ylab="Recognition Errors per 100", 
  trace.label="Templates",
  ylim=c(0,100),
  main="Recognition Errors per 100 by Algorithm, No. Templates", 
  lwd=3, 
  lty=1,
  col=rainbow(9)
))

# post hoc pairwise comparisons
summary(glht(m2, emm(pairwise ~ algoType*gestureType*ptype)), test=adjusted(type="holm"))


# not preferred
m3 = glmmTMB(test_mean_100 ~ algoType * gestureType * ptype * nTemplates + (1|pID), 
             data=df, 
             family=nbinom2, 
             ziformula=~algoType*gestureType*ptype*nTemplates
)
summary(m3)
Anova(m3, type=3)


## Gamma -- preferred
df$test_mean_1 = df$test_mean + 1

m = glmer(test_mean_1 ~ algoType * gestureType * ptype * nTemplates + (1|pID), 
          data=df, 
          family=Gamma(link="inverse")
)
summary(m)
Anova(m, type=3) ### preferred

# Response: test_mean_1
# Chisq Df Pr(>Chisq)    
# (Intercept)                           1129.4057  1  < 2.2e-16 ***
#   algoType                               192.6862  2  < 2.2e-16 ***
#   gestureType                             15.7443  2  0.0003812 ***
#   ptype                                    3.1040  1  0.0781003 .  
# nTemplates                              55.0981  1  1.147e-13 ***
#   algoType:gestureType                     2.4520  4  0.6532538    
# algoType:ptype                          29.8251  2  3.339e-07 ***
#   gestureType:ptype                        1.4022  2  0.4960418    
# algoType:nTemplates                     21.7434  2  1.899e-05 ***
#   gestureType:nTemplates                   0.4436  2  0.8010858    
# ptype:nTemplates                         0.5417  1  0.4617084    
# algoType:gestureType:ptype               6.6023  4  0.1584607    
# algoType:gestureType:nTemplates          1.1617  4  0.8843663    
# algoType:ptype:nTemplates                3.0248  2  0.2203751    
# gestureType:ptype:nTemplates             0.5394  2  0.7635910    
# algoType:gestureType:ptype:nTemplates    2.1287  4  0.7121091 


#Response: test_mean_1 OLD
#                                          Chisq Df Pr(>Chisq)    
#(Intercept)                           1698.2309  1  < 2.2e-16 ***
#algoType                               182.3948  2  < 2.2e-16 ***
#gestureType                             13.1051  2   0.001426 ** 
#ptype                                    3.4767  1   0.062237 .  
#nTemplates                              31.6487  1  1.847e-08 ***
#algoType:gestureType                     3.7452  4   0.441585    
#algoType:ptype                          24.4396  2  4.932e-06 ***
#gestureType:ptype                        2.0430  2   0.360057    
#algoType:nTemplates                      4.6405  2   0.098247 .  
#gestureType:nTemplates                   0.7735  2   0.679264    
#ptype:nTemplates                         0.4603  1   0.497495    
#algoType:gestureType:ptype               5.5265  4   0.237411    
#algoType:gestureType:nTemplates          3.0475  4   0.549900    
#algoType:ptype:nTemplates                2.3113  2   0.314858    
#gestureType:ptype:nTemplates             0.1392  2   0.932748    
#algoType:gestureType:ptype:nTemplates    2.2276  4   0.693988


# use Holm's sequential Bonferroni procedure to correct for multiple comparisons
summary(glht(m, emm(pairwise ~ algoType)), test=adjusted(type="holm"))

summary(glht(m, emm(pairwise ~ gestureType)), test=adjusted(type="holm"))

summary(glht(m, emm(pairwise ~ algoType*ptype)), test=adjusted(type="holm"))

# Post hoc pairwise comparisons, corrected with Holm's sequential Bonferroni procedure, 
# indicated that...


summary(glht(m, emm(pairwise ~ algoType*gestureType*ptype)), test=adjusted(type="none"))

# DEMO: manually conduct your pairwise comparisons by hand
p.adjust(c(
    0.0001,
    0.0005,
    0.0323),
  method="holm")



##
## 3. user_independent.csv
##
df <- read.csv("user_independent.csv")
df$pID = as.factor(df$pID)
df$algoType = as.factor(df$algoType)
df$ptype = as.factor(df$ptype)
df$nTemplates = as.factor(df$nTemplates)

contrasts(df$algoType) <- "contr.sum"
contrasts(df$ptype) <- "contr.sum"
contrasts(df$nTemplates) <- "contr.sum"

View(df)


# EDA
ddply(df, ~ algoType + ptype, function(data) c(
  "Nrows"=nrow(data),
  "Min"=min(data$test_mean),
  "Mean"=mean(data$test_mean), 
  "SD"=sd(data$test_mean),
  "Var"=var(data$test_mean),
  "Median"=median(data$test_mean),
  "IQR"=IQR(data$test_mean),
  "Max"=max(data$test_mean)
))

boxplot(test_mean ~ algoType, 
        xlab="Algorithm Type",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Algorithm",
        col=c("lightblue","lightgreen","lightyellow"),
        data=df)

m <- ddply(df, ~ algoType, function(data) c(
  "Mean"=mean(data$test_mean),  
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ algoType,  # b stores X midpoint of each bar
             xlab="Algorithm Type",
             ylab="Recognition Error Rate",
             ylim=c(0, 0.8),
             main="Recognition Error Rate by Algorithm",
             col=c("lightblue","lightgreen","lightyellow"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen","goldenrod"))


boxplot(test_mean ~ ptype, 
        xlab="Disability",
        ylab="Recognition Error Rate",
        ylim=c(0,1),
        main="Recognition Error Rate by Disability",
        col=c("lightblue","lightgreen"),
        data=df)

m <- ddply(df, ~ ptype, function(data) c(
  "Mean"=mean(data$test_mean), 
  "SD"=sd(data$test_mean)       
))
b <- barplot(Mean ~ ptype,  # b stores X midpoint of each bar
             xlab="Disability",
             ylab="Recognition Error Rate",
             ylim=c(0, 0.8),
             main="Recognition Error Rate by Disability",
             col=c("lightblue","lightgreen"), 
             data=m)
arrows(x0 = b,  # error bars
       y0 = m$Mean + m$SD,
       y1 = m$Mean - m$SD,
       angle=90,
       code=3,
       lwd=2,
       length=0.3,
       col=c("blue","darkgreen"))

with(df, interaction.plot(
  algoType, ptype, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  trace.label="Disability",
  ylim=c(0, 0.8),
  main="Recognition Error Rate by Algorithm, Disability", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen")
))
m <- ddply(df, ~ algoType + ptype, function(data) c(
  "Mean"=mean(data$test_mean),
  "SD"=sd(data$test_mean)
))
dx = 0.005  # nudge
arrows(x0=1-dx, y0=m[1,]$Mean - m[1,]$SD, x1=1-dx, y1=m[1,]$Mean + m[1,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=1+dx, y0=m[2,]$Mean - m[2,]$SD, x1=1+dx, y1=m[2,]$Mean + m[2,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=2-dx, y0=m[3,]$Mean - m[3,]$SD, x1=2-dx, y1=m[3,]$Mean + m[3,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=2+dx, y0=m[4,]$Mean - m[4,]$SD, x1=2+dx, y1=m[4,]$Mean + m[4,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")

arrows(x0=3-dx, y0=m[5,]$Mean - m[5,]$SD, x1=3-dx, y1=m[5,]$Mean + m[5,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkblue")
arrows(x0=3+dx, y0=m[6,]$Mean - m[6,]$SD, x1=3+dx, y1=m[6,]$Mean + m[6,]$SD, angle=90, code=3, lty=1, lwd=2, length=0.3, col="darkgreen")


## goodness-of-fit histograms

hist(df$test_mean, main="Histogram of Recognition Error Rate", freq=TRUE, xlab="Error Rate", xlim=c(0,1)) # frequency (counts)
hist(df$test_mean, main="Histogram of Recognition Error Rate", freq=FALSE, xlab="Error Rate", xlim=c(0,1)) # density (area sums to 1.00)

f = fitdistr(df$test_mean, "normal")$estimate  # normal
curve(dnorm(x, mean=f[1], sd=f[2]), col="blue", lty=1, lwd=3, add=TRUE) # add normal curve
ks.test(df$test_mean, "pnorm", mean=f[1], sd=f[2]) # p = 0.1768

f = fitdistr(df$test_mean, "lognormal")$estimate  # lognormal
curve(dlnorm(x, meanlog=f[1], sdlog=f[2]), col="purple", lty=1, lwd=3, add=TRUE) # add lognormal curve
ks.test(df$test_mean, "plnorm", meanlog=f[1], sdlog=f[2]) # p < 2.2e-16

f = fitdistr(round(df$test_mean,0), "Poisson")$estimate  # Poisson
curve(dpois(round(x,0), lambda=f[1]), col="gold", lty=1, lwd=3, add=TRUE) # add Poisson curve
ks.test(df$test_mean, "ppois", lambda=f[1]) # p < 2.2e-16

f = fitdistr(round(df$test_mean,0), "negative binomial", lower=1e-6)$estimate  # negative binomial
curve(dnbinom(round(x,0), size=f[1], mu=f[2]), col="darkgray", lty=1, lwd=3, add=TRUE) # add negative binomial curve
ks.test(df$test_mean, "pnbinom", size=f[1], mu=f[2]) # p < 2.2e-16

f = fitdistr(df$test_mean, "exponential")$estimate  # exponential
curve(dexp(x, rate=f[1]), col="darkred", lty=1, lwd=3, add=TRUE) # add exponential curve
ks.test(df$test_mean, "pexp", rate=f[1]) # p = 0.0004948

df$test_mean_1 = df$test_mean + 1  ## add new column
hist(df$test_mean_1, main="Histogram of Recognition Error Rate", freq=FALSE, xlab="Error Rate", xlim=c(1,2)) # density (area sums to 1.00)
f = fitdistr(df$test_mean_1, "gamma")$estimate  # Gamma
curve(dgamma(x, shape=f[1], rate=f[2]), col="darkgreen", lty=1, lwd=3, add=TRUE) # add Gamma curve
ks.test(df$test_mean_1, "pgamma", shape=f[1], rate=f[2]) # p = 0.2983


## ANOVA assumptions
m = aov_ez(dv="test_mean", within=c("algoType","nTemplates"), between="ptype", id="pID", type=3, data=df)
r = residuals(m$lm)
length(r) # 162
mean(r); sum(r) # should be ~0
plot(r[1:length(r)]); abline(h=0) # should be random
hist(r) # should be normal

hist(r, freq=FALSE)
f = fitdistr(r, "normal")$estimate  # normal
curve(dnorm(x, mean=f[1], sd=f[2]), col="blue", lty=1, lwd=3, add=TRUE) # add normal curve
ks.test(r, "pnorm", mean=f[1], sd=f[2]) # p-value = 0.6804 # p = 0.3995

rg = as.numeric(unlist(r + abs(min(r)) + 1))
hist(rg, freq=FALSE)
f = fitdistr(rg, "gamma")$estimate  # Gamma
curve(dgamma(x, shape=f[1], rate=f[2]), col="darkgreen", lty=1, lwd=3, add=TRUE) # add Gamma curve
ks.test(rg, "pgamma", shape=f[1], rate=f[2]) # p-value = 0.9216 # p = 0.5885

qqnorm(r); qqline(r) # Q-Q plot
shapiro.test(r)  # p-value = 0.009343 # p = 8.676e-05
ad.test(r) # p-value = 0.2144 # p = 0.0105

print(check_homogeneity(m)) # Levene's Test: p = 0.031
print(check_normality(m))   # p < .001



## LMM
m0 = lmer(test_mean ~ algoType * nTemplates * ptype + (1|pID), data=df)
summary(m0)
Anova(m0, type=3) ### 


## Gamma
df$test_mean_1 = df$test_mean + 1  ## add new column
m1 = glmer(test_mean_1 ~ algoType * nTemplates * ptype + (1|pID), data=df, family=Gamma(link="inverse"))
summary(m1)
Anova(m1, type=3) 

with(df, interaction.plot(
  algoType, ptype, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  trace.label="Disability",
  ylim=c(0,1),
  main="Recognition Error Rate by Algorithm, Disability", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen")
))

with(df, interaction.plot(
  algoType, nTemplates, test_mean, 
  xlab="Algorithm", 
  ylab="Recognition Error Rate", 
  trace.label="Templates",
  ylim=c(0,1),
  main="Recognition Error Rate by Algorithm, No. Templates", 
  lwd=3, 
  lty=1,
  col=c("darkblue","darkgreen","goldenrod")
))

# Response: test_mean_1
# Chisq Df Pr(>Chisq)    
# (Intercept)               1329.0791  1  < 2.2e-16 ***
#   algoType                   449.0180  2  < 2.2e-16 ***
#   nTemplates                  10.4172  2   0.005469 ** 
#   ptype                        3.2481  1   0.071505 .  
# algoType:nTemplates          1.1207  4   0.890970    
# algoType:ptype               9.1729  2   0.010189 *  
#   nTemplates:ptype             2.0943  2   0.350929    
# algoType:nTemplates:ptype    2.0338  4   0.729549  

#Response: test_mean_1 OLD
#                              Chisq Df Pr(>Chisq)    
#(Intercept)               1402.3562  1  < 2.2e-16 ***
#algoType                   427.2484  2  < 2.2e-16 ***
#nTemplates                   1.1574  2  0.5606263    
#ptype                        3.4927  1  0.0616389 .  
#algoType:nTemplates          0.5105  4  0.9724710    
#algoType:ptype              15.6071  2  0.0004083 ***
#nTemplates:ptype             0.0377  2  0.9813328    
#algoType:nTemplates:ptype    2.4149  4  0.6599332


# use Holm's sequential Bonferroni procedure to correct for multiple comparisons
summary(glht(m1, emm(pairwise ~ algoType)), test=adjusted(type="holm"))
summary(glht(m1, emm(pairwise ~ nTemplates)), test=adjusted(type="holm"))
summary(glht(m1, emm(pairwise ~ ptype)), test=adjusted(type="holm"))

summary(glht(m1, emm(pairwise ~ algoType*ptype)), test=adjusted(type="holm"))

# Post hoc pairwise comparisons, corrected with Holm's sequential Bonferroni procedure, 
# indicated that ...


