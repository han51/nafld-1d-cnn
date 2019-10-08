# R code for AUC plot and test

rm(list = ls())   # clear up
graphics.off()
library("pROC")
library(ggplot2)

setwd("C:\\Users\\aiguo\\Desktop\\Second_submission")
data <- read.csv("notgc.csv", colClasses=c("numeric", "numeric"), header=TRUE)    

tiff("Fig4a.tiff", units="in", width=4, height=4, res=1200)
ROC_obj1 <- roc(data$labels, data$scores, smoothed = FALSE, ci = TRUE, ci.alpha = 0.95, stratified = FALSE,
                plot = TRUE, auc.polygon = FALSE, max.auc.polygon = TRUE, grid = TRUE, print.auc = TRUE, 
                print.auc.x = 0.75, print.auc.y = 0.2,
                show.thres = TRUE, legacy.axes = TRUE, add = FALSE, xlim = c(1,0), ylim = c(0, 1))
sens.ci1 <- ci.se(ROC_obj1)
plot(sens.ci1, type = "shape", col = "lightblue")
dev.off()

data2 <- read.csv("tgc.csv", colClasses=c("numeric", "numeric"), header=TRUE)    
tiff("Fig4b.tiff", units="in", width=4, height=4, res=1200)
ROC_obj2 <- roc(data2$labels, data2$scores, smoothed = FALSE, ci = TRUE, ci.alpha = 0.95, stratified = FALSE,
                plot = TRUE, auc.polygon = FALSE, max.auc.polygon = TRUE, grid = TRUE, print.auc = TRUE, 
                print.auc.x = 0.75, print.auc.y = 0.2,
                show.thres = TRUE, legacy.axes = TRUE, add = FALSE, xlim = c(1,0), ylim = c(0,1))
sens.ci2 <- ci.se(ROC_obj2)
plot(sens.ci2, type = "shape", col = "lightblue")
dev.off()

roc.test(ROC_obj1, ROC_obj2, method=c("delong", "bootstrap",
                              "venkatraman", "sensitivity", "specificity"), sensitivity = NULL,
         specificity = NULL, alternative = c("two.sided", "less", "greater"),
         paired=NULL, reuse.auc=TRUE, boot.n=2000, boot.stratified=TRUE,
         ties.method="first", progress=getOption("pROCProgress")$name,
         parallel=FALSE)
