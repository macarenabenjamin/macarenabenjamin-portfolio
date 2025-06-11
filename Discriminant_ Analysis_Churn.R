
# # # # # # # # ANALISIS DISCRIMINANTE


install.packages("readxl")
library(readxl)

Data.AD <- read_excel("Desktop/analisis multivariado/Base de datos/BBDD - CLUSTERS.xlsx")


install.packages("descr")
install.packages("MASS")
install.packages("biotools")
install.packages("mvnormtest")


library(MASS)
library(readxl) 
library(descr)
library(biotools)
library(mvnormtest)

Base1 <- Data.AD
Base2 <- scale(Base1[2:7], center = T)
Base2 <- data.frame(Base2)
row.names(Base2) <- Base1$Suscriptores
Base2
Base3 <- cbind(Base2,Base1$Clusters)


fit <- lda(data=Base3,Base1$Clusters~MRR+p_loaded_hrs+act_users_q+p_act_users+avg_day_act_user+churn_risk)

#Coef de la fun de fisher
fit

#Obtenemos la pertenencia predicha y se anexa al fichero de datos
fit.p <- predict(fit)
fit.p <- predict(fit)$class
Base3 <- data.frame(Base3,fit.p)
Base3


#Tabla cruzada de aciertos y errores de clasificacion o Matriz de Confusion
CrossTable(Base3$Base1.Cluster,Base3$fit.p,digits=2,format="SPSS",
           prop.c=FALSE, prop.chisq=FALSE, prop.t=FALSE,
           dnn=c("Grupo real","Grupo pronosticado"))

#Constraste de hipotesis 

fit.manova<-manova(data=Base3,cbind(MRR,p_loaded_hrs,act_users_q,p_act_users,avg_day_act_user,churn_risk)~Base1$Clusters) 
summary((fit.manova),test="Wilks")
summary((fit.manova),test="Wilks")$SS

std.b.MRRLD1=(sqrt(summary(fit.manova)$SS$Residuals[1,1]/215))*fit$scaling[1,1] 
std.b.p_loaded_hrsLD1=(sqrt(summary(fit.manova)$SS$Residuals[2,2]/215))*fit$scaling[2,1] 
std.b.act_users_qLD1=(sqrt(summary(fit.manova)$SS$Residuals[3,3]/215))*fit$scaling[3,1] 
std.b.p_act_usersLD1=(sqrt(summary(fit.manova)$SS$Residuals[4,4]/215))*fit$scaling[4,1] 
std.b.avg_day_act_userLD1=(sqrt(summary(fit.manova)$SS$Residuals[5,5]/215))*fit$scaling[5,1]
std.b.churn_riskLD1=(sqrt(summary(fit.manova)$SS$Residuals[6,6]/215))*fit$scaling[6,1]


#Calculo de la matriz de estructura 
SCPC.residual<-summary((fit.manova),test="Wilks")$SS$Residuals 
SCPC.residual
SCPC.residual.varianzas<-SCPC.residual/215
SCPC.residual.varianzas

#Matriz de correlaciones 
SCPC.residual.correlaciones<-cov2cor(SCPC.residual.varianzas) 
SCPC.residual.correlaciones

#Matriz de estructura
VariablesLD1 =
  SCPC.residual.correlaciones[1,1]*std.b.MRRLD1+
  SCPC.residual.correlaciones[1,2]*std.b.p_loaded_hrsLD1+
  SCPC.residual.correlaciones[1,3]*std.b.act_users_qLD1+
  SCPC.residual.correlaciones[1,4]*std.b.p_act_usersLD1+
  SCPC.residual.correlaciones[1,5]*std.b.avg_day_act_userLD1+
  SCPC.residual.correlaciones[1,6]*std.b.churn_riskLD1+

grupo <- c(rep("3",32),rep("2",1), rep("3",2),rep("2",2), rep("3",10),rep("2",1),rep("3",1),rep("2",1),rep("3",4),rep("2",20),rep("1",4), rep("2",139))
etiqueta <- c(rep("3",32),rep("2",1), rep("3",2),rep("2",2), rep("3",10),rep("2",1),rep("3",1),rep("2",1),rep("3",4),rep("2",20),rep("1",4), rep("2",139))
individuo <- c(rep(1:217))

grupo 
etiqueta 
individuo

datos.grafico <-data.frame(Base3,grupo,etiqueta,individuo,predict(fit)$x[,1],predict(fit)$x[,2]) 



install.packages("ggrepel")
install.packages("ggplot2")
library(ggrepel) 
library(ggplot2)
library(plyr)

datos.grafico<-rename(datos.grafico,c('predict.fit..x...1.'='LD1','predict.fit..x...2.'='LD2')) 
datos.grafico
View(datos.grafico)

#Probabilidades a posteriori

library(MASS)
predict(fit)$posterior
View(predict(fit)$posterior)

#Graficamos


ggplot(datos.grafico,aes(x=LD1, y=LD2, colour=etiqueta, label=individuo))+
  geom_point(size=2)+scale_color_manual(values=c('black', 'red', 'blue'))+
  geom_text_repel()+expand_limits(x=c(-4,5),y=c(-2,3))+ labs(x='LD1',y='LD2')+
  guides(colour='legend', label=FALSE)+ theme(legend.title=element_blank())





