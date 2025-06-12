# # # # # # # # PCA

install.packages("readxl")
library(readxl)

# Lee el archivo Excel y almacena su contenido en un dataframe
DATA1 <- read_excel("Desktop/analisis multivariado/Base de datos/BBDD.xlsx")
DATA2 <- DATA1[complete.cases(DATA1), ]
DATA3 <- DATA2[,-1]

# Miro la matriz de correlaciones 
cor(DATA3)


# Miro los autovalores y autovectores

eigen(cor(scale(DATA3)))

# Analisis de PCA estandarizado

pca <- prcomp(DATA3, scale=TRUE)
summary(pca)

# Para graficar los resultados

#Biplot

library(FactoMineR)

grafico.pca <- PCA(scale(DATA3), graph = FALSE)

pca$rotation
biplot(pca, scale=0,
       col=c('blue', 'red'),
       cex=c(0.5,0.7),
       main = 'Resultados PCA',
       xlab = 'Primera Componente',
       ylab='Segunda Componente')

#PCA individuales

plot.PCA(grafico.pca,choix="ind")



# Pido el grafico de Scree y un graffico de codo para la eleccion del numero de componentes

#Grafico de codo

plot(pca,type="lines",ylim=c(0,5), main="Grafico de codo")

#Screeplot

screeplot(pca)


# Calculo la proporcion de la varianza explicada

ve = pca$sdev^2/sum(pca$sdev^2)

# Creo el scree con ggplot

library(ggplot2)
qplot(c(1:6),ve)+
  geom_line()+
  xlab("Componente principal")+
  ylab("Varianza Explicada")+
  ggtitle("Scree Plot")+
  ylim(0,1)

#Screeplot proporcion de la varianza explicada acumulado

qplot(c(1:6),cumsum(ve))+
  geom_line()+
  xlab("Componente principal")+
  ylab("Varianza Explicada")+
  ggtitle("Scree Plot Acumulado")+
  ylim(0,1)


#Vizualiacion de contribucion de cada variable


library(factoextra)
library(corrplot)

#Contribuciones
var<-get_pca_var(pca)
var$cor
corrplot(var$contrib,is.corr=FALSE)

#Contribuciones de las 3 componentes

fviz_contrib(pca,choice="var",axes=1,top=10,ylim=c(0,100))
fviz_contrib(pca,choice="var",axes=2,top=10,ylim=c(0,100))
fviz_contrib(pca,choice="var",axes=3,top=10,ylim=c(0,100))


