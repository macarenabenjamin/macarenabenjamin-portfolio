
# # # # # # # # CLUSTERS - METODO NO JERARQUICO

install.packages("readxl")
library(readxl)

# Lee el archivo Excel y almacena su contenido en un dataframe
DATA <- read_excel("Desktop/analisis multivariado/Base de datos/BBDD.xlsx")

# Miro las varianzas de las variables 
sapply(DATA,var)

# Como son muy distintas, voy a estandarizar las variables

Stan.Data1 <- scale(DATA[2:7], center=T) #Todas las variables

# Hacemos un boxplot y vemos que ahora las variables se parecen mas entre si
boxplot(Stan.Data1)


# Vamos a ver el numero de clusters que nos arrojan los indices. 
# A priori sabemos que no pueden ser menos de 2 ni mas de 4, con lo que el numero optimo tiene que estar cerca de los 2 o 3 clusters.

install.packages("NbClust")
install.packages("factoextra")
library(NbClust)
library(factoextra)

# Veamos los diferentes metodos para determinar el numero de clusters

# Metodo del Codo (suma de los cuadrados dentro)
wss <- fviz_nbclust (Stan.Data1, kmeans, method=c("wss"))
wss 

# Metodo de la Silueta
silueta <- fviz_nbclust (Stan.Data1, kmeans, method=c("silhouette"))
silueta 


# Metodo Analisis del Gap (k)
gap <- fviz_nbclust (Stan.Data1, kmeans, method=c("gap_stat"))
gap


# Aplicacion de paquete NbClust con metodo de K-means

ResClustMNJ <- NbClust (Stan.Data1, distance ="euclidean",min.nc = 2, max.nc = 4, method = "kmeans", index="alllong")
ResClustMNJ$Best.nc
ResClustMNJ_data <- as.data.frame(ResClustMNJ$Best.partition)
ResClustMNJ_data

# Elegimos 3 clusters para segmentar los datos usando k-means

clusters <-kmeans(Stan.Data1, centers=3, iter.max=30,  nstart=20)
clusters$centers
as.data.frame(clusters$cluster)
clusters$size


# Grafico

fviz_cluster(clusters, data = Stan.Data1, main = "3 clusters (KMeans)")



#Listado Suscriptores x Cluster
DataClust <-cbind(Stan.Data1,clusters$cluster) 
DataClust



