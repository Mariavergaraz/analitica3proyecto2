import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact
from sklearn import neighbors
import joblib
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import train_test_split


# Conectar BD
conn=sql.connect('db_movies2')
cur=conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

#---------------------------------------------------------------------#
#-------------------SISTEMAS BASADOS EN POPULARIDAD-------------------#
#---------------------------------------------------------------------#

# Películas mejores calificadas
pd.read_sql("""select title, 
            avg(rating) as avg_rat,
            count(*) as see_num
            from df
            where rating<>0
            group by title
            order by avg_rat desc
            limit 10
            """, conn)

# Películas más vistas con promedio de calificación
pd.read_sql("""select title, 
            avg(iif(rating = 0, Null, rating)) as avg_rat,
            count(*) as see_num
            from df
            group by title
            order by see_num desc
            limit 10
            """, conn)


# Películas mejores calificadas por tiempo de duración
pd.read_sql("""select timestamp, title, 
            avg(iif(rating = 0, Null, rating)) as avg_rat,
            count(iif(rating = 0, Null, rating)) as rat_numb,
            count(*) as see_num
            from df
            group by  timestamp, title
            order by timestamp desc, avg_rat desc 
            limit 10
            """, conn)

#---------------------------------------------------------------------#
#---------SISTEMAS BASADOS EN CONTENIDO DE UN SOLO PRODUCTO-----------#
#---------------------------------------------------------------------#

# Base de datos de películas
dfp=pd.read_sql("""select * from movies left join genres using (movieId)""", conn)
dfp.info()

# Eliminar columnas que no presentan características
mov_dum=dfp.drop(columns=['movieId','title','genres'])


# Ejemplo de recomendación para una sola película
pelicula='Toy Story (1995)'
ind_pelicula=dfp[dfp['title']==pelicula].index.values.astype(int)[0]
similar_movies=mov_dum.corrwith(mov_dum.iloc[ind_pelicula,:],axis=1)
similar_movies=similar_movies.sort_values(ascending=False)
top_similar_movies=similar_movies.to_frame(name="correlación").iloc[0:11,]
top_similar_movies['title']=dfp["title"]
print(top_similar_movies)


# Top 10 de recomendaciones para película seleccionada
def recomendacion(pelicula = list(dfp['title'])):
     
    ind_pelicula=mov_dum[dfp['title']==pelicula].index.values.astype(int)[0]   #### obtener indice de pelicula seleccionado de lista
    similar_movies = mov_dum.corrwith(mov_dum.iloc[ind_pelicula,:],axis=1) ## correlación entre pelicula seleccionado y todos los otros
    similar_movies = similar_movies.sort_values(ascending=False) #### ordenar correlaciones
    top_similar_movies=similar_movies.to_frame(name="correlación").iloc[0:11,] ### el 11 es número de peliculas recomendados
    top_similar_movies['title']=dfp["title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    
    return top_similar_movies


print(interact(recomendacion))

#---------------------------------------------------------------------#
#-------SISTEMAS BASADOS EN CONTENIDO KNN DE UN SOLO PRODUCTO---------#
#---------------------------------------------------------------------#

# Entrenamiento del modelo

# Similitud entre componentes basado en el coseno
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(mov_dum)
dist, idlist = model.kneighbors(mov_dum)

distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(pelicula)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde

# Ejemplo de recomendación para una sola película
movie_list_name = []
movie_name='Toy Story (1995)'
movie_id = dfp[dfp['title'] == movie_name].index ### extraer el indice del pelicula
movie_id = movie_id[0] ## si encuentra varios solo guarde uno

for newid in idlist[movie_id]:
        movie_list_name.append(dfp.loc[newid].title) ### agrega el nombre de cada una de los id recomendados

movie_list_name

def movieRecommender(movie_name = list(dfp['title'].value_counts().index)):
    movie_list_name = []
    movie_id = dfp[dfp['title'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        movie_list_name.append(dfp.loc[newid].title)
    return movie_list_name

print(interact(movieRecommender))

#---------------------------------------------------------------------#
#------SISTEMAS BASADOS EN CONTENIDO KNN DE TODOS LOS PRODUCTOS-------#
#---------------------------------------------------------------------#
movies=pd.read_sql('select * from df', conn )

# Seleccionar usuario para recomendaciones
usuarios=pd.read_sql('select distinct userId from df',conn)

user_id = 15 ### para ejemplo manual

def recomendar(user_id=list(usuarios['userId'].value_counts().index)):
    
    # Seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select * from ratings where userId=:user',conn, params={'user':user_id})
    
    # convertir ratings del usuario a array
    l_movies_r=ratings['rating'].to_numpy()
    
    # Agregar la columna del ID y title del pelicula a dummie para filtrar y mostrar nombre
    dfp[['movieId','title']]=movies[['movieId','title']]
    #dfp.drop('genres', axis = 1, inplace=True)
    
    ### filtrar peliculas calificados por el usuario
    movies_r=dfp[dfp['movieId'].isin(l_movies_r)]
    
    ## eliminar columna nombre e isbn
    movies_r=movies_r.drop(columns=['movieId','title'])
    movies_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    ##centroide o perfil del usuario
    centroide=movies_r.groupby("indice").mean()
    
    
    ### filtrar peliculas no leídos
    movies_nr=dfp[~dfp['movieId'].isin(l_movies_r)]
    ## eliminbar nombre e isbn
    movies_nr=movies_nr.drop(columns=['movieId','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_b=movies.loc[ids][['movieId','title']]
    leidos=movies[movies['movieId'].isin(l_movies_r)][['movieId','title']]
    
    return recomend_b

recomendar(15)

print(interact(recomendar))

#---------------------------------------------------------------------#
#--------------SISTEMAS BASADOS EN FILTROS COLABORATIVO---------------#
#---------------------------------------------------------------------#

# Selección de calificaciones explicitas
ratings=pd.read_sql('select * from df where rating>0', conn)


# Escala de calificación
reader = Reader(rating_scale=(0, 5)) 

# Columnas en orden estándar
data = Dataset.load_from_df(ratings[['userId','title','rating']], reader)


#####Existen varios modelos 
models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline()] 
results = {}

###knnBasiscs: calcula el rating ponderando por distancia con usuario/Items
###KnnWith means: en la ponderación se resta la media del rating, y al final se suma la media general
####KnnwithZscores: estandariza el rating restando media y dividiendo por desviación 
####Knnbaseline: calculan el desvío de cada calificación con respecto al promedio y con base en esos calculan la ponderación


#### función para probar varios modelos ##########
model=models[1]
for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
    
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result


performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')

###################se escoge el mejor knn withmeans#########################
param_grid = { 'sim_options' : {'name': ['msd','cosine'], \
                                'min_support': [5], \
                                'user_based': [False, True]}
             }

### se afina si es basado en usuario o basado en ítem

gridsearchKNNWithMeans = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], \
                                      cv=2, n_jobs=2)
                                    
gridsearchKNNWithMeans.fit(data)


gridsearchKNNWithMeans.best_params["rmse"]
gridsearchKNNWithMeans.best_score["rmse"]
gs_model=gridsearchKNNWithMeans.best_estimator['rmse'] ### mejor estimador de gridsearch


################# Entrenar con todos los datos y Realizar predicciones con el modelo afinado

trainset = data.build_full_trainset() ### esta función convierte todos los datos en entrnamiento, las funciones anteriores dividen  en entrenamiento y evaluación
model=gs_model.fit(trainset) ## se reentrena sobre todos los datos posibles (sin dividir)

predset = trainset.build_anti_testset() ### crea una tabla con todos los usuarios y los libros que no han leido
#### en la columna de rating pone el promedio de todos los rating, en caso de que no pueda calcularlo para un item-usuario
len(predset)

predictions = gs_model.test(predset) ### función muy pesada, hace las predicciones de rating para todos los libros que no hay leido un usuario
### la funcion test recibe un test set constriuido con build_test method, o el que genera crosvalidate

predictions_df = pd.DataFrame(predictions) ### esta tabla se puede llevar a una base donde estarán todas las predicciones
predictions_df.shape
predictions_df.head()
predictions_df['r_ui'].unique() ### promedio de ratings
predictions_df.sort_values(by='est',ascending=False)

####### la predicción se puede hacer para un libro puntual
model.predict(uid=269397, iid='0446353205',r_ui='') ### uid debía estar en número e isb en comillas


##### funcion para recomendar los 10 libros con mejores predicciones y llevar base de datos para consultar resto de información
def recomendaciones(user_id,n_recomend=10):
    
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace")
    
    recomendados=pd.read_sql('''select a.*, b.book_title 
                             from reco a left join books_final b
                             on a.iid=b.isbn ''', conn)

    return(recomendados)


 
recomendaciones(user_id=179733,n_recomend=10)