import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact
from sklearn import neighbors
import joblib


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
            """, conn)


# Películas mejores calificadas por tiempo de duración
pd.read_sql("""select timestamp, title, 
            avg(iif(rating = 0, Null, rating)) as avg_rat,
            count(iif(rating = 0, Null, rating)) as rat_numb,
            count(*) as see_num
            from df
            group by  timestamp, title
            order by timestamp desc, avg_rat desc limit 20
            """, conn)

#---------------------------------------------------------------------#
#---------SISTEMAS BASADOS EN CONTENIDO DE UN SOLO PRODUCTO-----------#
#---------------------------------------------------------------------#

dfp=pd.read_sql('select * from df', conn )
dfp.info()

# Escalar para que tiempo de duración esté en el mismo rango

sc=MinMaxScaler()
dfp[["timestamp"]]=sc.fit_transform(dfp[['timestamp']])


## eliminar filas que no se van a utilizar ###
mov_dum=dfp.drop(columns=['userId','rating','movieId','title'])


###### libros recomendadas ejemplo para un libro#####

libro='Toy Story (1995)'
ind_libro=dfp[dfp['title']==libro].index.values.astype(int)[0]
similar_movies=mov_dum.corrwith(mov_dum.iloc[ind_libro,:],axis=1)
similar_movies=similar_movies.sort_values(ascending=False)
top_similar_movies=similar_movies.to_frame(name="correlación").iloc[0:11,]
top_similar_movies['title']=dfp["title"]
    


#### libros recomendados ejemplo para visualización todos los libros

def recomendacion(pelicula = list(dfp['title'])):
     
    ind_libro=mov_dum[dfp['title']==pelicula].index.values.astype(int)[0]   #### obtener indice de libro seleccionado de lista
    similar_movies = mov_dum.corrwith(mov_dum.iloc[ind_libro,:],axis=1) ## correlación entre libro seleccionado y todos los otros
    similar_movies = similar_movies.sort_values(ascending=False) #### ordenar correlaciones
    top_similar_movies=similar_movies.to_frame(name="correlación").iloc[0:11,] ### el 11 es número de libros recomendados
    top_similar_movies['title']=dfp["title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    
    return top_similar_movies


print(interact(recomendacion))