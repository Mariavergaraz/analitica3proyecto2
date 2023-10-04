import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
import a_funciones as fn
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder

# Ejecutar sql y conectarse a BD movies 
conn=sql.connect('db_movies2')
cur=conn.cursor() # Para ejecutar consultas en BD

# Verificar tablas que hay disponibles
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()

# Convertir tablas de BD a DataFrames de pandas en python
movies= pd.read_sql("""select *  from movies""", conn)
movies_ratings = pd.read_sql('select * from ratings', conn)

# Movies es DataFrame con la columna 'genres'

# Convertir la columna 'genres' en una lista de listas de cadenas
genres_list = movies['genres'].str.split('|').tolist()

# Crear una instancia de TransactionEncoder
te = TransactionEncoder()

# Transformar los datos usando TransactionEncoder
genres_encoded = te.fit_transform(genres_list)

# Crear un DataFrame con los datos transformados y los nombres de las columnas
genres_df = pd.DataFrame(genres_encoded, columns=te.columns_)
genres_df['movieId'] = movies['movieId']

# Visualizar tablas
movies.head()
movies_ratings.head()
genres_df.head()

#-------------------- Exploración inicial --------------------#
# Identificar campos y verificar formatos
movies.info()
movies_ratings.info()
genres_df.info()

# Verificar duplicados
movies.duplicated().sum() 
movies_ratings.duplicated().sum()
genres_df.duplicated().sum()

# Cambio de nombres de las columnas de géneros
genres_df.rename(columns={'Film-Noir' : 'Film_noir', 'Sci-Fi' : 'Fiction', '(no genres listed)': 'SINref'}, inplace=True)

# Exportar generos como SQL
genres_df.to_sql('genres', conn, if_exists='replace', index=False)

# Unir título de la película a las calificaciones
cur.execute("create table if not exists df1 as select * from ratings left join movies using (movieId)")
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()

# Agregar a ratings los géneros dummies
df = pd.read_sql("""select * from df1 left join genres using (movieId)""", conn)
df.drop('genres', axis = 1, inplace=True)
df.fillna(0, inplace=True)
df.to_sql('df_ratings', conn, if_exists='replace', index=False)
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()

# Unir a las películas el promedio de calificación
cur.execute("create table if not exists df2 as select * from movies left join (select *, avg(rating) as prom_rating from ratings group by movieId) using (movieId)")
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()

# Agregar a ratings los géneros dummies
df = pd.read_sql("""select * from df2 left join genres using (movieId)""", conn)
df.drop('genres', axis = 1, inplace=True)
df.fillna(0, inplace=True)
df.to_sql('df_movies', conn, if_exists='replace', index=False)
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()

# Consultas SQL para contextualizar
# Número de películas
pd.read_sql("""select count(*) from df_movies""", conn)

# Número de calificaciones de los usuarios
pd.read_sql("""select count(*) from df_ratings""", conn)

# Número de usuarios que calificaron
pd.read_sql("""select count(distinct userId) from df_ratings""", conn)

# Calificación promedio por película
pd.read_sql("""select title, prom_rating
            from df_movies
            group by title""", conn)

# Películas sin evaluaciones
pd.read_sql("""select title, count(rating) as cnt from df_ratings
            where df_ratings.rating is null
            group by df_ratings.title 
            order by cnt asc """, conn)

# Películas con una sola evaluación
pd.read_sql("""select title, count(rating) as cnt from df_ratings
            group by df_ratings.title having cnt=1 order by cnt asc """, conn)

# Número de películas por género
pd.read_sql("""select sum(case when Action =True then 1 else 0 end) as total_accion,
                sum(case when Adventure =True then 1 else 0 end) as total_aventura,
                sum(case when Animation =True then 1 else 0 end) as total_animacion,
                sum(case when Children =True then 1 else 0 end) as total_infantiles,
                sum(case when Comedy =True then 1 else 0 end) as total_comedia,
                sum(case when Crime =True then 1 else 0 end) as total_crimen,
                sum(case when Documentary =True then 1 else 0 end) as total_documental,
                sum(case when Drama =True then 1 else 0 end) as total_Drama,
                sum(case when Fantasy =True then 1 else 0 end) as total_fantasia,
                sum(case when Horror =True then 1 else 0 end) as total_terror,
                sum(case when IMAX =True then 1 else 0 end) as total_IMAX,
                sum(case when Musical =True then 1 else 0 end) as total_musical,
                sum(case when Mystery =True then 1 else 0 end) as total_misterio,
                sum(case when Romance =True then 1 else 0 end) as total_romance,
                sum(case when Thriller =True then 1 else 0 end) as total_thriller,
                sum(case when War =True then 1 else 0 end) as total_guerra,
                sum(case when Western =True then 1 else 0 end) as total_oeste,
                sum(case when Film_noir = True then 1 else 0 end) as total_noir,
                sum(case when Fiction =True then 1 else 0 end) as total_ficcion,
                sum(case when SINref =True then 1 else 0 end) as total_sinref
                from df_movies""", conn)

# Distribución de calificaciones
df1=pd.read_sql(""" select rating, count(*) as cnt from df_ratings
               group by "rating"
               order by cnt desc""", conn)

data  = go.Bar( x=df1.rating,y=df1.cnt, text=df1.cnt, textposition="outside")
Layout=go.Layout(title="Número de películas en cada calificación", xaxis={'title':'Rating'}, yaxis={'title':'N° de películas'})
go.Figure(data,Layout)

# --- Se evidencian registros desde 0.5 a 5, con un buen rango de variación ---

# Cantidad de calificaciones por usuario
rating_users=pd.read_sql('''select userId, count(*) as cnt_rat
                         from df_ratings
                         group by userId
                         order by cnt_rat asc''',conn )

fig  = px.histogram(rating_users, x= 'cnt_rat', title= 'Número de calificaciones por usario')
fig.show() 

rating_users.describe()

# --- Se evidencian cantidades de calificaciones atípicas por usuario ---
# --- El 75% de los usuarios ha calificado 35 veces o más ---
# --- Sin embargo, hay usuarios hasta con 2698 calificaciones ---

# Filtrar usuarios con más de 20 y menos de 480 calificaciones (para tener calificaciones confiables)
rating_users2=pd.read_sql('''select userId, count(*) as cnt_rat
                         from df_ratings
                         group by userId
                         having cnt_rat>20 and cnt_rat <=480
                         order by cnt_rat asc''',conn )

# Visualizar distribución después de eliminar atípicos
fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Número de calificaciones por usario')
fig.show() 

rating_users2.describe()


# Cantidad de calificaciones por película
rating_movies=pd.read_sql('''select movieId, count(*) as cnt_rat
                         from df_ratings
                         group by movieId
                         order by cnt_rat desc''',conn)

fig  = px.histogram(rating_movies, x= 'cnt_rat', title= 'Número de calificaciones por película')
fig.show()  

rating_movies.describe()

# --- Se evidencian cantidades de calificaciones atípicas por película ---
# --- El 75% de las películas ha calificado 9 veces o menos ---
# --- Sin embargo, hay películas hasta con 329 calificaciones ---

# Filtrar peliculas que tengan más de 3 calificaciones
rating_movies2=pd.read_sql(''' select movieId, count(*) as cnt_rat
                         from df_ratings
                         group by movieId
                         having cnt_rat>3
                         order by cnt_rat desc''',conn )

fig  = px.histogram(rating_movies2, x= 'cnt_rat', title= 'Número de calificaciones por película')
fig.show()

rating_movies2.describe()

# Lectura de BD condensada, sin atípicos
fn.ejecutar_sql('preprocesamiento.sql', cur)
cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()

# Tabla final ratings
pd.read_sql('select count(*) from df_ratings', conn)
pd.read_sql('select count(*) from df_ratingsfinal', conn)

# Tabla finalmovies
pd.read_sql('select count(*) from df_movies', conn)
pd.read_sql('select count(*) from movies_sel', conn)

# Verificar tamaño, duplicados, información
ratings=pd.read_sql('select * from df_ratingsfinal',conn)
ratings.duplicated().sum()
ratings.info()
ratings.head()

movies=pd.read_sql('select * from df_moviesfinal',conn)
movies.duplicated().sum()
movies.info()
movies.head()