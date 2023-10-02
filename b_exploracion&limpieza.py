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
movieId = movies['movieId']
genres_df['movieId'] = movieId

# Exportar generos como SQL
genres_df.to_sql('genres', conn, if_exists='replace', index=False)

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

# Unión de tablas de movies y ratings
cur.execute("create table if not exists df1 as select * from movies left join ratings using (movieId)")
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()

# Unión de tablas de unión de df con genres
df = pd.read_sql("""select * from df1 left join genres using (movieId)""", conn)
df.drop('genres', axis = 1, inplace=True)
df.to_sql('df', conn, if_exists='replace', index=False)
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()

# Consultas SQL para contextualizar
# Número de películas
pd.read_sql("""select count(*) from movies""", conn)

# Número de calificaciones de los usuarios
pd.read_sql("""select count(*) from ratings""", conn)

# Número de usuarios que calificaron
pd.read_sql("""select count(distinct userId) from df""", conn)

# Calificación promedio por película
pd.read_sql("""select movieId, avg(rating)
            from df
            group by movieId""", conn)

# Películas sin evaluaciones
pd.read_sql("""select title, count(rating) as cnt from df
            where df.rating is null
            group by df.title 
            order by cnt asc """, conn)

# Películas con una sola evaluación
pd.read_sql("""select title, count(rating) as cnt from df
            group by df.title having cnt=1 order by cnt asc """, conn)

# Géneros con mayor cantidad de películas
pd.read_sql(""" select Action, count(case when Action ='True' then 1 end) as total_accion from df
            """, conn)

# Distribución de calificaciones
df1=pd.read_sql(""" select rating, count(*) as cnt from df
               group by "rating"
               order by cnt desc""", conn)

data  = go.Bar( x=df1.rating,y=df1.cnt, text=df1.cnt, textposition="outside")
Layout=go.Layout(title="Número de películas en cada calificación", xaxis={'title':'Rating'}, yaxis={'title':'N° de películas'})
go.Figure(data,Layout)

# --- Se evidencian registros desde 0.5 a 5, con un buen rango de variación ---

# Cantidad de calificaciones por usuario
rating_users=pd.read_sql('''select userId, count(*) as cnt_rat
                         from df
                         group by userId
                         order by cnt_rat asc''',conn )

fig  = px.histogram(rating_users, x= 'cnt_rat', title= 'Número de calificaciones por usario')
fig.show() 

rating_users.describe()

# --- Se evidencian cantidades de calificaciones atípicas por usuario ---
# --- El 75% de los usuarios ha calificado 168 veces o menos ---
# --- Sin embargo, hay usuarios hasta con 2698 calificaciones ---

# Filtrar usuarios con máx de 20 y menos de 240 calificaciones (para tener calificación confiable)
rating_users2=pd.read_sql('''select userId, count(*) as cnt_rat
                         from df
                         group by userId
                         having cnt_rat>20 and cnt_rat <=240
                         order by cnt_rat asc''',conn )

# Visualizar distribución después de eliminar atípicos
fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Número de calificaciones por usario')
fig.show() 

rating_users2.describe()


# Cantidad de calificaciones por película
rating_movies=pd.read_sql('''select movieId, count(*) as cnt_rat
                         from df
                         group by movieId
                         order by cnt_rat desc''',conn)

fig  = px.histogram(rating_movies, x= 'cnt_rat', title= 'Número de calificaciones por película')
fig.show()  

rating_movies.describe()

# --- Se evidencian cantidades de calificaciones atípicas por película ---
# --- El 75% de las películas ha calificado 9 veces o menos ---
# --- Sin embargo, hay películas hasta con 329 calificaciones ---

# Filtrar peliculas  que no tengan más de 50 calificaciones
rating_movies2=pd.read_sql(''' select movieId, count(*) as cnt_rat
                         from df
                         group by movieId
                         having cnt_rat<=50
                         order by cnt_rat desc''',conn )

fig  = px.histogram(rating_movies2, x= 'cnt_rat', title= 'Número de calificaciones por película')
fig.show()

rating_movies2.describe()