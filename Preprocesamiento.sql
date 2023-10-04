----procesamientos---

--- Crear tabla con usuarios con más de 20 y menos de 480 calificaciones de películas
drop table if exists usuarios_sel;
create table usuarios_sel as 
select userId, count(*) as cnt_rat
from df_ratings
group by userId
having cnt_rat>20 and cnt_rat <=480
order by cnt_rat asc;

--- Crear tablas con películas que tengan más de 3 calificaciones
drop table if exists movies_sel;
create table movies_sel as 
select movieId, count(*) as cnt_rat
from df_ratings
group by movieId
having cnt_rat>3
order by cnt_rat desc;

-------crear tablas filtradas de películas y calificaciones ----

drop table if exists df_ratingsfinal;
create table df_ratingsfinal as
select *
from df_ratings
inner join movies_sel
using (movieId)
inner join usuarios_sel
using (userId);

drop table if exists df_moviesfinal;
create table df_moviesfinal as
select *
from df_movies
inner join movies_sel
using (movieId)
inner join usuarios_sel
using (userId);