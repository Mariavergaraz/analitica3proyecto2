----procesamientos---

--- Crear tabla con usuarios con más de 20 y menos de 480 calificaciones de películas
drop table if exists usuarios_sel;
create table usuarios_sel as 
select userId, count(*) as cnt_calU --- Calificaciones que han dado los usuarios
from df_ratings
group by userId
having cnt_calU>20 and cnt_calU <=480
order by cnt_calU asc;

--- Crear tablas con películas que tengan más de 3 calificaciones
drop table if exists movies_sel;
create table movies_sel as 
select movieId, count(*) as cnt_calP --- Calificaciones que le han dado a la película
from df_ratings
group by movieId
having cnt_calP>3
order by cnt_calP desc;

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