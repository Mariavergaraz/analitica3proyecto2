----procesamientos---

---crear tabla con usuarios con más de 20 y menos de 240 calificaciones de películas

drop table if exists usuarios_sel;

create table usuarios_sel as 

select userId, count(*) as cnt_rat
from df
group by userId
having cnt_rat>20 and cnt_rat <=240
order by cnt_rat asc;

---Filtrar peliculas que tengan más de 3 calificaciones
drop table if exists movies_sel;



create table movies_sel as 

select movieId, count(*) as cnt_rat
from df
group by movieId
having cnt_rat>3
order by cnt_rat desc;


-------crear tablas filtradas de películas y calificaciones ----

drop table if exists df_final;

create table df_final as

select *
from df
inner join movies_sel
using (movieId)
inner join usuarios_sel
using (userId);