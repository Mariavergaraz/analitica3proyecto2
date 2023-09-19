def ejecutar_sql (db_movies, cur):
    sql_file=open(db_movies)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)

##para preprocesamiento hacer consultas de sql y guardarlas en tablas 