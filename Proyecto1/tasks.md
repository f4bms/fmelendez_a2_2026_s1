# Tareas pendientes
Se utilizará este documento para organizar de mejor manera las tareas pendientes del proyecto.

### Fine Grained

- Establecer realmente la parte de la semilla fija para el trigger del cambio de contexto
Ahorita se está utilizando algo deterministico(cada 10000 mul) lo cual no deberia de ser así
Se espera a la respuesta del profe para poder realmente implementar esto.


Ocupo agregarle una nueva variable llamada context switch latency que va a ser = 1 y se va a  sumar cada que se cambie de hilo, este valor suma a la cantidad de ciclos. Entonces tendriamos cantidad de mult + context switch latency + stall + fetch = cantidad de ciclos.

### Coarse Grained
- Al igual que el fine hay que realmente establecer el random para el trigger del cambio de contexto
Ahorita se está utilizando un random pero no debería de usarse cualquier random pensaría yo

Agregarle la cola que se usa en el fine al coarse para que se cuente dentro de los ciclos el fetch que se hace.

### SMT y CMP

- Se debe de probar por consola el uso real del hardware y que el programa creado realmente serve para probar dichas cosas y muy importante, que venga de la mano con los modelos del fine y coarse

### Generador de graficas 
- Prioritario
- Hay que realizar el script del make file que pueda poblar el archivo necesario y que se puedan generar las graficas. (Usando python).

### Revision de codigo
Eliminar codigo muerto que se ha ido quedando, reutilizar codigo que se usa en varios archivos y mejorar la modularidad



### PREGUNTAS

como estoy haciendo una red neuronal el primer calculo random de los pesos debe de ser igual en todas las ejecuciones(una semilla igual)