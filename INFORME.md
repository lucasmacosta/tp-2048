# Informe: agentes para el 2048 simplificado

Integrantes: [completar]

## 1. Comparación

| Agente                       | Puntuación media | Ficha máxima (distribución) | Tiempo |
| ---------------------------- | ---------------- | --------------------------- | ------ |
| Al azar                      | 19.26            | 16 (10.9%)                  | 0.0s   |
| Minimax, profundidad 1       | 57.14            | 32 (4.9%)                   | 0.2s   |
| Minimax, profundidad 2       | 57.98            | 32 (5.1%)                   | 1.1s   |
| Minimax, profundidad 3       | 57.98            | 32 (5.1%)                   | 6.7s   |
| Q-learning, 30 000 episodios | 66.46            | 32 (8.1%)                   | 1.9s   |

## 2. Minimax frente al azar real

El problema que tiene la implementación Minimax elegida es que asume que la nueva ficha siempre se va a ser la peor opción posible: `Minimax trata al ambiente como un adversario: supone que la ficha aparece siempre en el peor lugar posible para el jugador.`. Evidentemente esto no es realista ya que la próxima ficha no depende del estado actual del tablero sino que en realidad depende de una probabilidad que se puede configurar.

El problema principal del mecanismo usado para esta implementación está relacionado con los estados terminales y el hecho de elegir el primero que lo lleve a ese estado.

Si tomamos el tablero de ejemplo y aplicamos las únicas dos jugadas posibles sobre el mismo:

```
Estado inicial
| 8 2 |
| 4 . |

 --> Right (+0)    afterstate     ficha nueva
                   | 8 2 |        2 (p = 0,9) -> | 8 2 |   final, no quedan jugadas
                   | . 4 |                       | 2 4 |
                                  4 (p = 0,1) -> | 8 2 |   sigue (se pueden juntar los 4)
                                                 | 4 4 |

 --> Down (+0)     afterstate     ficha nueva
                   | 8 . |        2 (p = 0,9) -> | 8 2 |   sigue (se pueden juntar los 2)
                   | 4 2 |                       | 4 2 |
                                  4 (p = 0,1) -> | 8 4 |   final, no quedan jugadas
                                                 | 4 2 |
```

Con este tablero de ejemplo, tenemos que para ambas jugadas posibles `Right` y `Down` el peor caso es un estado terminal que da 0 puntos, y por la forma en que el algoritmo funciona siempre elegirá `Right` porque así el el orden arbitrario de las actions en la definición del problem. El problema es que cuando se elige esa opción, hay un 90% de probabilidad que se el siguiente estado sea un terminal (la probabilidad de que salga un 2), mientras que en el otro caso la probabilidad cae a un 10% (probabilidad de que salga un 4).

## 3. Profundidad

| Profundidad | Nodos expandidos por jugada | Puntuación media |
| ----------- | --------------------------- | ---------------- |
| 1           | 7.0                         | 57.14            |
| 2           | 54.9                        | 57.98            |
| 3           | 359.6                       | 57.98            |
| 4           | 1973.3                      | 57.98            |

Con los tres niveles de profundidad propuestos ya es evidente que luego de la profundidad 2 ya no se obtienen mejoras en la puntuación al aumentar su valor. De igual modo se agregó un nivel más para confirmar esta teoría y que sirve para dejar más evidencia todavía la tendencia al aumento de estados explorados de modo exponencial.

Además del problema descripto en el punto #2 respecto de la exploración a fondo de la misma jugada, tenemos otra limitación relacionada con la exploración a ciegas de ramas de juego que ya fueron exploradas con anterioridad. La alternativa más sencilla para evitar ir por ramas generadas por estados ya explorados consiste en almacenar los resultados de esas exploraciones previas en un diccionario, pero la viabilidad de esta opción depende de la cantidad de estados posibles. Por otra parte, se puede utilizar poda alfa-beta para evitar recorrer ramas que ya sabemos no van a ser mejores que las ya exploradas. Es importante notar que en ambos casos de mejoras, solamente se trata de optimizar el uso de recursos como espacio y velocidad, pero sin cambiar lo descripto en el punto #2 los puntajes serán los mismos.

## 4. Parámetros de Q-learning

Con los parámetros por default obtenemos la siguiente curva:

```
======================================================================
Q-learning on 2048 (2x2): alpha=0.1, gamma=1.0, epsilon=0.1
======================================================================
  episodes   mean score  Q entries  time (s)
       100        29.05        137       0.1
       300        30.72        219       0.1
      1000        52.84        315       0.3
      3000        58.49        422       0.5
     10000        61.36        556       1.0
     30000        66.71        704       2.3
```

Si incrementamos alfa a 0.01 y 1.0 tenemos las siguientes tablas:

```
======================================================================
Q-learning on 2048 (2x2): alpha=0.01, gamma=1.0, epsilon=0.1
======================================================================
  episodes   mean score  Q entries  time (s)
       100        28.78        137       0.1
       300        30.67        206       0.1
      1000        31.11        305       0.2
      3000        33.03        419       0.4
     10000        45.13        553       0.8
     30000        60.81        655       2.1

======================================================================
Q-learning on 2048 (2x2): alpha=1.0, gamma=1.0, epsilon=0.1
======================================================================
  episodes   mean score  Q entries  time (s)
       100        37.37        138       0.1
       300        42.10        228       0.2
      1000        52.03        357       0.3
      3000        54.74        487       0.5
     10000        57.72        656       1.0
     30000        58.10        789       2.2
```

Los cambios en las tablas se dan porque alfa es la tasa de aprendizaje, es decir, cuánto varían los valores de Q con cada update. Si se pone un valor demasiado bajo, como en la primera tabla, el proceso de aprendizaje es demasiado lento y son necesarios muchos episodios hasta que empieza a dar valores razonables. Y si es demasiado alto el aprendizaje es muy rápido con pocas repeticiones pero luego se vuelve un problema con muchos episodios ya que rebota entre valores y nunca llega a estabilizarse.

Ahora probemos con epsilon en 0.5 y 1.0:

```
======================================================================
Q-learning on 2048 (2x2): alpha=0.1, gamma=1.0, epsilon=0.5
======================================================================
  episodes   mean score  Q entries  time (s)
       100        27.97        226       0.1
       300        35.81        377       0.1
      1000        54.92        520       0.3
      3000        58.17        645       0.4
     10000        63.90        755       0.8
     30000        65.26        854       1.8

======================================================================
Q-learning on 2048 (2x2): alpha=0.1, gamma=1.0, epsilon=1.0
======================================================================
  episodes   mean score  Q entries  time (s)
       100        39.80        260       0.1
       300        40.91        375       0.2
      1000        54.62        512       0.3
      3000        58.11        607       0.4
     10000        61.57        701       0.7
     30000        62.62        786       1.4
```

El valor de epsilon (que por defecto está en 10%) determina la probabilidad con la que el algoritmo elegirá una action al azar y cuándo usará la mejor action disponible dependiendo de la reward obtenida. Con un valor de 50% vemos que tenemos un crecimiento mayor con pocos episodios porque básicamente se está imponiendo la suerte antes que el entrenamiento, es decir, que está visitando estados a los que nunca llegaría en condiciones normales usando una buena estrategia de juego pero que igual le ayudan a alcanzar buenos puntajes. Con un valor del 100% en definitiva el algoritmo está jugando de manera totalmente random, por eso la curva de crecimiento es más lineal y está directamente relacionada con la cantidad de episodios que ayudan a encontrar estados positivos aún en medio de toda esa aleatoriedad.

## 5. Un tablero más grande

Curva de aprendizaje de Q-learning y comparación entre algoritmos usando una grilla de 3x3:

```
======================================================================
Q-learning on 2048 (3x3): alpha=0.1, gamma=1.0, epsilon=0.1
======================================================================
  episodes   mean score  Q entries  time (s)
       100       183.37       2729       0.6
       300       179.71       6896       1.1
      1000       181.17      19493       1.8
      3000       185.00      45879       3.0
     10000       198.63     109964       5.6
     30000       223.01     231395      12.6
    100000       264.92     486839      38.1

======================================================================
2048 (3x3): 200 games per agent
======================================================================
Random                           mean score  179.92   (  0.1 s)
                                 max tile   4: 0.5%, 8: 3.5%, 16: 31.0%, 32: 57.5%, 64: 7.5%
Minimax (depth 1)                mean score  768.16   (  1.9 s)
                                 max tile   16: 0.5%, 32: 12.5%, 64: 48.0%, 128: 35.5%, 256: 3.5%
                                 expanded nodes per move 16.9
Q-learning (30000 episodes)      mean score  206.02   ( 10.1 s)
                                 max tile   8: 4.0%, 16: 31.0%, 32: 54.0%, 64: 11.0%
```

El rendimiento de Q-learning con tabla cuando se incrementa la grilla empeora básicamente por la cantidad de estados a cubrir. Como se puede apreciar en la tabla de entrenamiento, incluso con 100.000 todavía sigue encontrando estados nuevos para explorar. Si la tabla Q nunca deja de crecer entonces el algoritmo nunca llegará a la parte del aprendizaje.

El algoritmo minimax no tiene este problema porque en realidad no está aprendiendo, solamente calcula los siguientes estados a un solo nivel de profundidad y eso le alcanza para ser bastante bueno ya que tiene la cota que necesita para funcionar.

Posiblemente la alternativa más eficiente -además de una usar red neuronal- sea intentar reducir la cantidad de estados posibles a almacenar en Q. Por ejemplo, se podría intentar modelarlos de modo tal que se almacenen estados similares, rotados o espejados.
