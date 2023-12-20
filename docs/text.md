# Architectures for written-text processing

En este bloque se aborda el estudio de algunos modelos neuronales utilizados para procesar textos. El profesor de este bloque es Juan Antonio Pérez Ortiz. El bloque comienza con un repaso del funcionamiento del regresor logístico, que nos servirá para asentar los conocimientos necesarios para entender posteriores modelos. A continuación se estudia con cierto nivel de detalle *skip-grams*, uno de los algoritmos para la obtención de *embeddings* incontextuales de palabras. Después se repasa el funcionamiento de las arquitecturas neuronales *feedforward* y se estudia su aplicación a modelos de lengua. El objetivo último es abordar el estudio de la arquitectura más importante de los sistemas actuales de procesamiento de textos: el transformer. Una vez estudiadas estas arquitecturas, finalizaremos con un análisis del funcionamiento de los modelos preentrenados (modelos fundacionales), en general, y de los modelos de lengua, en particular.

Los materiales de clase complementan la lectura de algunos capítulos de un libro de texto ("Speech and Language Processing" de Dan Jurafsky y James H. Martin, borrador de la tercera edición, disponible online) con anotaciones realizadas por el profesor.

## Primera sesión (20 de diciembre de 2023)

### Contenidos a preparar antes de la sesión del 20/12/2023

Las actividades a realizar antes de esta clase son:

- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/regresor/) sobre regresión logística. Como verás, la página te indica qué contenidos has de leer del libro. Tras una primera lectura, lee las anotaciones del profesor, cuyo propósito es ayudarte a entender los conceptos clave del capítulo. Después, realiza una segunda lectura del capítulo del libro. En total, esta parte debería llevarte unas 3 horas 🕒️ de trabajo.
- Visionado y estudio de los tutoriales en vídeo de esta [playlist oficial de PyTorch](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN).  Estudia al menos los 4 primeros vídeos (“Introduction to PyTorch”, “Introduction to PyTorch Tensors”, “The Fundamentals of Autograd” y “Building Models with PyTorch”). En total, esta parte debería llevarte unas 2 horas 🕒️ de trabajo.
- Tras acabar con las dos partes anteriores, realiza este [test de evaluación](https://forms.gle/V3U9MTHo7c9DNhkc6) de estos contenidos. Son pocas preguntas y te llevará unos minutos.

### Contenidos para la sesión presencial del 20/12/2023

En la clase presencial (2,5 horas 🕒️ de duración), veremos cómo se implementa un regresor logístico en PyTorch siguiendo las implementaciones de un regresor logístico binario <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/logistic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab\"></a> y de uno multinomial <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/softmax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab\"></a> que se comentan en [este apartado](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-un-regresor-logistico-y-uno-multinomial).

La idea es que vayas estudiando y modificando ligeramente los notebooks que vayamos estudiando. En la última clase se presentará una práctica más avanzada que implicará modificar el código del transformer.

### Contenidos a preparar antes de la sesión del 10/01/2024

Las actividades a realizar antes de esta clase son:

- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/embeddings/) sobre los embeddings. Como verás, la página te indica qué contenidos has de leer del libro. Tras una primera lectura, lee las anotaciones del profesor, cuyo propósito es ayudarte a entender los conceptos clave del capítulo. Después, realiza una segunda lectura del capítulo del libro. En total, esta parte debería llevarte unas 4 horas 🕒️ de trabajo.
- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/ffw/) sobre las redes neuronales hacia delante y su uso como modelos de lengua muy básicos. Realiza al menos dos lecturas complementadas con las notas del profesor como en el punto anterior. En total, esta parte debería llevarte unas 2 horas 🕒️ de trabajo.
- Lectura y estudio de los contenidos de [esta página](https://dlsi.ua.es/~japerez/materials/transformers/attention/) de introducción a los transformers. Realiza, como siempre, al menos dos lecturas complementadas con las notas del profesor. En total, esta parte debería llevarte unas 4 horas 🕒️ de trabajo.
- Tras acabar con las partes anteriores, realiza este [test de evaluación](https://forms.gle/7KDwRtXcrpxsKjHp7) de estos contenidos. Son pocas preguntas y te llevará unos minutos.
- Aprovecha, si te queda tiempo, para repasar los contenidos de la primera sesión.

### Contenidos para la sesión presencial del 10/01/2024

En la clase presencial (5 horas 🕒️ de duración), veremos cómo se implementa en PyTorch el algoritmo de skip-grams, una red neuronal hacia delante y un transformer siguiendo las implementaciones que se comentan en estos apartados: [skip-grams](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-skip-grams), [redes hacia delante](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-un-modelo-de-lengua-con-redes-feedforward) y [transformers](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-un-transformer-del-proyecto-mingpt).

La idea es que vayas estudiando y modificando ligeramente los notebooks que vayamos estudiando. En la última clase se presentará una práctica más avanzada que implicará modificar el código del transformer.



