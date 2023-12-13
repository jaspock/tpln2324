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

En la clase presencial (2,5 horas 🕒️ de duración), veremos cómo se implementa un regresor logístico en PyTorch siguiendo la implementación de un regresor logístico binario y de uno multinomial que se comentan en [este apartado](https://dlsi.ua.es/~japerez/materials/transformers/implementacion/#codigo-para-un-regresor-logistico-y-uno-multinomial).

La idea es que vayas creando una serie de notebooks en Google Colab en los que comentes cada uno de los programas que vamos a ir viendo. En la última clase se presentará una práctica más avanzada que implicará modificar el código del transformer.

Assignments before class of Dec 20, 2023: this class will have two parts taught by different teachers; therefore, your assignments will deal with two different topics; firstly, read the contents of section x; after that, complete this test (deadline: 23:59 CET, Dec 19 2023); secondly, read the materials related to logistic regressors and PyTorch linked in this ![section](text.md#contenidos-a-preparar-antes-de-la-sesion-del-20/12/2023) and then complete [this test](https://forms.gle/V3U9MTHo7c9DNhkc6) (same deadline: 23:59 CET, Dec 19 2023)