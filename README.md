# APLICACIÓN DE DEEP LEARNING PARA LA MEJORA DE CALIDAD EN IMÁGENES ULTRASONIDO USANDO BEAMFORMING

Este es el código utilizado para el proyecto final del ramo, contiene 3 Jupyter notebooks, los cuales cumplen las siguientes funciones:

- `get_preprocesed_dataset.ipynb`: Este archivo nos ayuda a transformar el dataset original y obtener archivos .hdf5 con los inputs necesarios, además de los ground truth listos para entrenar la red. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Tio-Panda/inf557-project/blob/main/get_preprocesed_dataset.ipynb)

- `training.ipynb`: Este archivo contiene el código para entrenar la red habiendo obtenido el dataset procesado con el archivo anterior. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Tio-Panda/inf557-project/blob/main/training.ipynb)

- `testing.ipybn`: Este archivo nos permite obtener las figuras y tablas usadas en el trabajo con el dataset de testing (que es el dataset de PICMUS). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Tio-Panda/inf557-project/blob/main/testing.ipynb)

La estructura de las carpetas es la siguiente:

- `dataset_test`: Esta carpeta tiene datos pre-procesados, esto es lo que se podria obtener si procesamos los datos del dataset PICMUS con `get_preprocesed_dataset.ipynb`.

- `metric_tables`: Carpeta en donde se guardan los CSVs de las tablas generadas.

- `figures`: Carpeta en donde se guardan las figuras generadas.

- `modules`: Codigo de python necesario ya sea para cargar el dataset .hdf5, hacer las reconstrucciones con distintos metodos, construir la red y varias funciones auxiliares como por ejemplo obtener los B-mode de las reconstrucciones.

- `original_dataset`: Aquí irian los archivos .hdf5 descargados del [dataset PICMUS](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download) y el [dataset CUBDL](https://ieee-dataport.org/competitions/challenge-ultrasound-beamforming-deep-learning-cubdl-datasets#files).

`weights`: Aquí están los modelos en `.keras` para poder importarlos.

# Problema

Debido al peso del dataset original (22 GB) se me hace complicado dejarlo en el repositorio, por eso no va incluido o con un link directo a un Google Drive. Por esto, se optó por dejar el dataset test dentro del repositorio e incluir los modelos ya entrenados.