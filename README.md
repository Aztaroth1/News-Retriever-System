# Bienvenido al projecto de Recuperación de informacion.

## Entorno
Este proyecto es recomendable usar dentro de un entorno virtualizado.
Para ello, activalo de la siguiente manera

- source env_proyect/bin/activate

NOTA: Procura tener instalado venv para usaro el env. Caso contrario usa el comando:

- pip install venv

Y se lo has instalado procura crear el entorno de la siguiente manera:

- python3 -m venv env_proyect


## Requerimientos
Los requerimientos, estan ubicados en el txt requirements.txt, para instalarlos usa el comando

- pip install -r requeriments.txt

## Adicional
Este proyecto se esta trabajando mediante notebook, por lo cual se usa Jupyter Notebook.
Para instalarlo debes usar
- pip install notebook

Y usar

- jupyter notebook

Para poder acceder al servicio de notebook.

## Adicional 2
(Opcional si usas entorno virtual)
instala ipykernel

- pip install ipykernel

y agregalo en la lista de kernels diponibles.

- python -m ipykernel install --user --name=env_project --display-name "Python (env_project)"

Esto con el fin de que el kernel que usa jupyter sea el de entorno virtual.
