FROM python:3.10

# Crear usuario que ejecuta la app
# RUN adduser --disabled-password --gecos '' api-user

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el contenido actual del directorio al contenedor en /app
COPY . /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 5000 para que la aplicación Flask pueda ser accedida
EXPOSE 5000

# Define el comando por defecto que se ejecutará cuando el contenedor sea iniciado
CMD ["python", "app.py"]
