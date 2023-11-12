from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Esto permite solicitudes CORS desde cualquier origen

def before_request():
    g.sentimiento = ""
    g.generos = ""
    g.contador = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encuesta')
def encuesta():
    return render_template('encuesta.html')

@app.route('/genero')
def genero():
    return render_template('genero.html')

@app.route('/playlist')
def playlist():
    return render_template('playlist.html')

@app.route('/camweb')
def camweb():
    return render_template('camweb.html')

@app.route('/api/enviar-sentimiento', methods=['POST'])
def recibir_sentimiento():
    try:
        data = request.get_json()
        sentimiento = data.get('sentimiento')

        # Imprimir el sentimiento en la consola del servidor Flask
        print(f'Sentimiento recibido: {sentimiento}')

        # Mensaje a mostrar en el navegador
        mensaje_navegador = f'El sentimiento elegido por el usuario es {sentimiento}'
        g.sentimiento = sentimiento
        # Aquí puedes realizar acciones con el sentimiento recibido
        # En este ejemplo, simplemente lo devolvemos como parte de la respuesta
        return jsonify({'mensaje': mensaje_navegador}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enviar-generos', methods=['POST'])
def recibir_generos():
    try:
        data = request.get_json()
        generos = data.get('generos')
        g.generos =  generos
        # Imprimir los géneros en la consola del servidor Flask
        print(f'Géneros recibidos: {generos}')

        # Puedes realizar acciones con los géneros recibidos aquí

        # En este ejemplo, simplemente lo devolvemos como parte de la respuesta
        return jsonify({'mensaje': f'Géneros recibidos: {generos}'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generar-playlist', methods=['POST'])
def generar_playlist():
    try:
        data = request.get_json()
        contador = data.get('contador')

        # Aqui se hace la llamada del modelo que devuelve una play list segun sentimiento, generos y el contador
        playlist = ""

        if contador == 1:
            playlist = "<ul><li>nombre_cancion_1 - Almara</li><li>nombre_cancion_2 - Alejandro Sanz</li></ul>"
        elif contador == 2:
            playlist = "<ul><li>nombre_cancion_2 - Skakira </li><li>nombre_cancion_2 - Maluma</li></ul>"
        else: 
            playlist = ""
        # si la playlist esta vacia en la respuesta de javascript sabemos que hay volver a la pagina principal

        # Mensaje a mostrar en el navegador
        mensaje_navegador = f'La Playlist es {playlist}'

        print(f'Playlist recibidos: {playlist}')

        # Aquí puedes realizar acciones con el sentimiento recibido
        # En este ejemplo, simplemente lo devolvemos como parte de la respuesta
        return jsonify({'playlist': f'{playlist}'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)