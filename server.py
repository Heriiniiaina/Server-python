from flask import Flask, request, jsonify
import face_recognition
import os

app = Flask(__name__)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference_images')

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        print("Files reçus :", request.files)
        print("Form data reçue :", request.form)

        if 'image1' not in request.files or 'image2_name' not in request.form:
            return jsonify({'error': 'Image1 ou image2_name manquant'}), 400

        image2_name = request.form['image2_name']
        image2_path = os.path.join(REFERENCE_DIR, image2_name)

        if not os.path.exists(image2_path):
            return jsonify({'error': f"Le fichier '{image2_name}' n'existe pas sur le serveur"}), 400

        img1 = face_recognition.load_image_file(request.files['image1'])
        img2 = face_recognition.load_image_file(image2_path)

        encodings1 = face_recognition.face_encodings(img1)
        encodings2 = face_recognition.face_encodings(img2)

        if len(encodings1) == 0 or len(encodings2) == 0:
            return jsonify({'error': 'Aucun visage détecté dans une des images'}), 400

        distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
        score = max(0, 100 * (1 - distance))

        return jsonify({'confidence': round(score, 2)})

    except Exception as e:
        print("Erreur interne :", e)
        return jsonify({'error': 'Erreur serveur', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
