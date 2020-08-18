from flask import Flask, jsonify, request, render_template, redirect, url_for
import os
from proc_video import grey_image


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "very secret key"
app.config["IMAGE_UPLOADS"] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print(image)
            pth = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(pth)
            grey_image(pth)                                          
            return redirect(url_for('static', filename= 'uploads/' + image.filename), code=301)
    return render_template("upload_image.html")

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)