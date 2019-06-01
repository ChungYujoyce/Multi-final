import os
from flask import Flask, request, redirect, url_for,send_from_directory, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import shutil
import image_test
import uuid
import time

# Dataset names.
_ADE20K = 'ade20k'
_CITYSCAPES = 'cityscapes'
_MAPILLARY_VISTAS = 'mapillary_vistas'
_PASCAL = 'pascal'

#Model names
_Cityscapes_mobilenet_stride16 = 'models/mobilev2_restride16_100000.tar.gz'
_Pascal_deeplab = 'models/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz'
_Cityscapes_mobilenet = 'models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
_Pascal_mobilenet = 'models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'

# route names
#UPLOAD_FOLDER = './static/'
UPLOAD_FOLDER = 'C:/Users/USER/Desktop/Multi-final/webapp_mobilenet_test/static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])

app = Flask(__name__, static_url_path = "/static", static_folder = "static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                #-------------------------------------------------------------just a test
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
            else:
                
                image_path = UPLOAD_FOLDER + file.filename
                output_name = str(uuid.uuid4()) + '.png'
                output_name_color = str(uuid.uuid4()) + '.png'
                duration = image_test.image_inference(_Cityscapes_mobilenet, image_path, _CITYSCAPES, output_name, output_name_color)
               
                global original_image
                original_image = file.filename

                global output_image
                output_image = output_name

                global color_image
                color_image = output_name_color

                global time_spent
                time_spent = duration
                #------------------------------------------------------------------------
            return redirect(url_for('uploaded_file', filename=filename ))
    return render_template('main.html')

@app.route('/uploads/<filename>')
def uploaded_file( filename):
    global original_image
    global output_image
    global color_image
    global time_spent
    print("inside : ",time_spent)
    return render_template("showphoto.html", original_image = original_image, bicubic_image = color_image, result_image = output_image ,time_spent=time_spent)

@app.route('/return')
def get_ses():
 	return redirect(url_for('upload_file'))


if __name__ == '__main__':
    app.run(debug=True, port=8000)

