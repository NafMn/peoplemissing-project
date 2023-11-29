import string
import os
import random
import pathlib

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import heapq

from keras import backend
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten
from io import BytesIO
from flask import Flask, render_template, request, redirect, session, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from fungsi import get_random_string, prep, preds, pair_list, pred_image, visualize
from keras.layers import Layer
from keras.models import load_model




# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    


with tf.keras.utils.custom_object_scope({'L1Dist':L1Dist}):
    siamese_model = load_model('model/siamesemodelv2.h5')

# load model
# siamese_model = tf.keras.models.load_model('model/siamesemodelv2.h5',
#                                                custom_objects={'L1Dist':L1Dist,
#                                                    'BinaryCrossentropy':tf.losses.BinaryCrossentropy},
#                                                    compile=False)

app   = Flask(__name__, static_url_path='/static')
# app.config['SECRET_KEY'] = 'asd82a'
app.config['UPLOAD_FOLDER'] = 'static/pict_test/stored_image'
app.config['UPLOADED_FILES'] = 'static/pict_test/input_image'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/oranghilang'

# siamese_model = None
db = SQLAlchemy(app)

path_store = 'static/pict_test/stored_image'
path_input = 'static/pict_test/input_image'
dir_path = r'static/pict_test/stored_image/**/*.jpg*'
img_path = []
for file in glob.glob(dir_path, recursive=True):
    img_path.append(file)


class Orang(db.Model):
    __tablename__ = 'people'
    id = db.Column(db.Integer, primary_key=True)
    fotos = db.Column(db.String(100))
    nama = db.Column(db.String(50))
    usia = db.Column(db.Integer)
    tgl_lahir = db.Column(db.String(20))
    agama = db.Column(db.String(10))
    nama_pelapor = db.Column(db.String(50))
    kontak = db.Column(db.String(20))

    def __init__(self, fotos, nama, usia, tgl_lahir, agama, nama_pelapor, kontak):
        self.fotos = fotos
        self.nama = nama
        self.usia = usia
        self.tgl_lahir = tgl_lahir
        self.agama = agama
        self.nama_pelapor = nama_pelapor
        self.kontak = kontak

db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/addpeople', methods=['GET', 'POST'])
def addpeople():
    if request.method == "POST":
        fotos = request.files.getlist('fotos[]')
        nama = request.form.get('nama')
        pathlib.Path(app.config['UPLOAD_FOLDER'], nama).mkdir(exist_ok=True)

        for foto in fotos:     
            filename = secure_filename(foto.filename)
            ext = os.path.splitext(filename)[1]
            new_filename = get_random_string(20)

            foto.save(os.path.join(app.config['UPLOAD_FOLDER'], nama, new_filename+ext))
            
            people = Orang(fotos=os.path.join(app.config['UPLOAD_FOLDER'], nama, new_filename+ext), nama=request.form['nama'], usia=request.form['usia'], tgl_lahir=request.form['tgl_lahir'], agama=request.form['agama'], nama_pelapor=request.form['nama_pelapor'], kontak=request.form['kontak'])
            db.session.add(people)
            db.session.commit()

        success = "Berhasil menambahkan data."
        return render_template("register.html", success=success) 

@app.route('/find')
def find():
    return render_template('find.html')

@app.route('/findpeople', methods=['GET', 'POST'])
def findpeople():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded_img']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOADED_FILES'], img_filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOADED_FILES'], img_filename)

        img_file_path = session.get('uploaded_img_file_path', None)
        file_path = img_file_path

        strd_pth, inp_pth = pair_list(img_filename)
        df = pd.DataFrame(list(zip(inp_pth, strd_pth)),columns =['input_path', 'file_path'])

        # Predict
        y_pred = pred_image(df)
        df['pred'] = y_pred
        dd = df.loc[df['pred'] >= 0.5]

        # Maksimal 5 Photo
        n = []
        for i in range(len(dd)):
            n.append(dd['pred'].iloc[i])
        
        heapq.heapify(n)
        pred5 = heapq.nlargest(5, n)

        # Filtering
        strd = []
        for i in range(len(dd)):
            if dd['pred'].iloc[i] in pred5:
                strd.append(dd['file_path'].iloc[i])
        
        result = Orang.query.filter(Orang.fotos.in_(strd))
        return render_template('find.html', result=result)


@app.route('/data')
def data():
    data = Orang.query.all()
    return render_template('data.html', data=data)


if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)