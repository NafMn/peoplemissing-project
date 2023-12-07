import string
import os
import random
import pathlib
import cv2
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
from flask import Flask, render_template, request, jsonify, redirect, session, send_file
# from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from fungsi import get_random_string, prep, preds, pair_list, pred_image, visualize
from keras.layers import Layer
from keras.models import load_model

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# service account
cred = credentials.Certificate("./app/lokana/serviceAccountKey.json")

# init 
app = firebase_admin.initialize_app(cred)
db = firestore.client()

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    


with tf.keras.utils.custom_object_scope({'L1Dist':L1Dist}):
    siamese_model = load_model('model/siamese_model.h5', compile=False)

siamese_model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# load model
# siamese_model = tf.keras.models.load_model('model/siamesemodelv2.h5',
#                                                custom_objects={'L1Dist':L1Dist,
#                                                    'BinaryCrossentropy':tf.losses.BinaryCrossentropy},
#                                                    compile=False)

# configure path
app   = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/images/stored_image'
app.config['UPLOADED_FILES'] = 'static/images/input_image'

path_store = 'static/images/stored_image'
path_input = 'static/images/input_image'
dir_path = r'static/images/stored_image/**/*.jpg*'
img_path = []
for file in glob.glob(dir_path, recursive=True):
    img_path.append(file)
    
 
@app.route('/static/images/stored_image/<path:params>')
def get_store_image(params):
    parts = params.split('/')  # Membagi params menjadi bagian-bagian
    folder_name = parts[0]  # Menggunakan bagian pertama sebagai nama folder
    return send_file( os.path.join(path_store, folder_name, '/'.join(parts[1:]))) 

@app.route('/static/images/input_image/<path:params>')
def get_input_image(params):
    return send_file( os.path.join(path_input, params)) 
    
# insert
@app.route('/addpeople', methods=['POST'])
def add_people():
    try:
        if request.method == 'POST':
            # request photo
            fotos = request.files.getlist('fotos[]')
            nama = request.form.get('nama')
            pathlib.Path(app.config['UPLOAD_FOLDER'], nama).mkdir(exist_ok=True)
            foto_paths = []  # Inisialisasi array untuk menyimpan path setiap foto
            
            for foto in fotos:     
                filename = secure_filename(foto.filename)
                print(filename)
                ext = os.path.splitext(filename)[1]
                new_filename = get_random_string(20)
                foto.save(os.path.join(app.config['UPLOAD_FOLDER'], nama, new_filename+ext))
                file_path = os.path.join( app.config['UPLOAD_FOLDER'], nama, new_filename+ext)
                print(file_path)
                foto_paths.append(file_path)
        
            # Ambil data dari form
            nama = request.form['nama']
            umur = request.form['umur']
            tinggi = request.form['tinggi']
            berat_badan = request.form['berat_badan']
            ciri_fisik = request.form['ciri_fisik']
            nomor_dihubungi = request.form['nomor_dihubungi']
            sering_ditemukan_di = request.form['sering_ditemukan_di']
            kota = request.form['kota']
            gender = request.form['gender']
            isFound = request.form['isFound']
            
            # # Buat dokumen baru di koleksi 'people'
            addMisingPeople = {
                "foto" : foto_paths,
                "nama": nama,
                "umur" : umur,
                "tinggi": tinggi,
                "berat_badan": berat_badan,
                "ciri_fisik" : ciri_fisik,
                "nomor_dihubungi": nomor_dihubungi,
                "sering_ditemukan_di" : sering_ditemukan_di,
                'kota': kota,
                'gender': gender,
                "isFound": isFound 
            }
            db.collection('MissingPersons').add(addMisingPeople)
            
            return jsonify({"success": "Person added successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# find people to compare
@app.route('/findpeople', methods=['GET', 'POST'])
def findpeople():
    try:
        if request.method == 'POST':
            uploaded_img = request.files['uploaded_img']
            img_filename = secure_filename(uploaded_img.filename)
            # save to local
            uploaded_img.save(os.path.join(app.config['UPLOADED_FILES'], img_filename))
            
            # for compare
            img_file_path = os.path.join(app.config['UPLOADED_FILES'], img_filename) 
            file_path = img_file_path
            
            # save to firestore
            file_path_firestore = os.path.join( app.config['UPLOADED_FILES'], img_filename) 
            # add path image to firestore
            userFoto = {"foto" : file_path}
            db.collection("UserSubmittedPhotos").add(userFoto) 
            
            strd_pth, inp_pth = pair_list(img_filename)
            df = pd.DataFrame(list(zip(inp_pth, strd_pth)),columns =['input_path', 'file_path'])
            # # Predict
            y_pred = pred_image(df, img_file_path)
            df['pred'] = y_pred
            dd = df.loc[df['pred'] >= 0.5]

            # # Maksimal 5 Photo
            n = []
            for i in range(len(dd)):
                n.append(dd['pred'].iloc[i])
            
            heapq.heapify(n)
            pred5 = heapq.nlargest(5, n)

            # # Filtering
            strd = []
            for i in range(len(dd)):
                if dd['pred'].iloc[i] in pred5:
                    strd.append(dd['file_path'].iloc[i])
             
             
            MissingPersons = db.collection("MissingPersons") 
            query = MissingPersons.where("foto", "array_contains_any", strd).stream()

            # Lakukan iterasi pada hasil query untuk menampilkan dokumen yang memenuhi kondisi
            # Buat daftar untuk menyimpan data dokumen yang cocok
            matched_documents = []

            # Iterasi pada hasil query dan ambil data yang diperlukan dari setiap dokumen
            for doc in query:
                matched_documents.append(doc.to_dict())  # Menambahkan data dokumen ke dalam daftar

            # Return daftar yang berisi data dokumen yang cocok sebagai respons Flask
            return jsonify(matched_documents)
            
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500    


# Get people by criteria
@app.route('/getpeople', methods=['GET'])  
# /getpeople?nama=John   
# /getpeople?kota=Jakarta
# /getpeople?gender=Male
# /getpeople?nama=John&kota=Jakarta&gender=Male  
def get_people_by_criteria():
    try:
        # Mendapatkan parameter dari URL
        nama = request.args.get('nama')  
        kota = request.args.get('kota')
        gender = request.args.get('gender')

        # Inisialisasi query tanpa filter
        people_query = db.collection('MissingPersons')

        # Menambahkan filter berdasarkan nama jika parameter nama diberikan
        if nama:
            people_query = people_query.where('nama', '==', nama)

        # Menambahkan filter berdasarkan kota jika parameter kota diberikan
        if kota:
            people_query = people_query.where('kota', '==', kota)

        # Menambahkan filter berdasarkan gender jika parameter gender diberikan
        if gender:
            people_query = people_query.where('gender', '==', gender)

        # Menjalankan query
        people_collection = people_query.stream()

        # Membuat daftar untuk menyimpan data orang
        people_list = []

        # Mengambil data orang dari hasil query
        for person in people_collection:
            person_data = person.to_dict()
            people_list.append(person_data)

        return jsonify(people_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# edit person based on name
@app.route('/editpeople/<person_name>', methods=['PUT'])
def edit_people_by_name(person_name):
    try:
        if request.method == 'PUT':
            # Periksa apakah orang dengan nama tertentu ada
            query = db.collection('MissingPersons').where('nama', '==', person_name)
            results = query.stream()
            for doc in results:
                person_ref = doc.reference

            if not person_ref.get().exists:
                return jsonify({'error': 'Person not found'}), 404

            # request photo
            fotos = request.files.getlist('fotos[]')
            pathlib.Path(app.config['UPLOAD_FOLDER'], person_name).mkdir(exist_ok=True)

            foto_paths = []  # Inisialisasi array untuk menyimpan path setiap foto
            
            for foto in fotos:     
                filename = secure_filename(foto.filename)
                ext = os.path.splitext(filename)[1]
                new_filename = get_random_string(20)
                foto.save(os.path.join(app.config['UPLOAD_FOLDER'], person_name, new_filename+ext))
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], person_name, new_filename+ext)
                foto_paths.append(file_path)

            # Ambil data dari form
            nama = request.form.get('nama', '')
            umur = request.form.get('umur', '')
            tinggi = request.form.get('tinggi', '')
            berat_badan = request.form.get('berat_badan', '')
            ciri_fisik = request.form.get('ciri_fisik', '')
            nomor_dihubungi = request.form.get('nomor_dihubungi', '')
            sering_ditemukan_di = request.form.get('sering_ditemukan_di', '')
            kota = request.form.get('kota', '')
            gender = request.form.get('gender', '')
            isFound = request.form.get('isFound', '')

            # Update data orang
            person_ref.update({
                'foto': foto_paths,
                'nama': nama,
                'umur': umur,
                'tinggi': tinggi,
                'berat_badan': berat_badan,
                'ciri_fisik': ciri_fisik,
                'nomor_dihubungi': nomor_dihubungi,
                'sering_ditemukan_di': sering_ditemukan_di,
                'kota': kota,
                'gender': gender,
                'isFound': isFound
            })

            # Mendapatkan data yang sudah diupdate
            updated_data = person_ref.get().to_dict()

            return jsonify({'message': f'Person with name {person_name} updated successfully', 'updated_data': updated_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# delete person based on name
@app.route('/deletepeople/<person_name>', methods=['DELETE'])
def delete_people_by_name(person_name):
    try:
        # Periksa apakah orang dengan nama tertentu ada
        query = db.collection('MissingPersons').where('nama', '==', person_name)
        results = query.stream()
        for doc in results:
            doc.reference.delete()
        return jsonify({'message': f'Person with name {person_name} deleted successfully'}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)

