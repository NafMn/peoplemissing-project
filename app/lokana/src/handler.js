// tools
const fs = require("fs");
const util = require("util");
const tf = require('@tensorflow/tfjs');
const {customLayer} = require('./predictModel');
const readdir = util.promisify(fs.readdir);
// firebase
const { initializeApp, applicationDefault, cert } = require("firebase-admin/app");
const { getFirestore, Timestamp, FieldValue, Filter } = require("firebase-admin/firestore");


// send photo will be used for ML to folder images
const sendPhotoHandler = async (request, h) => {
    try {
      const db = getFirestore();
  
      const { photo } = request.payload;
  
      if (!photo) {
        return h.response("Upload Picture Failed");
      }
  
      const file = photo;
      const name = file.hapi.filename;
      const path = `./public/images/input/${name}`;
      const pathFinder = `images/input/${name}`;
  
      // Cek apakah file sudah ada
      const filesInDir = await readdir("./public/images/input");
      if (filesInDir.includes(name)) {
        return h.response("File already exists").code(409);
      }
  
      // Jika tidak ada, simpan file
      const fileStream = fs.createWriteStream(path);
      await file.pipe(fileStream);
  
      // // input to firestore -> MissingPeople
      // await db.collection("UserSubmittedPhotos").add({
      //   foto: `http://localhost:3000/${pathFinder}`,
      // });

      return h.response("Upload picture has been success");
    } catch (error) {
      // Log error di konsol atau tempatkan di file log
      console.error("Error occurred while uploading file:", error);
  
      // Memberikan respons yang lebih ramah pengguna
      return h.response("Something went wrong while uploading the file").code(500);
    }
  };

  const loadModelHandler = async (request, h) => {
    try {
      const layer = customLayer();
      const model = await tf.loadLayersModel('http://localhost:3000/model/model.json');
      console.log('Model loaded:', model);
      return 'Model loaded successfully!';
    } catch (error) {
      console.log('Error loading model', error);
      return h.response('Error loading model').code(500);
    }
  }


  module.exports = {sendPhotoHandler, loadModelHandler};

