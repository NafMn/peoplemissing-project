'use strict';

const Hapi = require('@hapi/hapi');
const Inert = require('@hapi/inert')
const routes = require('./routes');
const predictModel = require('./predictModel');
const Path = require('path');
var admin = require("firebase-admin");

var serviceAccount = require("../serviceAccountKey.json");



// allow admin 
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const init = async () => {

  const server = Hapi.server({
    port: 3000,
    host: 'localhost',
    routes: {
      files: {
        relativeTo: Path.join(__dirname, '../public')
      },
      cors: {
        origin: ['*'],
      },
    },
  });

  // load model
  // await loadModel();

  // register Inert
  await server.register(Inert);

  server.route(routes);

  await server.start();
  console.log('Server running on %s', server.info.uri);
};

process.on('unhandledRejection', (err) => {

  console.log(err);
  process.exit(1);
});

init();