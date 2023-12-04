const { sendPhotoHandler, loadModelHandler } = require('./handler');
const Path = require('path');


const routes = [
    {
        method: 'POST',
        path: '/find',
        config: {
            payload: {
                output: "stream",
                parse: true,
                multipart: true,
            },
        },
        handler: sendPhotoHandler,
        
    },
    {
        method: 'GET',
        path: '/images/input/{params*}',
        handler: {
           directory : {
            path: 'images/input',
            redirectToSlash: true
           }
        }
        
    },
    {
        method: 'GET',
        path: '/model/{params*}',
        handler: {
            directory: {
                path: Path.join(__dirname, '../model'),
                redirectToSlash: true,
                index: true,
            }
        }
    },
    {
        method: 'GET',
        path: '/loadModel',
        handler: loadModelHandler
    }
];

module.exports = routes;