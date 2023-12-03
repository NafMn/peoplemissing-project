const {sendPhotoHandler} = require('./handler');

const routes = [
    {
        method: 'POST',
        path: '/find',
        handler: sendPhotoHandler
    }
];

module.exports = routes;