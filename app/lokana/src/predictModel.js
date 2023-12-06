const tf = require('@tensorflow/tfjs');

// Define your custom layer class
class CustomLayer extends tf.layers.Layer {
    constructor(config) {
      super({});
      this.supportsMasking = true;
    }
  
    static get className() {
      return 'CustomLayer'; // Provide a unique name for your custom layer
    }
  }
  
  // Register the custom layer class
 tf.serialization.registerClass(CustomLayer);
    function customLayer() {
    return new CustomLayer();
  }


   module.exports = {customLayer};