# .pb -> .tflite

# tflite
# # Create a converter
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # Convert the model
# tflite_model = converter.convert()
# # Create the tflite model file
# tflite_model_name = "mymodel.tflite"
# open(tflite_model_name, "wb").write(tflite_model)

import tensorflow as tf

saved_model_dir = 'android/pb_cnn_lstm'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('android/tflite_cnn_lstm/converted_model.tflite', 'wb').write(tflite_model)


# Exception: We are continually in the process of adding support to TensorFlow Lite for more ops. 
# It would be helpful if you could inform us of how this conversion went by opening a github issue
# at https://github.com/tensorflow/tensorflow/issues/new?template=40-tflite-op-request.md and pasting the following:

# Some of the operators in the model are not supported by the standard TensorFlow Lite runtime and 
# are not recognized by TensorFlow. If you have a custom implementation for them you can disable this error with 
# --allow_custom_ops, or by setting allow_custom_ops=True when calling tf.lite.TFLiteConverter(). Here is a list of builtin
# operators you are using: CAST, CONV_2D, FULLY_CONNECTED, GATHER, MAX_POOL_2D, RESHAPE, SOFTMAX, STRIDED_SLICE. 
# Here is a list of operators for which you will need custom implementations: TensorListFromTensor, TensorListReserve, TensorListStack, While.
