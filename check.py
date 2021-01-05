import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="android/tflite_cnn_flatten/converted_model.tflite")
interpreter.allocate_tensors()

ss = interpreter.get_input_details()[0]['shape']
for s in ss:
    print(s)

print(interpreter.get_input_details()[0]['dtype'])

print(interpreter.get_output_details()[0]['shape'])
print(interpreter.get_output_details()[0]['dtype'])
