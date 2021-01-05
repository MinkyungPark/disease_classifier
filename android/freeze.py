from tensorflow import keras
model = keras.models.load_model('android/h5/android_lstm_0.935606_4.h5', compile=False)

export_path = 'android/pb_lstm'
model.save(export_path, save_format='tf')

# ValueError: Attempted to save a function b'__inference_forward_lstm_1_layer_call_fn_9811' 
# which references a symbolic Tensor Tensor("dropout/mul_1:0", shape=(None, 64), dtype=float32) 
# that is not a simple constant. This is not supported.