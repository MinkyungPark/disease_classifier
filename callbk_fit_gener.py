# # 1epoch마다 callbaks.ModelCheckpoint를 불러서 정해놓은 기능을 호출하는 함수
# import os
# from keras.callbacks import ModelCheckpoint

# MODEL_SAVE_FOLDER_PATH = './model/'
# if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
#     os.mkdir(MODEL_SAVE_FOLDER_PATH)
# model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
# cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbos=1, save_best_only=True) # save_weights_only, period(num of epoch)



# # model.fit_generator
# model.fit_generator(
#     train_generator,
#     steps_per_epoch = 1280 / 32,
#     epochs = 10,
#     validation_data = val_generator,
#     validation_steps = 320 / 32,
#     callbacks = [cb_checkpoint]
# )