import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

# 创建一个简单的 Keras 模型。
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=500, verbose=0)

model.summary()
result = model.predict([1, 3, 7])

# print(result)
plt.plot(x,y)
plt.plot([1, 3, 7], result)
# plt.scatter(x,y)
# plt.show()

# export_dir = 'saved_model/test'
# tf.saved_model.save(model, export_dir)
#
# 转换模型。
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# tflite_model_file = pathlib.Path('saved_model/model.tflite')
# tflite_model_file.write_bytes(tflite_model)
