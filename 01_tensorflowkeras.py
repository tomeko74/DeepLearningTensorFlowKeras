import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9

# unpacks images to x_train/x_test and labels to y_train/y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# tak można sobie podejrzec dane treningowe
# plt.imshow(x_train[0], cmap = plt.cm.binary)  # cm - colormap, binary - konwertuje do szarości
# plt.show()
# print(x_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)  # normalizowanie z wartości 0-255 do 0-1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # normalizowanie z wartości 0-255 do 0-1

# zwykła sieć, feed-forward bez pętli
model = tf.keras.models.Sequential()
# dodajemy warstwę wejściową, z 28x28 robimy płaski ciąg liczb, .shape to właśnie (28,28)
model.add(tf.keras.layers.Flatten(input_shape=x_train[0].shape))
# dodajemy warstwę ukrytą, 128 neuronów, sigmoid = relu - rectified linear unit
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# dodajemy drugą warstwę ukrytą, 128 neuronów, sigmoid = relu - rectified linear unit
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# dodajemy warstwę wyjściową, 10 neuronów (10 cyfr do rozpoznania), sigmoid = softmax bo to "probability distribution"
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# kompilujemy model podając parametry wykonania
model.compile(optimizer='adam',  # adam to sprawdzony default, opis  tu -> https://arxiv.org/abs/1412.6980v8
              loss='sparse_categorical_crossentropy',  # też taki default, dla cats vs dogs byłoby to binary nie sparse
              metrics=['accuracy'])  # metryka do śledzenia

# trenowanie modelu
model.fit(x_train, y_train, epochs=3)

# liczymy loss i accuracy dla zbioru testowego
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f"Loss testowe: {val_loss}, accuracy testowe: {val_acc}")

# zapiszemy teraz wygenerowany model
model.save('epic_num_reader.model')

# coś tam, coś tam :)

# wczytujemy jako new_model
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])

print(np.argmax(predictions[1]))
plt.imshow(x_test[1])
plt.show()
