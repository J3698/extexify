from models_tf import model1
from data import load_data
import tensorflow as tf
import os


loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam(1e-3)
metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)]

model = model1()
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

x_train, y_train, x_val, y_val, _, _ = \
            load_data(0.2, 0.1, "./dataX.npy", "./dataY.npy")

x_train = tf.ragged.constant(x_val)
x_val = tf.ragged.constant(x_val)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_val))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))


history = model.fit(train_dataset, epochs = 20, \
                    validation_data = val_dataset,
                    batch_size = 8, workers = os.cpu_count())
