
'''
Hyperparameter Tuning with Keras Tuner
'''
#pip install keras-tuner
import tensorflow as tf
import kerastuner
import matplotlib.pyplot as plt
import numpy as np
from keras_tuner import RandomSearch


#load dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()


#shape of the data splits
print(x_train.shape, x_test.shape)
print(set(y_train))


#view data sample
plt.imshow(x_train[0],cmap='binary')
plt.xlabel(y_train[0])
plt.show()


#define model
def create_model(hp):
  num_hidden_layers=1
  num_units=8
  dropout_rate=0.1
  learning_rate=0.01
  if hp:
    num_hidden_layers=hp.Choice('num_hidden_layers',values=[1,2,3])
    num_units=hp.Choice('num_units',values=[8,16,32,64])
    dropout_rate=hp.Float('dropout_rate',min_value=0.1,max_value=0.5)
    learning_rate=hp.Float('learning_rate',min_value=0.0001,max_value=0.01)

  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
  model.add(tf.keras.layers.Lambda(lambda x: x/255.))

  for _ in range(0,num_hidden_layers):
    model.add(tf.keras.layers.Dense(num_units,activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
  
  model.add(tf.keras.layers.Dense(10,activation='softmax'))

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=['accuracy'])
  return model
create_model(None).summary()


tuner = RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=20,
    directory='logs',
    project_name ='fashion_mnist',
    overwrite=True)
tuner.search_space_summary()

#fitting the tuner on train dataset
tuner.search(x_train,y_train,epochs=4,validation_data=(x_test,y_test))

#best model
model=tuner.get_best_models(num_models=1)[0]
model.summary()

tuner.results_summary()

