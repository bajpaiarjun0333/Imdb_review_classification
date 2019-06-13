#importing the libraries needed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras import layers

#loading the dataset and converting into train and test examples
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
#data has been loaded successfully
word_index=imdb.get_word_index()
#word index is a dictionary mapping words to integers

#now reverse map the index corressponding to the wrds
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
#sample decoded review
decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#preprocessing the data
def vectorize(sequences,dimensions=10000):
    results=np.zeros((len(sequences),dimensions))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

X_train=vectorize(train_data)
X_test=vectorize(test_data)

#vectorizing the ylabels data
Y_train=np.asarray(train_labels).astype('float32');
Y_test=np.asarray(test_labels).astype('float32');

#defining the model for work
model=Sequential()
#building the model
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


#creating the validation set
X_val=X_train[:10000]
partial_X_train=X_train[10000:]
Y_val=Y_train[:10000]
partial_Y_train=Y_train[10000:]

#compilin the model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#fittin the actual model
history=model.fit(partial_X_train,partial_Y_train,epochs=20,batch_size=512,
                  validation_data=(X_val,Y_val))

#plotting the training and validation losses
history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']

epochs=range(1,21)
plt.plot(epochs,loss_values,'bo',label='Training_Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation_Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plotting the training and validation accuracy
plt.clf()
acc_values=history_dict['acc']
val_acc_values=history_dict['val_acc']
plt.plot(epochs,acc_values,'bo',label='Training_Accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation_Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#finding out the predictions
model.evaluate(X_test,Y_test)
#till now we saw the model is suffering from the problem of overfitting 
#I use the process of early stopping which is reducing the number of the epochs
retrained_model=Sequential()
retrained_model.add(layers.Dense(16,activation='relu'))
retrained_model.add(layers.Dense(16,activation='relu'))
retrained_model.add(layers.Dense(1,activation='sigmoid'))
retrained_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
retrained_model.fit(partial_X_train,partial_Y_train,epochs=3,batch_size=512,
                  validation_data=(X_val,Y_val))
retrained_model.evaluate(X_test,Y_test)







