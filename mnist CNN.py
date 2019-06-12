#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


# In[2]:


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     number = 10000
#     x_train = x_train[0:number]
#     y_train = y_train[0:number]
    #x_train = x_train.reshape(number, 28*28)
    x_train = x_train.reshape(60000, 28, 28, 1)
    #x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_test = x_test.reshape(10000, 28, 28, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
#     y_train = np_utils.to_categorical(y_train,10)
#     y_test = np_utils.to_categorical(y_test,10)
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)
#     x_train = x_train
#     x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
#    x_test = np.random.normal(x_test)
    return (x_train, y_train), (x_test, y_test)


# In[3]:


(x_train, y_train), (x_test, y_test) = load_data()


# In[4]:


# model = Sequential()
y_test.shape


# In[5]:


model2 = Sequential()


# In[6]:


# model.add(Dense(input_dim=28*28, units=633, activation='relu'))
# #model.add(Dropout(0.7))
# model.add(Dense(units=633,activation='relu'))
# #model.add(Dropout(0.7))
# model.add(Dense(units=633,activation='relu'))
# #model.add(Dropout(0.7))

# model.add(Dense(units=10,activation='softmax'))


# In[7]:


#邊界模式=same (填補=0, 步幅(str)=1)
model2.add(Conv2D(filters=25, kernel_size=(3,3), input_shape=(28,28,1), activation='relu', padding='same'))
model2.add(MaxPooling2D(2,2))
# model2.add(Conv2D(32, 3, 3, input_shape = (128, 128, 1), activation = 'relu'))
# model2.add(MaxPooling2D(pool_size = (2, 2)))

model2.add(Conv2D(filters=50, kernel_size=(3,3), input_shape=(28,28,1), activation='relu', padding='same'))
model2.add(MaxPooling2D(2,2))
# model2.add(Conv2D(32, 3, 3, activation = 'relu'))
# model2.add(MaxPooling2D(pool_size = (2, 2)))
model2.add(Dropout(0.25)) 

model2.add(Flatten())

model2.add(Dense(output_dim=100, activation='relu'))
model2.add(Dropout(0.5)) 

model2.add(Dense(output_dim=10, activation='softmax'))


# In[8]:


#分類loss不適合用mse
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[9]:


model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


# model.fit(x_train, y_train, batch_size=100, epochs=20)


# In[11]:


#model2.fit(x_train, y_train, batch_size=100, epochs=20)
model2.fit(x=x_train,y=y_train,validation_split=0.2, epochs=10, batch_size=300,verbose=2)


# In[12]:


result = model2.evaluate(x_train, y_train)
print ('\ntrain Acc:', result[1])


# In[13]:


result = model2.evaluate(x_test, y_test)
print ('\nTest Acc:', result[1])


# In[ ]:





# In[ ]:




