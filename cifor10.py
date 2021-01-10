 import tensorflow as tf
 from tensorflow.keras import datasets, layers ,models
 from tensorflow.keras.preprocessing.image  import ImageDataGenerator ,img_to_array,load_img
 import matplotlib.pyplot as plt
 import random

 import numpy as np

 (x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
 
 classes=["airplane","automibile","bird","cat","deer","dog","frog","horse","ship","truck"]


 def plot_sample(x,y,index):
   plt.figure(figsize=(15,2))
   plt.imshow(x[index])
   plt.xlabel(classes[y[index]])
   
 x_train=x_train.astype('float32')
 x_test=x_test.astype('float32')
 x_train=x_train/255
 x_test=x_test/255
 y_train=tf.keras.utils.to_categorical(y_train,num_classes=10)
 y_test=tf.keras.utils.to_categorical(y_test,num_classes=10)

 cnn = models.Sequential([
                        layers.Conv2D(filters=32,kernel_size=(3,3),activation='sigmoid',input_shape=(32,32,3)),
                        layers.MaxPool2D((2,2)),
                        layers.Dropout(0.1),
                     
                          layers.Conv2D(filters=64,kernel_size=(3,3),activation='sigmoid'),
                        layers.MaxPool2D((2,2)),
                         layers.Dropout(0.1),
                        
                        #cnn
                        layers.Flatten(),
                        layers.Dense(64,activation='sigmoid'),
                        layers.Dropout(0.1),
                        layers.Dense(10,activation='softmax')
 ])
 cnn.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])
 #cnn.fit(x_train,y_train,epochs=10)
 #cnn.evaluate(x_test,y_test)
 #//////////////////////////////////
 detage=ImageDataGenerator()
 detage.fit(x_train)
 history = cnn.fit_generator(detage.flow(x_train,y_train,batch_size=64),steps_per_epoch=x_train.shape[0]//64,epochs=15,validation_data=(x_test,y_test)
                   )
cnn.evaluate(x_test,y_test,64)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summariz
layer_output=[layer.output for layer in cnn.layers[1:]]
visu_model=tf.keras.models.Model(inputs=cnn.input,outputs=layer_output)
img=load_img('dog.png')
x=img_to_array(img=img)
print(x.shape)
x=x.reshape((1,32,32,3))
x=x/255
feature_map=visu_model.predict(x)
print(len(feature_map))
layer_name=[layer.name for layer in cnn.layers]
print(layer_name)
for layer_names,feature_maps in zip(layer_name,feature_map):
  print(feature_maps.shape)
  if(len(feature_maps.shape)==4):
    channel=feature_maps.shape[-1]
    size=feature_maps.shape[1]
    disply_g=np.zeros((size,size*channel))
    for i in range(channel):
      x=feature_maps[0,:,:,i]
      x-=x.mean()
      x/=x.std()
      x*=16
      x+=32
      x=np.clip(x,0,255).astype('uint8')
      disply_g[:,i*size:(i+1)*size]=x
    scale=20. / channel
    plt.figure(figsize=(scale*channel,scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(disply_g,aspect='auto',cmap='viridis')
    plt.show()


 #///////////////////////////////////
 y_test=y_test.reshape(-1,)
 y_pred=cnn.predict(x_test)
 y_class=[np.max(element) for element in y_pred]
 #print(y_class[0:10])
