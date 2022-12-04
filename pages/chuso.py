from tensorflow.keras.models import model_from_json
from tensorflow import keras
import cv2
from pyexpat import model
import streamlit as st
import numpy as np

model_config = "data1/digit_config.json"
model_weight = "data1/digit_weight.h5"
model = model_from_json(open(model_config).read())
model.load_weights(model_weight)


model.compile(optimizer='Adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
size = 150

X_test_chuanHoa = X_test / 255.0

RESHAPED = 784

X_test_chuanHoa = X_test.reshape(10000, RESHAPED)
#index = np.random.randint(0, 10000, size)

def taoAnh(sh):
    X_test_index = np.zeros((size,28,28), dtype = np.uint8)
    for i in range(0, size):
        X_test_index[i,:,:] = X_test[sh.index[i],:,:]

    X_test_image = np.zeros((10*28,15*28), np.uint8)
    for i in range(0, size):
        m = i // 15
        n = i % 15
        X_test_image[m*28:(m+1)*28,n*28:(n+1)*28] = X_test[sh.index[i],:,:]
    cv2.imwrite('image/chu_so_ngau_nhien.jpg', X_test_image)
    sh.write('Hình ảnh: ')
    img = sh.image('image/chu_so_ngau_nhien.jpg')
    sh.write('Kết quả nhận dạng')
    X_test_index = np.zeros((size,RESHAPED), dtype=np.float64)
    for i in range(size):
        X_test_index[i] = X_test_chuanHoa[sh.index[i]]
    prediction = model.predict(X_test_index)
    s = ''
    for i in range(0, size):
        ket_qua = np.argmax(prediction[i])
        s=s+str(ket_qua)
        print(ket_qua, end = ' ')
        if (i+1) % 15 == 0:
            s=s+'\n' 
    sh.text(s)

if __name__ == "__main__":    
    sh = st.container()
    sh.index = np.random.random_integers(0, 10000, size)
    sh.header('Nhận Diện Chữ Số Viết Tay')
    btn_taoCaiAnh = sh.button('Tạo ảnh và nhận dạng')
    if btn_taoCaiAnh:
        taoAnh(sh)