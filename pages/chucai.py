import json
import pathlib
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

DATA_PATH = 'data\\data2'
TRAIN_JSON = 'data\\labels.json'
RAW_FOLDER = 'data\\data'

with open(TRAIN_JSON,'r', encoding='utf8') as f:
    train_labels = json.load(f)


#find all characters in labels (so we don't blindly add chacters not even existed in our dataset)
char_list= set()
for label in train_labels.values():
    char_list.update(set(label))
char_list=sorted(char_list)

print(char_list)
def encode_to_labels(txt):
    # encoding each output word into digits of indexes
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print("No found in char_list :", char)
        
    return dig_lst

train_image_path = []

for item in pathlib.Path(RAW_FOLDER).glob('**/*'):
    if item.is_file() and item.suffix not in [".json"]:
        train_image_path.append(str(item))

dict_filepath_label={}
raw_data_path = pathlib.Path(RAW_FOLDER)
for item in raw_data_path.glob('**/*.*'):
    file_name=str(os.path.basename(item))
    if file_name != "labels.json":
      label = train_labels[file_name]
      dict_filepath_label[str(item)]=label
print(dict_filepath_label)
label_lens= []
for label in dict_filepath_label.values():
    label_lens.append(len(label))

max_label_len = max(label_lens)

all_image_paths = list(dict_filepath_label.keys())

widths = []
heights = []
for image_path in all_image_paths:
    img = cv2.imread(image_path)
    (height, width, _) = img.shape
    heights.append(height)
    widths.append(width)

min_height = min(heights)
max_height = max(heights)
min_width = min(widths)
max_width = max(widths)


# being a good Data Scientist, we need to have train set and test set
test_size = 0.2
train_image_paths, val_image_paths = train_test_split(all_image_paths, test_size=test_size, random_state=42)

TIME_STEPS = 240

#lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []
resize_max_width=0

for val_img_path in val_image_paths:
    # print(f_name)
    # read input image and convert into gray scale image
    img = cv2.cvtColor(cv2.imread(val_img_path), cv2.COLOR_BGR2GRAY)
    
    # in this dataset, we don't need to do any resize at all here.
    img = cv2.resize(img,(int(118/height*width),118))
    
    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]
        
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')
    
    # YOUR PART: Blur it
    img = cv2.GaussianBlur(img, (5,5), 0)

    # YOUR PART: Threshold the image using adapative threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    # add channel dimension
    img = np.expand_dims(img , axis = 2)
    
    # Normalize each image
    img = img/255.

    label = dict_filepath_label[val_img_path]

    valid_orig_txt.append(label)   
    valid_label_length.append(len(label))

    # our time steps for valid input
    valid_input_length.append(TIME_STEPS)
    valid_img.append(img)

    # convert words to digits based on charlist
    valid_txt.append(encode_to_labels(label))

max_label_len = TIME_STEPS 

inputs = Input(shape=(118,2167,1))
 
# Block 1
x = Conv2D(64, (3,3), padding='same')(inputs)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_1 = x 

# Block 2
x = Conv2D(128, (3,3), padding='same')(x)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_2 = x

# Block 3
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_3 = x

# Block4
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_3])
x = Activation('relu')(x)
x_4 = x

# Block5
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_5 = x

# Block6
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_5])
x = Activation('relu')(x)

# Block7
x = Conv2D(1024, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3, 1))(x)
x = Activation('relu')(x)

# pooling layer with kernel size (2,2) to make the height/2 #(1,9,512)
x = MaxPool2D(pool_size=(3, 1))(x)
 
# # to remove the first dimension of one: (1, 31, 512) to (31, 512) 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
 
# # # bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

# # this is our softmax character proprobility with timesteps 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time

act_model = Model(inputs, outputs)

act_model.summary()

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length) # all must be equal length to T timesteps
valid_label_length = np.array(valid_label_length) 
data = 'E:/matkhau/trituenhantao/thucHanh/doan/a/checkpoint_weights.hdf5'

act_model.load_weights(os.path.join(data))

if __name__ == "__main__":
    import streamlit as st
    import numpy as np
    sh = st.container()
    sh.header('Nhận Diện Chữ Cái Viết Tay')
    OFFSET= sh.number_input('Điểm khởi đầu: ',0 ,10)
    NO_PREDICTS = sh.number_input('Số lượng ảnh: ', OFFSET+1, 50)
    chay = sh.button('Chạy')

    if chay:
        prediction = act_model.predict(valid_img[OFFSET:OFFSET+NO_PREDICTS])
        #prediction.shape
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=True)[0][0])
        # see the results
        all_predictions =[]
        i = 0
        for x in out:
            pred = ""
            for p in x:  
                if int(p) != -1:
                    pred += char_list[int(p)]
            all_predictions.append(pred)
            i+=1

        for n in range(NO_PREDICTS):
            sh.image(valid_img[n][:,:,0])
            sh.title(f"Label {n}: "+valid_orig_txt[n+OFFSET])
            sh.text(f"Prediction {n}: "+all_predictions[n+OFFSET])
        
        