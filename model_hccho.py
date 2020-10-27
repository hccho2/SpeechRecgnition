
'''
참고: https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ


- 다운받은 features_3_sec.csv로 MyModel(MLP) train하면 train acc = 95.08, test acc 91.23% 나온다.
- 직접만든 feature로 하면 train acc=93.3, test acc=94.35

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json,time
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os, csv
import librosa
    
DATASET_PATH = r'D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\genres_original'   # 음악 장르별 wav 디렉토리
JSON_PATH = r"D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\data_10.json"         # save_mfcc로 생성된 json파일
CSV_PATH = r"D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\features_3_sec.csv"    # 다운받은 feature 파일.


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into---> 30초 wav를 num_segments개로 나누어, 독립적인 data로 만든다.
        :return:
        """

    SAMPLE_RATE = 22050
    TRACK_DURATION = 30 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION  # load했을 때, data 길이


    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)  # 1개 파일을 num_segments로 나누었을 때, 각각의 data길이
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)  # mel spectrogram 길이(frame 수).

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

        # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)  # i=0일 때는 dataset_path 자체이고, i=1일대 첫번째 하위디렉토리 이므로...
                        print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w",encoding='utf8') as fp:
        json.dump(data, fp, indent=4)






def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def json_load_test():
    json_file = JSON_PATH
    with open(json_file, "r",encoding='utf8') as fp:
        data = json.load(fp)
    
    print(data['mapping'])  # ['D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\blues', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\classical', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\country', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\disco', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\hiphop', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\jazz', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\metal', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\pop', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\reggae', 'D:\\SpeechRecognition\\DeepLearningForAudioWithPython-musikalkemist\\Data\\genres_original\\rock']

    print(len(data['mfcc']), len(data['labels']))  # 9986개
    
    print(data['labels'][:5], np.array(data['mfcc']).shape)  # mfcc는 transpose되어서 저장된 것. shape: (130,num_mfcc=13)
    
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((data['mfcc'], data['labels']))
    train_dataset = train_dataset.shuffle(buffer_size=10)
    train_dataset = train_dataset.batch(batch_size=4,drop_remainder=True)
    
    
    def mapping_fn(X,Y):
        print(X,X.shape)
        return X,Y
    
    
    
    train_dataset = train_dataset.map(mapping_fn)
    
    
    
    iterator = iter(train_dataset)
    
    for i in range(3):
        x,y = iterator.get_next()
        
        print(x.shape, y)

def load_csv_data(csv_filename,start_col=1):
    
    data = pd.read_csv(csv_filename)  # (999,60)
    data = data.iloc[0:, 1:]  # (9990, 59)
    print(data.head(5))
    
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels  ===> 58개 featrues
    
    #### NORMALIZE X ####
    
    # Normalize so everything is on the same scale. 
    
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_x = min_max_scaler.fit_transform(X)   # return numpy array (9990,58)
    np_x = np_x[:,start_col:] # 첫번째 column은 'length'로 모두 같은 값이라 제거.  ===> 57개 features


    le = LabelEncoder()
    np_y = le.fit_transform(data['label'])  # 0,..,9까지 변환.
    mapping_table = le.classes_
    print(mapping_table)

     
    X_train, X_test, y_train, y_test = train_test_split(np_x, np_y, test_size=0.2, random_state=42)   # Frame in Frame out


    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

def load_json_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y


def train_mlp():
    
    #csv_filename = CSV_PATH  # start_col=1
    csv_filename = r'D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\data_adv_3_sec_hccho.csv'  # start_col=0
    
    
    
    X_train, X_test, y_train, y_test = load_csv_data(csv_filename,start_col=0)  ##### json 파일 Load
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size=32,drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
    val_dataset = val_dataset.shuffle(buffer_size=1000)
    val_dataset = val_dataset.batch(batch_size=32,drop_remainder=True)

    input_dim = X_train.shape[-1]


    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            
            drop_rate = 0.25
            # mode=1, drop_rate=0 --> val acc가 88%까지 나온다(train acc는 98%).
            # mode=1, drop_rate=0.25 --> lr=0.0001 ---> val acc가 91%까지 나온다(train acc는 95%).
            #                            lr=0.001 ---> val acc가 95%까지 나온다(train acc는 97%).
            
            model_mode = 1
            
            if model_mode ==1:
               # validation acc가 90%까지 나온다.
                self.model = tf.keras.Sequential([
                    
                    # 1st dense layer
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dropout(drop_rate),
                    # 2nd dense layer
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(drop_rate),
                    # 3rd dense layer
                    tf.keras.layers.Dense(64, activation='relu'),      
                    tf.keras.layers.Dropout(drop_rate),
                    # output layer
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
            else:
                # regularization을 적용한 이 모텔은 dropout만 적용한 모델에 비해, 성능이 나오지 않는다.
                self.model = tf.keras.Sequential([
                
                    # 1st dense layer
                    tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
                    tf.keras.layers.Dropout(0.1),
                    # 2nd dense layer
                    tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
                    tf.keras.layers.Dropout(0.1),

                    # 3rd dense layer
                    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
                    tf.keras.layers.Dropout(0.1),     
                    # output layer
                    tf.keras.layers.Dense(10, activation='softmax')
                ])            
            
        def call(self,inputs):
            return self.model(inputs)

    model = MyModel()

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #model.summary()  # ----> error. 모델이 build되지 않은 상황에서 sumary가 call되면 안된다.
    model.model.build(input_shape=(None,input_dim))
    model.model.summary()
    
    
    # train model
    history = model.fit(train_dataset, validation_data=val_dataset, batch_size=32, epochs=300)
    # plot accuracy and error as a function of the epochs
    plot_history(history)



def prepare_datasets(test_size, validation_size,add_axis=False):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_json_data(JSON_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    if add_axis:
        # add an axis to input sets for CNN
        X_train = X_train[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train_cnn():
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15, 0.15,add_axis=True)
    
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            
            # build network topology
            model = tf.keras.Sequential()
        
            # 1st conv layer
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
        
            # 2nd conv layer
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
        
            # 3rd conv layer
            model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
        
            # flatten output and feed it into dense layer
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.3))
        
            # output layer
            model.add(tf.keras.layers.Dense(10, activation='softmax'))
            
            self.model = model
            
        def call(self,inputs):
            return self.model(inputs)
    
    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = MyModel()

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.model.build(input_shape=input_shape)
    model.model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=100, epochs=100)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

def train_rnn():
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15, 0.15,add_axis=False)
    
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            
            # build network topology
            model = tf.keras.Sequential()
        
            # 2 GRU layers
            model.add(tf.keras.layers.GRU(100, input_shape=input_shape, return_sequences=True))
            model.add(tf.keras.layers.GRU(500))
        
            # dense layer
            model.add(tf.keras.layers.Dense(100, activation='relu'))
            #model.add(keras.layers.Dropout(0.3))
        
            # output layer
            model.add(tf.keras.layers.Dense(10, activation='softmax'))

            
            self.model = model
            
        def call(self,inputs):
            return self.model(inputs)
    
    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = MyModel()

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.model.build(input_shape=input_shape)
    model.model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=100, epochs=100)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)    

def train_xgboost_mfcc():
    # 직접 만든, mfcc만으로 train.
    # music-genre-classification-xgboost.py ----> xgboost_train() ---> 96.66까지 나온다.
    from xgboost import XGBClassifier, XGBRFClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    DATA_PATH = r"D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\data_10.json"   # test acc = 81.10
    X, y  = load_json_data(DATA_PATH)  ##### CSV 파일 Load
    
    
    mode = 2
    if mode ==1:    
        X = np.mean(X,axis=1)
    else:
        X = np.concatenate([np.mean(X,axis=1), np.var(X,axis=1)],axis=-1)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    
    
    # Final model
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    
    
    preds = xgb.predict(X_test)   # array(['hiphop', 'jazz', 'blues', ....],dtype=object)
    
    print('Test Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')
    
def extract_features():
    # download받은 "features_3_sec.csv", "features_30_sec.csv"를 직접 만들수 있을까?
    # wav 파일들로 부터, feature 생성
    # filename,chroma_stft,rmse,spectral_centroid,spectral_bandwidth,rolloff,zero_crossing_rate,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,mfcc17,mfcc18,mfcc19,mfcc20,label
    DATASET_PATH = r'D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\genres_original'
    
    num_segments = 1

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    
    
    
    n_mfcc=20
    
    
    for i in range(1, n_mfcc+1):
        header += f' mfcc{i}'
    header += ' label'   # filename,chroma_stft,rmse,spectral_centroid,spectral_bandwidth,rolloff,zero_crossing_rate,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,mfcc17,mfcc18,mfcc19,mfcc20,label
    header = header.split()
    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:
        for filename in os.listdir(os.path.join(DATASET_PATH,g)):
            songname = os.path.join(DATASET_PATH,g,filename)
            y, sr = librosa.load(songname, mono=True, duration=30)
            
            samples_per_segment = int(len(y) / num_segments)
            
            for d in range(num_segments):
                
                # 1. filename + segment  ---> blues.00000.0.wav
                segment_filename = os.path.splitext(filename)  # ('blues.00000', '.wav')
                segment_filename = segment_filename[0]+'.'+str(d)+segment_filename[1]  #                 
                
                
                start = samples_per_segment * d
                finish = start + samples_per_segment
                
                y_segment = y[start:finish]
                
                # 2. chroma_shift
                chroma_stft = librosa.feature.chroma_stft(y=y_segment, sr=sr)
                
                # 3. rmse
                rms = librosa.feature.rms(y=y)
                
                # 4. spectral_centroid
                spec_cent = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
                
                # 5. spectral_bandwidth
                spec_bw = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
                
                # 6. rolloff
                rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
                
                # 7. zero_crossing_rate
                zcr = librosa.feature.zero_crossing_rate(y_segment)
                
                # 8~27. mfcc(20개)
                mfcc = librosa.feature.mfcc(y=y_segment, sr=sr,n_mfcc=20)
                

                
                
                to_append = f'{segment_filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                for e in mfcc:  # 20개의 mfcc 계수에 대하여, 모든 frame에 대한 평균을 계산한다.
                    to_append += f' {np.mean(e)}'
                
                
                to_append += f' {g}'
                
                
                
                file = open('data.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())
    


def extract_features_advanced():
    # download받은 "features_3_sec.csv", "features_30_sec.csv"를 직접 만들수 있을까?
    # wav 파일들로 부터, feature 생성
    # 평균 + 분산 + 몇가지 feature 추가.
    # mfcc의 분산을 빼먹었네.....
    
    
    DATASET_PATH = r'D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\genres_original'
    output_filename = 'data_adv_3_sec.csv'   # 300초 x 10개 = 50분 소요.
    num_segments = 10

    header = 'filename chroma_stft_mean chroma_stft_var rmse_mean rmse_var spectral_centroid_mean spectral_centroid_var'
    header += ' spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var'
    header += ' harmony_mean harmony_var percussive_mean percussive_var tempo'
    
    
    
    n_mfcc=20
    
    
    for i in range(1, n_mfcc+1):
        header += f' mfcc{i}'
    header += ' label'   # filename,chroma_stft,rmse,spectral_centroid,spectral_bandwidth,rolloff,zero_crossing_rate,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,mfcc17,mfcc18,mfcc19,mfcc20,label
    header = header.split()
    file = open(output_filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    
    s_time = time.time()
    with open(output_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for g in genres:
            for filename in os.listdir(os.path.join(DATASET_PATH,g)):
                songname = os.path.join(DATASET_PATH,g,filename)
                y, sr = librosa.load(songname, mono=True, duration=30)
                
                samples_per_segment = int(len(y) / num_segments)
                
                for d in range(num_segments):
                    
                    # 1. filename + segment  ---> blues.00000.0.wav
                    segment_filename = os.path.splitext(filename)  # ('blues.00000', '.wav')
                    segment_filename = segment_filename[0]+'.'+str(d)+segment_filename[1]  #                 
                    
                    
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    
                    y_segment = y[start:finish]
                    
                    # 2. chroma_shift
                    chroma_stft = librosa.feature.chroma_stft(y=y_segment, sr=sr)
                    
                    # 3. rmse
                    rms = librosa.feature.rms(y=y)
                    
                    # 4. spectral_centroid
                    spec_cent = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
                    
                    # 5. spectral_bandwidth
                    spec_bw = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
                    
                    # 6. rolloff
                    rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
                    
                    # 7. zero_crossing_rate
                    zcr = librosa.feature.zero_crossing_rate(y_segment)
                    
                    
                    #8,9 harmony, 
                    harmonic,percussive = librosa.effects.hpss(y_segment)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    
                    # 8~27. mfcc(20개)
                    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr,n_mfcc=20)
                    
    
                    
                    
                    to_append =  f'{segment_filename} {np.mean(chroma_stft)} {np.var(chroma_stft)}'
                    to_append += f' {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)}'
                    to_append += f' {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)}'
                    to_append += f' {np.mean(harmonic)} {np.var(harmonic)} {np.mean(percussive)} {np.var(percussive)}'
                    to_append += f' {tempo}'
                    for e in mfcc:  # 20개의 mfcc 계수에 대하여, 모든 frame에 대한 평균을 계산한다.
                        to_append += f' {np.mean(e)}'
                    
                    
                    to_append += f' {g}'
                    
                    
                    writer.writerow(to_append.split())
            print(g,' done ......', time.time() - s_time)


def extract_features_advanced2():
    # download받은 "features_3_sec.csv", "features_30_sec.csv"를 직접 만들수 있을까?
    # wav 파일들로 부터, feature 생성
    # 평균 + 분산 + 몇가지 feature 추가.
    # mfcc의 분산추가
    
    
    DATASET_PATH = r'D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data\genres_original'
    output_filename = 'data_adv_3_sec.csv'   # 300초 x 10개 = 50분 소요.
    num_segments = 10

    header = 'filename chroma_stft_mean chroma_stft_var rmse_mean rmse_var spectral_centroid_mean spectral_centroid_var'
    header += ' spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var'
    header += ' harmony_mean harmony_var percussive_mean percussive_var tempo'
    
    
    
    n_mfcc=20
    
    
    for i in range(1, n_mfcc+1):
        header += f' mfcc{i}_mean mfcc{i}_var'
    header += ' label'   # filename,chroma_stft,rmse,spectral_centroid,spectral_bandwidth,rolloff,zero_crossing_rate,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,mfcc17,mfcc18,mfcc19,mfcc20,label
    header = header.split()
    file = open(output_filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    
    s_time = time.time()
    with open(output_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for g in genres:
            for filename in os.listdir(os.path.join(DATASET_PATH,g)):
                songname = os.path.join(DATASET_PATH,g,filename)
                y, sr = librosa.load(songname, mono=True, duration=30)
                
                samples_per_segment = int(len(y) / num_segments)
                
                for d in range(num_segments):
                    
                    # 1. filename + segment  ---> blues.00000.0.wav
                    segment_filename = os.path.splitext(filename)  # ('blues.00000', '.wav')
                    segment_filename = segment_filename[0]+'.'+str(d)+segment_filename[1]  #                 
                    
                    
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    
                    y_segment = y[start:finish]
                    
                    # 2. chroma_shift
                    chroma_stft = librosa.feature.chroma_stft(y=y_segment, sr=sr)
                    
                    # 3. rmse
                    rms = librosa.feature.rms(y=y)
                    
                    # 4. spectral_centroid
                    spec_cent = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
                    
                    # 5. spectral_bandwidth
                    spec_bw = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
                    
                    # 6. rolloff
                    rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
                    
                    # 7. zero_crossing_rate
                    zcr = librosa.feature.zero_crossing_rate(y_segment)
                    
                    
                    #8,9 harmony, 
                    harmonic,percussive = librosa.effects.hpss(y_segment)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    
                    # 8~27. mfcc(20개)
                    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr,n_mfcc=20)
                    
    
                    
                    
                    to_append =  f'{segment_filename} {np.mean(chroma_stft)} {np.var(chroma_stft)}'
                    to_append += f' {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)}'
                    to_append += f' {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)}'
                    to_append += f' {np.mean(harmonic)} {np.var(harmonic)} {np.mean(percussive)} {np.var(percussive)}'
                    to_append += f' {tempo}'
                    for e in mfcc:  # 20개의 mfcc 계수에 대하여, 모든 frame에 대한 평균/분산을 계산한다.
                        to_append += f' {np.mean(e)} {np.var(e)}'
                    
                    
                    to_append += f' {g}'
                    
                    
                    writer.writerow(to_append.split())
            print(g,' done ......', time.time() - s_time)


if __name__ == "__main__":
    s_time = time.time()
    
    #save_mfcc(DATASET_PATH, JSON_PATH,num_mfcc=40, num_segments=10)
    
    #json_load_test()
    #load_csv_data()
    #train_mlp()
    #train_cnn()
    #train_rnn()
    
    
    train_xgboost_mfcc()
    
    
    
    #extract_features()
    #extract_features_advanced()
    #extract_features_advanced2()
    
    
    print('elapsed: ', time.time() - s_time, 'sec')




