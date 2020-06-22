# -*- coding: utf-8 -*-
# コンクリートのひび割れ写真データをもとに、ひび割れがあるかどうかを判別するプログラム
# 参考サイト:https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
# データについて：https://qiita.com/FukuharaYohei/items/11d4cdce824c2a0e04aa
# 評価関数について：https://qiita.com/FukuharaYohei/items/f7df70b984a4c7a53d58

import os
from keras import layers, models
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import keras.callbacks as callback
from sklearn.metrics import f1_score
from keras.callbacks import Callback
import csv
import pickle

# 保存先出力pathの指定
output_path = "Output/Debug2000data/test2_EqualData/"
#output_path = "Output/Debug200data/test3_F-measure/"

# 処理時間の計測開始
start_time = time.time()

# ディープラーニングモデルの構築
model = models.Sequential()

# 畳み込み層の追加
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# 二次元配列を一次元配列に整列させる。
model.add(layers.Flatten())

# dropoutの追加
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(6, activation="sigmoid"))   # 分類先の種類分設定

# モデル構成の確認
model.summary()
with open(output_path + "modelSummary.txt", "w") as fp:
    model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

# csvに結果の保存
csv_logger = callback.CSVLogger(output_path + 'training.log.csv', separator=',', append=True)

# ---------------------------------------------------------------------
# 参考サイト：https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
from keras import backend as K

# 混合行列の算出
def Conf_matrix(y_true, y_pred):
    Conf_metrics = confusion_matrix(y_true, y_pred)
    return Conf_metrics

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # これ計算合ってる？
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# モデルのコンパイル
model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc", recall_m, precision_m, f1_m])
    # metricの引数は(y_true, y_pred),y_trueは真値で、y_predは予測値
# ---------------------------------------------------------------------

# データの準備
#categories = ["CD", "UD"]
categories = ["CD", "UD", "CP", "UP", "CW", "UW"]
nb_classes = len(categories)

# データの読み込み
#X_train, X_test, y_train, y_test = np.load("data/tea_data.npy", allow_pickle=True)
#X_train, X_test, y_train, y_test = np.load("data/tea_Debug-data4_200.npy", allow_pickle=True)
#X_train, X_test, y_train, y_test = np.load("data/tea_Debug-data1_2000.npy", allow_pickle=True)

with open("data/Crackdata1_2000_2.pickle", "rb") as f2:
    X_train, X_test, y_train, y_test = pickle.load(f2)

# データの確認
#print("X_test is:")
#print(X_test)
#print("y_test is:")
#print(y_test)
#print("X_train is:")
#print(X_train)
#print("y_train is:")
#print(y_train)


# データの正規化
X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

# kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


# ---------------------------------------------------------------------
# 参考サイト：https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
# コールバック

class F1Callback(Callback):
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs):
        pred = self.model.predict(self.X_val)
        f1_val = f1_score(self.y_val, np.round(pred), average='micro')  # averageについて：https://qiita.com/isobe_mochi/items/beb4357d69f6886ac05d
        print("f1_val =", f1_val)
        # 以下チェックポイントなど必要なら書く

# モデルの学習
model = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test),  callbacks=[F1Callback(model, X_test, y_test), csv_logger])
# ---------------------------------------------------------------------

# 学習結果を変数に格納
acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']


# 学習結果をグラフ表示
epochs = range(len(acc))
'''
count_epochs = 1
for i in epochs:
    count_epochs = int(count_epochs + i)
'''
plt.plot(epochs, acc, 'bo', label='Training acc')   # 恐らくbは青色、oはサークルマーカー(点)を意味する
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlim(0, 20)
plt.ylim(0.1, 1)
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(output_path + 'Training-and-validation-accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlim(0, 20)
plt.ylim(0.1, 1)
plt.title('Training and validation loss')
plt.legend()
plt.savefig(output_path + 'Training-and-validation-loss')

# 学習結果の表示
for i in range(20):
    print("epochs", i, ", acc:", acc[i], ", val_acc:", val_acc[i], ", loss:", loss[i], ", val_loss:", val_loss[i])

# モデルの保存
json_string = model.model.to_json()
open(output_path + 'tea_predict.json', 'w').write(json_string)

# 重みの保存
hdf5_file = output_path + "tea_predict.hdf5"
model.model.save_weights(hdf5_file)

# 計測時間の表示
process_time = time.time() - start_time
print("total processing time is:", process_time, " [ms]\n")

# 計算時間の追記
with open(output_path + 'training.log.csv', 'a', encoding='utf-8') as f:
    f.write("total processing time is:,")
    f.write(str(process_time))
    f.write(",[ms]\n")
f.close()
