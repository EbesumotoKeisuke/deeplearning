# -*- coding: utf-8 -*-
# LoadCrackDataで読み込んだデータをもとに、学習データのnpyファイルを作成するプログラム

# データについて:https://qiita.com/FukuharaYohei/items/11d4cdce824c2a0e04aa

from PIL import Image
import os, glob
import numpy as np
import random, math
import pickle


# ひび割れのカテゴリ名(ディレクトリ名)
#categories = ["CD", "UD"]
categories = ["CD", "UD", "CP", "UP", "CW", "UW"]

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []

# 画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)

# 渡された画像データを読み込んでXに格納し、また、画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((250, 250))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

print("start program")
# 全データ格納用配列
allfiles = []

# データ容量が大きくなった時用の処理
with open("data/Crackdata.pickle", "rb") as f:
    CrackData = pickle.load(f)

# シャッフル後、学習データと検証データに分ける
random.shuffle(CrackData)
th = math.floor(len(CrackData) * 0.8)    # 学習データの割合
train = CrackData[0:th]
test = CrackData[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)
print ("data import.")

# データの確認
#print("X_test is:")
#print(X_test)
#print("y_test is:")
#print(y_test)
#print("X_train is:")
#print(X_train)
#np.set_printoptions(threshold=np.inf)
#print("y_train is:")
#print(y_train)


# データを保存する（データの名前を「tea_data.npy」としている）
#np.save("data/tea_data.npy", xy)
#np.save("data/tea_Debug-data.npy", xy)
np.save("data/tea_Debug-data1_2000.npy", xy)
print("data save.")
