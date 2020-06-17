# -*- coding: utf-8 -*-
# コンクリートのひび割れ写真データをもとに、学習データを作成するプログラム
# 参考サイト:https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
# データについて:https://qiita.com/FukuharaYohei/items/11d4cdce824c2a0e04aa

from PIL import Image
import os, glob
import numpy as np
import random, math
import pickle


# 画像が保存されているルートディレクトリのパス
root_dir = "data/SDNET2018_edit"
#root_dir = "data/SDNET2018_DebugTest3_1000"  # 各ディレクトリ500枚ずつ入れたデータ

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

# カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):  # enumerateで要素とインデックスを同時に取得する
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))
print("data storage.")

# シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)    # 学習データの割合
train = allfiles[0:th]
test = allfiles[th:]
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

