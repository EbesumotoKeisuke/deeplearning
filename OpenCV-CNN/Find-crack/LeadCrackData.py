# -*- coding: utf-8 -*-
# Crackのデータを読み込み、pickleのファイルに格納するプログラム
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

# データ容量が大きくなった時用の処理
with open("data/Crackdata.pickle", "wb") as f:
    pickle.dump(allfiles, f, protocol=4)