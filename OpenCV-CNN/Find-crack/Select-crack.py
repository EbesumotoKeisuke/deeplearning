# -*- coding: utf-8 -*-
# 学習データをもとに、撮影した写真のひび割れを判定する。
# 参考サイト:https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import sys
from datetime import datetime

# 結果保存先出力pathの指定
output_path = "Output/Debug2000data/test1_FullData/"

#保存したモデルの読み込み
model = model_from_json(open(output_path + 'tea_predict.json').read())
#保存した重みの読み込み
model.load_weights(output_path + 'tea_predict.hdf5')

# 識別種類
categories = ["CD", "UD", "CP", "UP", "CW", "UW"]

#画像を読み込む
#print("Please input image.")
#img_path = str(input())
img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(250, 250, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

# 現在時刻の取得
now_time = datetime.now()

#結果の保存。予測結果によって処理を分ける
with open(output_path + 'Analysis-Result.csv', 'a', encoding='utf-8') as f:
    if features[0, 0] == 1:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",This is the data with cracks in the bridge.\n")
        print ("This is the data with cracks in the bridge.")

    elif features[0, 1] == 1:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",This is the data without cracks in the bridge.\n")
        print ("This is the data without cracks in the bridge.")

    elif features[0, 2] == 1:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",This is the data with paint cracks.\n")
        print ("This is the data with paint cracks.")

    elif features[0, 3] == 1:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",This is the data with no paint cracks.\n")
        print ("This is the data with no paint cracks.")

    elif features[0, 4] == 1:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",This is data with concrete cracks.\n")
        print ("This is data with concrete cracks.")

    elif features[0, 5] == 1:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",This is the data without cracking of concrete.\n")
        print ("This is the data without cracking of concrete.")

    else:
        f.write(str(now_time))
        f.write(",")
        f.write(img_path)
        f.write(",I couldn't tell. Is it really a picture of a concrete wall?\n")
        print("I couldn't tell. Is it really a picture of a concrete wall?")
f.close()