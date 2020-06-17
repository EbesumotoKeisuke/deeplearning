# -*- coding: utf-8 -*-
# 学習データをもとに、撮影した写真のひび割れを判定する。
# 参考サイト:https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('Output/Debug500data/test4_Verification0.1/tea_predict3.json').read())
#保存した重みの読み込み
model.load_weights('Output/Debug500data/test4_Verification0.1/tea_predict3.hdf5')

# 識別種類
categories = ["CD", "UD", "CP", "UP", "CW", "UW"]

#画像を読み込む
img_path = str(input())
img = image.load_img(img_path, target_size=(250, 250, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

#予測結果によって処理を分ける
if features[0, 0] == 1:
    print ("This is the data with cracks in the bridge.")

elif features[0, 1] == 1:
    print ("This is the data without cracks in the bridge.")

elif features[0, 2] == 1:
    print ("This is the data with paint cracks.")

elif features[0, 3] == 1:
    print ("This is the data with no paint cracks.")

elif features[0,4] == 1:
    print ("This is data with concrete cracks.")

elif features[0,5] == 1:
    print ("This is the data without cracking of concrete.")

else:
    print("I couldn't tell. Is it really a picture of a concrete wall?")