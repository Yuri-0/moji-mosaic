import os
import cv2
import numpy as np
import random
import ast
from PIL import Image

print('テキストをモザイクイメージに変えてくれるアプリケーション。「文字モザイク」です。')
print('原本となるテキストファイルを「.txt」形式で「mosaic.py」と同じフォルダの中にいれ、テキストファイルを名前をご指定くださいますと「kinoko.png」のモザイクイメージを出力いたします。')
print('作者：Yuri')

text_file = input('原本のテキストファイルの名前、または位置をご入力ください。:')

with open(text_file, 'r', encoding='utf-8') as raw:
    text = raw.read()
    text = text.replace('\n', '')

color_dic = {}
Pic_pos = []
Pic_used = []

f = open('color_dic.txt', 'a+')
f.seek(0)
for ln in f.readlines()[1:]:
    line = ln.replace('\n', '')
    line = line.split(':')
    color_dic[line[0]] = ast.literal_eval(line[1])

for s in range(1, 11):
    if len(text) <= (2 ** s) ** 2 or s == 10:
        if s == 10 and len(text) > 1048576:
            text[:1048576]
        width = height = 2 ** s
        data = np.zeros((width, height, 3), dtype=np.uint8)
        data.fill(255)
        break

for i in range(width):
    for ii in range(height):
        Pic_pos.append((i, ii))

for i in text:
    if i not in list(color_dic.keys()):
        color = list(np.random.choice(range(256), size=3, replace=False))
        while color in list(color_dic.values()):
            color = list(np.random.choice(range(256), size=3, replace=False))
        new_color = '\n{0}:{1}'.format(i, color)
        f.write(new_color)
        color_dic[i] = color

    X_pos = random.sample(list(map(lambda x: x[0], Pic_pos)), 1)
    Y_pos = random.sample(list(map(lambda x: x[1], Pic_pos)), 1)

    while (X_pos[0], Y_pos[0]) in Pic_used:
        X_pos = random.sample(list(map(lambda x: x[0], Pic_pos)), 1)
        Y_pos = random.sample(list(map(lambda x: x[1], Pic_pos)), 1)

    Pic_used.append((X_pos[0], Y_pos[0]))

    data[X_pos[0], Y_pos[0]] = color_dic[i]

f.close()

data = cv2.resize(data, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
img = Image.fromarray(data)
img.save('kinoko.png')

img = Image.open('kinoko.png')
img.show()
