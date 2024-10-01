<h1 align="center">Moji Mosaic</h1>
<p align="center"><i>A Simple Way to Text to Art</i></p>
<div align="center">

![Static Badge](https://img.shields.io/badge/python-3.7-blue)

</div>

Moji Mosaic is a unique text-to-image generator that transforms arbitrary text into colorful, abstract art pieces. Each generated mosaic is a representation of the input text to encrypted color combination, creating a visually appealing and enigmatic artwork.

## How It Works
### Core Workflow

1. **Color Allocation**: Each letter in the input text is assigned a unique color.
2. **Randomization**: Letters are mixed randomly to obscure the original text.
4. **Placement**: The mixed letters are placed on a square white palette.
5. **Sizing**: The palette size is determined by the number of letters in the input text.

### Color Dictionary System
- Moji Mosaic utilizes a dynamic color dictionary to assign colors to characters.
- Initially, the color dictionary is empty and grows as new characters are encountered.
- For each new character:
  - If it exists in the dictionary, its predefined color is used.
  - If it's new, a random color (not already in use) is assigned and added to the dictionary.

_Each user ends up with their own unique color dictionary from the initial one finally, so even when generating a mosaic from the same text, completely different color combinations may appear depending on the color dictionary possessed by the user._

###  Encryption Effect
Mosaics generated through Moji Mosaic have an encryption effect that makes it difficult to infer the text from the mosaic, as long as the input text is not too short, by a random placement of characters replaced with colors.

## Getting Start

### Prerequisites
This project is designed and implemented using Python and OpenCV primarily, it requried following dependancy.
- python 3.7 or later
- numpy 1.18.5
- opencv-python 4.2.0.34
- pillow 7.1.2

### Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/moji-mosaic.git
cd moji-mosaic
```
2. Install required dependencies:
```
pip install -r requirements.txt
```

### Usage
1. Prepare a text file (.txt) with the content you want to convert into a image. Please refer below preparation guidance.
  - The resolution of the mosaic generated through a text file is the minimum number of pixels that can form a square when all characters in the file are replaced with corresponding colors and allocated to each pixel.</br>(i.e. If a mosaic is generated from a file with 10 characters, it will have a resolution of 4x4, and if a mosaic is generated from a file with 20 characters, it will have a resolution of 6x6)
  - If `the number of pixels required to form a square > the number of all characters in the file`, empty pixels that are not assigned a color will be white (255, 255, 255 in RGB).
  - The maximum number of characters that Moji Mosaic can process is 1,048,576. If the number of characters in the input text file is larger than this, only up to the 1,048,576th character from the beginning will be processed.
2. Run the script:
```
python3 mosaic.py
# テキストをモザイクイメージに変えてくれるアプリケーション。「文字モザイク」です。
# 原本となるテキストファイルを「.txt」形式で「mosaic.py」と同じフォルダの中にいれ、テキストファイルを名前をご指定くださいますと「kinoko.png」のモザイクイメージを出力いたします。
# 作者：Yuri
```
3. When prompted, enter the name or path of your text file:
```
原本のテキストファイルの名前、または位置をご入力ください。:<path-of-text-file>
```
4. The script will process your text and generate a  image named `kinoko.png` in the same directory.
