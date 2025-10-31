from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from glob import glob


# https://hituji-ws.com/code/python/face_reco/
# 顔検出のAI
# image_size: 顔を検出して切り取るサイズ
# margin: 顔まわりの余白
mtcnn=MTCNN(image_size=160, margin=10)
 
print('切り取った顔を512個の数字にするAI')
print('1回目の実行では学習済みのモデルをダウンロードしますので、少し時間かかります。')
resnet=InceptionResnetV1(pretrained='vggface2').eval()

# 1つ目を(仮)カメラで取得した方した人として(検索したい不明人物)
image_path1="./folder/Fukuyama1.jpg"
img1=Image.open(image_path1) 
# 顔データを160×160に切り抜き
img_cropped1=mtcnn(img1)
# save_pathを指定すると、切り取った顔画像が確認できます。
# img_cropped1 = mtcnn(img1, save_path="cropped_img1.jpg")
# 切り抜いた顔データを512個の数字に
img_embedding1=resnet(img_cropped1.unsqueeze(0))

# 512個の数字にしたものはpytorchのtensorという型なので、numpyの方に変換
p1=img_embedding1.squeeze().to('cpu').detach().numpy().copy()

# 類似度の関数
def cos_similarity(p1, p2): 
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

# 氏名と顔が既知のファイル群
files1=glob('./folder/face/**/*.jpg', recursive=True)

print('類似度(0から1.0の値をとる。似ているほうが数字が1.0に近づく)')

for file in files1:
    image_path2=file
    img2=Image.open(image_path2)
    img_cropped2=mtcnn(img2)
    img_embedding2=resnet(img_cropped2.unsqueeze(0))
    # 512個の数字にしたものはpytorchのtensorという型なので、numpyの方に変換
    p2=img_embedding2.squeeze().to('cpu').detach().numpy().copy()
    # 類似度を計算して顔認証
    img1vs2=cos_similarity(p1, p2)
    print("類似度", img1vs2, file)