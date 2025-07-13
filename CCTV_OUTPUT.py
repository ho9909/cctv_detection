import pandas as pd
from geopy.distance import geodesic
import cv2
import os
from image_similarity_measures.quality_metrics import rmse, ssim, sre, fsim, sam, uiq

# CSV 파일에서 데이터를 읽어옵니다.
data = pd.read_csv('cctv_data.csv', encoding='cp949') #cp949

# 사용자로부터 입력값인 위도와 경도를 받습니다.
user_latitude = float(input("위도를 입력하세요: "))
user_longitude = float(input("경도를 입력하세요: "))

# 사용자로부터 입력받은 위치와 모든 CCTV의 거리를 계산합니다.
distances = []
for index, row in data.iterrows():
    cctv_latitude = row['위도']
    cctv_longitude = row['경도']
    distance = geodesic((user_latitude, user_longitude), (cctv_latitude, cctv_longitude)).km
    distances.append(distance)

# 반경 1km 내에 있는 CCTV의 RTSP 주소를 출력합니다.
total = []
for index, distance in enumerate(distances):
    if distance <= 1:
        cctv_address = data.loc[index, 'RTSP 주소']
        total.append(cctv_address)
        print(cctv_address)


import subprocess

def run_anaconda_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('utf-8', errors='replace')
    error = error.decode('utf-8', errors='replace')
    return output, error

# 사용 예시
for i in range(len(total)):
    command = 'conda activate AS && cd C:\\Users\\jo\\Desktop\\pythonProject5\\Vehicle-Detection && python detect.py ' \
              '--weights runs/train/exp16/weights/best.pt --source ' +str(total[i]) + ' --save-crop'
    output, error = run_anaconda_command(command)
    if output:
        print(f"Output:\n{output}")
    if error:
        print(f"Error:\n{error}")



# input_path = './img/qwer.png'
# output_path = 'result.png'
#
# input = Image.open(input_path)
# output = remove(input)
#
# output.save(output_path)

img = 'a.jpg'

test_img = cv2.imread('img/' + img)

ssim_measures = {}
sre_measures = {}
fsim_measures = {}


scale_percent = 100  # percent of original img size
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)
test_img = cv2.resize(test_img, (61, 68), interpolation=cv2.INTER_LINEAR)
data_dir = 'dataset/crops/medium'

for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    resized_img = cv2.resize(data_img, (61, 68), interpolation=cv2.INTER_LINEAR)
    ssim_measures[img_path] = ssim(test_img, resized_img)
    sre_measures[img_path] = sre(test_img, resized_img)
    fsim_measures[img_path] = fsim(test_img, resized_img)

def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
        closest = max(dict.values())
    else:
        closest = min(dict.values())

    for key, value in dict.items():
        print("The difference between ", key, " and the original image is : \n", value)
        if (value == closest):
            result[key] = closest

    print("The closest value: ", closest)
    print("######################################################################")
    return result


ssim = calc_closest_val(ssim_measures, True)
sre = calc_closest_val(sre_measures, True)
fsim = calc_closest_val(fsim_measures, True)

print("The most similar according to SSIM: ", ssim)
print("The most similar according to SRE: ", sre)
print("The most similar according to fsim: ", fsim)




