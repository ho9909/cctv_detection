import os
import glob

# file_path = 'C:\\Users\\as\\PycharmProjects\\pythonProject5\\Vehicle-Detection\\Dataset\\custom_dataset\\ttt*.jpg*'
file_path = glob.glob(".\\Dataset\\custom_dataset\\test\\*.txt")

i = 1
for name in file_path:
    if not os.path.isdir(name):
        dst = str(i) + '.txt'
        print(name, i)
        os.rename(name, dst)
    i += 1

    # src = os.path.join(file_path, name)
    # dst = str(i) + '.jpg'
    # dst = os.path.join(file_path, dst)
    # os.rename(src, dst)
    # i += 1