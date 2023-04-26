import cv2
import os

from tqdm import tqdm

IDs = ["n000307", "n000309", "n000337", "n000353", "n000359", "n003021", "n003197", "n005546", "n006579", "n006634"]

test_set_path = "/home/cry/J-20/Datasets/VGGFace2/train_split/test.txt"
image_dir = "/home/cry/J-20/Datasets/VGGFace2/train_align_arcface"


with open(test_set_path,"r") as file:
    datas = file.readlines()

for data in tqdm(datas):
    id_pid = data.split("/")[0]

    if id_pid in IDs:
        image_path = os.path.join(
            image_dir, data.split(" ")[0]
        )

        image = cv2.imread(image_path)

        cv2.imwrite(data.split(" ")[0], image)