# -- coding: utf-8 --**
import os
import json
from tqdm import tqdm
import numpy as np

explanation_method = "submodular_results/cub-fair-resnet/superpixel-seeds-1.0-1.0-10.0-1.0/json"
# explanation_method = "explanation_insertion_results/cub-fair-vgg19/ScoreCAM"
eval_list = "datasets/CUB/eval_fair-resnet.txt"
steps = 49
percentage = 0.25
number = int(percentage * steps)
# 

def main():
    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    highest_acc = []
    region_area = []

    for info in tqdm(infos[:]):
        json_file_path = os.path.join(explanation_method, info.split(" ")[0].replace(".jpg", ".json"))

        with open(json_file_path, 'r', encoding='utf-8') as f:
            f_data = json.load(f)
        
        data = f_data["consistency_score"][:number]

        highest_conf = max(data)
        highest_acc.append(highest_conf)

        area = (data.index(highest_conf) + 1) / steps
        region_area.append(area)

    mean_highest_acc = np.array(highest_acc).mean()
    std_highest_acc = np.array(highest_acc).std()

    mean_region_area = np.array(region_area).mean()
    std_region_area = np.array(region_area).std()
    print("The avg. highest confidence is {}, std:{}, the retention percentage at highest confidence is {}, std:{}".format(
        mean_highest_acc, std_highest_acc, mean_region_area, std_region_area
    ))
    return

main()