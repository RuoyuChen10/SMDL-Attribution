# -- coding: utf-8 --**
import os
import json
from tqdm import tqdm
import numpy as np


explanation_method = "explanation_insertion_results/cub-fair-efficientnet/KernelShap"
eval_list = "datasets/CUB/eval_fair-efficientnet.txt"
# steps = 49
# percentage = 0.25
# number = int(percentage * steps)
# 

def main(percentage):
    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    highest_acc = []
    region_area = []

    for info in tqdm(infos[:]):
        # if "CUB" in eval_list:
        #     json_file_path = os.path.join(explanation_method, info.split(" ")[0].split("/")[-1].replace(".jpg", ".json").replace(".JPEG", ".json").replace(".jpeg", ".json"))
        # else:
        json_file_path = os.path.join(explanation_method, info.split(" ")[0].replace(".jpg", ".json").replace(".JPEG", ".json").replace(".jpeg", ".json"))

        
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            f_data = json.load(f)
        
        steps = len(f_data["recognition_score"])
        number = int(percentage * steps)
        
        data = f_data["recognition_score"][:number]

        highest_conf = max(data)
        highest_acc.append(highest_conf)

        area = (data.index(highest_conf) + 1) / steps
        region_area.append(area)

    mean_highest_acc = np.array(highest_acc).mean()
    std_highest_acc = np.array(highest_acc).std()

    mean_region_area = np.array(region_area).mean()
    std_region_area = np.array(region_area).std()
    print("When percentage is {}, the avg. highest confidence is {}, std:{}, the retention percentage at highest confidence is {}, std:{}".format(
        percentage, mean_highest_acc, std_highest_acc, mean_region_area, std_region_area
    ))
    return

main(0.25)
main(0.5)
main(0.75)
main(1.0)