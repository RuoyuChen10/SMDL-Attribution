# -- coding: utf-8 --**
import os
import json
from tqdm import tqdm
import numpy as np

explanation_method = "./submodular_results_iclr_baseline/imagenet-languagebind-false/grad-10x10-4"
# explanation_method = "explanation_insertion_results/imagenet-fair-clip-vitl/Rise"
eval_list = "datasets/imagenet/val_languagebind_2k_false.txt"

# percentage = 1.


def main(percentage):
    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    highest_acc = []
    region_area = []

    for info in tqdm(infos[:]):
        npy_file_path = os.path.join(
            os.path.join(explanation_method+"/npy", info.split(" ")[1])
            , info.split(" ")[0].replace(".jpg", ".npy").replace(".JPEG", ".npy").replace(".jpeg", ".npy"))
        json_file_path = os.path.join(
            os.path.join(explanation_method+"/json", info.split(" ")[1])
            , info.split(" ")[0].replace(".jpg", ".json").replace(".JPEG", ".json").replace(".jpeg", ".json"))
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                f_data = json.load(f)
        except:
            # print("{} not found!".format(json_file_path))
            continue
        
        insertion_area = []
        submodular_image_set = np.load(npy_file_path)
        insertion_ours_image = submodular_image_set[0] - submodular_image_set[0] # baseline
        for smdl_sub_mask in submodular_image_set:
            insertion_ours_image += smdl_sub_mask
            insertion_area.append(
                (insertion_ours_image.sum(-1)!=0).sum() / (insertion_ours_image.shape[0] * insertion_ours_image.shape[1]))
    
        number = (np.array(insertion_area) <= percentage).sum()
        
        data = f_data["consistency_score"][:number]
        
        if len(data) == 0:
            continue

        highest_conf = max(data)
        highest_acc.append(highest_conf)

        # area = (data.index(highest_conf) + 1) / steps
        region_area.append(insertion_area[data.index(highest_conf)])

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