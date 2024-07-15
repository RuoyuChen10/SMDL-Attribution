#!/bin/bash

dataset="datasets/imagenet/ILSVRC2012_img_val"
eval_list="datasets/imagenet/val_clip_vitl_5k_true.txt"
lambda1=0
lambda2=0.05
lambda3=1
lambda4=1

declare -a cuda_devices=("0" "1")

# GPU numbers
gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."

# text length
line_count=$(wc -l < "$eval_list")
echo "Evaluation on $line_count instances."

line_count_per_gpu=$(( (line_count + gpu_numbers - 1) / gpu_numbers ))
echo "Each GPU should process at least $line_count_per_gpu lines."

gpu_index=0
for device in "${cuda_devices[@]}"
do
    begin=$((gpu_index * line_count_per_gpu))
    if [ $gpu_index -eq $((gpu_numbers - 1)) ]; then
        end=-1  # 最后一个 GPU，设置 end 为 -1
    else
        end=$((begin + line_count_per_gpu))
    fi

    CUDA_VISIBLE_DEVICES=$device python -m submodular_attribution.efficientv1-smdl_explanation_imagenet_clip_superpixel \
    --Datasets $dataset \
    --eval-list $eval_list \
    --lambda1 $lambda1 \
    --lambda2 $lambda2 \
    --lambda3 $lambda3 \
    --lambda4 $lambda4 \
    --begin $begin \
    --end $end &

    gpu_index=$((gpu_index + 1))
done

wait
echo "All processes have completed."