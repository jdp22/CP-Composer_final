#!/bin/bash

CONFIG_PATH="/home/jiangdapeng/PepGLAD/configs/pepbench/test_prompt_codesign.yaml"
CKPT_PATH="./ckpts/LDM_codesign/version_325/checkpoint/epoch37_step85234.ckpt"
GPU=2
BASE_SAVE_DIR="/data/private/jdp/PepGLAD/results"

for guidance_strength in {3..6}; do
    echo "Running with guidance_strength=${guidance_strength}"

    # 修改配置文件中的 guidance_strength
    sed -i "s/guidance_strength: [0-9]\+/guidance_strength: ${guidance_strength}/" "$CONFIG_PATH"

    # 运行 Python 代码
    SAVE_DIR="${BASE_SAVE_DIR}/condition2_w${guidance_strength}_40samples"
    python generate.py --config "$CONFIG_PATH" --ckpt "$CKPT_PATH" --gpu $GPU --save_dir "$SAVE_DIR"

    echo "Finished run with guidance_strength=${guidance_strength}, results saved in ${SAVE_DIR}"
done

echo "All experiments completed!"
