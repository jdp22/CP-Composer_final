import pandas as pd
import os

def split(index_file,full_prompt_file,target_file):
    # 读取两个txt文件
    file1 = index_file  # 第一个文档路径
    file2 = full_prompt_file  # 第二个文档路径

    # 读取文件并分隔空格，每行的第一列为蛋白质名称
    df1 = pd.read_csv(file1, sep = '\t', header=None)
    df2 = pd.read_csv(file2, sep = '\t', header=None)

    # 设置第一列为蛋白质名称索引，便于后续操作
    df1.set_index(0, inplace=True)
    df2.set_index(0, inplace=True)


    # 找出两个文档共有的蛋白质
    common_proteins = df1.index.intersection(df2.index)

    # 挑选出共有蛋白质并合并后面的内容
    merged_df = pd.concat([df1.loc[common_proteins], df2.loc[common_proteins]], axis=1)

    if os.path.exists(target_file):
        os.remove(target_file)
    merged_df.reset_index().to_csv(target_file, sep="\t", index=False, header=False)

if __name__ == '__main__':
    split(index_file = "/data/private/jdp/PepGLAD/datasets/train_valid/processed/train_index.txt",
          full_prompt_file="/data/private/jdp/PepGLAD/datasets/train_valid/processed/prompts_distance.txt",
          target_file="/data/private/jdp/PepGLAD/datasets/train_valid/processed/prompt_train_distance_index.txt")