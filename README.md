# CC3M-Helper, how to prepare CC3M dataset for [gill](https://github.com/kohjingyu/gill)

## Download raw data in caption-url pair
Please download `Training split` and `Validation split` via [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download)

> Note: there is another tool [CC3M auto download](https://huggingface.co/spaces/flax-community/dalle-mini/commit/75b01a0a3a29bb2eb6962f5f2fdf160e5c784647 ), but seems not works cos the data pair structure is different.

## Download raw train and val data
ref: [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)

### Install img2dataset
``` shell
pip install img2dataset
```

### Add head to .tsv files
``` shell
apt install sed

sed -i '1s/^/caption\turl\n/' Train_GCC-training.tsv
sed -i '1s/^/caption\turl\n/' Validation_GCC-1.1.0-Validation.tsv
```

### Download image data
ref : [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT/blob/main/data/T-X_pair_data/cc3m/prepare.md)
``` shell
# Make a dir
mkdir cc3m

# Download training image
img2dataset --url_list Train_GCC-training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m/training --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True

# Download validation image
img2dataset --url_list Validation_GCC-1.1.0-Validation.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m/validation --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
```
**Note that: **
- `url_list` A file with the list of url of images to download. It can be a folder of such files. (required)
- `image_size` The size to resize image to (default 256)
- `output_folder` The path to the output folder. (default "images")
- `processes_count` The number of processes used for downloading the pictures. This is important to be high for performance. (default 1)
- `thread_count` The number of threads used for downloading the pictures. This is important to be high for performance. (default 256)
- `output_format` decides how to save pictures (default files)
  - `files saves` as a set of subfolder containing pictures
  - `webdataset` saves as tars containing pictures
  - ...
- `url_col` the name of the url column for parquet and csv (default url)
- `caption_col` the name of the caption column for parquet and csv (default None)
- `enable_wandb` whether to enable wandb logging (default False)

After these two commands, you will have 331 .tar files and 2 .tar, benchmark you could find here [cc3m download benchmark](https://wandb.ai/rom1504/img2dataset/reports/Download-cc3m-with-img2dataset--VmlldzoxMjE5MTE4)

### decompress all data
Define a shell script
``` shell
vi untar.sh
```
type in,
``` shell
for file in *.tar; do
	tar -xzvf "$file" 
done
```
give executive permission,
``` shell
chmod 777 untar.sh
```
decompress training and validation image dataset,
``` shell
cp untar.sh cc3m/training
cd cc3m/training
./untar.sh

cp untar.sh cc3m/validation
cd ../cc3m/validation
./untar.sh 
```

## Generate new .tsv files
### Create a Python script
```shell
vi gen_train_val_tsv.py
```
, type in :
```python
import json
import os
from tqdm import tqdm

def process_json_files(directory, output_file):
    # 检查输出文件是否已存在，如果存在则删除
    # Check if the output file exists, and delete it if it does
    if os.path.exists(output_file):
        os.remove(output_file)

    # 创建并写入列标题
    # Create and write the column headers
    with open(output_file, 'a') as out_file:
        out_file.write('caption\timage\n')

    # 遍历指定目录下的所有文件
    # Iterate over all files in the specified directory
    for filename in tqdm(os.listdir(directory),desc="Parsing dataset..."):
        if filename.endswith('.json'):
            # 构建完整的文件路径
            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # 打开并读取 JSON 文件
            # Open and read the JSON file
            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    continue

                # 检查 status 字段是否为 'success'
                # Check if the status field is 'success'
                if data.get('status') == 'success':
                    # 提取 caption 和 key 字段的值
                    # Extract the values of the caption and key fields
                    caption = data.get('caption', '')
                    key = data.get('key', '') + '.jpg'

                    # 将提取的数据写入到输出文件
                    # Write the extracted data to the output file
                    with open(output_file, 'a') as out_file:
                        out_file.write(f"{caption}\t{key}\n")


# 调用函数，指定目录和输出文件的路径
# Call the function, specifying the directory and output file paths
process_json_files('path/to/your/train/json/files', 'cc3m_train.tsv')
process_json_files('path/to/your/val/json/files', 'cc3m_val.tsv')
```
, run the Python file:
``` shell
python gen_train_val_tsv.py
```

After this, please move these two files (`cc3m_train.tsv` and `cc3m_val.tsv`) to your gill home path in gill/datasets to replace existing examples.

Then go back to [gill - Precomputing Text Embeddings](https://github.com/kohjingyu/gill/tree/main#precomputing-text-embeddings)

