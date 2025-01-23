import os
from tqdm import tqdm

str_set = set()
original_line_count = 0
data_folder = "../../../../data/raw_data/zinc20/canonicalized_data"

for file in os.listdir(data_folder):
    with open(os.path.join(data_folder, file), "r") as f:
        for line in tqdm(f, desc=f"Processing {file}"):
            str_set.add(line.strip())
            original_line_count += 1

    print(f"Original line count: {original_line_count}, New line count: {len(str_set)}")
