import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import logging
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--length_list_dir', default='../../../data/raw_data/zinc20/shuffled_data/zinc_length_list', help='The directory to the raw data')
parser.add_argument('--filter_length', default=512, type=int, help='The length to filter the mols by.')

args = parser.parse_args()


zinc_length_data_dir = args.length_list_dir
filter_length = args.filter_length

logger = logging.getLogger('dataset_info_logger')
logger.setLevel(level=logging.INFO)

# Create a file handler
handler = logging.FileHandler(os.path.join(zinc_length_data_dir, 'dataset_info.log'))

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


zinc_length_data = []
zinc_length_data_short = []
for file in os.listdir(zinc_length_data_dir):
    if file.endswith(".npy"):
        sub_zinc_length_data = np.load(os.path.join(zinc_length_data_dir, file))
        zinc_length_data.extend(sub_zinc_length_data)
        zinc_length_data_short.extend([length for length in sub_zinc_length_data if length < filter_length])
        logger.warning(f"Loaded {file} with {len(sub_zinc_length_data)} mols.")

logger.info("ZINC20 info...")
logger.info(f"    Max length: {np.max(zinc_length_data)}")
logger.info(f"    Min length: {np.min(zinc_length_data)}")
logger.info(f"    Num of mols: {len(zinc_length_data)}")
logger.info(f"    Total_tokens: {np.sum(zinc_length_data)}")
logger.info(f"    Num of mols shorter than {filter_length} is {len(zinc_length_data_short)}, which is {len(zinc_length_data_short)/len(zinc_length_data)*100:.2f}% of the total mols.")
logger.info(f"    Total tokens in mols shorter than {filter_length} is {np.sum(zinc_length_data_short)}.")

logger.info("Plotting...")
fig, ax = plt.subplots()
sns.histplot(zinc_length_data, color='r', bins=100, label="ZINC20", ax=ax)
plt.legend()

plt.savefig(os.path.join(zinc_length_data_dir, "length_distribution.png"))
