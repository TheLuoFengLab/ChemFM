import os


data_path = "../../../../data/raw_data/unichem/smiles"

# list all txt files in the data_path
count = 0
files = os.listdir(data_path)
for file in files:
    if file.endswith(".txt"):
        print(file)
        count += len(open(os.path.join(data_path, file)).readlines())

print(f"Total number of lines: {count}")