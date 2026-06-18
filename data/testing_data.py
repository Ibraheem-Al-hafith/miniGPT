from dataset import download_and_load_file
from dataset import data_split

data = download_and_load_file("instruction_data.json", "https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json")

train_data, test_data, val_data = data_split("instruction_data.json")

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))