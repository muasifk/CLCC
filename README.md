

# Curriculum Learning for Crowd Counting - Is it worthy?

✍️ PyTorch implementation (official code) of the paper "**M. A. Khan, H. Menouar, R. Hamila, "Curriculum Learning for Crowd Counting - Is it worthy?", 2023"**".


![alt text](https://github.com/muasifk/CLCC/blob/main/clcc.jpg?raw=true)

## Getting Started

The list of files and their description is explained as follows:

### 📂 Directories

- checkpoints: contains checkpoints crated during training
- figures: contained figures e.g., predictions, pacing functions etc.
- networks: contains PyTorch implementation of crowd models.


### 💻 Files (used in standard training)

- cc_cl.ipynb:  The main notebook to train and test models. You can run all experiments using this notebook while changing parameters such as dataset, crowd model, training iterations etc.
- cc_load_data.py:  python function to load dataset.
- cc_trainer: trainer function using standard training.
- train.py: training loop implementation.
- validate.py:  validation loop implementation
- CrowdDataset.py:  Python class to create dataset.
- cc_utils.py: Utilities functions e.g., display image, prediction, etc.

### 💻 Files (used in Curriculum Learning)

- cc_cl_trainer:  trainer function using Curriculum Learning.
- get_pacing_function:  function to chose pacing function.
- sort_data:  scoring function to sort data by difficulty.

## 🛢 Datasets

You can download datasets from the following links.
**ShanghaiTech:**  https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0


## ⭐️ Authors

- Muhammad Asif Khan  📧 feel free to write <asifk@ieee.org>

## License

This project is licensed under the GNU general public License.
