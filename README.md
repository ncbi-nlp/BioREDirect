# BioREDirect

This project provides the implementation of our work "Enhancing Biomedical Relation Extraction with Directionality"

## Updates

Updated BioREDirect to the Python 3.11 version

## Environment

- GPU: Our experiments were conducted on an Nvidia A100 GPU. BioREDirect was also tested on Nvidia V100 and RTX 3080 GPUs. However, for fine-tuning on these GPUs, a smaller batch size, such as 8 for the V100 and 4 for the RTX 3080, may be required.
- Python: Python 3.11
- OS: Linux/Windows WSL2 with Conda

## Installation

Open a terminal or Anaconda Prompt and create a new environment. Then install Torch-gpu (check the latest from https://pytorch.org/get-started/locally/):

```bash
conda create -n bioredirect python=3.11
conda activate bioredirect
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

(Optional) Use the following command to check whether PyTorch can access the GPU after installation.

```bash
python src/run_check_torch_gpu.py
```

## Train and evaluate BioREDirect

### Step 1: Download the datasets

You can download [our converted datasets](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREDirect/datasets.zip), and unzip it to 

```
datasets/
```

(Optional) If you want to convert the datasets by yourself, you can use the below script to convert original datasets into our input format.

```bash
bash scripts/build_biored_dataset.sh
```

You can change the above script to build_cdr_dataset.sh for the BC5CDR task experiment.

### Step 2: Download the pre-trained model

Please download the model [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/biorex_biolinkbert_pt.zip)

Unzip it into 

```
biorex_biolinkbert_pt/
```

### Step 3: Running the BioRED task experiment

To train and evaluate the model, execute script/run_biored_bc8_exp.sh

```bash
bash scripts/run_biored_bc8_exp.sh
```

You can also change the above script to run_cdr_exp.sh for the BC5CDR task experiment.

## Predicting New Data:

If you only wish to use our tool for predicting new data without the need for training, please follow the steps outlined below:

Download the BioREDirect pre-trained model [BioREDirect BioRED model](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREDirect/bioredirect_biored_pt.zip) file and place it in the "BioREDirect/" directory.
Open the "scripts/run_test_pred.sh" file and modify the values of the variables "in_pubtator_file" and "out_pubtator_file" to match your input PubTator file (with NER/ID annotations) and the desired output PubTator file (where predicted relations will be stored).

Execute the following script to initiate the prediction process:

```
bash scripts/run_test_pred.sh
```

## Citing BioREDirect

* Po-Ting Lai, Chih-Hsuan Wei, Shubo Tian, Robert Leaman, Zhiyong Lu, Enhancing biomedical relation extraction with directionality, Bioinformatics, Volume 41, Issue Supplement_1, July 2025, Pages i68–i76, https://doi.org/10.1093/bioinformatics/btaf226
```
@article{10.1093/bioinformatics/btaf226,
    author = {Lai, Po-Ting and Wei, Chih-Hsuan and Tian, Shubo and Leaman, Robert and Lu, Zhiyong},
    title = {Enhancing biomedical relation extraction with directionality},
    journal = {Bioinformatics},
    volume = {41},
    number = {Supplement_1},
    pages = {i68-i76},
    year = {2025},
    month = {07},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf226},
    url = {https://doi.org/10.1093/bioinformatics/btaf226},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/Supplement\_1/i68/63745428/btaf226.pdf},
}
```

## Acknowledgments

This research was supported by the NIH Intramural Research Program, National Library of Medicine.

## Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
