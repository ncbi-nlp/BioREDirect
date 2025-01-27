# BioREDirect

This project provides the implementation of our work "Enhancing Biomedical Relation Extraction with Directionality"

## Getting Started

These instructions will guide you through setting up and running the project on your local machine for development and experimentation.

### Prerequisites

- OS: Linux/Windows WSL2 with Anaconda environment
- GPU: Our experiments were conducted on an Nvidia A100 GPU. Additionally, the model was tested on Nvidia V100 and RTX 3080 GPUs. However, for fine-tuning on these GPUs, a smaller batch size, such as 8 for the V100 and 4 for the RTX 3080, may be required.

### Creating a Conda Environment

Open a terminal or Anaconda Prompt and create a new environment:

```bash
conda create -n bioredirect python=3.10
conda activate bioredirect
```

### Installing Dependencies

Install Torch-gpu (check the latest from https://pytorch.org/get-started/locally/):

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

After installed, please check your GPU availability:

```bash
python src/run_check_torch_gpu.py
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

### Download the datasets

You can download [our converted datasets](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREDirect/datasets.zip), and unzip it to 

```
datasets/
```

(Optional) If you want to convert the datasets by yourself, you can use the below script to convert original datasets into our input format.

```bash
bash scripts/build_biored_dataset.sh
```

You can change the above script to build_cdr_dataset.sh for the BC5CDR task experiment.

### Download the pre-trained model

Please download the model [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/biorex_biolinkbert_pt.zip)

Unzip it into 

```
biorex_biolinkbert_pt/
```

### Running the BioRED task experiment

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

* Lai P. T., Wei C. H., Tian S., Robert L. and Lu Z. Enhancing Biomedical Relation Extraction with Directionality. 2025.
```
@misc{lai2025enhancingbiomedicalrelationextraction,
      title={Enhancing Biomedical Relation Extraction with Directionality}, 
      author={Po-Ting Lai and Chih-Hsuan Wei and Shubo Tian and Robert Leaman and Zhiyong Lu},
      year={2025},
      eprint={2501.14079},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.14079}, 
}
```

## Acknowledgments

This research was supported by the NIH Intramural Research Program, National Library of Medicine.

## Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
