# RWKVQA: No-Reference Image Quality Assessment Based on RWKV Architecture

This project implements a vision model based on the RWKV architecture (RWKV6.0), which significantly improves inference speed while maintaining assessment accuracy. The approach leverages the efficient recurrent neural network structure of RWKV for image quality evaluation tasks.

## Environment Dependencies

The following dependencies are required:
- Python 3.8+
- CUDA 11.3
- PyTorch 1.12.1+cu113
- pandas 2.2.2
- numpy 1.26.4
- scipy 1.13.0
- mmcls 0.25.0
- torchsummary 1.5.1
- tqdm 4.66.4

## How to Use

### Code Structure
The main functionality is implemented in:
- `main.py`: Contains complete training, validation, and testing code
- `dataset/IQA_dataloader.py`: Handles data loading and preprocessing

### Dataset Preparation
Organize your dataset in the following structure:  
dataset    
├── csv  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train.csv  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── val.csv  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── test.csv   
└── Images       
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── image1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── image2.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── ...


Each CSV file should contain 2 columns:
- `Image_name`: File name of the image (e.g., "image1.jpg")
- `mos`: Mean Opinion Score for the corresponding image

### Execution
Run the main script with appropriate parameters:
```bash
python main.py

