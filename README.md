# No-Reference Video Quality Assessment 


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



Each CSV file should contain 2 columns:
- `Video_name`: File name of the Video
- `mos`: Mean Opinion Score for the corresponding image

### Execution
Run the main script with appropriate parameters:
```bash
python main.py

