# SeMANTIC
This is an offical implementation of "Sample Efficiency Matters: Training Multimodal Conversational Recommendation Systems in a Small Data Setting"
![image](https://github.com/user-attachments/assets/bd873479-273a-44c9-bb69-20540dbfb760)

# Dataset
MMD-v3 dataset can be downloaded from google drive [link](https://drive.google.com/file/d/1DcrDGtI2OjPOeAoCYdkxr6j-XtbcpvVu/view?usp=drive_link)

# Configs
Dataset config can be setup in config/dataset_config.py

Training config can be setup in config/train_config.py

# Run
Run experiments on MMD-v3:
```bash
# Preprocess MMD-v3 dataset
python preprocess_mmd-v3.py

# Start training
python train_mmd.py <model_file_name> <device>
```

Run experiments on MMD-v2:
```bash
# Preprocess MMD-v2 dataset
python preprocess_mmd-v2.py

# Start training
python train_mmd.py <model_file_name> <device>
```

Run experiments on SIMMC:
```bash
# Preprocess SIMMC dataset
python preprocess_simmc.py

# Start training
python train_simmc.py <model_file_name> <device>
```
