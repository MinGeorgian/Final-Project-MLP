<div align="center">
<h1>
  AI Research Paper Implementation and Review
  <br> (MLP Final Project)
</h1>
</div>

# About this project
This project is the MLP Final Project for the Fall Semester of the 2023 AIDI Program at Georgian College.
The paper we have selected is **'FACILITATING NSFW TEXT DETECTION IN OPEN-DOMAIN DIALOGUE SYSTEMS VIA KNOWLEDGE DISTILLATION'**. 
<br> Detailed information, including the paper and the code, can be found at the following link.

<p align="center">
📄 <a href="https://arxiv.org/pdf/2309.09749.pdf" target="_blank">Paper</a> • 
🤗 <a href="https://github.com/qiuhuachuan/CensorChat" target="_blank">Model</a> 
</p>

# Limitations of Implementation
The original training data for this model consists of 71,997 entries. Due to the extensive time required to train this model with our available hardware, we focused on training and evaluating the model using a sampled dataset of 1,000 entries, maintaining the original data type and label ratio.

# Contents
Our implementation of this paper and the tasks we have undertaken are as follows:

1. Observe the original model results with new data.
   - We evaluated new data based on the model trained using 1000 sampled Data.
   - The new data was divided in half from the existing ‘test.json’ data, one was used as a reference, and the other was used as our new data.
2. Compare performance by changing Hyper parameters in the existing methodology.
   - We compared the performance after changing the lr_scheduler_type, one of the training Hyper parameters, from Linear to cosine model.
3. After applying our own new idea in the existing methodology, we compared the performance.
   - The original model used the BERT model as a text classifier.
   - We compared the performance after changing to the ALBERT model, which can derive faster results.

# Instructions for Code Execution and Notes
1. Download all attached files.
   - For necessary libraries and versions, refer to the 'requirements.txt' file.
2. Training
   - In the terminal, type and execute `python finetune.py`.
   - The input files are located in the project's subfolder named 'data'.
   - The output, which is the trained model, will be saved in the subfolder 'out'.
3. Model Evaluation
   - In the terminal, type and execute `python eval.py`.
   - The input files are 'pytorch_model.bin' from the previously trained 'out' folder and 'test.json' from the 'data' folder.
   - The output will display the performance results of 'test.json' on the terminal.
     
※ This code utilizes a GPU accelerator, 'CUDA'.
We used CUDA version 12.1 and installed a compatible version of torch. 
<br> It's recommended to upgrade pip before installing the torch library. `python -m pip install --upgrade pip`
<br> For torch installation, refer to the following website or use the command:
<br> Link : https://pytorch.org/get-started/locally/
<br> Command : `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
