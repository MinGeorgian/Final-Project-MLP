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

# Task
Our implementation of this paper and the tasks we have undertaken are as follows:

1. Observe the original model results with new data.
   - We evaluated new data based on the model trained using 1000 sampled Data.
   - The new data was divided in half from the existing ‘test.json’ data, one was used as a reference, and the other was used as our new data.
2. Compare performance by changing Hyper parameters in the existing methodology.
   - We compared the performance after changing the lr_scheduler_type, one of the training Hyper parameters, from Linear to cosine model.
  
```
parser.add_argument('--lr_scheduler_type',
                    type=SchedulerType,
                    # default='linear',
                    default='cosine',
                    help='The scheduler type to use.',
                    choices=[
                        'linear', 'cosine', 'cosine_with_restarts',
                        'polynomial', 'constant', 'constant_with_warmup'
                    ])
```
<br>     
3. After applying our own new idea in the existing methodology, we compared the performance.
   - The original model used the BERT model as a text classifier.
   - We compared the performance after changing to the ALBERT model, which can derive faster results.
   - Comment out the existing BERT model as follows and modify the required libraries, config, and tokenizer for applying the ALBERT model.

**`@finetune_ALBERT.py`**

```
# ALBERT Model
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer

...

parser.add_argument(
    '--model_name_or_path',
    type=str,
    help=
    'Path to pretrained model or model identifier from huggingface.co/models.',
    # default='bert-base-cased')
    default='albert-base-v2')
    
...

    # config = BertConfig.from_pretrained(args.model_name_or_path,
    #                                     num_labels=num_labels,
    #                                     finetuning_task='text classification')

    # tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
    #                                           use_fast=False,
    #                                           never_split=['[user]', '[bot]'])

    # ALBERT Model
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_labels)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', use_fast=False, never_split=['[user]', '[bot]'])

    # tokenizer.vocab['[user]'] = tokenizer.vocab.pop('[unused1]')
    # tokenizer.vocab['[bot]'] = tokenizer.vocab.pop('[unused2]')

    # ALBERT Model
    additional_tokens = ['<[user]>', '<[bot]>']
    tokenizer.add_tokens(additional_tokens)

    MODEL_CLASS = MODEL_CLASS_MAPPING[args.model_name_or_path]

    # model = MODEL_CLASS(config, args.model_name_or_path)
```
**`@eval_ALBERT.py`**
```
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification

...

# config = BertConfig.from_pretrained(args.model_name_or_path,
#                                     num_labels=2,
#                                     finetuning_task='text classification')
# tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
#                                           use_fast=False,
#                                           never_split=['[user]', '[bot]'])

# ALBERT Model
config = AlbertConfig.from_pretrained('albert-base-v2', num_labels=2)
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', use_fast=False, never_split=['[user]', '[bot]'])

# tokenizer.vocab['[user]'] = tokenizer.vocab.pop('[unused1]')
# tokenizer.vocab['[bot]'] = tokenizer.vocab.pop('[unused2]')

# ALBERT Model
additional_tokens = ['<[user]>', '<[bot]>']
tokenizer.add_tokens(additional_tokens)

# MODEL_CLASS = MODEL_CLASS_MAPPING[args.model_name_or_path]

# model = MODEL_CLASS(config, args.model_name_or_path)
# PATH = f'out/pytorch_model.bin'
# model.load_state_dict(torch.load(PATH))

# ALBERT Model
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', config=config, ignore_mismatched_sizes=True)
# model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
PATH = 'out/albert_model.bin' 
model.load_state_dict(torch.load(PATH), strict=False)

```

# Instructions for Code Execution and Notes

## Basic Code Execution Procedure
1. Download all attached files.
   - For necessary libraries and versions, refer to the 'requirements.txt' file.
2. Training
   - In the terminal, type and execute `python finetune.py`.
   - The input files are 'train.json' and 'valid.json' located in the 'data' folder under the project directory.
   - The results will be saved as 'pytorch_model.bin' in the 'out' folder under the project directory.
3. Model Evaluation
   - In the terminal, type and execute `python eval.py`.
   - The input files are 'pytorch_model.bin' from the previously trained 'out' folder and 'test.json' from the 'data' folder.
   - The output will display the performance results of 'test.json' on the terminal.

## Task-Specific Execution Methods
1. Task 1:
   - The input files are as follows:
   - Ref: 'test_Ori.json', new: 'test_new.json'
   - Modify and save the names of the input files appropriately in the 'finetune.py' function, then execute it.
   - E.g., with open('./data/test_Ori.json', 'r', encoding='utf-8') as f:
   - E.g., with open('./data/test_New.json', 'r', encoding='utf-8') as f:
2. Task 2:
   - Change the method of the hyperparameter '--lr_scheduler_type' from 'linear' to 'cosine' and save it before execution.

3. Task 3:
   - To apply the LABERT model, execute 'finetune_ALBERT.py' to train the input data.
   - After training, evaluate the performance of the modified model by running 'eval_ALBERT.py'."
     
※ our trained model : https://drive.google.com/drive/folders/1vahWzy9F4Zx1lb1NRqW2zmdtANEjAFx1?usp=sharing

※ This code utilizes a GPU accelerator, 'CUDA'.
We used CUDA version 12.1 and installed a compatible version of torch. 
<br> It's recommended to upgrade pip before installing the torch library. `python -m pip install --upgrade pip`
<br> For torch installation, refer to the following website or use the command:
<br> Link : https://pytorch.org/get-started/locally/
<br> Command : `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
