import random
import argparse
import os
import openai
import time
import re
import json
import multiprocessing
import numpy as np
import torch
from statistics import mean

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, args=None, data=None):
        super().__init__()
        self.questions, self.answers = data_reader(data=data)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def create_demo_text(args, cot_flag):
    x, z, y = [], [], []
    
    # example sentences ...    
    if True:
        
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")
    
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text

def setup_data_loader(data=None):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(0)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, 4)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(data=data)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=20,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                  worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)
    return dataloader

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args