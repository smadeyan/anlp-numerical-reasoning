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

from utils import create_demo_text, MyDataset, setup_data_loader

def decoder_for_gpt3(model, input, max_length, i, k, client):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    # time.sleep(1)
    time.sleep(1)
    
    # https://beta.openai.com/account/api-keys
    openai.api_key = os.getenv("OPENAI_API_KEY")
    #print(openai.api_key)
    
    # Specify engine ...
    # Instruct GPT3
    if model == "gpt3":
        engine = "text-ada-001"
    elif model == "gpt3-medium":
        engine = "text-babbage-001"
    elif model == "gpt3-large":
        engine = "text-curie-001"
    elif model == "gpt3-xl":
        engine = "text-davinci-002"
    elif model == '':
        engine = 'gpt-3.5-turbo'
    else:
        raise ValueError("model is not properly defined ...")
        
    # response = openai.Completion.create(
    #   engine=engine,
    #   prompt=input,
    #   max_tokens=max_length,
    #   temperature=0,
    #   stop=None
    # )
    
    response = client.completions.create(
        model=engine,
        prompt=input,
        max_tokens=max_length,
        temperature=0,
        stop=None
    )
    
    return response["choices"][0]["text"]

class Decoder():
    def __init__(self, args):
        self.client = openai.OpenAI(api_key = os.getenv("PROXY_API_KEY"), base_url = os.getenv("PROXY_BASE_URL"))
        
    def decode(self, args, input, max_length, i, k):
        response = decoder_for_gpt3(args.model, input, max_length, i, k, self.client)
        return response

def answer_cleansing(args, pred):

    print("pred_before : " + pred)
    
    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]
        
    if args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    print("pred_after : " + pred)
    
    return pred