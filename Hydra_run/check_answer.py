from tqdm import tqdm
import argparse
from utils import *
# from compress_G.py import *
import random
from cot_prompt_list import main_path_select_prompt
from subgraph_utilts import *
from collections import defaultdict
import pickle
import json
import os 
import os
import json
import asyncio
import os
import json
import sqlite3
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ast
from tqdm import tqdm
import math
import tiktoken
import argparse
from utils import *
import random
from cot_prompt_list import main_path_select_prompt
from subgraph_utilts import *
from pog2_utilts import *
from resp_process import *
from collections import defaultdict
import pickle
import json
import os
import os
import json
import asyncio
import os
import json
import sqlite3
import multiprocessing
import time
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ast
import time
import psutil
from wiki_func import *
from agent_prompt import *
from multi_source import *
from search import *
from client import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


LLM_model = "llama70b"





def check_in_path(paths, answer_list):
    for i in paths:
        for answer in answer_list:
            if answer in i:
                return True
    return False


import argparse
import os
import torch.multiprocessing as mp

# ------------------------------------------------------------
# 1. Build the argument parser
# ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Return an argument parser for the Hydra main program (argparse version).
    All comments and help messages are in English.
    """
    parser = argparse.ArgumentParser(
        description="Hydra main program (argparse version)."
    )

    # ---------- Required positional argument ----------
    parser.add_argument(
        "file_name",
        help=(
            "Dataset file name prefix, for example webqsp "
            "(CWQ, hotpot(AdvHotpotQA), qald(QALD10-en), simpleqa(SimpleQA), webqsp(WebQSP), webquestions(WebQuestions), zeroshotre(ZeroShot-RE))."
        ),
    )

    # ---------- Feature switches: off by default, enable with flags ----------
    parser.add_argument(
        "--allsource",
        action="store_true",
        help="Enable all sources (default: False).",
    )
    parser.add_argument(
        "--allr",
        action="store_true",
        help="Enable all relations (Hydra). Default is single relation (Hydra-E).",
    )
    parser.add_argument(
        "--incomplete",
        action="store_true",
        help="Include incomplete data splits (default: False).",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        choices=[100, 80, 50, 30],
        default=100,
        help="If --incomplete on, Sampling incomplete KG with ratio in percent (default: 100).",
    )

    # ---------- Data sources: enabled by default, disable with --no-* ----------
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary generation.",
    )
    parser.add_argument(
        "--no-freebase",
        action="store_true",
        help="Disable the Freebase knowledge graph.",
    )
    parser.add_argument(
        "--no-wikikg",
        action="store_true",
        help="Disable the Wiki knowledge graph.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web search.",
    )
    parser.add_argument(
        "--no-wikidocu",
        action="store_true",
        help="Disable retrieval from Wikipedia documents.",
    )

    # ---------- Other options ----------
    parser.add_argument(
        "--model",
        choices=["gpt3", "gpt4", "llama", "deepseek", "llama70b"],
        default="llama70b",
        help="Select the large language model (default: llama70b).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help="Maximum search depth (default: 3).",
    )

    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = build_parser().parse_args()


    if_using_all_r       = args.allr
    using_all_source     = args.allsource
    incomplete     = args.incomplete
    incomplete_ratio = args.ratio/100

    using_summary    = not args.no_summary
    using_freebase   = not args.no_freebase
    using_wikiKG     = not args.no_wikikg
    using_web        = not args.no_web
    using_wkidocument= not args.no_wikidocu
    ori_source       = (using_freebase, using_wikiKG, using_web, using_wkidocument)

    # 其他参数
    file_name     = args.file_name
    Global_depth  = args.depth
    
    


    using_tree_search    = True
    uisng_branch_reduced = False
    error_summary = {}
    datas, question_string,Q_id = prepare_dataset(file_name)

    # for data in tqdm(datas[20+4+13+2+4+10+10+11+22+8+2+29+69+22+9+68+11:1000]):
    changemode(args.model)
    change_depth(Global_depth)
    LLM_model = display_LLM_model()


    print("Global_depth:", Global_depth)
    if using_all_source:
        answer_db = f'../answer/allsource_{LLM_model}_{file_name}_{using_summary}_{if_using_all_r}_{using_freebase}_{using_wikiKG}_{using_web}_{using_wkidocument}_{Global_depth}.db'
        if incomplete:
            answer_db = f'../answer/allsource_{incomplete_ratio}_{LLM_model}_{file_name}_{using_summary}_{if_using_all_r}_{using_freebase}_{using_wikiKG}_{using_web}_{using_wkidocument}_{Global_depth}.db'
    else:
        if incomplete:
            answer_db = f'../answer/new_{incomplete_ratio}_{LLM_model}_{file_name}_{using_summary}_{if_using_all_r}_{using_freebase}_{using_wikiKG}_{using_web}_{using_wkidocument}_{Global_depth}.db'
        else:
            answer_db = f'../answer/{LLM_model}_{file_name}_{using_summary}_{if_using_all_r}_{using_freebase}_{using_wikiKG}_{using_web}_{using_wkidocument}_{Global_depth}.db'
    online_search_db = f'../online_search/{file_name}_online_search.db'
    initialize_large_database(answer_db)
    initialize_large_database(online_search_db)



    obtained_answer = 0
    total_question = 0
    error_reasoning = 0


    TTT_a = 0
    for data in tqdm(datas):
        depth, path, graph_storage, NL_formatted_paths, NL_subgraph = None, None, None, None, None
        question = data[question_string]
        topic_entity = data['topic_entity']
        question_id = data[Q_id] 
        TTT_a += 1
        answer = load_from_large_db(answer_db, question_id)
        question_real_answer = check_answerlist(file_name, question_string, question, datas,data)
        if answer:
            answer_dict = answer
            answer_list = check_answerlist(file_name, question_string, answer['question'], datas,data)
            total_question += 1
            obtain = check_answer(answer_dict, answer_list)
            if obtain:
                obtained_answer += 1

            else:
                final_kg = answer_dict['final_entity_path'] + answer_dict.get('wiki_document_param', []) + answer_dict.get('web_document_param', [])
                if check_in_path(final_kg, answer_list):
                    error_reasoning += 1
    print("TTT_a:", TTT_a)
    print("Global_depth:", Global_depth)
    print("total_question:", total_question)
    print("obtained_answer:", obtained_answer)
    acc = round(obtained_answer/total_question*100,2)
    print("accuracy in % (2 decimal):", round(obtained_answer/total_question*100,2))
