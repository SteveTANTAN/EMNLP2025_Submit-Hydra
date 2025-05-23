from tqdm import tqdm
import math
import tiktoken
import argparse
import random
from subgraph_utilts import *
from subgraph_helper import *

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
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ast
import time
import psutil
from wiki_client import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'





import sys
import sys
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    file_name = sys.argv[1]

    # db_path = f'{file_name}_subgraph.db'
    subgraph_db = f'../freebase_subgraph/{file_name}_main_Subgraphs.db'
    NL_subgraph_db = f'../freebase_subgraph/{file_name}_main_nl_Subgraphs.db'

    wiki_subgraph_db = f'../wiki_subgraph/{file_name}_main_Subgraphs.db'
    wiki_NL_subgraph_db = f'../wiki_subgraph/{file_name}_main_nl_Subgraphs.db'
    

    datas, question_string,Q_id = prepare_dataset(file_name)

    with open("server_urls.txt", "r") as f:
        server_addrs = f.readlines()
        server_addrs = [addr.strip() for addr in server_addrs]
    wiki_client = MultiServerWikidataQueryClient(server_addrs)
    wiki_client.test_connections()

    initialize_large_database(subgraph_db)
    initialize_large_database(NL_subgraph_db)
    initialize_large_database(wiki_subgraph_db)
    initialize_large_database(wiki_NL_subgraph_db)


    for data in tqdm(datas[0:50]):

        depth, path, graph_storage, NL_formatted_paths, NL_subgraph = None, None, None, None, None
        wiki_topic_entity = data['QID']
        question = data[question_string]
        topic_entity = data['topic_entity']
        question_id = data[Q_id] 
        # obtained question subgraph from freebase
        graph, total_id_to_name_dict = load_and_check_subgraph(question, question_id,subgraph_db, 
        2, NL_subgraph_db, question_string, data, topic_entity)

        # obtained question subgraph from Wikipedia KG
        wiki_graph, wiki_total_id_to_name_dict = wiki_load_and_check_subgraph(question, question_id,wiki_subgraph_db, 
        2, wiki_NL_subgraph_db, question_string, data, wiki_topic_entity, wiki_client)
