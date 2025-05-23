from tqdm import tqdm
import math
import tiktoken
import argparse
from utilts import *
import random
from cot_prompt_list import *
from subgraph_utilts import *
from utilts2 import *
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
import os
os.environ["PYTORCH_NO_META_TENSOR"] = "1"
os.environ["PYTORCH_META_DEVICE_TENSOR_MODE"] = "0"
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ast
import time
import psutil
import urllib3
import torch.nn as nn
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["PYTORCH_NO_META_DEVICE_TENSOR"] = "1"  

from sentence_transformers import SentenceTransformer
import torch

import psutil
import gc



def Beam_search_step1(query_sentence, NL_CoT_all_paths, total_id_to_name_dict, model, top_k_value = 80):

    if not NL_CoT_all_paths or len(NL_CoT_all_paths) == 0:
        return []
    if not isinstance(NL_CoT_all_paths[0], str): 
        NL_path_tobe_del = format_paths_to_natural_language_id_with_name(NL_CoT_all_paths,total_id_to_name_dict)
    else:
        NL_path_tobe_del = NL_CoT_all_paths
    if len(NL_path_tobe_del) >= top_k_value:
        model = SentenceTransformer('msmarco-distilbert-base-tas-b')


        print("none-LLM model loaded")
        candidate_sentences = NL_path_tobe_del

        candidate_embeddings = model.encode(candidate_sentences, batch_size=64, show_progress_bar=True)
        query_embedding = model.encode([query_sentence])

        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        top_k = top_k_value
        top_k_indices = similarities.argsort()[-top_k:][::-1]

        print(f"finish obtained the top {top_k} sentences.\n") 
        NL_path_tobe_del = [candidate_sentences[i] for i in top_k_indices]
    return NL_path_tobe_del


def Beam_search_step3(NL_CoT_all_paths, question, split_question, data, top_k_value = 3):
    
    print('''
        +++++++++++++++++++++  LLM-aware Path Selection     +++++++++++++++++++++
        ''')
    while len(NL_CoT_all_paths) > top_k_value:
        token_length = num_tokens_from_string("".join(NL_CoT_all_paths), "cl100k_base")
        input_token_length(token_length)
        NL_CoT_all_paths = beam_path_expand_select(question, split_question,data, NL_CoT_all_paths, [])

    return NL_CoT_all_paths

def Beam_search(question, split_answer,query_sentence, data, total_id_to_name_dict, CoT_all_paths, model,final_entity_path = []):

    if not CoT_all_paths:
        return [],[]
    if not isinstance(CoT_all_paths[0], str): 
        NL_CoT_all_paths = format_paths_to_natural_language_id_with_name(CoT_all_paths,total_id_to_name_dict)
        
    else:
        NL_CoT_all_paths = CoT_all_paths
    

    NL_CoT_all_paths = Beam_search_step1(query_sentence, NL_CoT_all_paths, total_id_to_name_dict,model, 80)

    initial_path = NL_CoT_all_paths

    NL_CoT_all_paths = Beam_search_step3(NL_CoT_all_paths,  question, split_answer, data, 3)

    return NL_CoT_all_paths,initial_path

def beam_path_expand_select_help(question, split_answer, data, to_be_deleted_path, 
                                existing_path, prompt_formate=explored_path_select_prompt, 
                                test_size_ori = 40,max_token_length=16000, web=False):
    prunned_path_number = []
    llm_run_time = 0
    i = 0
    LLM_model = display_LLM_model()
    if web:
        prompt_formate = main_path_select_web_prompt
    else:
        prompt_formate = explored_path_select_prompt
    # global LLM_model
    print("LLM model is:", LLM_model)
    if LLM_model == "gpt4":
        max_token_length = 8192    
    while i < len(to_be_deleted_path):
        test_size = test_size_ori
        
        while test_size >= 5:
            end_index = min(i + test_size, len(to_be_deleted_path))
            to_be_deleted_path_prompt = ""
            for idx in range(i, end_index):
                if web:
                    to_be_deleted_path_prompt += f"Web {idx+1}: " + to_be_deleted_path[idx] + "\n"
                else:
                    to_be_deleted_path_prompt += f"Candidate Edge {idx+1}: " + to_be_deleted_path[idx] + "\n"
            prompt_Cot = prompt_formate + "\nQ: " + question + f"\n {split_answer} ?\n"
            if len(existing_path) > 0:
                existing_path_prompt = ""
                for j in range(len(existing_path)):
                    existing_path_prompt += f"existing_knowledge {j+1}: " + existing_path[j] + "\n"
                prompt_Cot += existing_path_prompt
            if web:
                prompt_Cot += f"\n\nThe retrived web is: \n" + to_be_deleted_path_prompt
            else:
                prompt_Cot += f"\n\nThe Candidate Edge list is: \n" + to_be_deleted_path_prompt
            num_token_size = num_tokens_from_string(to_be_deleted_path_prompt, "cl100k_base")
            if num_token_size <= max_token_length:
                
                try:
                    question_answer = run_LLM(prompt_Cot, LLM_model)
                    llm_run_time += 1
                    if web:
                        prunned_path_number += extract_top_web(question_answer)
                    else:
                        prunned_path_number += extract_top_list(question_answer)
                except ValueError as e:
                    print(f"Error: {e}")
                i = end_index 
                break 
            else:
                print(f"Token size exceeds {max_token_length}, size-5")
                test_size = test_size-5
        else:
            i = end_index
    answer_list = []
    for num in prunned_path_number:
        num_int = int(num)
        if num_int > 0 and num_int <= len(to_be_deleted_path):
            answer_list.append(num_int)
    new_returned_answer = [to_be_deleted_path[i-1] for i in answer_list]
    print("select path number:", len(new_returned_answer))
    return new_returned_answer

def beam_path_expand_select(question, split_answer, data, to_be_deleted_path, 
                                existing_path, prompt_formate=explored_path_select_prompt, 
                                test_size_ori = 40,max_token_length=16000, web=False):
    tring_time = 0
    result = []
    increment(1)
    while len(result) == 0 and tring_time < 3:
    
        result = beam_path_expand_select_help(question, split_answer, data, to_be_deleted_path, existing_path,prompt_formate, test_size_ori,max_token_length, web)

        tring_time += 1
    if len(result) == 0:
        input_error("select error, anwer formate error, refuse answering\n")
        result = Beam_search_step1(thinking_cot_line, to_be_deleted_path, total_id_to_name_dict,sb_model, 3)
    return result

def check_n_explor(question, split_question, data, topic_entity_path, CoT_entity_path, prompt_formate):

    prompt_Cot = prompt_formate + "\nQ: " + question +f"\n {split_question[2:]}"
    # + "\n"
    # global LLM_model
    LLM_model = display_LLM_model()

    if len(topic_entity_path) > 0:

        prompt_topic_entity_path = "\nTopic entity path: \n" 
        for i in range(1, len(topic_entity_path)+1):
            prompt_topic_entity_path += f"Path{i}: " + topic_entity_path[i-1] + "\n"
        prompt_Cot += prompt_topic_entity_path
    
    if len(CoT_entity_path) > 0:
        prompt_CoT_entity_path = "\nSupplementary Edges: \n"
        for i in range(1, len(CoT_entity_path)+1):
            prompt_CoT_entity_path += f"Edge{i}: " + CoT_entity_path[i-1] + "\n"
        prompt_Cot += prompt_CoT_entity_path
    try:
        # print(f"Q: {question}\n ")
        # print("answer is ", data["answer"])
        # print(prompt_Cot)

        question_answer = run_LLM(prompt_Cot, LLM_model,0)
        increment()

        return question_answer

    except ValueError as e:
        print(f"Error: {e}")
        return []

def check_n_explor_v4(question, data,  split_answer, related_edge, CoT, prompt_formate):

    prompt_Cot = prompt_formate + "\nQ:\n Question: " + question+ "?\n"
    prompt_Cot += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n"
    LLM_model = display_LLM_model()

    prompt_topic_entity_path = ""
    if len(split_answer) > 0:
        prompt_topic_entity_path += f"\n {split_answer}"
    
    prompt_related_edge_path = ""
    if len(related_edge) > 0:
        prompt_related_edge_path += "\nRelated edge: \n"
        for i in range(1, len(related_edge)+1):
            prompt_related_edge_path += f"Related Path:{i} " + related_edge[i-1] + "\n\n"
    prompt_CoT_entity_path = ""
    if len(CoT) > 0:
        prompt_CoT_entity_path += "\nLLM_generated CoT: \n"
        for i in range(1, len(CoT)+1):
            prompt_CoT_entity_path += f"CoT Path:{i} " + CoT[i-1] + "\n\n"

    prompt_Cot = prompt_Cot + prompt_topic_entity_path +prompt_related_edge_path+ prompt_CoT_entity_path

    
    try:

        print("-------- start summarization ---------")
        question_answer = run_LLM(prompt_Cot, LLM_model)
        increment()

        return question_answer
    except ValueError as e:
        print(f"Error: {e}")
        return []


def extract_entities_from_strings(paths):
    entities_dict = {}


    entity_pattern = r'(m\.\w+): ([^,}]+)'

    for path in paths:
        entities = re.findall(entity_pattern, path)
        
        for entity_id, name in entities:
            entities_dict[entity_id] = name.strip()

    return entities_dict

def extract_path_length_from_text(text):
    back_up = text
    tokens = re.split(r'\s*-\s*', text.strip())
    # 计算路径长度
    path_length = (len(tokens) - 1) // 2

    match = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)
    match2 = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)
    match3 = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)

    # 输出结果
    if match:
        thinking_cot_line = match.group(1).strip()
        # print('提取的文本是：')
        # print(thinking_cot_line)
    else:
        return 0, ""

    return path_length, thinking_cot_line



import re



    
def extract_top_web(text):
    """Extracts the top 3 paths from a given text.

    Args:
    text: The input text containing the top_list dictionary.

    Returns:
    A list of the top 3 paths.
    """

    match = re.search(r'ist:\s*\{([^}]+)\}', text)
    if not match:
        return []

    top_list_str = match.group(1)

    numbers = re.findall(r'\b\d+\b', top_list_str)

    return list(map(int, numbers))




def document_path_generation(split_answer, data, related_web_paragraphs):
    LLM_model = display_LLM_model()
    if len(related_web_paragraphs) == 0:
        return []

    related_web_paragraphs = beam_path_expand_select(split_answer, " ", data, related_web_paragraphs, [], " ",40, 16000, web=True)

    prompt_online_search = From_web_para_to_path_prompt + "\nQ:\n Question: " + split_answer + "\n"
    prompt_online_search += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n \n"
    for index in range(0, len(related_web_paragraphs)):
        prompt_online_search += 'Paragraph '+ str(index+1)+': '+ related_web_paragraphs[index] + "\n"
    prompt_online_search += "\n" +"A:\n"

    prompt_online_search_result = run_LLM(prompt_online_search, LLM_model)
    web_kg_paths = extract_KGpaths(prompt_online_search_result)
    return web_kg_paths


def online_search_paragaph(question, question_id, data, thinking_cot_line, online_search_db, emb_model, timeout=6, k=3):
    online_search_result = load_from_large_db(online_search_db, question_id)

    if not online_search_result:
        online_search = search_google(question)
        online_search_result_ = {}
        online_search_result_[question] = online_search
        delete_data_by_question_id(online_search_db, question_id)
        save_to_large_db(online_search_db, question_id, online_search_result_)
    else:
        if not online_search_result.get(question):
            online_search = search_google(question)
            online_search_result[question] = online_search
            delete_data_by_question_id(online_search_db, question_id)
            save_to_large_db(online_search_db, question_id, online_search_result)
        # online_search = online_search_result[question]
    online_search_mainquestion = []
    online_search = load_from_large_db(online_search_db, question_id)
    # print("online_search:", online_search)
    for question_name, result in online_search.items():
        search_mainquestion = get_title_and_snippet(result)
        online_search_mainquestion.extend(search_mainquestion)
        # print("online_search_mainquestion:", search_mainquestion)
    online_search_mainquestion = list(set(online_search_mainquestion))

    print("number of online_search_mainquestion:", len(online_search_mainquestion))

    # online_search_mainquestion = get_title_and_snippet(online_search.get(question))

    # print("online_search_mainquestion:", online_search_mainquestion)
    # print("number of online_search_mainquestion:", len(online_search_mainquestion))
    # for i in range(3):
    #     if i < len(online_search_mainquestion):
    #         print("online_search_mainquestion:", online_search_mainquestion[i])
    search_text_final = beam_path_expand_select(question, " ", data, online_search_mainquestion, [], " ",40, 16000, web=True)
    # print("number of search_text_final:", len(search_text_final))
    # for i in range(3):
    #     if i < len(search_text_final):
    #         print("search_text_final:", search_text_final[i])
    # search_text_final = online_search_mainquestion
    search_url_final = []
    search_paragrah = []
    # online_search = online_search.get(question)
    for each_web in search_text_final:
        if "Related question" in each_web:
            for question_name, websearch_reult in online_search.items():
                for result in websearch_reult.get("related_questions", []):
                    if result.get("question") in each_web:
                        if "link" in result:
                            search_url_final.append(result["link"])
                        break
        else:
            for question_name, websearch_reult in online_search.items():
                for result in websearch_reult.get("organic_results", []):
                    if result.get("title") in each_web:
                        if "link" in result:
                            search_url_final.append(result["link"])

                        break
    search_url_final = list(set(search_url_final))
    print("number of search_url_final:", len(search_url_final))

    result_dict = process_search_results(thinking_cot_line, search_url_final, text_emb_name, text_emb_model,timeout, k)
    related_web_paragraphs = []
    for url_item, top_paragraphs in result_dict.items():
        for idx, para in enumerate(top_paragraphs, 1):
            related_web_paragraphs.append(para)
            temp_dict = {
                "path_text":para,
                "source":"webDoc",
                "doc_id":url_item
            }
            search_paragrah.append(temp_dict)
    return related_web_paragraphs,search_paragrah




def get_name_to_id(name1, total_id_to_name_dict):
    ids = []
    for id, name in total_id_to_name_dict.items():
        if name == name1:
            return id

def wiki_search(wiki_topic_entity, thinking_cot_line, emb_model,timeout=6, k=3):
    wiki_source_db = []
    paragraph_Related = []

    wikipidia_url = []
    if len(wiki_topic_entity) > 0:
        for entity_id in wiki_topic_entity:
            entity_name = wiki_topic_entity[entity_id]
            entity_name_= entity_name.replace(" ", "_")
            wikipedia_url = 'https://en.wikipedia.org/wiki/{}'.format(entity_name_)
            print('wikipedia_url  ' + wikipedia_url)
            wikipidia_url.append(wikipedia_url)

        result_dict = process_search_results(thinking_cot_line, wikipidia_url,text_emb_name, text_emb_model,timeout, k)
        # 输出结果
        for url_item, top_paragraphs in result_dict.items():
            for idx, para in enumerate(top_paragraphs, 1):
                paragraph_Related.append(para)
                temp_dict = {
                    "path_text":para,
                    "source":"wikiDoc",
                    "doc_id":url_item
                }
                wiki_source_db.append(temp_dict)

    else:
        print("No related sentences found in Wikipedia.")
    return paragraph_Related,wiki_source_db


def get_predict_entity(predict_entity, topic_entity, 
wiki_topic_entity, total_id_to_name_dict, wiki_total_id_to_name_dict, 
predict_CoT_list, final_path_toal, Global_depth, using_freebase, using_wikikg):
    wiki_id_CoT = []
    free_id_CoT = []

    if using_freebase and len(topic_entity) > 0:        
        top_result = get_most_similar_entities_bert(total_id_to_name_dict, predict_entity, top_k=3)
        print("top_result:", top_result)
        if len(top_result) > 0:
            for (predict_id, name, sim,original_name) in top_result:
                if sim < 0.8 or "nnamed" in name :
                    continue
                CoT_indicter = ""
                for i in predict_CoT_list:
                    if original_name in i:
                        CoT_indicter = i
                        break
                sorted_CoT_entity_name = reorder_entities(CoT_indicter, list(topic_entity.values())+[original_name])
                sorted_CoT_entity_id = []
                wiki_sorted_CoT_entity_id = []
                for temp_name in sorted_CoT_entity_name:
                    found = False
                    for te_id, te_entity in topic_entity.items():
                        if te_entity == temp_name:
                            found = True
                            sorted_CoT_entity_id.append(te_id)
                            break
                    if found == False:
                        sorted_CoT_entity_id.append(predict_id)
                free_id_CoT.append((sorted_CoT_entity_id, CoT_indicter))
        
    if using_wikikg and len(wiki_topic_entity) > 0:
        top_result = get_most_similar_entities_bert(wiki_total_id_to_name_dict, predict_entity, top_k=3)
        print("top_result:", top_result)
        
        if len(top_result) > 0:
            for (wiki_predict_id, name, sim,original_name) in top_result:
                if sim < 0.8 or "nnamed" in name :
                    continue
                CoT_indicter = ""
                for i in predict_CoT_list:
                    if original_name in i:
                        CoT_indicter = i
                        break
                sorted_CoT_entity_name = reorder_entities(CoT_indicter, list(topic_entity.values())+[original_name])
                wiki_sorted_CoT_entity_id = []                        
                for temp_name in sorted_CoT_entity_name:
                    found = False
                    for te_id, te_entity in wiki_topic_entity.items():
                        if te_entity == temp_name:
                            found = True
                            wiki_sorted_CoT_entity_id.append(te_id)
                            break
                    if found == False:
                        wiki_sorted_CoT_entity_id.append(wiki_predict_id)

                wiki_id_CoT.append((wiki_sorted_CoT_entity_id, CoT_indicter))
    
    ToTal = []
    for ori_cot in predict_CoT_list:
        free_id = []
        wiki_id = []
        for (id, CoT_line) in free_id_CoT:
            if ori_cot in CoT_line:
                free_id = id
                break
        for (id, CoT_line) in wiki_id_CoT:
            if ori_cot in CoT_line:
                wiki_id = id
                break
        ToTal.append((free_id, wiki_id, ori_cot))

    print("ToTal:", ToTal)
    return ToTal
                        


def iteration_expand(question, split_answer, data, main_path = None, if_used_for_request = False):
    def re_query(question,split_answer, last_step_result, data, main_path, if_used_for_request = False, iteration = 0):
        if if_used_for_request:
            prompt_Cot = re_genquery_prompt + "\n for the given Question: " + question+ "?\n  the text need to be formated is:\n" + last_step_result + "\n"
        else:
            prompt_Cot = re_predict_prompt + "\n for the given Question: " + question+ "?\n  the text need to be formated is:\n" + last_step_result + "\n"

        LLM_model = display_LLM_model()

        try:
            question_answer = run_LLM(prompt_Cot, LLM_model,0)
            print(question_answer)
            if not "No" in question_answer:
                if if_used_for_request:
                    query_new_predict, CoTlist = question_regen_help(question_answer)
                else:
                    query_new_predict, CoTlist = get_predicted_result(question_answer)

                if iteration == 3:
                    return query_new_predict, CoTlist
                if len(CoTlist) == 0 or len(query_new_predict) == 0:
                    return iteration_expand(question, split_answer, data, main_path, if_used_for_request)
                else:
                    return query_new_predict, CoTlist
            else:
                print("LLM reasonning error, requsting again")
                print(question_answer)
                if iteration == 3:
                    return [], []
                return iteration_expand(question, split_answer, data, main_path, if_used_for_request)
        except ValueError as e:
            print(f"Error: {e}")
            return [],[]

    if if_used_for_request:
        prompt_Cot = generated_new_questionRAG + "\nQ:\n Question: " + question + "?"+ "\n"
    else:
        prompt_Cot = Generated_predict_answer + "\nQ:\nQuestion: " + question + "?"+ "\n"
    prompt_Cot += "Main Topic Entities: " + str(data['topic_entity']) + "\n" + f"\n {split_answer}\n"
    LLM_model = display_LLM_model()


    if main_path:
        main_path_prompt = "\nTopic entity clue path: \n"
        for i in range(1, len(main_path)+1):
            main_path_prompt += f"Path {i}: " + main_path[i-1] + "\n"
        prompt_Cot += main_path_prompt
    try:
        question_answer = run_LLM(prompt_Cot, LLM_model)
        print(f"A: {question_answer}")
        if if_used_for_request:
            query_new_predict, CoTlist = question_regen_help(question_answer)
        else:
            query_new_predict, CoTlist = get_predicted_result(question_answer)
        if len(CoTlist) == 0:
            print("LLM reasonning error, requsting again")
            print(question_answer)
            input_error("select error, anwer formate error, refuse answering\n")
            return re_query(question,split_answer, question_answer, data, main_path, if_used_for_request, 1)
        return query_new_predict, CoTlist
    
    except ValueError as e:
        print(f"Error: {e}")
        return [], question_answer




def run_split_question_prompt(question, data):
    LLM_model = display_LLM_model()
    
    prompt_split = split_question_prompt + "\nQ:\n Question: " + question + "\n"
    if len(data['topic_entity']) > 0:
        prompt_split += "Main Topic Entities: \n" + str(data['topic_entity']) 
    else:
        prompt_split += "Main Topic Entities: \n" + str(data['QID']) + "\n"
    split_answer = run_LLM(prompt_split + "\n" +"A:\n", LLM_model)[2:]

    predict_length, thinking_cot_line = extract_path_length_from_text(split_answer)
    
    while predict_length == 0:
        split_answer = run_LLM(prompt_split + "\n" +"A:\n", LLM_model)[2:]
        predict_length, thinking_cot_line = extract_path_length_from_text(split_answer)
    split_question = extract_split_questions(split_answer)
    return split_question, predict_length, thinking_cot_line, split_answer

def path_generation_kg(question_cot, sorted_topic_entity_id, graph, total_id_to_name_dict,question_real_answer,
                     Global_depth, predict_length, if_using_all_r=True, using_tree_search =False):

    final_entity_path = []            

    predict_length = min(predict_length, Global_depth)
    intersection = None

    path_gen_time = time.time()
    if len(sorted_topic_entity_id) >1:
        intersection = set(bfs_with_intersection_only(graph, sorted_topic_entity_id, predict_length))
                    
    if not intersection:
        print("no intersection found, returning paths separately")
        if using_tree_search:

            limit_path = 1000
            if len(graph) > 100000:
                limit_path = 500
                if len(graph) > 1000000:
                    limit_path = 100
            final_entity_path = multi_entity_tree_search(
                graph,
                sorted_topic_entity_id,
                max_hop=predict_length,
                question_cot=question_cot,
                id_to_name_dict=total_id_to_name_dict,
                model=sb_model,
                topk=limit_path,
                if_using_all_r=True,
                bfs_inter_fn=bfs_with_intersection_only     # 传你自己的检测函数
            )

        else:           
            final_entity_path = find_all_paths_bibfs_itersection(graph, sorted_topic_entity_id, predict_length,if_using_all_r, question_cot, total_id_to_name_dict,sb_model, 80)
            final_entity_path= Beam_search_step1(question_cot, final_entity_path, total_id_to_name_dict, sb_model, 80)

    else:
        print("search with depth :", predict_length)
        if using_tree_search:
            limit_path = 1000
            if len(graph) > 100000:
                limit_path = 500
                if len(graph) > 1000000:
                    limit_path = 100
            final_entity_path = multi_entity_tree_search(
                graph,
                sorted_topic_entity_id,
                max_hop=predict_length,
                question_cot=question_cot,
                id_to_name_dict=total_id_to_name_dict,
                model=sb_model,
                topk=limit_path,
                if_using_all_r=True,
                bfs_inter_fn=bfs_with_intersection_only     # 传你自己的检测函数
            )
        
        
        else:
            check_degree = False
            for i in sorted_topic_entity_id:
                if len(graph[i]) > 1000:
                    check_degree = True
                    break
            if check_degree:
                final_entity_path = multi_entity_tree_search(graph, sorted_topic_entity_id, predict_length, question_cot, total_id_to_name_dict, sb_model,5000, 300, if_using_all_r)
            else:
                final_entity_path = find_all_paths_bibfs_itersection(graph, sorted_topic_entity_id, predict_length,if_using_all_r, question_cot, total_id_to_name_dict, sb_model, 80)

    print("all_paths:", len(final_entity_path))
    print("========================  path generation time:", time.time() - path_gen_time)
    return final_entity_path

def main_rag_process_multi(question_web, thinking_cot_line, question_id, data, split_answer, 
    online_search_db, emb_model, timeout=6, k=10,
    sorted_topic_entity_id=None,topic_entity=None,wiki_topic_entity=None,
    using_web=True, using_wkidocument=True, using_freebase=True,using_wikiKG=True, if_using_all_r=True, using_tree_search =False,
    question_real_answer=None, Global_depth=3, predict_length=2, return_topN_ = 10,
    total_id_to_name_dict=None, graph=None, wiki_total_id_to_name_dict=None, wiki_graph=None
    ):
    import threading

    related_web_paragraphs = []
    web_doc_paths_tobe_del = []
    related_wiki_paragraphs = []
    wiki_doc_paths_tobe_del = []
    freebase_result_holder = {}
    wiki_result_holder = {}
    wiki_topic_entity_ids = list(wiki_topic_entity)
    def web_thread_func():
        print("========starting the web search thread")
        web_time = time.time()
        if using_web:
            nonlocal related_web_paragraphs, web_doc_paths_tobe_del
            related_web_paragraphs, web_doc_paths_tobe_del = online_search_paragaph(
                question_web, question_id, data, thinking_cot_line, online_search_db, emb_model, timeout, k)
            
        print("========web search thread finished, with time:", time.time() - web_time)

    def wiki_thread_func():
        print("========starting the wiki search thread")
        wiki_time = time.time()
        if using_wkidocument and len(wiki_topic_entity) > 0:
            nonlocal related_wiki_paragraphs, wiki_doc_paths_tobe_del
            if len(wiki_topic_entity) > 0:
                related_wiki_paragraphs, wiki_doc_paths_tobe_del  = wiki_search(
                    wiki_topic_entity, thinking_cot_line, emb_model, timeout, k)
            else:
                related_wiki_paragraphs, wiki_doc_paths_tobe_del  = wiki_search(
                    topic_entity, thinking_cot_line, emb_model, timeout, k)
            
        print("========wiki search thread finished, with time:", time.time() - wiki_time)

    def freebase_thread_func():
        final_entity_path = []
        freebase_time = time.time()
        if using_freebase and len(topic_entity) > 0:
            final_entity_path = path_generation_kg(thinking_cot_line, sorted_topic_entity_id, graph, 
            total_id_to_name_dict,question_real_answer,
            Global_depth, predict_length, if_using_all_r, using_tree_search)
            print("number of freebase paths:", len(final_entity_path))

        freebase_result_holder["paths"] = final_entity_path
        
        print("========freebase search thread finished, with time:", time.time() - freebase_time)

    def wikiKG_thread_func():

        wiki_all_paths = []
        print("========starting the wikiKG search thread")
        wikiKG_time = time.time()
        if using_wikiKG and len(wiki_topic_entity) > 0:

        
            wiki_all_paths = path_generation_kg (thinking_cot_line,list(wiki_topic_entity), wiki_graph, 
            wiki_total_id_to_name_dict,question_real_answer,
            Global_depth, predict_length, if_using_all_r,using_tree_search)
            
        wiki_result_holder["paths"] = wiki_all_paths


        print("========wikiKG search thread finished, with time:", time.time() - wikiKG_time)

    t_web = threading.Thread(target=web_thread_func)
    t_wiki = threading.Thread(target=wiki_thread_func)
    t_freebase = threading.Thread(target=freebase_thread_func)
    t_wikiKG = threading.Thread(target=wikiKG_thread_func)



    t_web.start()
    t_wiki.start()
    t_freebase.start()
    t_wikiKG.start()

    t_web.join()
    t_wiki.join()
    t_freebase.join()
    t_wikiKG.join()


    total_docuemnt_paragraphs = related_web_paragraphs + related_wiki_paragraphs
    wik_kG_gen_path = document_path_generation(split_answer, data,  related_wiki_paragraphs)
    web_doc_gen_path = document_path_generation(split_answer, data,  related_web_paragraphs)
    for i in wik_kG_gen_path:
        print("wik_kG_gen_path:", i)
        temp_dict = {
            "path_text": i,
            "source": "wikiDoc",
            "doc_id": i
        }
        wiki_doc_paths_tobe_del.append(temp_dict)
    
    for i in web_doc_gen_path:
        print("web_doc_gen_path:", i)
        temp_dict = {
            "path_text": i,
            "source": "webDoc",
            "doc_id": i
        }
        web_doc_paths_tobe_del.append(temp_dict)

    freebase_KG_paths_tobe_del = []
    wiki_KG_paths_tobe_del = []
    Freebase_path = freebase_result_holder.get("paths", [])
    wiki_path = wiki_result_holder.get("paths", [])

    if Freebase_path:
        for p in Freebase_path:
            temp = {"path_text": p, "source": "freebaseKG", "doc_id": "freebaseKG"}
            freebase_KG_paths_tobe_del.append(temp)

    if wiki_path:
        for p in wiki_path:
            temp = {"path_text": p, "source": "wikiKG", "doc_id": "wikiKG"}
            wiki_KG_paths_tobe_del.append(temp)


    top20 = multi_source_beam_search(
        query_sentence=thinking_cot_line,
        freebase_KG_paths=freebase_KG_paths_tobe_del,
        wiki_KG_paths=wiki_KG_paths_tobe_del,
        wiki_doc_paths=wiki_doc_paths_tobe_del,
        web_doc_paths=web_doc_paths_tobe_del,
        kg_reference_texts=Freebase_path + wiki_path,
        return_topN = return_topN_
    )
    return top20

def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    torch.cuda.empty_cache()

    import gc
    gc.collect()

    import multiprocessing as mp
    for p in mp.active_children():
        p.join(timeout=1)


import sys
import sys
import torch.multiprocessing as mp

"""
Hydra main program entry point.
"""

import argparse
import os
import torch.multiprocessing as mp



import argparse

import argparse

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
    
    

    embed = "minilm"
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


    # subgraph_db       = f'../free_subgraph/{file_name}_multi_main_Subgraphs.db'
    # NL_subgraph_db    = f'../free_subgraph/{file_name}_multi_main_nl_Subgraphs.db'
    # path_db           = f'../free_subgraph/{file_name}_multi_path_db.db'
    # wiki_subgraph_db  = f'../wiki_subgraph/{file_name}_main_Subgraphs.db'
    # wiki_NL_subgraph_db = f'../wiki_subgraph/{file_name}_main_nl_Subgraphs.db'


    subgraph_db       = f'/data1/xingyut/PoG2/PoG/free_subgraph/{file_name}_multi_main_Subgraphs.db'
    NL_subgraph_db    = f'/data1/xingyut/PoG2/PoG/free_subgraph/{file_name}_multi_main_nl_Subgraphs.db'
    wiki_subgraph_db  = f'/data1/xingyut/PoG2/PoG/wiki_subgraph/{file_name}_main_Subgraphs.db'
    wiki_NL_subgraph_db = f'/data1/xingyut/PoG2/PoG/wiki_subgraph/{file_name}_main_nl_Subgraphs.db'


  

    using_beam_step1_3 = True


    datas, question_string,Q_id = prepare_dataset(file_name)
    

    text_emb_name = embed
    text_emb_model = embed
    sb_model = "paraphrase-multilingual-MiniLM-L12-v2"

    initialize_large_database(subgraph_db)
    initialize_large_database(NL_subgraph_db)
    initialize_large_database(wiki_subgraph_db)
    initialize_large_database(wiki_NL_subgraph_db)

    # for Global_depth in [Global_depth]:
    print("Global_depth:", Global_depth)
    online_search_db = f'../online_search/{file_name}_online_search.db'
    initialize_large_database(answer_db)
    initialize_large_database(online_search_db)
    for data in tqdm(datas[0:2]):
        (using_freebase, using_wikiKG, using_web, using_wkidocument) =  ori_source
        depth, path, graph_storage, NL_formatted_paths, NL_subgraph = None, None, None, None, None
        wiki_topic_entity = data['QID']
        question = data[question_string]
        if file_name == "zeroshotre":
            question = question[0]
        topic_entity = data['topic_entity']
        question_id = data[Q_id] 
        question_real_answer = check_answerlist(file_name, question_string, question, datas,data)
        inital_num()
        answer = load_from_large_db(answer_db, question_id)
        # if answer:
        #     print("qurstion_id:", question)
        #     print("answer is found in the database")
        #     continue
        LLM_model = display_LLM_model()
        print("\n Question:", question)
        print("answer:", question_real_answer)
        print("topic_entity:", topic_entity)
        split_question, predict_length, thinking_cot_line, split_answer = run_split_question_prompt(question, data)
        if len(topic_entity) > 0:
            topic_entity_length = len(topic_entity)
        elif len(wiki_topic_entity) > 0:
            topic_entity_length = len(wiki_topic_entity)
        else:
            topic_entity_length = 1
        print("split_question:", split_question)
        print("predict CoT length:", predict_length)
        print("thinking_cot_line:", thinking_cot_line)
        sorted_topic_entity_id = list(topic_entity)
        free_predict_length = 1
        wiki_predict_length = 1
        if len(topic_entity) >1:
                free_predict_length = math.ceil(predict_length/len(topic_entity))
        if len(wiki_topic_entity) >1:
                wiki_predict_length = math.ceil(predict_length/len(wiki_topic_entity))
        predict_length = max(wiki_predict_length, free_predict_length)
        sorted_topic_entity_name = reorder_entities(thinking_cot_line, list(topic_entity.values()))
        sorted_topic_entity_id = []
        for name in sorted_topic_entity_name:
            for id, entity in topic_entity.items():
                if entity == name:
                    sorted_topic_entity_id.append(id)
                    break
        print("sorted_topic_entity_id:", sorted_topic_entity_id)
        print("sorted_topic_entity_id_name:", sorted_topic_entity_name)

        wiki_sorted_topic_entity_name = reorder_entities(thinking_cot_line, list(wiki_topic_entity.values()))
        wiki_sorted_topic_entity = {}
        for name in wiki_sorted_topic_entity_name:
            for id, entity in wiki_topic_entity.items():
                if entity == name:
                    temp_dict = {id: entity}
                    wiki_sorted_topic_entity.update(temp_dict)
                    break
        if len(wiki_sorted_topic_entity) == 0 or len(wiki_sorted_topic_entity_name) != len(wiki_topic_entity):
            wiki_sorted_topic_entity = wiki_topic_entity

        wiki_topic_entity = wiki_sorted_topic_entity

        print("sorted_topic_entity_id:", wiki_sorted_topic_entity_name)
        print("sorted_topic_entity:", wiki_sorted_topic_entity)

        print("explore the graph")
        wiki_document_kg_paths = []
        web_kg_paths = []
        total_document_kg = []
        total_id_to_name_dict = {}
        
        if using_all_source:
            using_freebase = True
            using_wikiKG = True
            using_web = True
            using_wkidocument = True
        else:
            using_freebase, using_wikiKG, using_web, using_wkidocument = select_source_agent(str(question + split_answer),
            topic_entity, wiki_topic_entity,LLM_model,using_freebase, using_wikiKG, using_web, using_wkidocument)
        Num_run_LLM += 1

        graph, total_id_to_name_dict = load_and_check_subgraph(question, question_id,subgraph_db, 
        Global_depth, NL_subgraph_db, question_string, data, topic_entity, incomplete, incomplete_ratio)
        
        wiki_graph, wiki_total_id_to_name_dict = wiki_load_and_check_subgraph(question, question_id,wiki_subgraph_db, 
        Global_depth, wiki_NL_subgraph_db, question_string, data, wiki_topic_entity, None, incomplete, incomplete_ratio)
        


        

        question_start = time.time()

        top20  = main_rag_process_multi(question, thinking_cot_line, question_id, data, split_answer, 
        online_search_db,  sb_model, 6, 10,
        sorted_topic_entity_id, topic_entity, wiki_topic_entity,
        using_web, using_wkidocument, using_freebase,using_wikiKG, if_using_all_r, using_tree_search,
        question_real_answer, Global_depth, predict_length, 40,
        total_id_to_name_dict, graph, wiki_total_id_to_name_dict, wiki_graph)


        print("========main_rag_process finished")
        
        final_KG_paths = [item["path_text"] for item in top20 if item.get("source") in ["freebaseKG", "wikiKG"]]
        web_kg_param = [item["path_text"] for item in top20 if item.get("source") == "webDoc"]
        wiki_document_kg_param = [item["path_text"] for item in top20 if item.get("source") == "wikiDoc"]
        # text to paths
        total_document_kg_paths = document_path_generation(split_answer, data,  wiki_document_kg_param + web_kg_param)
        candidate_paths = []
        candidate_paths = final_KG_paths + total_document_kg_paths
        # combine all the paths
        fianl_selected_path,_ = Beam_search(
        question, split_answer, thinking_cot_line,data, total_id_to_name_dict,
        candidate_paths, sb_model,[])

        Summary_main_entity_path = []
        print('''
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        +++++++++++++++++++++        Check the answer         ++++++++++++++++
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ''')

        if using_summary:
            result = check_n_explor_v4(question, data, split_answer, fianl_selected_path,[],Summary_COT_w_splitQ_prompt)

            Summary_main_entity_path = extract_cots_as_strings(result)
            candidate_paths += Summary_main_entity_path
        final_path_toal =Summary_main_entity_path + fianl_selected_path

        final_result = check_n_explor(question, split_answer, data, final_path_toal, [], answer_n_explore_prompt)
        
        
        
        # print("final_result:", final_result)
        if "{Yes}" in final_result:
        
        # if False:
            print("Yes in the answer")
            print("========the real answer is:", question_real_answer)
            result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct)
            print("Result:", result)
            LLM_call = display_LLM_calls()
            error_message = display_error_status()
            total_reasonning_token_input = display_input_token_length()
            process = psutil.Process(os.getpid())
            memory_in_gb = process.memory_info().rss / 1024 / 1024 / 1024
            answer_dict = {
                "LLM_answer": result,
                "real_answer": question_real_answer,
                "question": question,
                "split_answer": split_answer,
                "final_path_toal": candidate_paths,
                "final_entity_path": final_KG_paths,
                "LLM_call": LLM_call,
                "web_document_param": web_kg_param,
                "wiki_document_param": wiki_document_kg_param,
                "freebase_kg": 1 if using_freebase else 0,
                "wiki_kg":1 if using_wikiKG else 0,
                "wiki_docuemnt":1 if using_wkidocument else 0,
                "web_docuemnt": 1 if using_web else 0,
                "error_message": error_message,
                "memory": memory_in_gb,
                "run_time": time.time() - question_start,
                "number_of_iteration": 1,
                "finished_depth": math.ceil(predict_length/topic_entity_length),
                "total_reasonning_token_input": total_reasonning_token_input
            }
            print(("run_time:", time.time() - question_start))
            delete_data_by_question_id(answer_db, question_id)
            save_to_large_db(answer_db, question_id, answer_dict)
            continue

        current_iteration = 2
        new_explored_length = predict_length


        (using_freebase, using_wikiKG, using_web, using_wkidocument) = ori_source 

        while current_iteration <= 3:
            print(f"""
                    =========================
                    ==========  Iteration {current_iteration} ==========
                    =========================

                    """)
            # print("current_iteration:%", )
            new_top20 = []
            if current_iteration == 2:
                if using_web:
                    new_question, new_question_cot = iteration_expand(question, split_question, data, final_path_toal, True)
                    if len(new_question) == 0:
                        new_question = question
                        new_question_cot = thinking_cot_line
                    print("new_question:", new_question)
                    print("new_question_cot:", new_question_cot)
                else:
                    new_question = question
                    new_question_cot = thinking_cot_line
                    print("new_question:", new_question)
                    print("new_question_cot:", new_question_cot)
                new_top20  = main_rag_process_multi(new_question, new_question_cot, question_id, data, split_answer, 
                online_search_db,  sb_model, 6, 10,
                sorted_topic_entity_id, topic_entity, wiki_topic_entity,
                using_web, using_wkidocument, using_freebase,using_wikiKG, if_using_all_r, using_tree_search,
                question_real_answer, Global_depth, Global_depth, 40,
                total_id_to_name_dict, graph, wiki_total_id_to_name_dict, wiki_graph)
            else:
                predict_entity, predict_CoT_list = iteration_expand(question, split_answer, data, final_path_toal, False)
                print("predict_entity:", predict_entity)
                print("predict_CoT_list:", predict_CoT_list)
                if len(predict_entity) == 0:
                    print("No new entity found, stop searching")
                    break
                ToTal = get_predict_entity(predict_entity, topic_entity, 
                wiki_topic_entity, total_id_to_name_dict, wiki_total_id_to_name_dict, 
                predict_CoT_list, final_path_toal, Global_depth, using_freebase, using_wikiKG)
                if len(ToTal) == 0:
                    print("No new entity found, stop searching")
                    break
                for (free_id, wiki_id, CoT) in ToTal:
                    print("CoT:", CoT)
                    print("free_id:", free_id)
                    print("wiki_id:", wiki_id)
                    temp_wiki_dict = {}
                    for w_id in wiki_id:
                        temp_wiki_dict[w_id] = wiki_total_id_to_name_dict[w_id]
                    if CoT == "":
                        CoT = thinking_cot_line
                    if min(len(wiki_id), len(free_id)) > 2:
                        new_search_depth = 2
                    else:
                        new_search_depth = 3
                    new_top20  += main_rag_process_multi(question, thinking_cot_line, question_id, data, split_answer, 
                    online_search_db,  sb_model, 6, 10,
                    sorted_topic_entity_id, free_id, temp_wiki_dict,
                    using_web, using_wkidocument, using_freebase,using_wikiKG, if_using_all_r, using_tree_search,
                    question_real_answer, Global_depth, new_search_depth, 40,
                    total_id_to_name_dict, graph, wiki_total_id_to_name_dict, wiki_graph)
                    


            new_final_KG_paths = [item["path_text"] for item in new_top20 if item.get("source") in ["freebaseKG", "wikiKG"]]
            new_web_kg_param = [item["path_text"] for item in new_top20 if item.get("source") == "webDoc"]
            new_wiki_document_kg_param = [item["path_text"] for item in new_top20 if item.get("source") == "wikiDoc"]
            
            new_total_document_kg = document_path_generation(split_answer, data, new_wiki_document_kg_param + new_web_kg_param)
            candidate_paths += new_final_KG_paths + new_total_document_kg
            new_fianl_selected_path, _ = Beam_search (
            question, split_answer, thinking_cot_line,data,total_id_to_name_dict,
            candidate_paths,sb_model ,[])


            new_Summary_main_entity_path = []
            if using_summary:
                result = check_n_explor_v4(question, data, split_answer, new_fianl_selected_path,[],Summary_COT_w_splitQ_prompt)

                new_Summary_main_entity_path = extract_cots_as_strings(result)
                candidate_paths += new_Summary_main_entity_path
            final_path_toal =new_Summary_main_entity_path+ new_fianl_selected_path 
            
            for i in final_path_toal:
                print("final_path_toal:", i)
            
            final_result = check_n_explor(question, split_answer, data, new_Summary_main_entity_path+ new_fianl_selected_path, [], answer_n_explore_prompt)
            final_KG_paths = final_KG_paths + new_final_KG_paths
            web_kg_param = web_kg_param + new_web_kg_param
            wiki_document_kg_param = wiki_document_kg_param + new_wiki_document_kg_param
            fianl_selected_path = list(set(fianl_selected_path + new_fianl_selected_path))
            Summary_main_entity_path = list(set(Summary_main_entity_path + new_Summary_main_entity_path))
            current_iteration += 1

            print('answer:', final_result)
            if "{Yes}" in final_result:
            # if False:
                print("Yes in the answer")
                print("========the real answer is:", question_real_answer)

                result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct)
                print("Result:", result)

                print("the real answer is:", question_real_answer)

                LLM_call = display_LLM_calls()
                error_message = display_error_status()
                total_reasonning_token_input = display_input_token_length()
                process = psutil.Process(os.getpid())
                memory_in_gb = process.memory_info().rss / 1024 / 1024 / 1024

                answer_dict = {
                    "LLM_answer": result,
                    "real_answer": question_real_answer,
                    "question": question,
                    "split_answer": split_answer,
                    "final_path_toal": final_path_toal,
                    "final_entity_path": final_KG_paths,
                    "LLM_call": LLM_call,
                    "web_document_param": web_kg_param,
                    "wiki_document_param": wiki_document_kg_param,
                    "freebase_kg": 1 if using_freebase else 0,
                    "wiki_kg":1 if using_wikiKG else 0,
                    "wiki_docuemnt":1 if using_wkidocument else 0,
                    "web_docuemnt": 1 if using_web else 0,
                    "error_message": error_message,
                    "finished_depth": math.ceil(new_explored_length/topic_entity_length),
                    "number_of_iteration": current_iteration,
                    
                    "memory": memory_in_gb,
                    "run_time": time.time() - question_start,
                    "total_reasonning_token_input": total_reasonning_token_input
                }
                print(("run_time:", time.time() - question_start))

                delete_data_by_question_id(answer_db, question_id)
                save_to_large_db(answer_db, question_id, answer_dict)
                break

        if "{Yes}" in final_result:
            continue

        print("========the real answer is:", question_real_answer)

        result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct)
        print("Result:", result)

        print("the real answer is:", question_real_answer)
        current_iteration += 1
        LLM_call = display_LLM_calls()
        error_message = display_error_status()
        total_reasonning_token_input = display_input_token_length()
        process = psutil.Process(os.getpid())
        memory_in_gb = process.memory_info().rss / 1024 / 1024 / 1024
        answer_dict = {
            "LLM_answer": result,
            "real_answer": question_real_answer,
            "question": question,
            "split_answer": split_answer,
            "final_path_toal": candidate_paths,
            "final_entity_path": final_KG_paths,
            "LLM_call": LLM_call,
            "web_document_param": web_kg_param,
            "wiki_document_param": wiki_document_kg_param,
            "freebase_kg": 1 if using_freebase else 0,
            "wiki_kg":1 if using_wikiKG else 0,
            "wiki_docuemnt":1 if using_wkidocument else 0,
            "web_docuemnt": 1 if using_web else 0,
            "error_message": error_message,
            "memory": memory_in_gb,
            "finished_depth": math.ceil(new_explored_length/topic_entity_length),

            "number_of_iteration": 5,

            "run_time": time.time() - question_start,
            "total_reasonning_token_input": total_reasonning_token_input
        }
        print(("run_time:", time.time() - question_start))
        print("answer_db:", answer_db)
        delete_data_by_question_id(answer_db, question_id)
        save_to_large_db(answer_db, question_id, answer_dict)

            