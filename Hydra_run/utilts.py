from __future__ import annotations
from typing import List, Tuple, Dict
import heapq
from collections import defaultdict
from freebase_func import *
from cot_prompt_list import *
from wiki_client import *
from openai import OpenAI
import json
import re
import time
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import re
import time
import tiktoken 
from subgraph_helper import *

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def run_LLM(prompt, model,temperature=0.4):
    result = ''
    if "google" in model:
        genai.configure(api_key="your_api_key")

        # model = genai.GenerativeModel('gemini-1.5-flash')
        model = genai.GenerativeModel("gemini-1.5-flash")
        system_message = "You are an AI assistant that helps people find information."

        chat = model.start_chat(
            history=[
                {"role": "user", "parts": system_message},
            ]
        )

        try_time = 0
        while try_time<3:
            try:
                response = chat.send_message(prompt)
                print("Google response: ")
                return (response.text)
                break
            except Exception as e:
                error_message = str(e)
                print(f"Google error: {error_message}")
                print("Retrying in 2 seconds...")
                try_time += 1
                time.sleep(40)
                    

    # openai_api_base = "http://localhost:8000/v1"
    elif "gpt" in model:
        openai_api_key = "your_api_key"
        if model == "gpt4":
            model = "gpt-4-turbo"
        else:
            model = "gpt-3.5-turbo-0125"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            # base_url=openai_api_base,
        )

    elif "llama70b" in model:
        openai_api_key = "secret_pog2_dcce439883564bf490aa837029cae09f.rK8W9U49J6Kau1QRzjz8Fu8gskLgSMyr"
        openai_api_base = "https://api.lambda.ai/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        model = "llama3.1-70b-instruct"

        # openai_api_base = "http://localhost:6666/v1"
        # openai_api_key = "EMPTY"
        # client = OpenAI(
        #     # defaults to os.environ.get("OPENAI_API_KEY")
        #     api_key=openai_api_key,
        #     base_url=openai_api_base,
        # )
        # model = "Meta-Llama-3.1-70B-Instruct"
    elif "llama" in model:
        openai_api_base = "http://localhost:6666/v1"
        openai_api_key = "EMPTY"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        model = "Meta-Llama-3.1-8B-Instruct"
    elif "deep" in model:
        
        openai_api_key = "your_api_key"
        openai_api_base = "https://api.deepseek.com"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        model = "deepseek-chat"
    elif "qwen" in model:
        print("using local qwen")
        openai_api_base = "your_api_key"
        openai_api_key = "EMPTY"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        model = "qwen2.5-7b-instruct-1m"

    else:
        print("using local openai")
        openai_api_base = "http://localhost:6666/v1"
        openai_api_key = "your_api_key"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        model = "Meta-Llama-3-8B-Instruct"
    print("model:", model)
    system_message = "You are an AI assistant that helps people find information."
    messages = [{"role": "system", "content": system_message}]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    try_time = 0
    while try_time<3:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=256,
                frequency_penalty= 0,
                presence_penalty=0
            )
            result = response.choices[0].message.content
            break
        except Exception as e:
            error_message = str(e)
            print(f"OpenAI error: {error_message}")
            print("Retrying in 2 seconds...")
            try_time += 1
            time.sleep(2)

    print(f"{model} response: ")

        # print("end openai")

    return result

def select_source_agent(analysised_question, topic_entity, wiki_topic_entity, LLM_model, using_freebase, using_wikiKG, using_web, using_wkidocument):
    data_avaliavle = ""
    if using_freebase or using_wikiKG:
        if len(topic_entity) > 0 or len(wiki_topic_entity) > 0:
            data_avaliavle += "KG on (" + str([i for i in topic_entity.values()]) + "), "
    if using_wkidocument:
        if len(wiki_topic_entity) > 0:
            data_avaliavle += "wiki on (" + str([i for i in wiki_topic_entity.values()]) + "), "
    if using_web:
        data_avaliavle += "Web"
    prompt_source_select = source_select.format(analysised_question, data_avaliavle)
    text = run_LLM(prompt_source_select, LLM_model)

    decision_marker = "cision"
    start_index = text.find(decision_marker)
    source_select_answer = ""
    if start_index != -1:
        result =  text[start_index + len(decision_marker):].strip()
        if len(result) > 0:
            # print(f"Decision found: {result}")
            source_select_answer = result
        else:
            source_select_answer = text
    else:
        source_select_answer = text
    
        # print("prompt_source_select:", prompt_source_select)
    print(source_select_answer.lower())

    if "kg" not in source_select_answer.lower():
        if "action1" not in source_select_answer.lower():
            using_freebase = False
            using_wikiKG = False
            print("not using KG")
    if "web" not in source_select_answer.lower():
        if "action3" not in source_select_answer.lower():
            using_web = False
            print("not using web")
    if "wiki" and "action2" not in source_select_answer.lower():
        if  "action2" not in source_select_answer.lower():
            using_wkidocument = False
            print("not using wiki")

    if not (using_freebase or using_wikiKG or using_web or using_wkidocument):
        print("not using any KG or web or wiki; turn it on")
        using_freebase = True
        using_wikiKG = True
        using_web = True
        using_wkidocument = True

    return using_freebase, using_wikiKG, using_web, using_wkidocument



def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = 'ID'

    elif dataset_name == 'cwq_multi':
        with open('../data/cwq_multi.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'ID'
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        ID = "QuestionId"
        # answer = ""
    elif dataset_name == 'webqsp_multi':
        with open('../data/webqsp_multi.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = 'qid'
    elif dataset_name == 'fever':
        with open('../data/fever_1000_entities_azure.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'claim'
        ID = 'id'
    elif dataset_name == 'hotpot':
        with open('../data/hotpotadv.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = "qas_id"
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
        ID = 'question'   
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
        ID = 'question'

    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'question'
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        # question_string = 'input' 
        question_string = "template_questions"
        ID = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string, ID
import time
import concurrent.futures
import time

import torch

device_ = "cuda" if torch.cuda.is_available() else "cpu"
# _SBERT_PATH = "emb_model/MiniLM-L12-v2"
_SBERT_PATH = "paraphrase-multilingual-MiniLM-L12-v2"

_ST_MODEL = SentenceTransformer(_SBERT_PATH, device=device_)
_ST_MODEL.eval()
if True:
    _ST_MODEL.half()
hf = _ST_MODEL[0].auto_model
if hasattr(hf.config, "sdpa_kernel"):
    hf.config.sdpa_kernel = "flash"
else:
    hf.config.attn_implementation = "flash_attention_2"
# _ST_MODEL.to(device_)


def get_st_score_model(
    name=_SBERT_PATH,
    device=device_,
    use_fp16=True,
    model=_ST_MODEL
):
    return model

###############################################################################
# 1. Sub-scores (f1, f2, f3). NO f4 in this version.
###############################################################################

def infer_source_reliability(source_name: str) -> float:
    """
    f1: A simple mapping from the source label to a reliability in [0,1].
    Adjust as you see fit.
    """
    lower_name = source_name.lower()
    if "freebase" in lower_name:
        return 1
    elif "wikikg" in lower_name:
        return 1
    elif "wikidoc" in lower_name:
        return 0.8
    elif "webdoc" in lower_name:
        return 0.7
    return 0.4


def compute_alignment_to_kg(path_text: str, kg_refs: list, source_label: str, threshold=0.8):

    if source_label.lower() in ("freebasekg", "wikikg"):
        return 1.0
    if not kg_refs:
        return 1.0

    model = get_st_score_model()
    with torch.no_grad():
        cur = model.encode([path_text], device=model.device, show_progress_bar=False)[0]
        refs = model.encode(kg_refs, device=model.device, show_progress_bar=False)
    sims = cosine_similarity([cur], refs)[0]
    return float(max(0.0, min(1.0, max(sims))))

def cross_source_verification(record, all_records, threshold=0.8):

    texts, ids = [], []
    text0 = record.get("path_text", "")
    doc0  = record.get("doc_id", None)
    for r in all_records:
        if r is record:
            continue
        texts.append(r.get("path_text",""))
        ids.append(r.get("doc_id", None))
    if not texts:
        return 0.5

    model = get_st_score_model()
    with torch.no_grad():
        v0 = model.encode([text0], device=model.device, show_progress_bar=False)[0]
        vs = model.encode(texts, device=model.device, show_progress_bar=False)
    sims = cosine_similarity([v0], vs)[0]

    cnt = sum(1 for i,s in enumerate(sims) if s>= threshold and ids[i]!=doc0)
    return {0:0.6,1:0.7,2:0.8}.get(cnt, 0.9 if cnt>=3 else 0.6)
###############################################################################
# 2. Combined Confidence:  f = (f1 + f2 + f3) / 3    (no f4 here)
###############################################################################

def compute_confidence(record, all_records, kg_reference_texts):
    text = record.get("path_text","")
    source_name = record.get("source","webDoc")
    f1 = infer_source_reliability(source_name)
    f2 = cross_source_verification(record, all_records)
    f3 = compute_alignment_to_kg(text, kg_reference_texts, source_name)

    record["f1_source_reliability"] = f1
    record["f2_cross_source_verification"] = f2
    record["f3_alignment_to_kg"] = f3

    confidence_val = (f1 + f2 + f3) / 3.0
    record["confidence"] = max(0.0, min(1.0, confidence_val))
    return record["confidence"]


def compute_relevance(path_text: str, query: str, batch_size=64):

    model = get_st_score_model()
    with torch.no_grad():
        qv = model.encode(
            [query],
            batch_size=batch_size,
            max_length=256,
            truncation=True,
            show_progress_bar=False,
            device=model.device
        )
        tv = model.encode(
            [path_text],
            batch_size=batch_size,
            max_length=256,
            truncation=True,
            show_progress_bar=False,
            device=model.device
        )
    torch.cuda.empty_cache()
    sim = cosine_similarity(qv, tv)[0][0]
    return max(0.0, min(1.0, sim))

def beam_search_step1(query_sentence, path_list, top_k_value=80):
    """
    Step1: Relevance-only pruning.
    """
    if not path_list:
        return []

    # model = SentenceTransformer('msmarco-distilbert-base-tas-b')
    # model = get_st_socre_model()
    model = None
    for record in path_list:
        text = record.get("path_text","")
        record["relevance"] = compute_relevance(text, query_sentence)

    path_list.sort(key=lambda x: x["relevance"], reverse=True)
    return path_list[:top_k_value]

def beam_search_step2(path_list, query_sentence, kg_reference_texts, top_k_value=40):
    """
    Step2: Confidence-based re-ranking => combined_score = alpha * relevance + beta * confidence
    """
    if not path_list:
        return []

    model = None

    for record in path_list:
        _ = compute_confidence(record, path_list, kg_reference_texts)

    alpha, beta = 0.7, 0.3
    for record in path_list:
        rel = record.get("relevance", 1)
        conf = record.get("confidence", 1)
        record["combined_score"] = alpha*rel + beta*conf

    path_list.sort(key=lambda x: x["combined_score"], reverse=True)
    return path_list[:top_k_value]

def beam_search_step3(path_list, top_k_value=3):
    """
    Step3: final short list
    """
    if not path_list:
        return []
    path_list.sort(key=lambda x: x["combined_score"], reverse=True)
    return path_list[:top_k_value]

###############################################################################
# 5. The Main Multi-source Function
###############################################################################

def multi_source_beam_search(
    query_sentence,
    freebase_KG_paths,    # list[dict]
    wiki_KG_paths,        # list[dict]
    wiki_doc_paths,       # list[dict]
    web_doc_paths,        # list[dict]
    kg_reference_texts=None,
    return_topN=40
):
    """
    Merge the four lists, run 3-step beam search.
    Notably, each record should have:
        "path_text": "...some text..."
        "source": "freebaseKG" / "wikiKG" / "wikiDoc" / "webDoc" / ...
        "doc_id": a unique identifier for the doc or KG snippet
                  so cross-source verification doesn't get inflated 
                  by the same doc repeated

    Step1 => top80 by relevance
    Step2 => top40 by combined_score
    Step3 => final3
    Return topN from step2 plus final3
    """
    all_paths = []
    all_paths.extend(freebase_KG_paths)
    all_paths.extend(wiki_KG_paths)
    all_paths.extend(wiki_doc_paths)
    all_paths.extend(web_doc_paths)
    # get_st_socre_model()
    # If no reference, we pass an empty list
    if not kg_reference_texts:
        kg_reference_texts = []
    print(f"Total paths: {len(all_paths)}")
    # Step1
    if len(all_paths) > 100:
        step1_res = beam_search_step1(query_sentence, all_paths, top_k_value=100)
        print(f"Step1: {len(step1_res)} paths after relevance filtering.")
    else:
        step1_res = all_paths
        print(f"Step1: {len(step1_res)} paths after relevance filtering (no filtering).")
    # Step2
    if len(step1_res) > 40:
        step2_res = beam_search_step2(step1_res, query_sentence, kg_reference_texts, top_k_value=return_topN)
        print(f"Step2: {len(step2_res)} paths after confidence re-ranking.")
    else:
        step2_res = step1_res
        print(f"Step2: {len(step2_res)} paths after confidence re-ranking (no filtering).")
    return step2_res



# Example usage (outside of function definitions):
# results = explore_graph_from_entities(["entity1", "entity2"])
def Multi_relation_search(entity_id, head=True):
    # Fetch head relations
    if (head == True):
        sparql_relations_extract_head = sparql_head_relations % (format(entity_id))
        head_relations = execurte_sparql(sparql_relations_extract_head)
        relations = replace_relation_prefix(head_relations)
    else:
    # Fetch tail relations
        sparql_relations_extract_tail = sparql_tail_relations % (format(entity_id))
        tail_relations = execurte_sparql(sparql_relations_extract_tail)
        relations = replace_relation_prefix(tail_relations)


    # # Prune unnecessary relations
    # if args.remove_unnecessary_rel:
    #     head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
    #     tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    return relations


def Multi_entity_search(entity, relation, head=True):
    if head:
        sparql_query = sparql_tail_entities_extract % (format(entity), format(relation))
    else:
        # sparql_query = sparql_head_entities_extract % (format(entity), format(relation))
        sparql_query = sparql_head_entities_extract % (format(relation), format(entity))

    entities = execurte_sparql(sparql_query)  # ensure this function is correctly spelled as `execute_sparql`
    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity



def bfs_expand_one_hop(entity, graph_storage, is_head):
    """Perform a single hop expansion for a given entity."""
    # start = time.time()
    # print(f"Expanding {entity} in {'head' if is_head else 'tail'} direction...")
    relations = Multi_relation_search(entity, is_head)
    # end = time.time()
    # print(f"Time taken to fetch {entity} in {'head' if is_head else 'tail'}  relations: {end - start:.2f} seconds")
    new_entities = set()
    if (len(relations) > 0):
        for relation in relations:
            # start2 = time.time()
            connected_entities = Multi_entity_search(entity, relation, is_head)
            # end2_2 = time.time()
            # print(f"Time taken to fetch entities: {end2_2 - start2:.2f} seconds")
            if len(connected_entities) > 0:
                if is_head:
                    if graph_storage.get((entity, relation)) is None:
                        graph_storage[(entity, relation)] = connected_entities
                    else:
                        for connected_entity in connected_entities:
                            if connected_entity not in graph_storage[(entity, relation)]:
                                graph_storage[(entity, relation)].append(connected_entity)
                        # graph_storage[(entity, relation)].extend(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        if graph_storage.get((connected_entity, relation)) is None:
                            graph_storage[(connected_entity, relation)] = [entity]
                        else:
                            for entity in connected_entities:
                                if entity not in graph_storage[(connected_entity, relation)]:

                                     graph_storage[(connected_entity, relation)].append(entity)
                # graph_storage[(entity, relation)] = connected_entities
                new_entities.update(connected_entities)
    #         end2_3 = time.time()
    #         # print(f"Time taken to update graph storage: {end2_3 - end2_2:.2f} seconds")
    #     end2 = time.time()
    # print(f"Time taken to fetch {entity} in {'head' if is_head else 'tail'} entities: {end2 - end:.2f} seconds")

    return new_entities

from concurrent.futures import ThreadPoolExecutor


def replace_prefix1(data):
    if data is None:
        print("Warning: No data available to process in replace_prefix1.")
        return []
    # Function to process results and replace prefixes or format data
    return [{key: value['value'].replace("http://rdf.freebase.com/ns/", "") for key, value in result.items()} for result in data]


def search_relations_and_entities(entity_id, head=True):
    if head:
        sparql_query = sparql_head_entities_and_relations % entity_id
    else:
        sparql_query = sparql_tail_entities_and_relations % entity_id
    results = execute_sparql(sparql_query)
    return replace_prefix1(results)


import concurrent.futures
from threading import Lock
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
def search_relations_and_entities_combined(entity_id):
    sparql_query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation ?connectedEntity ?direction
    WHERE {
        {
            ns:%s ?relation ?connectedEntity .
            BIND("tail" AS ?direction)
        }
        UNION
        {
            ?connectedEntity ?relation ns:%s .
            BIND("head" AS ?direction)
        }
    }
    """ % (entity_id, entity_id)
    results = execute_sparql(sparql_query)
    return replace_prefix1(results)
# Ensure you have proper implementations for search_relations_and_entities_combined and are_entities_connected

def search_relations_and_entities_combined_1(entity_id):
    sparql_query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation ?connectedEntity ?connectedEntityName ?direction
    WHERE {
        {
            ns:%s ?relation ?connectedEntity .
            OPTIONAL {
                ?connectedEntity ns:type.object.name ?name .
                FILTER(lang(?name) = 'en')
            }
            BIND(COALESCE(?name, "Unnamed Entity") AS ?connectedEntityName)
            BIND("tail" AS ?direction)
        }
        UNION
        {
            ?connectedEntity ?relation ns:%s .
            OPTIONAL {
                ?connectedEntity ns:type.object.name ?name .
                FILTER(lang(?name) = 'en')
            }
            BIND(COALESCE(?name, "Unnamed Entity") AS ?connectedEntityName)
            BIND("head" AS ?direction)
        }
    }
    """ % (entity_id, entity_id)
    results = execute_sparql(sparql_query)
    return replace_prefix1(results)


def explore_graph_from_entities_by_hop_neighbor_1(entity_ids, max_depth=5):
    current_entities = set(entity_ids)
    all_entities = set(entity_ids)
    entity_names = {entity: id2entity_name_or_type(entity) for entity in entity_ids}  # 默认所有初始实体名称为"unnamedentity"
    graph = {entity: {} for entity in all_entities}  # 初始化图
    storage_lock = Lock()  # 创建线程安全锁
    if len(entity_ids) == 1:
        connect = True
    else:
        connect = False
    hopnumber = 5

    for depth in range(1, max_depth + 1):
        print(f"Exploring entities at depth {depth}...")
        start = time.time()
        new_entities = set()

        with ThreadPoolExecutor(max_workers=80) as executor:
            futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}
            for future in as_completed(futures):
                results = future.result()
                entity = futures[future]
                for result in results:
                    relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']

                    if connected_entity.startswith("m."):
                        with storage_lock:
                            # 更新或添加实体名称
                            entity_names[connected_entity] = connected_name
                            # 确保图中包含相关实体和关系
                            if entity not in graph:
                                graph[entity] = {}
                            if connected_entity not in graph:
                                graph[connected_entity] = {}
                            if connected_entity not in graph[entity]:
                                graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                            if entity not in graph[connected_entity]:
                                graph[connected_entity][entity] = {'forward': set(), 'backward': set()}
                            # 更新关系
                            if direction == "tail":
                                graph[entity][connected_entity]['forward'].add(relation)
                                graph[connected_entity][entity]['backward'].add(relation)
                            else:  # direction is "head"
                                graph[entity][connected_entity]['backward'].add(relation)
                                graph[connected_entity][entity]['forward'].add(relation)
                        new_entities.add(connected_entity)


        new_entities.difference_update(all_entities)
        all_entities.update(new_entities)
        current_entities = new_entities
        end = time.time()
        print(f"Time taken to explore depth {depth}: {end - start:.2f} seconds")
        if connect == False:
            connect = are_entities_connected(graph, entity_ids, all_entities)
            if connect:
                print(f"All entities are connected within {depth} hops.")
                hopnumber = depth

    print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (connect, graph, hopnumber,all_entities, current_entities, entity_names, False)
# Ensure you have proper implementations for search_relations_and_entities_combined and are_entities_connected



def wiki_search_relations_and_entities_combined_1(entity_id, wiki_client, pre_relations=None, pre_head=False):
    if pre_relations is None:
        pre_relations = set()

    # 1) Obtain all relations for the given entity
    total_relations = wiki_relation_search(
        entity_id=entity_id,
        entity_name="",  # You can adjust how the entity name is handled if needed
        pre_relations=pre_relations,
        pre_head=pre_head,
        remove_unessary_rel=True,
        wiki_client=wiki_client
    )
    # 2) Collect all connected entities from each relation
    all_results = []
    for item in total_relations:
        # 'head' in wiki code indicates if the entity is subject (head=True) or object (head=False)
        # Map that to the freebase style direction:
        #   if head=True => direction="tail"
        #   if head=False => direction="head"
        if item['head']:
            direction = "tail"
        else:
            direction = "head"

        candidate_list = wiki_entity_search(
            entity_id=item['entity_id'],
            relation=item['relation'],
            wiki_client=wiki_client,
            head=item['head']
        )

        # 3) Build the output list with the same fields as the original function
        for candidate in candidate_list:
            all_results.append({
                'relation': item['relation'],
                'connectedEntity': candidate['id'],
                'connectedEntityName': candidate['name'],
                'direction': direction
            })

    return all_results


def wiki_explore_graph_from_entities_by_hop_neighbor_1(entity_set, wiki_client,
                                                       max_depth=5):
    entity_ids = list(entity_set)
    current_entities = set(entity_ids)
    all_entities = set(entity_ids)
    found_answer = False
    entity_names = entity_set.copy()
    # entity_names = {entity.key(): entity.value() for entity in entity_ids}
    graph = {entity: {} for entity in all_entities}
    storage_lock = Lock()
    empty_set = set()

    # If there is only one starting entity, consider them connected by default
    connect = True if len(entity_ids) == 1 else False
    hopnumber = 5

    for depth in range(1, max_depth + 1):
        print(f"Exploring entities at depth {depth}...")
        start = time.time()
        new_entities = set()

        with ThreadPoolExecutor(max_workers=80) as executor:
            futures = {}
            for entity in current_entities:
                # You can adjust the parameters passed to wiki_search_relations_and_entities_combined_1 if needed
                futures[executor.submit(
                    wiki_search_relations_and_entities_combined_1,
                    entity,  # entity_id
                    wiki_client,
                )] = entity

            for future in as_completed(futures):
                results = future.result()
                entity = futures[future]
                for result in results:
                    relation = result['relation']
                    connected_entity = result['connectedEntity']
                    connected_name = result['connectedEntityName']
                    direction = result['direction']

                    if connected_entity and not connected_entity.startswith("[FINISH_ID]"):
                        with storage_lock:
                            entity_names[connected_entity] = connected_name

                            if entity not in graph:
                                graph[entity] = {}
                            if connected_entity not in graph:
                                graph[connected_entity] = {}
                            if connected_entity not in graph[entity]:
                                graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                            if entity not in graph[connected_entity]:
                                graph[connected_entity][entity] = {'forward': set(), 'backward': set()}

                            # Update the graph based on direction
                            if direction == "tail":
                                graph[entity][connected_entity]['forward'].add(relation)
                                graph[connected_entity][entity]['backward'].add(relation)
                            else:
                                graph[entity][connected_entity]['backward'].add(relation)
                                graph[connected_entity][entity]['forward'].add(relation)

                        new_entities.add(connected_entity)

        new_entities.difference_update(all_entities)
        all_entities.update(new_entities)
        current_entities = new_entities

        end = time.time()
        print(f"Time taken to explore depth {depth}: {end - start:.2f} seconds")

        if not connect:
            connect = are_entities_connected(graph, entity_ids, all_entities)
            if connect:
                print(f"All entities are connected within {depth} hops.")
                hopnumber = depth

    print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (connect, graph, hopnumber, all_entities, current_entities, entity_names, False)



def wiki_explore_graph_from_one_topic_entities(
    current_entities, graph, entity_names, exlored_entities, all_entities,
    wiki_client
):
    storage_lock = Lock()
    print("Exploring entities ...")
    start = time.time()
    new_entities = set()

    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {
            executor.submit(
                wiki_search_relations_and_entities_combined_1,
                entity,
                wiki_client
            ): entity
            for entity in current_entities
        }

        exlored_entities.update(current_entities)

        for future in as_completed(futures):
            results = future.result()
            entity = futures[future]
            for result in results:
                relation = result['relation']
                connected_entity = result['connectedEntity']
                connected_name = result['connectedEntityName']
                direction = result['direction']

                # Adjust this check depending on how you treat wiki entities and placeholders
                if not connected_entity or connected_entity.startswith("[FINISH_ID]"):
                    continue

                if connected_entity in exlored_entities:
                    continue

                with storage_lock:
                    entity_names[connected_entity] = connected_name

                    if entity not in graph:
                        graph[entity] = {}
                    if connected_entity not in graph:
                        graph[connected_entity] = {}
                    if connected_entity not in graph[entity]:
                        graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                    if entity not in graph[connected_entity]:
                        graph[connected_entity][entity] = {'forward': set(), 'backward': set()}

                    if direction == "tail":
                        graph[entity][connected_entity]['forward'].add(relation)
                        graph[connected_entity][entity]['backward'].add(relation)
                    else:  # direction == "head"
                        graph[entity][connected_entity]['backward'].add(relation)
                        graph[connected_entity][entity]['forward'].add(relation)

                new_entities.add(connected_entity)

        new_entities.difference_update(exlored_entities)
        all_entities.update(new_entities)
        current_entities = new_entities

    end = time.time()
    print(f"Time taken to explore entities: {end - start:.2f} seconds")

    return (graph, all_entities, exlored_entities, current_entities, entity_names)


def bfs_expand_one_hop2(entity, graph_storage, is_head, executor):
    relations = Multi_relation_search(entity, is_head)
    new_entities = set()
    if relations:
        future_to_relation = {executor.submit(Multi_entity_search, entity, relation, is_head): relation for relation in relations}
        results = {}
        for future in concurrent.futures.as_completed(future_to_relation):
            relation = future_to_relation[future]
            try:
                results[relation] = future.result()
            except Exception as e:
                print(f"Error processing {relation}: {e}")
                continue

        for relation, connected_entities in results.items():
            if connected_entities:
                # with threading.Lock():  # 确保线程安全
                if is_head:
                    graph_storage.setdefault((entity, relation), set()).update(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        graph_storage.setdefault((connected_entity, relation), set()).add(entity)
                new_entities.update(connected_entities)
    return new_entities

def explore_graph_from_entities2(total_entities, max_depth=5):
    graph_storage = {}
    current_entities = set(total_entities)
    all_entities = set(total_entities)
    def process_entity(entity):
        # Both head and tail expansion for a single entity
        new_head_entities = bfs_expand_one_hop2(entity, graph_storage, True, executor)
        new_tail_entities = bfs_expand_one_hop2(entity, graph_storage, False, executor)

        return new_head_entities | new_tail_entities  # Union of sets 

    with ThreadPoolExecutor(max_workers=150) as executor:
        for depth in range(1, max_depth + 1):
            print(f"Exploring entities at depth {depth}...")
            future_to_entity = {executor.submit(process_entity, entity): entity for entity in current_entities}
            next_entities = set()
            for future in concurrent.futures.as_completed(future_to_entity):
                next_entities.update(future.result())

            new_current_entities = next_entities - all_entities
            all_entities.update(next_entities)
            current_entities = new_current_entities

            print(f"Checking connectivity at depth {depth}...")
            if are_entities_connected(graph_storage, total_entities):
                print(f"All entities are connected within {depth} hops.")
                return (True, graph_storage, all_entities, depth)

    print("Entities are not fully connected within the maximum allowed hops.")
    return (False, graph_storage, all_entities, max_depth)



# Global ThreadPoolExecutor

def bfs_expand_one_hop3(entity, graph_storage, is_head):
    executor1 = concurrent.futures.ThreadPoolExecutor(max_workers = 80)

    """Perform a single hop expansion for a given entity."""
    relations = Multi_relation_search(entity, is_head)
    new_entities = set()
    if relations:
        # Perform entity searches in parallel
        future_to_relation = {executor1.submit(Multi_entity_search, entity, relation, is_head): relation for relation in relations}
        results = {}
        for future in concurrent.futures.as_completed(future_to_relation):
            relation = future_to_relation[future]
            results[relation] = future.result()

        # Update graph_storage and new_entities
        for relation, connected_entities in results.items():
            if connected_entities:
                if is_head:
                    if graph_storage.get((entity, relation)) is None:
                        graph_storage[(entity, relation)] = set(connected_entities)
                    else:
                        graph_storage[(entity, relation)].update(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        if graph_storage.get((connected_entity, relation)) is None:
                            graph_storage[(connected_entity, relation)] = {entity}
                        else:
                            graph_storage[(connected_entity, relation)].add(entity)
                new_entities.update(connected_entities)
    return new_entities

def explore_graph_from_entities3(total_entities, max_depth=5):
    graph_storage = {}
    current_entities = set(total_entities)  # Start with the initial set of entities
    all_entities = set(total_entities)      # To track all discovered entities

    for depth in range(1, max_depth + 1):
        print(f"Exploring entities at depth {depth}...")
        next_entities = set()
        
        for entity in current_entities:
            new_head_entities = bfs_expand_one_hop3(entity, graph_storage, True)
            new_tail_entities = bfs_expand_one_hop3(entity, graph_storage, False)
            
        next_entities.update(new_head_entities)
        next_entities.update(new_tail_entities)

        # Calculate new current_entities before updating all_entities
        new_current_entities = next_entities - all_entities

        # Update the set of all entities
        all_entities.update(next_entities)

        # Update current_entities to only include newly discovered entities
        current_entities = new_current_entities

        print("Checking connectivity at depth {depth}...")
        if are_entities_connected(graph_storage, total_entities):
            print(f"All entities are connected within {depth} hops.")
            # all_paths = find_all_paths(graph_storage, total_entities, all_entities)
            # for path in all_paths:
            #     print("Path:", " -> ".join(path))
            return (True, graph_storage, all_entities, depth)

    print("Entities are not fully connected within the maximum allowed hops.")
    return (False, graph_storage, all_entities, depth)



from collections import deque
# from collections import deque

def are_entities_connected(graph, total_entities, all_entities):
    """
    Check if starting from the first entity in total_entities, all other entities in total_entities can be visited.
    graph: Dictionary with entity as key and another dictionary {connected_entity: {'forward': set(), 'backward': set()}} as value.
    total_entities: Set of initial entities to check connectivity from.
    """
    if not total_entities:
        return True  # If no entities are provided, they are trivially connected.

    total_entities_set = set(total_entities)
    if len(total_entities_set) == 1:
        return True  # Only one entity, trivially connected to itself.

    start_entity = next(iter(total_entities_set))  # Start BFS from any entity in the set
    visited = set()
    queue = deque([start_entity])

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            # Early termination check
            if total_entities_set.issubset(visited):
                return True

            # Add connected entities to the queue
            for connected_entity, relations in graph[current].items():
                if connected_entity not in visited:
                    queue.append(connected_entity)

    # Final check in case not all entities are connected
    return False



# def are_entities_connected1(graph_storage, total_entities):
    """
    Check if starting from the first entity in total_entities, all other entities in total_entities can be visited.
    graph_storage: Dictionary storing connections (head, relation) -> [connected_entities]
    total_entities: List or Set of initial entities to check connectivity from.
    """
    if not total_entities:
        return True  # If no entities are provided, they are trivially connected.

    total_entities_set = set(total_entities)

    if len(total_entities_set) == 1:
        return True  # Only one entity, trivially connected to itself.

    start_entity = next(iter(total_entities_set)) 

    visited = set()
    queue = deque([start_entity])  # Using deque for efficient pops from the front

    while queue:
        current = queue.popleft()  # O(1) time complexity
        if current in visited:
            continue
        visited.add(current)

        if total_entities_set.issubset(visited):
            return True

        # Process each connection where the current entity is involved
        for (head, relation), tails in graph_storage.items():
            if head == current:
                for tail in tails:
                    if tail not in visited:
                        queue.append(tail)
            elif current in tails and head not in visited:
                queue.append(head)

    return False

import concurrent.futures
import time

# Global ThreadPoolExecutor


def bfs_expand_one_hop1(entity, graph_storage, is_head):
    """Perform a single hop expansion for a given entity."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    relations = Multi_relation_search(entity, is_head)
    new_entities = set()
    if relations:
        # Perform entity searches in parallel
        future_to_relation = {executor.submit(Multi_entity_search, entity, relation, is_head): relation for relation in relations}
        results = {}
        for future in concurrent.futures.as_completed(future_to_relation):
            relation = future_to_relation[future]
            results[relation] = future.result()

        # Update graph_storage and new_entities
        for relation, connected_entities in results.items():
            if connected_entities:
                if is_head:
                    if graph_storage.get((entity, relation)) is None:
                        graph_storage[(entity, relation)] = set(connected_entities)
                    else:
                        graph_storage[(entity, relation)].update(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        if graph_storage.get((connected_entity, relation)) is None:
                            graph_storage[(connected_entity, relation)] = {entity}
                        else:
                            graph_storage[(connected_entity, relation)].add(entity)
                new_entities.update(connected_entities)
    return new_entities




def initialize_graph(graph_storage, all_entities):
    graph = {entity: {} for entity in all_entities}
    for (head, relation), tails in graph_storage.items():
        for tail in tails:
            if tail not in graph[head]:
                graph[head][tail] = {}
            if 'forward' not in graph[head][tail]:
                graph[head][tail]['forward'] = set()
            graph[head][tail]['forward'].add(relation)

            # 存储反向关系
            if tail not in graph:
                graph[tail] = {}
            if head not in graph[tail]:
                graph[tail][head] = {}
            if 'backward' not in graph[tail][head]:
                graph[tail][head]['backward'] = set()
            graph[tail][head]['backward'].add(relation)
    return graph


from functools import lru_cache




from collections import deque


from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import itertools

def process_node(start_paths, goal_paths, node):
    local_paths = []
    for f_path in start_paths[node]:
        for b_path in goal_paths[node]:
            # 确保 f_path 和 b_path 始终为列表
            f_path = f_path if isinstance(f_path, list) else [f_path]
            b_path = b_path if isinstance(b_path, list) else [b_path]
            # local_paths.append(tuple(f_path))
            # local_paths.append(tuple(b_path))
            try:
                if len(b_path) > 1:
                    combined_path = f_path + b_path[::-1][1:]  # 确保 b_path 长度大于1
                else:
                    combined_path = f_path  # 如果 b_path 只有一个元素，只取 f_path
                if len(combined_path)>1:
                    local_paths.append(tuple(combined_path))
            except TypeError as e:
                print(f"TypeError combining paths: {e}")
                print(f"f_path: {f_path}, b_path: {b_path}")  # 输出问题数据

    return local_paths


def node_expand_with_paths(graph, start, hop):
    queue = deque([(start, [start])])  # 存储节点和到该节点的路径
    visited = {start: [start]}  # 记录到达每个节点的所有路径

    while queue:
        current_node, current_path = queue.popleft()
        if current_node not in graph:
            print(f"Skipping non-existent node {current_node}")
            continue
        current_layer = len(current_path) - 1
        if current_layer < hop:  # 只扩展到给定的层数
            for neighbor in graph[current_node]:
                if neighbor in current_path:
                    continue  # 跳过已经在路径中的节点，防止回环
                new_path = current_path + [neighbor]
                if neighbor not in visited:
                    visited[neighbor] = []
                    queue.append((neighbor, new_path))
                visited[neighbor].append(new_path)  # 记录到此节点的每条路径

    return visited


# def bfs_with_intersection_only(graph, entity_list, hop):
#     # 使用多线程并行执行node_expand
#     with ThreadPoolExecutor() as executor:
#         futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
#         paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}
    
#     # 计算所有实体的路径交集
#     intersection = set.intersection(*(set(paths.keys()) for paths in paths_dict.values()))
#     return intersection

def bfs_with_intersection_only(graph, entity_list, hop):
    # run node expansion in parallel
    with ThreadPoolExecutor() as executor:
        future_to_entity = {
            executor.submit(node_expand_with_paths, graph, ent, hop): ent
            for ent in entity_list
        }

        # collect the results
        paths_dict = {
            future_to_entity[future]: future.result()
            for future in as_completed(future_to_entity)
        }

    # make a list of the key sets that were found
    key_sets = [set(paths.keys()) for paths in paths_dict.values() if paths]

    # if any entity had zero reachable keys, or none were found at all,
    # the intersection is empty
    if len(key_sets) < len(entity_list) or not key_sets:
        return set()

    # otherwise compute the common nodes
    return set.intersection(*key_sets)

def create_subgraph_through_intersection3s(graph, entity_list, hop):
    from collections import defaultdict
    import copy

    # Initialize a function to safely add nodes and relationships to the subgraph
    def safe_add_edge(subgraph, src, dst, relation, direction):
        if src not in subgraph:
            subgraph[src] = {}
        if dst not in subgraph[src]:
            subgraph[src][dst] = {'forward': set(), 'backward': set()}
        subgraph[src][dst][direction].add(relation)

    # Use ThreadPoolExecutor to handle node expansion
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    subgraph = {}

    # 计算所有实体的路径交集
    intersection = set.intersection(*(set(paths.keys()) for paths in paths_dict.values()))
    print("Find all the intersection nodes")

    # Iterate through each entity's paths to build the subgraph
    for node in intersection:
        for paths in paths_dict.values():
            for path in paths[node]:
                path = path if isinstance(path, list) else [path]
                if len(path) > 1:
                    for i in range(0, len(path) - 1):
                        src, dst = path[i], path[i + 1]
                        if src in graph and dst in graph[src]:
                            for direction in ['forward', 'backward']:
                                for relation in graph[src][dst][direction]:
                                    safe_add_edge(subgraph, src, dst, relation, direction)
                                for relation in graph[dst][src][direction]:
                                    safe_add_edge(subgraph, dst, src, relation, direction)
                        else:
                            print(path)
                            print(f"Missing edge or node in the graph from {src} to {dst}")

    return subgraph


def create_subgraph_through_intersections(graph, entity_list, intersection, total_id_to_name_dict, hop):
    # 使用多线程并行执行 node_expand_with_paths


    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    complete_subgraph = {}
    reduce_entity_names = {}

    for paths in paths_dict.values():
        for end_node, all_paths in paths.items():
            if end_node in intersection:
                # 将所有通过交集节点的路径加入到子图中
                for path in all_paths:
                    path = path if isinstance(path, list) else [path]
                    if len(path) > 1:
                        try:
                            for i in range(len(path) - 1):
                                head, tail = path[i], path[i + 1]
                                if head not in complete_subgraph:
                                    complete_subgraph[head] = {}
                                if tail not in complete_subgraph[head]:
                                    complete_subgraph[head][tail] = graph[head][tail].copy()
                                if tail not in complete_subgraph:
                                    complete_subgraph[tail] = {}
                                if head not in complete_subgraph[tail]:
                                    complete_subgraph[tail][head] = graph[tail][head].copy()
                        except KeyError as e:
                            print(f"An error occurred when processing edge from {head} to {tail}: {e}")
    for en_id in complete_subgraph.keys():
        reduce_entity_names[en_id] = total_id_to_name_dict.get(en_id, "Unnamed Entity")

    return complete_subgraph,reduce_entity_names

def bfs_with_intersection_backup(graph, entity_list, hop):
    # Perform node expansion in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    if len(entity_list) == 1:
        paths = set()
        for nei in paths_dict[entity_list[0]].values():
            for f_path in nei:
                f_path = f_path if isinstance(f_path, list) else [f_path]
                if len(f_path) > 1:
                    paths.add(tuple(f_path))
        print("Only one entity, return all paths")
        return list(paths)
    
    combination_path_dict = {}
    total_paths = None
    
    for i in range(len(entity_list) - 1):
        start_entity = entity_list[i]
        target_entity = entity_list[i+1]
        start_entity_paths = paths_dict[start_entity]
        target_entity_paths = paths_dict[target_entity]
        intersection = set(start_entity_paths.keys()) & set(target_entity_paths.keys())
        if not intersection:

            return_path = set()
            for path in start_entity_paths.values():
                for f_path in path:
                    f_path = f_path if isinstance(f_path, list) else [f_path]
                    if len(f_path) > 1:
                        return_path.add(tuple(f_path))
            for path in target_entity_paths.values():
                for b_path in path:
                    b_path = b_path if isinstance(b_path, list) else [b_path]
                    if len(b_path) > 1:
                        return_path.add(tuple(b_path))
            return list(return_path)
        temp_paths = []
        for node in intersection:
            temp_paths.extend(process_node(start_entity_paths, target_entity_paths, node))
        if total_paths is None:
            total_paths = temp_paths
        else:
            combined_paths = []
            for path1 in total_paths:
                for path2 in temp_paths:
                    if path1[-1] == path2[0]:
                        combined_paths.append(path1 + path2[1:])
            total_paths = combined_paths
            if not total_paths:
                return []

    return total_paths

# from hydra_main import Beam_search_step1

import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_prefix_text_fast(text):
    """
    Return the first ten words of a string efficiently.
    """
    parts = text.split(' ', 10)
    return ' '.join(parts[:10])


def format_paths_fast(paths, name_map, include_ids=False):
    """
    Convert full paths (alternating entity IDs and relation tokens) to readable strings.
    """
    formatted = []
    for path in paths:
        segs = []
        for idx, token in enumerate(path):
            if idx % 2 == 1:
                segs.append(token)
            else:
                if token.startswith('{'):
                    ids = token[1:-1].split(',')
                    seen = set(); uniq = []
                    for eid in ids:
                        eid = eid.strip()
                        name = name_map.get(eid) or "Unnamed Entity"
                        snippet = extract_prefix_text_fast(name)
                        val = f"{eid}: {snippet}" if include_ids else snippet
                        if val not in seen:
                            seen.add(val); uniq.append(val)
                    segs.append("{" + ", ".join(uniq) + "}")
                else:
                    name = name_map.get(token) or "Unnamed Entity"
                    snippet = extract_prefix_text_fast(name)
                    segs.append(f"{{{token}: {snippet}}}" if include_ids else f"{{{snippet}}}")
        formatted.append(' - '.join(segs))
    return formatted


def combine_all_relations_fast(graph, entity_seq):
    """
    Yield every full path by expanding all relations for a pure-entity sequence.
    """
    rel_opts = []
    for u, v in zip(entity_seq, entity_seq[1:]):
        rd = graph.get(u, {}).get(v, {})
        opts = []
        for direction, rels in rd.items():
            sym = '->' if direction == 'forward' else '<-'
            items = rels if isinstance(rels, (list, set)) else [rels]
            for r in items:
                opts.append(f"{{{sym} {r} {sym}}}")
        rel_opts.append(opts or ["{->}"])

    for combo in itertools.product(*rel_opts):
        full = []
        for u, rel in zip(entity_seq, combo):
            full.extend([u, rel])
        full.append(entity_seq[-1])
        yield full


def combine_top_relation_fast(graph, entity_seq):
    """
    Build one path selecting the lexicographically first relation for each edge.
    """
    full = []
    for u, v in zip(entity_seq, entity_seq[1:]):
        rd = graph.get(u, {})
        rels_dict = rd.get(v, {})
        candidates = []
        for direction, rs in rels_dict.items():
            sym = '->' if direction == 'forward' else '<-'
            items = rs if isinstance(rs, (list, set)) else [rs]
            for r in items:
                candidates.append(f"{{{sym} {r} {sym}}}")
        pick = sorted(candidates)[0] if candidates else "{->}"
        full.extend([u, pick])
    full.append(entity_seq[-1])
    return full


def merge_by_relation_fast(paths):
    """
    Merge full paths by identical relation patterns, uniting their entity IDs.
    """
    buckets = defaultdict(list)
    for p in paths:
        key = tuple(p[i] for i in range(1, len(p), 2))
        buckets[key].append(p)

    merged = []
    for group in buckets.values():
        L = len(group[0])
        comb = []
        for i in range(0, L, 2):
            # only using top 20 entities for each relation
            ents = {p[i] for p in group}
            ent_str = "{" + ", ".join(sorted(ents)[:20]) + "}" if len(ents) > 1 else ents.pop()
            comb.append(ent_str)
            if i < L - 1:
                comb.append(group[0][i+1])
        merged.append(comb)
    return merged


def process_node(paths1, paths2, node):
    """
    Chain two lists of paths at a common node.
    """
    merged = []
    for p1 in paths1[node]:
        for p2 in paths2[node]:
            merged.append(p1 + p2[1:])
    return merged



# from try_new_path import (
#     combine_all_relations_fast,
#     format_paths_fast,
#     combine_top_relation_fast,
#     merge_by_relation_fast,
# )
def bfs_with_intersection(graph, entity_list, hop):
    # Parallel node expansion for each entity
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, ent, hop): ent for ent in entity_list}
        paths_dict = {futures[f]: f.result() for f in as_completed(futures)}

    # Single entity: return all paths longer than 1
    if len(entity_list) == 1:
        result = set()
        for paths in paths_dict[entity_list[0]].values():
            for p in paths:
                seq = p if isinstance(p, list) else [p]
                if len(seq) > 1:
                    result.add(tuple(seq))
        print("Only one entity, returning all paths")
        return list(result)

    merged_paths = None
    intersect_entities = set()

    # Compare each adjacent pair and chain intersecting paths
    for i in range(len(entity_list) - 1):
        ent1 = entity_list[i]
        ent2 = entity_list[i + 1]
        paths1 = paths_dict[ent1]
        paths2 = paths_dict[ent2]
        common_nodes = set(paths1.keys()) & set(paths2.keys())

        if common_nodes:
            # mark entities involved
            intersect_entities.add(ent1)
            intersect_entities.add(ent2)
            # intersect_entities.add([ent1, ent2])
            pair_paths = []
            for node in common_nodes:
                pair_paths.extend(process_node(paths1, paths2, node))
            if merged_paths is None:
                merged_paths = pair_paths
            else:
                new_merged = []
                for p1 in merged_paths:
                    for p2 in pair_paths:
                        if p1[-1] == p2[0]:
                            new_merged.append(p1 + p2[1:])
                merged_paths = new_merged
                # if not merged_paths:
                #     print("Chained intersection resulted in no paths, returning empty list")
                #     return []
        # continue checking other pairs

    # If no intersections found at all, return all entities' direct paths
    if merged_paths is None:
        all_paths = set()
        for ent in entity_list:
            for paths in paths_dict[ent].values():
                for p in paths:
                    seq = p if isinstance(p, list) else [p]
                    if len(seq) > 1:
                        all_paths.add(tuple(seq))
        print("No intersections found for any pairs, returning merged paths of all entities")
        return list(all_paths)

    # Build final result: chains + direct paths for non-connected entities
    result_set = set(tuple(p) for p in merged_paths)
    for ent in entity_list:
        if ent not in intersect_entities:
            for paths in paths_dict[ent].values():
                for p in paths:
                    seq = p if isinstance(p, list) else [p]
                    if len(seq) > 1:
                        result_set.add(tuple(seq))

    print("Returning chained paths plus isolated entities' paths")
    return list(result_set)
import os
import gc
import psutil

THRESHOLD = 200 * 1024 ** 3        # 200 GB（字节）
process = psutil.Process(os.getpid())

def maybe_collect():
    if process.memory_info().rss > THRESHOLD:
        gc.collect()

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_all_paths_bibfs_itersection(graph, total_entities, hop, if_using_all_r, question_cot, total_id_to_name_dict,model, topk):
    raw = []
    # entity_list = sorted(total_entities, key=lambda x: len(graph.get(x, {})))

    raw = bfs_with_intersection(graph, total_entities, hop)
    if if_using_all_r:
        expanded = [full for seq in raw for full in combine_all_relations_fast(graph, seq)]
    else:
        expanded = [combine_top_relation_fast(graph, seq) for seq in raw]
    merged = merge_by_relation_fast(expanded)
    formatted = format_paths_fast(merged, total_id_to_name_dict)
    return formatted 


from typing import List, Tuple
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from sentence_transformers import SentenceTransformer

import threading

import torch
from sentence_transformers import SentenceTransformer

device_ = "cuda" if torch.cuda.is_available() else "cpu"
# _SBERT_PATH = "emb_model/MiniLM-L12-v2"
_SBERT_PATH = "paraphrase-multilingual-MiniLM-L12-v2"
_ST_MODEL   = None

_ST_MODEL = SentenceTransformer(_SBERT_PATH, device=device_)
_ST_MODEL.eval()
if True:
    _ST_MODEL.half()
# 可选：开启 flash attention
hf = _ST_MODEL[0].auto_model
if hasattr(hf.config, "sdpa_kernel"):
    hf.config.sdpa_kernel = "flash"
else:
    hf.config.attn_implementation = "flash_attention_2"
# _ST_MODEL.to(device_)


def get_st_model_path_P(
    name=_SBERT_PATH,
    device=device_,
    use_fp16=True,
    model=_ST_MODEL
):
    return model




import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

###############################################################################
# 0.  Fast intersection‑depth finder (exp. probe + binary search)              #
###############################################################################

def min_hop_with_intersection(graph: Dict[str, List[str]],
                              entities: List[str],
                              max_hop: int,
                              bfs_inter_fn) -> int:
    """Return the smallest hop ≤ `max_hop` with non‑empty intersection."""
    if max_hop <= 1:
        return max_hop
    hop = 1
    while hop <= max_hop and not bfs_inter_fn(graph, entities, hop):
        hop <<= 1  # exponential step
    if hop > max_hop:
        return max_hop  # no intersection within bound
    low, high = hop >> 1, hop
    while high > low + 1:
        mid = (low + high) // 2
        if bfs_inter_fn(graph, entities, mid):
            high = mid
        else:
            low = mid
    return high

###############################################################################
# 1.  Token‑overlap lexical pre‑filter                                         #
###############################################################################



def _tokenize(text: str) -> set:
    """Lower‑case whitespace tokenisation; can be swapped for better tokenizer."""
    return set(text.lower().split())

def lexical_prefilter(question: str, texts: List[str], keep: int = 500) -> List[int]:
    """
    Very cheap overlap score: |intersect| / |question_tokens|.
    Returns indices of the *keep* best texts.  If len(texts) <= keep, returns all.
    """
    if not texts or len(texts) <= keep:
        return list(range(len(texts)))
    if not question or len(question) ==0:
        return list(range(len(texts)))

    q_tok = _tokenize(question)
    scores = []
    for i, t in enumerate(texts):
        inter = q_tok & _tokenize(t)
        # normalise by question length to favour texts sharing *many* unique tokens
        scores.append((len(inter) / (len(q_tok) + 1e-9), i))
    # 取分最高的 keep 条
    scores.sort(key=lambda x: -x[0])
    return [i for _, i in scores[:keep]]

###############################################################################
# 2.  SBERT encoder with global cache                                          #
###############################################################################

# _emb_cache: defaultdict[str, np.ndarray] = defaultdict(lambda: None)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


_emb_cache: defaultdict = defaultdict(lambda: None)  # text -> np.ndarray

def encode_with_cache(model: SentenceTransformer, str_list: List[str], batch_size: int = 256) -> np.ndarray:
    """Return embeddings in the same order as str_list, reusing cache when possible."""
    to_encode = [s for s in str_list if _emb_cache[s] is None]
    if to_encode:
        vecs = model.encode(to_encode, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        for s, v in zip(to_encode, vecs):
            _emb_cache[s] = v
    return np.stack([_emb_cache[s] for s in str_list])

###############################################################################
# 3.  Adaptive lexical‑keep & streaming top‑K cosine                           #
###############################################################################

def _auto_lexical_keep(total: int,
                       topk: int,
                       sparse_prop: float = 0.5,
                       dense_prop: float = 0.3,
                       sparse_thresh: int = 10000,
                       widen: int = 8,
                       cap: int = 5000) -> int:
    prop = sparse_prop if total < sparse_thresh else dense_prop
    base = max(int(total * prop), topk * widen, topk * 2, 500)
    return min(cap, base)


def _streaming_topk_sim(model: SentenceTransformer,
                        q_vec: np.ndarray,
                        texts: List[str],
                        topk: int,
                        batch_size: int = 256) -> set[int]:
    heap: List[Tuple[float, int]] = []  # (score, idx)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        sims = cosine_similarity(q_vec, encode_with_cache(model, chunk)).ravel()
        for i, s in enumerate(sims):
            idx = start + i
            if len(heap) < topk:
                heapq.heappush(heap, (s, idx))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, idx))
    return {idx for _, idx in heap}

###############################################################################
# 4.  One‑layer pruning                                                        #
###############################################################################

def prune_paths_by_beam_search(paths: List[List[str]],
                               question_cot: str,
                               id_to_name_dict: Dict[str, str],
                               model: SentenceTransformer,
                               topk: int,
                               graph,
                               if_using_all_r: bool,
                               *,
                               sbert_batch: int = 256,
                               **auto_kw) -> List[List[str]]:
    """Keep paths whose *any* translation ranks in top‑`topk`."""
#     # 1. 生成 (path, candidate_text) 对 -------------------------------------------------
    pairs: List[Tuple[List[str], str]] = []
    for path in paths:
        temps: List[str] = []
        if if_using_all_r:
            temps.extend(add_relations_to_path_with_all_R(graph, path))
        else:
            temps = [add_relations_to_path1(graph, path)]
        # translate entity IDs to readable strings
        temps = format_paths_to_natural_language_id_with_name(temps, id_to_name_dict)
        for text in temps:
            pairs.append((path, text))

    if len(pairs) == 0:
        return []

    # 2. 提早返回：候选数 <= topk ------------------------------------------------------
    if len(pairs) <= topk:
        print("No need to prune, all paths are kept.")
        return list({tuple(p) for p, _ in pairs})  # 去重后直接返回
    total = len(pairs)
    keep = _auto_lexical_keep(total, topk, **auto_kw)
    all_txts = [t for _, t in pairs]
    kept_idx = lexical_prefilter(question_cot, all_txts, keep)
    kept_pairs = [pairs[i] for i in kept_idx]
    kept_txts  = [all_txts[i] for i in kept_idx]
    print("Pre-filtered texts:", len(kept_idx), "out of", len(all_txts))
    model = get_st_model_path_P()
    q_vec = encode_with_cache(model, [question_cot])[0].reshape(1, -1)
    best_idx = _streaming_topk_sim(model, q_vec, kept_txts, topk)
    best_txt_set = {kept_txts[i] for i in best_idx}
    print("Selected texts:", len(best_txt_set), "out of", len(kept_txts))
    # print first 3 selected texts
    # for i in range(3):
    #     print("Selected text:", kept_txts[i])
    #     print("Selected path:", kept_pairs[i][0])
    # 3. 收集至少有一个翻译入选的原始节点路径 ----------------------------------------
    # 4. 去重
    # 5. 返回
    pruned, seen = [], set()
    for p, t in kept_pairs:
        key = tuple(p)
        if t in best_txt_set and key not in seen:
            pruned.append(p)
            seen.add(key)
    return pruned

###############################################################################
# 5.  Tree‑based peeling (single root)                                         #
###############################################################################

def tree_based_peeling_search(graph,
                              start: str,
                              max_hop: int,
                              question_cot: str,
                              id_to_name_dict: Dict[str, str],
                              model: SentenceTransformer,
                              *,
                              topk: int = 80,
                              if_using_all_r: bool = False,
                              **kw) -> List[List[str]]:
    # current = [[start]]
    # for _ in range(max_hop):
    #     expanded = [p + [nei]
    #                 for p in current
    #                 for nei in graph.get(p[-1], [])
    #                 if nei not in p]
    #     if not expanded:
    #         break
    #     current = prune_paths_by_beam_search(expanded, question_cot, id_to_name_dict,
    #                                          model, topk, graph, if_using_all_r, **kw)
        
    #     if not current:
    #         break
    # return current
    current = [[start]]
    for _ in range(max_hop):
        expanded = []
        for p in current:
            for nei in graph.get(p[-1], []):
                if nei in p:
                    continue
                expanded.append(p + [nei])
        if len(expanded) > topk:
            current = prune_paths_by_beam_search(expanded, question_cot, id_to_name_dict,
                                             model, topk, graph, if_using_all_r, **kw)
        
        else:
            current = expanded
        if not current:
            break
    return current
###############################################################################
# 6.  Multi‑entity peeling & merge                                             #
###############################################################################
from concurrent.futures import ThreadPoolExecutor, as_completed


def multi_entity_tree_search(graph,
                             entity_list: List[str],
                             max_hop: int,
                             question_cot: str,
                             id_to_name_dict: Dict[str, str],
                             model: SentenceTransformer,
                             *,
                             topk: int = 80,
                             if_using_all_r: bool = False,
                             bfs_inter_fn=None,
                             **kw) -> List[str]:
    # 0. minimal hop optimisation
    # if bfs_inter_fn is not None:
    #     max_hop = min_hop_with_intersection(graph, entity_list, max_hop, bfs_inter_fn)

    # 1. peel each entity in parallel
    with ThreadPoolExecutor() as pool:
        future_map = {pool.submit(tree_based_peeling_search, graph, ent, max_hop,
                                  question_cot, id_to_name_dict, model, topk=topk,
                                  if_using_all_r=if_using_all_r, **kw): ent for ent in entity_list}
        ent_paths = {future_map[f]: f.result() for f in as_completed(future_map)}

    # 2. merge paths across entities
    if len(entity_list) == 1:
        node_paths = ent_paths[entity_list[0]]
    else:
        merged, connected = None, set()
        for a, b in zip(entity_list, entity_list[1:]):
            pa, pb = ent_paths[a], ent_paths[b]
            end_a, end_b = defaultdict(list), defaultdict(list)
            for p in pa:
                end_a[p[-1]].append(p)
            for p in pb:
                end_b[p[-1]].append(p)
            common = set(end_a) & set(end_b)
            if common:
                connected.update([a, b])
                comb = [x + y[1:] for c in common for x in end_a[c] for y in end_b[c]]
                merged = comb if merged is None else [m + q[1:] for m in merged for q in comb if m[-1] == q[0]]
        if merged is None:
            node_paths = [p for plist in ent_paths.values() for p in plist]
        else:
            node_paths = merged
            for ent in entity_list:
                if ent not in connected:
                    node_paths.extend(ent_paths[ent])

    # 3. translate node paths
    translate = add_relations_to_path_with_all_R if if_using_all_r else add_relations_to_path1
    rel_txts = []
    for p in node_paths:
        rel_txts.extend(translate(graph, p))

    rel_txts = merge_paths_by_relations(rel_txts)
    return format_paths_to_natural_language_id_with_name(rel_txts, id_to_name_dict)



def multi_entity_tree_search_backup(graph, entity_list, max_hop,
                             question_cot, id_to_name_dict, therashold = 200,
                             topk=80, if_using_all_r=False):
    """
    并行对每个实体做 peeling，按末端对齐拼接各自的节点路径，
    最后一步直接把所有节点路径翻译成“节点+边”形式返回。
    """
    from concurrent.futures import ThreadPoolExecutor
    # 并行单实体剥离
    with ThreadPoolExecutor() as ex:
        futures = [
            ex.submit(
                tree_based_peeling_search,
                graph, ent, max_hop,
                question_cot, id_to_name_dict,model, therashold,
                topk, if_using_all_r
            )
            for ent in entity_list
        ]
        paths_dict = [f.result() for f in futures]

    result_list = []
    if len(entity_list) == 1:
        paths = set()
        for nei in paths_dict[entity_list[0]].values():
            for f_path in nei:
                f_path = f_path if isinstance(f_path, list) else [f_path]
                if len(f_path) > 1:
                    paths.add(tuple(f_path))
        print("Only one entity, return all paths")
        result_list = list(paths)
    else:
        combination_path_dict = {}
        total_paths = None
        
        for i in range(len(entity_list) - 1):
            start_entity = entity_list[i]
            target_entity = entity_list[i+1]
            start_entity_paths = paths_dict[start_entity]
            target_entity_paths = paths_dict[target_entity]
            intersection = set(start_entity_paths.keys()) & set(target_entity_paths.keys())
            if not intersection:

                return_path = set()
                for path in start_entity_paths.values():
                    for f_path in path:
                        f_path = f_path if isinstance(f_path, list) else [f_path]
                        if len(f_path) > 1:
                            return_path.add(tuple(f_path))
                for path in target_entity_paths.values():
                    for b_path in path:
                        b_path = b_path if isinstance(b_path, list) else [b_path]
                        if len(b_path) > 1:
                            return_path.add(tuple(b_path))
                result_list = list(return_path)
            temp_paths = []
            for node in intersection:
                temp_paths.extend(process_node(start_entity_paths, target_entity_paths, node))
            if total_paths is None:
                total_paths = temp_paths
            else:
                combined_paths = []
                for path1 in total_paths:
                    for path2 in temp_paths:
                        if path1[-1] == path2[0]:
                            combined_paths.append(path1 + path2[1:])
                total_paths = combined_paths
                if not total_paths:
                    return []
        result_list = total_paths
    # 拼接多实体路径
    if len(lists) == 1:
        node_paths = lists[0]
    else:
        node_paths = lists[0]
        for nxt in lists[1:]:
            combined = []
            for p1 in node_paths:
                for p2 in nxt:
                    if p1[-1] == p2[0]:
                        combined.append(p1 + p2[1:])
            node_paths = combined
            if not node_paths:
                return []


    # 最终翻译
    translated = []
    if if_using_all_r:
        for path in node_paths:
            translated.extend(add_relations_to_path_with_all_R(graph, path))
    else:
        for path in node_paths:
            translated.append(add_relations_to_path1(graph, path))
    translated =  merge_paths_by_relations(translated)
    final_entity_path = format_paths_to_natural_language_id_with_name(translated,id_to_name_dict)
    return final_entity_path



def find_all_paths_tree_search(graph, total_entities, hop, if_using_all_r):
    all_paths = []
    # entity_list = sorted(total_entities, key=lambda x: len(graph.get(x, {})))

    raw_paths = bfs_with_tree_based_search(graph, total_entities, hop)
    if if_using_all_r:
        for path in raw_paths:
            all_paths.extend(add_relations_to_path_with_all_R(graph, path))

        # all_paths = all_paths.extend(add_relations_to_path_with_all_R(graph, path) for path in raw_paths)
    else:
        all_paths = [add_relations_to_path1(graph, path) for path in raw_paths]

    return merge_paths_by_relations(all_paths)
    # return all_paths


def find_all_paths_bibfs_itersection_limit(graph, total_entities, hop, if_using_all_r):
    all_paths = []
    # entity_list = sorted(total_entities, key=lambda x: len(graph.get(x, {})))

    raw_paths = bfs_with_intersection_inter(graph, total_entities, hop)

    # raw_paths = bfs_with_intersection_testv1(graph, total_entities, hop)
    if if_using_all_r:
        for path in raw_paths:
            all_paths.extend(add_relations_to_path_with_all_R(graph, path))
    else:
        all_paths = [add_relations_to_path1(graph, path) for path in raw_paths]

    return merge_paths_by_relations(all_paths)
    # return all_paths

import difflib

def find_best_matching_substring(entity, cot_line):
    len_entity = len(entity)
    len_cot = len(cot_line)

    # Consider substrings within reasonable lengths
    min_len = max(1, len_entity // 2)
    max_len = min(len_cot, len_entity * 2)

    best_score = 0
    best_start = -1

    for length in range(min_len, max_len + 1):
        for start in range(len_cot - length + 1):
            substring = cot_line[start:start + length]
            score = difflib.SequenceMatcher(None, entity, substring).ratio()
            if score > best_score:
                best_score = score
                best_start = start

    return best_score, best_start

def reorder_entities(cot_line, topic_entity_dict):
    entity_positions = []

    for entity in topic_entity_dict:
        score, position = find_best_matching_substring(entity, cot_line)
        # Assign a high position if no match is found
        if position != -1:
            entity_positions.append((position, entity))
        else:
            entity_positions.append((float('inf'), entity))

    # Sort entities based on their positions in cot_line
    entity_positions.sort()
    sorted_entities = [entity for position, entity in entity_positions]
    return sorted_entities

def bfs_with_intersection_inter(graph, entity_list, hop):
    # 使用多线程并行执行node_expand
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}
    paths = set()
    
    # 计算所有实体的路径交集
    if len(entity_list) == 1:
        local_paths = []
        for nei in paths_dict[entity_list[0]].values():
            for f_path in nei:
                # 确保 f_path 和 b_path 始终为列表
                f_path = f_path if isinstance(f_path, list) else [f_path]
                try:

                    paths.add(tuple(f_path))
                except TypeError as e:
                    print(f"TypeError combining paths: {e}")
        print("Only one entity, return all paths")
        return list(paths)
    
    intersection = set.intersection(*(set(paths.keys()) for paths in paths_dict.values()))
    print("Find all the intersection nodes")
    if not intersection:
        return []

    combination_path_dict = {}
    with ThreadPoolExecutor() as executor:
        for i in range(1, len(entity_list)):
            futures = []
            start_entity_paths = paths_dict[entity_list[i - 1]]
            target_entity_paths = paths_dict[entity_list[i]]
            # intersection = set(start_entity_paths.keys()) & set(target_entity_paths.keys())
            if not intersection:
                return []
            for node in intersection:
                futures.append(executor.submit(process_node, start_entity_paths, target_entity_paths, node))
            
            # Collect results for this entity pair
            combination_paths = []
            for future in as_completed(futures):
                combination_paths.extend(future.result())
            combination_path_dict[(entity_list[i - 1], entity_list[i])] = combination_paths

    # Combine all paths
    total_paths = combine_all_paths(combination_path_dict, entity_list)
    print(entity_list)
    return total_paths


def node_expand_with_paths_tree_search(graph, start, hop):
    queue = deque([(start, [start])])  # 存储节点和到该节点的路径
    visited = {start: [start]}  # 记录到达每个节点的所有路径

    while queue:
        current_node, current_path = queue.popleft()
        if current_node not in graph:
            print(f"Skipping non-existent node {current_node}")
            continue
        current_layer = len(current_path) - 1
        if current_layer < hop:  # 只扩展到给定的层数
            for neighbor in graph[current_node]:
                if neighbor in current_path:
                    continue  # 跳过已经在路径中的节点，防止回环
                new_path = current_path + [neighbor]
                if neighbor not in visited:
                    visited[neighbor] = []
                    queue.append((neighbor, new_path))
                visited[neighbor].append(new_path)  # 记录到此节点的每条路径

    return visited


def bfs_with_tree_based_search(graph, entity_list, hop):
    # Perform node expansion in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths_tree_search, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    if len(entity_list) == 1:
        paths = set()
        for nei in paths_dict[entity_list[0]].values():
            for f_path in nei:
                f_path = f_path if isinstance(f_path, list) else [f_path]
                if len(f_path) > 1:
                    paths.add(tuple(f_path))
        print("Only one entity, return all paths")
        return list(paths)
    
    combination_path_dict = {}
    total_paths = None
    
    for i in range(len(entity_list) - 1):
        start_entity = entity_list[i]
        target_entity = entity_list[i+1]
        start_entity_paths = paths_dict[start_entity]
        target_entity_paths = paths_dict[target_entity]
        intersection = set(start_entity_paths.keys()) & set(target_entity_paths.keys())
        if not intersection:
            return paths_dict[start_entity] + paths_dict[target_entity]
        temp_paths = []
        for node in intersection:
            temp_paths.extend(process_node(start_entity_paths, target_entity_paths, node))
        if total_paths is None:
            total_paths = temp_paths
        else:
            combined_paths = []
            for path1 in total_paths:
                for path2 in temp_paths:
                    if path1[-1] == path2[0]:
                        combined_paths.append(path1 + path2[1:])
            total_paths = combined_paths
            if not total_paths:
                return []

    return total_paths



def combine_all_paths(combination_path_dict, entity_list):
    # Start with the paths between the first pair
    
    total_paths = combination_path_dict.get((entity_list[0], entity_list[1]), [])
    for i in range(2, len(entity_list)):
        next_paths = combination_path_dict.get((entity_list[i - 1], entity_list[i]), [])
        combined_paths = []
        for path1 in total_paths:
            for path2 in next_paths:
                # Ensure the paths can be connected
                if path1[-1] == path2[0]:
                    # Avoid duplicate nodes
                    combined_path = path1 + path2[1:]
                    combined_paths.append(combined_path)
        if not combined_paths:
            return []
        total_paths = combined_paths
    return total_paths


def add_relations_to_path1(graph, path):
    """Add relation information to a completed path."""
    full_path = []
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        relations_dict = None
        if node not in graph or next_node not in graph[node]:
            # print(f"KeyError: No relation found between {node} and {next_node}")
            if next_node in graph and node in graph[next_node]:
                relations_dict = graph[next_node][node]
            else:
                relations_dict = {}
                relations_dict['forward'] = set()
                relations_dict['forward'].add("unnamed relation")
                # graph[node][next_node] = relations_dict
        if relations_dict is None:
            relations_dict = graph[node][next_node]
        relation_strings = []
        for direction, relations in relations_dict.items():
            direction_symbol = " ->" if direction == 'forward' else " <-"
            if isinstance(relations, set):
                relations = list(relations) 
            for relation in relations:
                relation_strings.append(f"{direction_symbol} {relation} {direction_symbol}")
        relation_strings.sort()
        top1 = relation_strings[0]
        relation_string = "{" + (top1) + "}"

        # relation_string = "{" + ", ".join(relation_strings) + "}"
        full_path.append(node)
        full_path.append(relation_string)
    full_path.append(path[-1])
    return full_path

def add_relations_to_path_with_all_R(graph, path):
    """Add all relation information to a completed path, generating different paths accordingly."""
    import itertools

    # Build a list of possible relations between each pair of nodes
    relations_list = []
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        try:
            relations_dict = graph[node][next_node]
            # relations_dict = graph[node][next_node]
            relation_strings = []
            for direction, relations in relations_dict.items():
                direction_symbol = " ->" if direction == 'forward' else " <-"
                if isinstance(relations, set):
                    relations = list(relations)
                for relation in relations:
                    relation_strings.append(f"{direction_symbol} {relation} {direction_symbol}")
        except KeyError:
            # print(f"KeyError: No relation found between {node} and {next_node}")
            relation_strings = ["{" + "->" + "}"]
            # continue
        relations_list.append(relation_strings)

    # Generate all combinations of relations
    relation_combinations = list(itertools.product(*relations_list))

    # For each combination, build the full path
    paths = []
    for combination in relation_combinations:
        full_path = []
        for i in range(len(path) - 1):
            node = path[i]
            relation_string = "{" + combination[i] + "}"
            full_path.append(node)
            full_path.append(relation_string)
        full_path.append(path[-1])
        paths.append(full_path)
    return paths


def task(graph, entity_list, hop):
    if start in graph and end in graph:
        raw_paths = bfs_with_intersection(graph, entity_list, hop)
        return [add_relations_to_path1(graph, path) for path in raw_paths]
    return []



def expand_node(node, path, graph):
    """扩展给定节点，返回所有可能的下一步"""
    expansions = []
    for next_node, relations_dict in graph[node].items():
        if next_node not in path:  # 防止环路
            new_path = path + [next_node]
            expansions.append((next_node, new_path))
    return expansions



def create_relation_strings(relations_dict, reverse=False):
    relation_strings = []
    for direction, relations in relations_dict.items():
        direction_symbol = " ->" if direction == 'forward' else " <-"
        if reverse:
            direction_symbol = direction_symbol[::-1]  # Reverse the arrow directions
        for relation in set(relations):
            relation_strings.append(f"{direction_symbol} {relation} {direction_symbol}")
    relation_strings.sort()
    return "{" + ", ".join(relation_strings) + "}"

def merge_paths(graph, path_from_start, path_from_goal, is_direct_meet):
    # Ensure that the last element of path_from_start and the first element of path_from_goal are nodes
    last_node_start = path_from_start[-2] if len(path_from_start) > 1 else path_from_start[0]
    first_node_goal = path_from_goal[-2] if len(path_from_goal) > 1 else path_from_goal[0]
    
    if is_direct_meet:
        return path_from_start[:-1] + path_from_goal[::-1]

    try:
        middle_relation = create_relation_strings(graph[last_node_start][first_node_goal], reverse=True)
        return path_from_start + [middle_relation] + path_from_goal[::-1][1:]
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        return []  # Return an empty list or handle the error as appropriate for your application



def merge_paths_by_relations(paths):
    from collections import defaultdict
    import itertools

    # Organize paths by their relation sequences
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Merge paths with the same relation sequences
    merged_paths = []
    for relations, paths in paths_by_relations.items():
        # This will hold the final merged path
        merged_path = []
        # We know all paths have the same length and relations, so we iterate through entities
        for i in range(0, len(paths[0]), 2):  # Only iterate over entity indices
            # Gather all entities at this position across all paths
            entities = {path[i] for path in paths}
            if len(entities) > 1:
                merged_entity = "{" + ", ".join(sorted(set(entities))) + "}"
            else:
                merged_entity = list(entities)[0]  # Just take the single entity
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:  # Add the relation if it's not the last element
                merged_path.append(paths[0][i+1])

        merged_paths.append(merged_path)

    return merged_paths



def merge_paths_custom_format(paths, intersection_nodes):
    from collections import defaultdict

    # First, separate paths by intersection node presence and collect segments after the first intersection node
    segments_by_intersection = defaultdict(list)
    for path in paths:
        for idx, node in enumerate(path):
            if node in intersection_nodes:
                before = tuple(path[:idx+1])  # Segment up to and including intersection node
                after = tuple(path[idx+1:])  # Everything after intersection node
                segments_by_intersection[before].append(after)
                break

    # Prepare to merge and format the paths
    merged_paths = []
    for before, afters in segments_by_intersection.items():
        merged_path = list(before)  # Start with the segment before the intersection node, excluding it
        # Handle multiple different segments after the intersection
        for after in afters:
            if after:  # Ensure there is a segment to process
                merged_path += ['{'+'AND}', before[-1]] + list(after)  # Include 'AND', the intersection node, and the segment

        # Remove the first 'AND' for proper formatting
        if merged_path[0] == '{'+'AND}':
            merged_path = merged_path[2:]

        merged_paths.append(merged_path)

    return merged_paths

def merge_and_format_paths(paths, intersection_nodes):
    from collections import defaultdict

    # First, merge by relations
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Initial merge based on relation sequences
    preliminary_merged_paths = []
    for relations, paths in paths_by_relations.items():
        merged_path = []
        for i in range(0, len(paths[0]), 2):  # Iterate over entity indices
            entities = {path[i] for path in paths}
            merged_entity = "{" + ", ".join(sorted(entities)) + "}" if len(entities) > 1 else list(entities)[0]
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:
                merged_path.append(paths[0][i+1])
        preliminary_merged_paths.append(merged_path)

    # Then, merge by intersection and format
    segments_by_intersection = defaultdict(list)
    for path in preliminary_merged_paths:
        for idx, node in enumerate(path):
            if node.strip('{}').split(', ')[0] in intersection_nodes:  # Adjusting for merged entities
                before = tuple(path[:idx+1])
                after = tuple(path[idx+1:])
                segments_by_intersection[before].append(after)
                break

    # Final merging and formatting
    final_merged_paths = []
    for before, afters in segments_by_intersection.items():
        merged_path = list(before)  # Exclude the intersection node initially
        for after in afters:
            if after:  # Ensure there is something to process
                merged_path += ['{'+'AND}', before[-1]] + list(after)
        if merged_path[0] == '{'+'AND}':
            merged_path = merged_path[2:]
        final_merged_paths.append(merged_path)

    return final_merged_paths




def merge_and_format_paths_segmented(paths, intersection_nodes, main_entities):
    from collections import defaultdict

    # First, merge by relations
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Initial merge based on relation sequences
    preliminary_merged_paths = []
    for relations, paths in paths_by_relations.items():
        merged_path = []
        for i in range(0, len(paths[0]), 2):  # Iterate over entity indices
            entities = {path[i] for path in paths}
            merged_entity = "{" + ", ".join(sorted(entities)) + "}" if len(entities) > 1 else list(entities)[0]
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:
                merged_path.append(paths[0][i+1])
        preliminary_merged_paths.append(merged_path)

    # Then, merge by intersection and format
    segments_by_intersection = defaultdict(list)
    for path in preliminary_merged_paths:
        for idx, node in enumerate(path):
            if node.strip('{}').split(', ')[0] in intersection_nodes:  # Adjusting for merged entities
                before = tuple(path[:idx+1])
                after = tuple(path[idx+1:])
                segments_by_intersection[before].append(after)
                break

    # Final merging and formatting
    final_merged_paths = []
    for before, afters in segments_by_intersection.items():
        merged_path = list(before)  # Exclude the intersection node initially
        initial_entities = set(before[::2])  # Capture initial entities in the path before the intersection
        for after in afters:
            # Append only main entities not in the initial segment
            filtered_entities = [ent for ent in main_entities if ent not in initial_entities and ent in after]
            if after:  # Ensure there is something to process
                merged_path += ['{'+'AND}', before[-1]]  # Add 'AND' and the intersection node
                for ent in filtered_entities:
                    # Append each entity exactly once along with its associated relations if any
                    ent_idx = after.index(ent)
                    if ent_idx < len(after) - 1:  # Ensure there is a relation to follow
                        merged_path += [ent, after[ent_idx + 1]]
        final_merged_paths.append(merged_path)

    return final_merged_paths

from collections import deque
from concurrent.futures import ThreadPoolExecutor


from collections import deque







def extract_first_ten_words(text):
    # 将字符串按空格分割成单词列表
    words = text.split()
    
    # 提取前10个单词
    first_ten_words = words[:10]
    
    # 将单词列表重新组合成字符串
    return ' '.join(first_ten_words)
def format_paths_to_natural_language_id_with_name(paths, entity_id_to_name, version =1, without_entity_id = True):
    natural_language_paths = []
    # print("version", version)

    for path in paths:
        formatted_path = []
        # print(type(path))    

        for i, element in enumerate(path):
            if i % 2 == 0:  # Assuming even indices are entities and odd are relations
                try:
                    # print(element)
                    # print(type(element))    
                    if element.startswith('{'):

                        entities = element.strip('{}').split(', ')
                        formatted_entities = []
                        for e in entities[:20]:
                            if version == 2:
                                # entity_name = entity_id_to_name[element] if element in entity_id_to_name else id2entity_name_or_type(element)

                                entity_name = entity_id_to_name.get(e.strip(), id2entity_name_or_type(e.strip()))
                            else:
                                entity_name = entity_id_to_name.get(e.strip())
                            # formatted_entities.append(e.strip() + ": " + extract_first_ten_words(entity_name))
                            if without_entity_id:
                                formatted_entities.append(extract_first_ten_words(entity_name))
                            else:
                                formatted_entities.append(e.strip() + ": " + extract_first_ten_words(entity_name))
                        # Limiting to first 5 unique entities if more than 5 are present
                        formatted_entities = list(set(formatted_entities))
                        formatted_path.append("{" + ", ".join(formatted_entities) + "}")
                    else:
                        # Single entity handling
                        # entity_name = entity_id_to_name.get(element, id2entity_name_or_type(element))
                        # print("element 2")
                        if version == 2:
                            # entity_name = entity_id_to_name[element] if element in entity_id_to_name else id2entity_name_or_type(element)

                            entity_name = entity_id_to_name.get(element, id2entity_name_or_type(element))
                        else:
                            entity_name = entity_id_to_name.get(element)
                        if without_entity_id:
                            formatted_path.append("{"  + extract_first_ten_words(entity_name) + "}")
                        else:
                            formatted_path.append("{" + element + ": " + extract_first_ten_words(entity_name) + "}")
                except:
                    print(type(element))
                    print(path)
                    print(f"KeyError encountered for element {element}")
                    exit()
            else:
                # Adding relation as is
                formatted_path.append(element)
        # Creating a readable natural language path
        natural_language = " - ".join(formatted_path)
        natural_language_paths.append(natural_language)

    return natural_language_paths


def merge_paths_by_relations_remove_usless(paths):
    from collections import defaultdict
    import itertools

    # Organize paths by their relation sequences
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Merge paths with the same relation sequences
    merged_paths = []
    for relations, paths in paths_by_relations.items():
        # This will hold the final merged path
        merged_path = []
        # We know all paths have the same length and relations, so we iterate through entities
        for i in range(0, len(paths[0]), 2):  # Only iterate over entity indices
            # Gather all entities at this position across all paths
            entities = {path[i] for path in paths}
            if len(entities) > 1:

                merged_entity = "{" + ", ".join(sorted(set(entities))) + "}"

            else:
                merged_entity = list(entities)[0]  # Just take the single entity
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:  # Add the relation if it's not the last element
                merged_path.append(paths[0][i+1])

        merged_paths.append(merged_path)

    return merged_paths



def find_1_hop_relations_and_entities(entity, graph,entity_id_to_name, ifusing_all_R):
    # results = search_relations_and_entities_combined(entity)

    all_path = []
    for r_entity in graph[entity]:
        # continue
        if ifusing_all_R:


            path = add_relations_to_path_with_all_R(graph, [entity, r_entity])   
            all_path.extend(path)

        else:
            path = add_relations_to_path1(graph, [entity, r_entity])
            all_path.append(path)

    merge_path = merge_paths_by_relations_remove_usless(all_path)
    new_nl_related_paths = format_paths_to_natural_language_id_with_name(merge_path,entity_id_to_name)

    # if id2entity_name_or_type(entity) == "Unnamed Entity":
    #     format_paths_to_natural_language_new_parall(merge_path)
    # new_nl_related_paths1, entity_id_to_name = format_paths_to_natural_language_new_parall_remove_nonname(merge_path, entity_id_to_name)


    return new_nl_related_paths





def explore_graph_from_one_topic_entities(current_entities, graph, entity_names, exlored_entities,all_entities):
    # all_entities = set(entity_ids)
    storage_lock = Lock()  # 创建线程安全锁
    print(f"Exploring entities ...")
    start = time.time()
    new_entities = set()

    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}
        exlored_entities.update(current_entities)
        for future in as_completed(futures):
            results = future.result()
            entity = futures[future]
            for result in results:
                relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']
                if connected_entity.startswith("m."):
                    if connected_entity in exlored_entities:
                        continue
                    with storage_lock:
                        # 更新或添加实体名称
                        entity_names[connected_entity] = connected_name
                        # 确保图中包含相关实体和关系
                        if entity not in graph:
                            graph[entity] = {}
                        if connected_entity not in graph:
                            graph[connected_entity] = {}
                        if connected_entity not in graph[entity]:
                            graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                        if entity not in graph[connected_entity]:
                            graph[connected_entity][entity] = {'forward': set(), 'backward': set()}
                        # 更新关系
                        if direction == "tail":
                            graph[entity][connected_entity]['forward'].add(relation)
                            graph[connected_entity][entity]['backward'].add(relation)
                        else:  # direction is "head"
                            graph[entity][connected_entity]['backward'].add(relation)
                            graph[connected_entity][entity]['forward'].add(relation)
                    new_entities.add(connected_entity)


        new_entities.difference_update(exlored_entities)
        all_entities.update(new_entities)
        current_entities = new_entities
        # print ((all_entities))
        # print((exlored_entities))
        # print ((current_entities))

    # print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (graph, all_entities, exlored_entities, current_entities, entity_names)



# Ensure you have proper implementations for search_relations_and_entities_combined and are_entities_connected
def extract_brace_contents(path):
    """
    提取路径中所有大括号内的内容。
    """
    return re.findall(r'\{([^}]+)\}', path)

def concatenate_paths_with_unlinked(list1, list2):

    list2_dict = {}
    for path2 in list2:
        braces2 = extract_brace_contents(path2)
        if braces2:
            first_brace2 = braces2[0]
            if first_brace2 not in list2_dict:
                list2_dict[first_brace2] = []
            list2_dict[first_brace2].append(path2)
    
    concatenated_paths = []
    linked_list1 = set()
    linked_list2 = set()
    
    for idx1, path1 in enumerate(list1):
        braces1 = extract_brace_contents(path1)
        if braces1:
            last_brace1 = braces1[-1]
            if last_brace1 in list2_dict:
                for path2 in list2_dict[last_brace1]:
                    braces2 = extract_brace_contents(path2)
                    if braces2:
                        concatenated_braces = braces1 + braces2[1:]
                        concatenated_path = " - ".join(f'{{{brace}}}' for brace in concatenated_braces)
                        concatenated_paths.append(concatenated_path)
                        linked_list1.add(idx1)
                        linked_list2.add(list2.index(path2))
    
    unlinked_paths = []
    
    for idx1, path1 in enumerate(list1):
        if idx1 not in linked_list1:
            unlinked_paths.append(path1)
    
    for idx2, path2 in enumerate(list2):
        if idx2 not in linked_list2:
            unlinked_paths.append(path2)
    
    result = concatenated_paths + unlinked_paths
    
    return result



def check_answerlist(dataset_name, question_string, ori_question, ground_truth_datas, origin_data):
    answer_list= []
    if dataset_name == 'cwq':
        answer_list.append(origin_data["answer"])

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    elif dataset_name == 'hotpot':
        answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
    elif dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)
    return list(set(answer_list))

def check_answer(answer, answer_list):
    if not answer or not answer["LLM_answer"]:
        return False

    lower_answer = answer["LLM_answer"].strip().replace(" ","").lower()
    getanswer = clean_results(lower_answer)
    for answer_name in answer_list:
        lower_answer_name = answer_name.strip().replace(" ","").lower()
        if lower_answer_name in lower_answer:
            return True

    if len(getanswer) > 0:
        for getanswer_e in getanswer:
            for answer_name in answer_list:
                lower_answer_name = answer_name.strip().replace(" ","").lower()
                if getanswer_e in lower_answer_name:
                    return True
    return False


def clean_results(string):

    if "answer:{" not in string:
        return []
    # else:   
        # print("+++==========++++++++++")
    sections = string.split('answer:')
    
    all_answers = []
    
    for section in sections[1:]:  # Skip the first part since it doesn't contain an answer

        string = section.split("\n")[0]
        replace_string = string.replace("{",",").replace("}",",")

        answers_list = [answer.strip() for answer in replace_string.split(',')]
        # Add to the overall list
        all_answers.extend(answers_list)
    
    # Remove duplicates and return the final list
    all_answers = list(set(all_answers))
    all_answers = [x for x in all_answers if x != ""]
    return list(set(all_answers))





def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False




def _add_edge(g, src, dst, rel, direction):
    if src not in g:
        g[src] = {}
    if dst not in g[src]:
        g[src][dst] = {"forward": set(), "backward": set()}
    g[src][dst][direction].add(rel)

def _collect_all_edges(graph):
    edges = []
    for src, nbrs in graph.items():
        for dst, dirs in nbrs.items():
            for d in ("forward", "backward"):
                for rel in dirs.get(d, ()):
                    edges.append((src, dst, rel, d))
    return edges

def sample_graph_edges(original_graph, keep_ratio, seed=None):
    """
    随机保留 keep_ratio 比例的边，直接“搭”出新图，原图不变。
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio 必须在 (0,1] 之间")

    rng = random.Random(seed)
    edges = _collect_all_edges(original_graph)
    if not edges:
        return {}

    keep_num = max(1, int(len(edges) * keep_ratio))
    kept_edges = rng.sample(edges, keep_num)

    new_graph = defaultdict(dict)
    for src, dst, rel, d in kept_edges:
        _add_edge(new_graph, src, dst, rel, d)
        mirror = "backward" if d == "forward" else "forward"
        _add_edge(new_graph, dst, src, rel, mirror)

    return dict(new_graph)



def build_incomplete_graph(graph, ratio = 0.8, seeds=42):

    g80 = sample_graph_edges(graph, ratio, seeds)

    return g80


from subgraph_utilts import *
def load_and_check_subgraph(question, question_id,subgraph_db, Global_depth, NL_subgraph_db, question_string, data, topic_entity, build_imcomplete_graph = False, ratio = 0.8):
    
    subgraph_dict = load_from_large_db(subgraph_db, question_id)
    sub_cap = False
    total_id_to_name_dict = None
    graph = None
    NL_subgraph = load_from_large_db(NL_subgraph_db, question_id)

    if subgraph_dict and NL_subgraph:
        print("Data found in the database.")
        print(subgraph_dict.keys())
        
        graph = subgraph_dict['subgraph']
        all_entities = subgraph_dict['all_entities']
        depth = subgraph_dict['hop']
        outter_entity = subgraph_dict['outter_entity']
        if "sub_cap" in subgraph_dict:
            sub_cap = subgraph_dict['sub_cap']
        if NL_subgraph:
            total_id_to_name_dict = NL_subgraph['total_id_to_name_dict']
            NL_name_set = NL_subgraph['NL_name_set']
    else:
        print("Data not found in the database. Exploring the graph...")
        sub_cap, graph, depth,all_entities, outter_entity, dict, if_inside = explore_graph_from_entities_by_hop_neighbor_1(list(topic_entity), Global_depth)

        total_id_to_name_dict = dict
        NL_name_set = set(dict.values())
        
        delete_data_by_question_id(NL_subgraph_db, question_id)
        NL_subgraph= {
            "total_id_to_name_dict": total_id_to_name_dict,
            "NL_name_set": NL_name_set
        }
        save_to_large_db(NL_subgraph_db, question_id, NL_subgraph)
        NL_subgraph = None

        subgraph_dict = {
            "question": question,
            "machine_question": data[question_string],
            "question_id": question_id,
            "topic_entity": topic_entity,
            "hop": depth,
            "subgraph": graph,
            "all_entities": all_entities,
            "outter_entity": outter_entity,
            "sub_cap": sub_cap
        }
        delete_data_by_question_id(subgraph_db, question_id)
        save_to_large_db(subgraph_db, question_id, subgraph_dict)
    using_graph_reduction = True
    if build_imcomplete_graph:
        print("graph nodes number before incomplete:", len(graph.keys()))

        new_graph = build_incomplete_graph(graph, ratio, seeds=42)
        del(graph)
        gc.collect()
        graph = new_graph
        print("graph nodes number after incomplete:", len(graph.keys()))

    if using_graph_reduction:
        test_depth = 3
        # while test_depth <= 3:
        intersection = set(bfs_with_intersection_only(graph,  list(topic_entity), test_depth))
        
        intersection_new = set()
        itersection_name = set()
        if total_id_to_name_dict is None:
            raise ValueError("total_id_to_name_dict is None; "
                        "load_and_check_subgraph did not return a mapping.")

        for i in intersection:
            name = total_id_to_name_dict.get(i)
            if name is None:   
                print("not found inside the entity dict")       # identifier not found in the mapping
                continue
            itersection_name.add(total_id_to_name_dict[i])
            intersection_new.add(i)

        intersection = intersection_new
        if len(intersection) > 0:
            # print("no intersection")
            print("graph nodes number before:", len(graph.keys()))
            reduced_graph, reduced_name_dict = create_subgraph_through_intersections(graph, list(topic_entity), intersection, total_id_to_name_dict,test_depth)
            del(graph)
            del(total_id_to_name_dict)
            gc.collect()
            total_id_to_name_dict = reduced_name_dict
            graph = reduced_graph
            print("graph nodes number after:", len(graph.keys()))
            # else:
            if len(graph) > 1000000:
                print("graph nodes number is too large, we need to reduce it")
                if test_depth >1:
                    test_depth =  test_depth -1
                    intersection = set(bfs_with_intersection_only(graph,  list(topic_entity), test_depth))
        
                    intersection_new = set()
                    itersection_name = set()
                    for i in intersection:
                        itersection_name.add(total_id_to_name_dict[i])
                        intersection_new.add(i)
                    if len(intersection) > 0:
                        if len(intersection_new) > 0:
                            del(intersection)
                            print("answer is still in the intersection")
                            intersection = intersection_new
                        else:
                            del(intersection_new)
                        reduced_graph, reduced_name_dict = create_subgraph_through_intersections(graph, list(topic_entity), intersection, total_id_to_name_dict,test_depth)
                        
                        del(graph)
                        del(total_id_to_name_dict)
                        gc.collect()
                        total_id_to_name_dict = reduced_name_dict
                        graph = reduced_graph
                        subgraph_dict["subgraph"] = graph
                        NL_name_set = set(total_id_to_name_dict.values())
                        NL_subgraph= {
                            "total_id_to_name_dict": total_id_to_name_dict,
                            "NL_name_set": NL_name_set
                        }
            return graph, total_id_to_name_dict
            
        else:
            print("no intersection")
    return graph, total_id_to_name_dict

    # return graph, total_id_to_name_dict



def wiki_load_and_check_subgraph(question, question_id,subgraph_db, Global_depth, NL_subgraph_db, question_string, data, topic_entity, wiki_client, build_imcomplete_graph = False, ratio = 0.8):
    
    subgraph_dict = load_from_large_db(subgraph_db, question_id)
    sub_cap = False
    total_id_to_name_dict = None
    graph = None

    if subgraph_dict:
        print("Data found in the database.")
        print(subgraph_dict.keys())
        
        graph = subgraph_dict['subgraph']
        all_entities = subgraph_dict['all_entities']
        depth = subgraph_dict['hop']
        outter_entity = subgraph_dict['outter_entity']
        if "sub_cap" in subgraph_dict:
            sub_cap = subgraph_dict['sub_cap']
        NL_subgraph = load_from_large_db(NL_subgraph_db, question_id)
        if NL_subgraph:
            total_id_to_name_dict = NL_subgraph['total_id_to_name_dict']
            NL_name_set = NL_subgraph['NL_name_set']

        
    else:
        print("Data not found in the database. Exploring the graph...")
        sub_cap, graph, depth,all_entities, outter_entity, dict, if_inside = wiki_explore_graph_from_entities_by_hop_neighbor_1(topic_entity, wiki_client, Global_depth)

        total_id_to_name_dict = dict
        NL_name_set = set(dict.values())
        
        delete_data_by_question_id(NL_subgraph_db, question_id)
        NL_subgraph= {
            "total_id_to_name_dict": total_id_to_name_dict,
            "NL_name_set": NL_name_set
        }
        save_to_large_db(NL_subgraph_db, question_id, NL_subgraph)
        NL_subgraph = None

        subgraph_dict = {
            "question": question,
            "machine_question": data[question_string],
            "question_id": question_id,
            "topic_entity": topic_entity,
            "hop": depth,
            "subgraph": graph,
            "all_entities": all_entities,
            "outter_entity": outter_entity,
            "sub_cap": sub_cap
        }
        delete_data_by_question_id(subgraph_db, question_id)
        save_to_large_db(subgraph_db, question_id, subgraph_dict)
    using_graph_reduction = True
    if build_imcomplete_graph:
        new_graph = build_incomplete_graph(graph, ratio, seeds=42)
        del(graph)
        gc.collect()
        graph = new_graph
    if using_graph_reduction:
        test_depth = 3
        # while test_depth <= 3:
        intersection = set(bfs_with_intersection_only(graph,  list(topic_entity), test_depth))
        
        intersection_new = set()
        itersection_name = set()
        if total_id_to_name_dict is None:
            raise ValueError("total_id_to_name_dict is None; "
                        "load_and_check_subgraph did not return a mapping.")

        for i in intersection:
            name = total_id_to_name_dict.get(i)
            if name is None:   
                print("not found inside the entity dict")       # identifier not found in the mapping
                continue
        for i in intersection:
            itersection_name.add(total_id_to_name_dict[i])
            intersection_new.add(i)

        intersection = intersection_new
        if len(intersection) > 0:
            # print("no intersection")
            print("graph nodes number before:", len(graph.keys()))
            reduced_graph, reduced_name_dict = create_subgraph_through_intersections(graph, list(topic_entity), intersection, total_id_to_name_dict,test_depth)
            
            del(graph)
            del(total_id_to_name_dict)
            gc.collect()
            total_id_to_name_dict = reduced_name_dict
            graph = reduced_graph
            print("graph nodes number after:", len(graph.keys()))

            # break
            if len(graph) > 1000000:
                print("graph nodes number is too large, we need to reduce it")
                if test_depth >1:
                    test_depth =  test_depth -1
                    intersection = set(bfs_with_intersection_only(graph,  list(topic_entity), test_depth))
        
                    intersection_new = set()
                    itersection_name = set()
                    for i in intersection:
                        itersection_name.add(total_id_to_name_dict[i])
                        intersection_new.add(i)

                    if len(intersection) > 0:
                        if len(intersection_new) > 0:
                            del(intersection)
                            print("answer is still in the intersection")
                            intersection = intersection_new
                        else:
                            del(intersection_new)
                        reduced_graph, reduced_name_dict = create_subgraph_through_intersections(graph, list(topic_entity), intersection, total_id_to_name_dict,test_depth)
                        del(graph)
                        del(total_id_to_name_dict)
                        gc.collect()
                        total_id_to_name_dict = reduced_name_dict
                        graph = reduced_graph
                        subgraph_dict["subgraph"] = graph
                        NL_name_set = set(total_id_to_name_dict.values())
                        NL_subgraph= {
                            "total_id_to_name_dict": total_id_to_name_dict,
                            "NL_name_set": NL_name_set
                        }
            return graph, total_id_to_name_dict
        else:
            print("no intersection")
        

    return graph, total_id_to_name_dict



