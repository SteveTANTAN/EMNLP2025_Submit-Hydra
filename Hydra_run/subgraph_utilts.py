from tqdm import tqdm
import argparse
from utilts import *
import random
from cot_prompt_list import *
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
from multiprocessing import Process, Queue
import sqlite3
import time
import pickle

def entity_need_explore(topic_entity,path_list, high_sub_entities):
    """从subgraph_data中根据高相似实体找到相关路径"""
    # 构建一个字典以存储每个实体的最大相似度
    # print(path_list)

    entity_similarity = defaultdict(float)
    for entity1, entity2, similarity in high_sub_entities:
        if entity1 == entity2:
            entity_similarity[entity1] = 1.1
        entity_similarity[entity1] = max(entity_similarity[entity1], similarity)
        # entity_similarity[entity2] = max(entity_similarity[entity2], similarity)

    need_to_find = []
    for entity, similarity in sorted(entity_similarity.items(), key=lambda x: x[1], reverse=True):
        # print(entity)
        if entity in topic_entity.values():
            continue
        need_to_find.append(entity)
    return need_to_find


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def find_subgraph_entities(data, question_id):
    """ 根据question_id在subgraph数据中查找实体 """
    for item in data:
        if item['question_id'] == question_id:
            entities = set()
            if 'NL_subgraph' in item:
                for key, values in item['NL_subgraph'].items():
                    head_entity = key.split(':')[1].strip().split(',')[0]
                    entities.add(head_entity)
                    for value in values:
                        tail_entity = value.split(':')[1].strip()
                        entities.add(tail_entity)
            print("number of NL_entities:", len(entities))
            # print("number of all entity:", len())
            
            return list(entities)
    return []

def extract_main_entity(data, question_id):
    """ 从答案中提取对应question_id的主实体 """
    for item in data:
        if item['ID'] == question_id:
            return item['entities']
            start_idx = answer_text.find('{') + 1
            end_idx = answer_text.find('}')
            if start_idx > 0 and end_idx > 0:
                return answer_text[start_idx:end_idx]
    return None

def calculate_cosine_similarity(list1, list2, value=0.5):
    """计算两个实体列表之间的余弦相似度"""
    if not list1 or not list2:
        return []
    vectorizer = TfidfVectorizer()
    all_entities = list(list1) + list(list2)
    tfidf_matrix = vectorizer.fit_transform(all_entities)
    
    list1_vecs = tfidf_matrix[:len(list1)]
    list2_vecs = tfidf_matrix[len(list1):]
    cosine_sim = cosine_similarity(list1_vecs, list2_vecs)
    
    similar_entities = []
    for i, entity1 in enumerate(list1):
        for j, entity2 in enumerate(list2):
            if cosine_sim[i, j] > value:  # 设置相似度阈值为0.5
                similar_entities.append((entity1, entity2, cosine_sim[i, j]))
                # print(f"高相似度: {entity1} -> {entity2} ({cosine_sim[i, j]})")
    return similar_entities

def compress_path(path):
    """压缩路径中相同实体之间的不同关系"""
    segments = path.split(" -> ")
    compressed_path = []
    i = 0
    while i < len(segments):
        if i + 2 < len(segments) and segments[i] == segments[i + 2]:
            relations = [segments[i + 1]]
            j = i + 3
            while j < len(segments) and segments[j] == segments[i]:
                relations.append(segments[j - 1])
                j += 2
            compressed_path.append(f"{segments[i]} -> {{{', '.join(relations)}}} -> {segments[j - 1]}")
            i = j
        else:
            compressed_path.append(segments[i])
            i += 1
    return " -> ".join(compressed_path)

def find_related_paths(subgraph_data, question_id, high_sub_entities):
    """从subgraph_data中根据高相似实体找到相关路径"""
    # 构建一个字典以存储每个实体的最大相似度
    Q_enrities = question_entity(question_id)
    print(Q_enrities)
    entity_similarity = defaultdict(float)
    for entity1, entity2, similarity in high_sub_entities:
        entity_similarity[entity1] = max(entity_similarity[entity1], similarity)
        # entity_similarity[entity2] = max(entity_similarity[entity2], similarity)
    
    for item in subgraph_data:
        if item['question_id'] == question_id:
            if 'NL_path' in item:
                related_paths = []
                for entity, similarity in sorted(entity_similarity.items(), key=lambda x: x[1], reverse=True):
                    if entity not in Q_enrities.values():
                    
                        for path in item['NL_path']:
                        # print(path)
                            if entity in path:
                                print("entity: " + str(entity) + "; similarity" + str(similarity))
                                # compressed_path = compress_path(path)
                                # related_paths.append((compressed_path, similarity))
                                # print("Compressed Path: " + compressed_path)
                                related_paths.append(path)
                                print("Path: " + path)

                                # break
                # 按相似度从高到低排序路径
                # related_paths.sort(key=lambda x: x[1], reverse=True)
                # return [path for path, _ in related_paths]
                return path
    return []

def generate_kg_path(question, search_text):
    # """调用 OpenAI API 生成知识图谱路径（KG Path）。"""
    prompt = f"""
    {new_key_question_prompt}
    
    Question: {question}
    
    online search results:
    {search_text}
    
    A:
    """
    return run_LLM(prompt, "gpt-3",temperature=0.4)

# from utilts import bfs_with_intersection_only



if __name__ == '__main__':

    # 路径需要根据实际情况调整
    subgraph_data = cg_load_jsonl('subgraph/Final_Subgraph_cwq_multi_0.jsonl')
    answer_data = cg_load_jsonl('revise_CoT_GPT.jsonl')
    # exp_data = cg_load_jsonl("../data/cwq_multi.json")


    # 假设从答案中选取一个问题ID
    question_id = answer_data[0]['ID']  # 示例中使用列表的第一个元素
    subgraph_entities = find_subgraph_entities(subgraph_data, question_id)
    main_entities  = extract_main_entity(answer_data, question_id)
    # print("Subgraph Entities:", subgraph_entities)
    # print("Main Entities:", main_entities )

    # 计算并输出相似度
    high_sub_entities = calculate_cosine_similarity(subgraph_entities, main_entities)

    # 查找相关路径
    related_paths = find_related_paths(subgraph_data, question_id, high_sub_entities)
    print("Related Paths:", related_paths)

