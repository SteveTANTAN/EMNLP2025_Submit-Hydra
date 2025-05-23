from tqdm import tqdm
import math
import tiktoken
import argparse
from utilts import *
import random
from cot_prompt_list import *
from subgraph_utilts import *
from utilts2 import *
from hydra_main import *

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
import urllib3


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
import tiktoken
from typing import List, Dict

# ---------- 1. 计算 Chat 消息 token ----------
def num_tokens_from_messages(messages: List[Dict[str, str]], model: str) -> int:
    """
    依据 OpenAI Chat API 的格式统计整条 messages 的 token 数。
    代码改编自 openai‑cookbook 示例。
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 新型号若尚未内置，回退到 cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")

    # GPT‑3.5 / GPT‑4 系列使用同一条规则：
    tokens_per_message, tokens_per_name = 3, 1
    if model.startswith("gpt-3.5-turbo-0301"):
        tokens_per_message, tokens_per_name = 4, -1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message          # <im_start>{role}
        for key, value in message.items():
            # role / name / content
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name     # 角色被 name 取代
    num_tokens += 2  # 每个对话结尾的 <im_end>
    return num_tokens
def extract_top_list(text):
    """Extracts the top 3 paths from a given text.

    Args:
    text: The input text containing the top_list dictionary.

    Returns:
    A list of the top 3 paths.
    """

    # 找到 top_list 的内容
    # match = re.search(r'top_list:\{([^}]*)\}', text)
    match = re.search(r'ist:\s*\{([^}]+)\}', text)
    if not match:
        # match = re.search(r'list:\s*\{([^}]+)\}', text)
        return []
    
    top_list_str = match.group(1)
    
    # 提取所有的数字
    numbers = re.findall(r'\b\d+\b', top_list_str)
    
    # 转换为 int 并返回
    return list(map(int, numbers))


def extract_entities(text):
    # 使用正则表达式匹配 entities: 后面的花括号中的内容
    match = re.search(r"entities:\s*{([^}]*)}", text)
    if match:
        entities_str = match.group(1)
        # 分割字符串并去除多余的空格
        entities_list = [entity.strip() for entity in entities_str.split(",")]
        return entities_list
    return []


def extract_possible_entities(text):
    # 使用正则表达式匹配 entities: 后面的花括号中的内容
    # 修改正则表达式以匹配花括号或双引号内的全部内容
    # 更新正则表达式以确保能匹配包括花括号和逗号在内的所有内容
    print ("planing extract_possible_entities")
    pattern = r'edicted:\s*(?:"(.*?)"|{(.*?)})'
    matches = re.findall(pattern, text)
    
    results = []
    for match in matches:
        # 处理从正则表达式捕获的每个组
        combined = ' '.join([m.strip() for m in match if m])
        # 处理中英文逗号作为分隔符
        # split_items = [item.strip() for item in re.split(r'[,]', combined)]
        entities_list = [entity.strip() for entity in combined.split(",")]
        results.extend(entities_list)
        
    print ("extract_possible_entities", results)
    return results


def extract_unique_entities_from_backet(text):
    # 使用正则表达式提取所有 {} 中的内容
    entities = re.findall(r'\{(.*?)\}', text)
    
    # 使用 set 去重，然后转换回 list
    unique_entities = list(set(entities))
    
    return unique_entities
# # 示例用法
# text = "Answer: top_list:{Path:2, Path:1, Path:3} Explanation: ..."
# result = extract_top_list(text)
# print(result)  # 输出：['2', '1', '3']




def extract_split_questions(text):
    # 将文本按行分割
    lines = text.strip().split('\n')
    questions = []

    for line in lines:
        # 去除行中的所有空格
        line_no_spaces = line.replace(' ', '')
        # 检查行中是否包含 'split'（忽略大小写）
        if re.search(r'split', line_no_spaces, re.IGNORECASE):
            # 使用 ':' 分割，提取问题部分
            parts = line.split(':', 1)
            if len(parts) > 1:
                question = parts[1].strip()
                questions.append(question)
            else:
                # 如果没有 ':'，整个行作为问题
                questions.append(line.strip())

    return questions

def extract_entities_from_sentence(sentence):
    words = sentence.split()
    entities = []
    inside_entity = False
    current_entity = ''
    for word in words:
        if word.startswith('"') and word.endswith('"'):
            # 单词被双引号包围，直接提取
            entities.append(word.strip('"'))
        elif word.startswith('"'):
            # 实体开始
            inside_entity = True
            current_entity = word.lstrip('"') + ' '
        elif word.endswith('"'):
            # 实体结束
            current_entity += word.rstrip('"')
            entities.append(current_entity.strip())
            current_entity = ''
            inside_entity = False
        elif inside_entity:
            # 实体内部
            current_entity += word + ' '
    return entities
def extract_wiki_entities_from_strings(paths):
    # 创建一个字典来存储实体 ID 和名称
    entities_dict = {}

    # 正则表达式用于匹配实体
    entity_pattern = r'(Q\.\w+): ([^,}]+)'

    # 遍历每个路径字符串
    for path in paths:
        # 使用正则表达式找到所有匹配的实体
        entities = re.findall(entity_pattern, path)

        # 将实体添加到字典中
        for entity_id, name in entities:
            entities_dict[entity_id] = name.strip()

    return entities_dict
def find_top_similar_entities(
    entity_id_to_name,
    query_sentence,
    top_k=3,
    topic_exsiting=[],
    sbert_model=None,
    ner_pipeline=None,
    device='cuda'  # 新增参数
):
    """
    Finds the top_k entities from entity_id_to_name that are most similar to the topic entities extracted from query_sentence.

    Parameters:
    - entity_id_to_name (dict): A dictionary mapping entity IDs to entity names.
    - query_sentence (str): The input sentence containing the topic entities.
    - top_k (int): The number of top similar entities to retrieve (default is 3).
    - sbert_model (SentenceTransformer, optional): Pre-loaded SentenceTransformer model. If None, a default model will be loaded.
    - ner_pipeline (transformers.pipeline, optional): Pre-loaded NER pipeline. If None, a default English NER model will be loaded.
    - device (str or int): The device to run the models on. 'cuda' or 'cpu' or GPU index (default is 'cuda').

    Returns:
    - List[Tuple]: A list of tuples containing (entity_id, entity_name, similarity_score).
    """

    from transformers import pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    # Load models if not provided
    # if sbert_model is None:
    #     sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    # if ner_pipeline is None:
    #     if device == 'cuda':
    #         device_index = 0
    #     elif device == 'cpu':
    #         device_index = -1
    #     elif isinstance(device, int):
    #         device_index = device
    #     else:
    #         device_index = 0
    #     ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device_index)

    # Prepare data
    entity_names = list(entity_id_to_name.values())
    entity_ids = list(entity_id_to_name.keys())

    # Extract topic entities using NER
    # entities = ner_pipeline(query_sentence)

    topic_entities = []
    current_entity = ''
    current_label = ''

    # for entity in entities:
    #     word = entity['word']
    #     label = entity['entity_group']

    #     if label in ['ORG', 'LOC', 'PER', 'MISC']:  # 根据需要调整实体类型
    #         if word.startswith('##'):
    #             word = word[2:]
    #             current_entity += word
    #         else:
    #             if current_entity != '':
    #                 topic_entities.append(current_entity)
    #                 current_entity = ''
    #             current_entity = word
    #     else:
    #         if current_entity != '':
    #             topic_entities.append(current_entity)
    #             current_entity = ''

    # if current_entity != '':
    #     topic_entities.append(current_entity)
    # for i in topic_entities:
    #     for j in entity_names:
    #         if i in j:
    #             topic_entities.remove(i)
    #             break

    # parts = re.split(r'\s-\s(?!\d)', query_sentence)
    
    # # 去除每个部分前后的空白字符
    # parts = [part.strip().strip('"#/') for part in parts]
    
    # # 选择位于奇数位置的元素
    # topic_entities = [parts[i] for i in range(len(parts)) if i % 2 == 0]

    parts = re.split(r'\s-\s(?![^()]*\))', query_sentence)
    
    # 清理每个部分，移除引号和前后空白
    topic_entities = [re.sub(r'[“”"]', '', part).strip() for part in parts]

    for i in topic_entities:
        if i in topic_exsiting:
            topic_entities.remove(i)
    print("Extracted topic entities:", topic_entities)
    if len(topic_entities) == 0:
        return [],[]
    

    # 将实体名称和主题实体合并为语料库
    corpus = entity_names + topic_entities

    # 创建 TF-IDF 矢量化器
    vectorizer = TfidfVectorizer()

    # 拟合并转换语料库
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 分割 TF-IDF 矩阵为实体和主题实体的向量
    entity_tfidf = tfidf_matrix[:len(entity_names)]
    topic_tfidf = tfidf_matrix[len(entity_names):]

    # 计算主题实体向量的平均值
    topic_vector = topic_tfidf.mean(axis=0)

    # **修改这里，转换为 np.array**
    topic_vector = topic_vector.A1  # 或者使用 .toarray().ravel()

    # 计算实体向量与主题向量之间的余弦相似度
    similarities = cosine_similarity(entity_tfidf, topic_vector.reshape(1, -1))

    # 将相似度数组展平
    similarities = similarities.flatten()

    # 获取具有最高相似度分数的 top_k 索引
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # 汇总结果
    top_entities = []
    for idx in top_k_indices:
        entity_id = entity_ids[idx]
        entity_name = entity_names[idx]
        similarity_score = similarities[idx]
        if similarity_score > 0.8:
            if entity_name not in topic_exsiting:
                top_entities.append((entity_id, entity_name, similarity_score))

    return top_entities, topic_entities
    # # Encode entity names
    # entity_embeddings = sbert_model.encode(entity_names, batch_size=64, show_progress_bar=False)

    # # Encode topic entities
    # if len(topic_entities) > 0:
    #     topic_entity_embeddings = sbert_model.encode(topic_entities)
    #     topic_entity_embedding = np.mean(topic_entity_embeddings, axis=0)
    # else:
    #     print("No topic entities extracted. Using the entire query sentence for encoding.")
    #     topic_entity_embedding = sbert_model.encode([query_sentence])[0]

    # # Compute similarities
    # similarities = cosine_similarity([topic_entity_embedding], entity_embeddings)[0]

    # # Get top_k similar entities
    # top_k_indices = similarities.argsort()[-top_k:][::-1]

    # # Compile results
    # top_entities = []
    # for idx in top_k_indices:
    #     entity_id = entity_ids[idx]
    #     entity_name = entity_names[idx]
    #     similarity_score = similarities[idx]
    #     top_entities.append((entity_id, entity_name, similarity_score))

    # return top_entities
from typing import Tuple

def question_regen_help(text):
    """
    Extracts the query and cot strings from a block of text.

    Parameters
    ----------
    text : str
        The whole text that contains 'query: {...}' and 'cot: {...}'.

    Returns
    -------
    Tuple[str, str]
        A two‑item tuple: (query_string, cot_string). Empty strings are
        returned if a section is missing.
    """
    query_lines, cot_lines = [], []
    section = None          # 当前读取段落: None / "query" / "cot"

    for line in text.splitlines():
        # 判断是否进入新的段落
        if re.match(r'^\s*query\s*:', line, flags=re.I):
            section = "query"
            # 去掉前缀，只保留内容
            query_lines.append(re.sub(r'(?i)^\s*query\s*:\s*', '', line))
            continue
        if re.match(r'^\s*cot\s*:', line, flags=re.I):
            section = "cot"
            cot_lines.append(re.sub(r'(?i)^\s*cot\s*:\s*', '', line))
            continue

        # 如果正在读取某段，就把当前行加入对应列表
        if section == "query":
            query_lines.append(line)
        elif section == "cot":
            cot_lines.append(line)

    query_str = "\n".join(query_lines).strip()
    cot_str   = "\n".join(cot_lines).strip()
    query_str =query_str.replace("{", "")  # 将换行符替换为空格
    query_str = query_str.replace("}", "")  # 将换行符替换为空格
    return query_str, cot_str
# def question_regen_help(response_text):
    # # Helper function to remove leading/trailing curly braces if present.
    # def remove_braces(text):
    #     text = text.strip()
    #     if text.startswith('{') and text.endswith('}'):
    #         return text[1:-1].strip()
    #     return text
    # query = ""
    # raw_keywords = ""
    # # Regex for the line containing 'query:'
    # query_pattern = r"(?m)^query:\s*(.*)$"
    # # Regex for the line containing 'keywords:'
    # keywords_pattern = r"(?m)^CoT:\s*(.*)$"
    # if not keywords_pattern:
    #     keywords_pattern = r"(?m)^cot:\s*(.*)$"
    #     if not keywords_pattern:
    #         keywords_pattern = r"(?m)^cot*\d+: (.*)$"

        
    # # Search for the patterns in the response text

    # query_match = re.search(query_pattern, response_text)
    # keywords_match = re.search(keywords_pattern, response_text)

    # query = query_match.group(1).strip() if query_match else None
    # if query:
    #     query = remove_braces(query)

    # if keywords_match:
    #     raw_keywords = keywords_match.group(1)


    # return query, raw_keywords

def extract_cots_as_strings(text):
    # Use regular expression to find all occurrences of lines that start with "CoT" followed by a number and a colon.
    cot_patterns = re.findall(r'CoT*\d+: .*', text)
    if not cot_patterns:
        cot_patterns = re.findall(r'cot*\d+: .*', text)
        if not cot_patterns:
            cot_patterns = re.findall(r'cot: .*', text)
                
    # Initialize a list to hold the contents of each CoT found.
    cots = []
    # Iterate over each CoT pattern found in the text.
    for cot in cot_patterns:
        # Extract the entire line following "CoT<digit>: "
        cot_content = re.search(r'CoT*\d+: (.*)', cot)
        if not cot_content:
            cot_content = re.search(r'cot*\d+: (.*)', cot)
            if not cot_content:
                cot_content = re.search(r'cot: (.*)', cot)
        if cot_content:
            cots.append(cot_content.group(1))
    return cots


def get_predicted_result (text):
    entity = extract_possible_entities(text)
    Cot = extract_cots_as_strings(text)
    return entity, Cot

def get_name_to_id(name1, total_id_to_name_dict):
    ids = []
    for id, name in total_id_to_name_dict.items():
        if name == name1:
            ids += [id]
    return ids


