import re
from cot_prompt_list import *
# from subgraph_utilts import *
from subgraph_helper import *
# from search import *
serpapi_Key = "your own keys"

Num_run_LLM = 0
Global_depth = 3
Global_error_status = ""
reasoning_input_token_length = 0
LLM_model =''
url_db = "../online_search/url_db.db"
initialize_large_database(url_db)

def change_depth(depth = 3):
    global Global_depth
    Global_depth = depth
    print(f"Global depth is changed to {Global_depth}")

def changemode(name): 
    # display_LLM_model() 
    global LLM_model
    LLM_model = str(name)
    print(f"LLM model is changed to {LLM_model}")
    # display_LLM_model()

def display_LLM_model():
    global LLM_model
    # print(f"LLM model: {LLM_model}")
    return LLM_model

def increment(num = 1):
    global Num_run_LLM
    Num_run_LLM += num
def input_error(error_message="format error, "):
    global Global_error_status
    if error_message not in Global_error_status:
        Global_error_status += error_message
def input_token_length(length):
    global reasoning_input_token_length
    reasoning_input_token_length += length


def inital_num():
    global Num_run_LLM
    Num_run_LLM = 0
    global Global_error_status
    Global_error_status = ""
    global reasoning_input_token_length
    global LLM_model
    
def display_LLM_calls():
    global Num_run_LLM
    print(f"LLM calls time: {Num_run_LLM}")
    return Num_run_LLM

def display_error_status():
    global Global_error_status
    print(f"Error status: {Global_error_status}")
    return Global_error_status
def display_input_token_length():
    global reasoning_input_token_length
    print(f"input_token_length: {reasoning_input_token_length}")
    return reasoning_input_token_length



def extract_keywords(text):
    """
    Extract keywords from the given text within curly braces and return as a list.
    
    Parameters:
    text (str): Input text containing keywords in curly braces.
    
    Returns:
    list: A list of extracted keywords.
    """
    matches = re.findall(r'\{([^{}]*)\}', text)
    return [match.strip("{}") for match in matches]


import openai
import requests
from serpapi import GoogleSearch

def search_google(query, num_results=10):
    # """使用 Google 搜索 API 获取前 num_results 个搜索结果的文本。"""
    # params = {
    #     "q": query,
    #     "num": num_results,
    #     "api_key": "5680327412d0c587b8a60a19ae41f671d674d0163e63d60159eb276ce78b6e9f"
    # }
    
    # search = GoogleSearch(params)
    # results = search.run()
    # print(results)
    text_data = []


    params = {
        "q": query,
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        # "num": num_results,
        # "hl": "en",
        # "gl": "us",
        "api_key": serpapi_Key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results


def get_title_and_snippet(results):
    text_data = []
    for result in results.get("related_questions", []):
        if result.get("question") and result.get("snippet"):
            text_data.append("Related question: "+ result.get("question") + "\nSnippet: "+ result.get("snippet", ""))
            # text_data.append("Snippet: "+ result.get("snippet", ""))

    for result in results.get("organic_results", []):
        if result.get("title") and result.get("snippet"):

            text_data.append("organic title: "+ result.get("title", "") + "\nSnippet: "+ result.get("snippet", ""))
            # text_data.append("Snippet: "+ result.get("snippet", ""))
    return text_data
    return "\n".join(text_data)

def question_to_kg_path(question):
    # """主函数，完成问题到 KG Path 的转换。"""
    search_text = search_google(question)
    print("search_text:", search_text)
    kg_path = generate_kg_path(question, search_text)
    return kg_path

def extract_KGpaths(text):
    # 提取 keywords
    # keywords_match = re.search(r'keywords:\s*{(.*?)}', text)
    # keywords = [kw.strip() for kw in keywords_match.group(1).split(',')] if keywords_match else []
    paths_in_brackets = re.findall(r'\[(.*?)\]', text, flags=re.DOTALL)

    # Strip whitespace and store results in a list
    result_list = [p.strip() for p in paths_in_brackets]
    return result_list
    print(result_list)

from bs4 import BeautifulSoup
import bs4.builder
# ensure attribute_dict_class exists so unpickling works
if not hasattr(bs4.builder.HTMLParserTreeBuilder, "attribute_dict_class"):
    bs4.builder.HTMLParserTreeBuilder.attribute_dict_class = dict
from sentence_transformers import SentenceTransformer, util
import re


def split_wiki_page(soup):
    if not soup:
        return "Not Found!"
    content_div = soup.find("div", {"id": "bodyContent"})
    # Remove script and style elements
    for script_or_style in content_div.find_all(["script", "style"]):
        script_or_style.decompose()


    summary_content = ""
    for element in content_div.find_all(recursive=False):
        if element.name == "h2":
            break
        summary_content += element.get_text()

    return summary_content.strip()
    

import requests
from bs4 import BeautifulSoup


import requests
from bs4 import BeautifulSoup
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

# Global header declaration
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(level=logging.INFO, filename="fetch_web_content.log")

def is_lxml_installed():
    try:
        import lxml
        return True
    except ImportError:
        return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_web_content_test(url, timeout=6):
    """
    Fetches the real-time web content and returns a BeautifulSoup object.
    No cache is used, always fetches from the web.
    """
    try:
        # Skip loading from cache, always fetch from the web
        print("Fetching real-time content from the web")

        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code == 200:
            html = response.text
            delete_data_by_question_id(url_db, url)
            save_to_large_db(url_db, url, html)
            # Return the parsed content using BeautifulSoup
            return BeautifulSoup(html, "lxml" if is_lxml_installed() else "html.parser")
        else:
            print(f"Cannot access {url}: HTTP {response.status_code}")
            return None

    except Exception as exc:
        logging.error(f"Connect error for {url}: {exc}")
        print(f"Connect error for {url}: {exc}")
        return None

def fetch_web_content(url, timeout=6):
    """
    获取网页 HTML 字符串并返回 BeautifulSoup 对象
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise RuntimeError("beautifulsoup4 is not installed") from e

    try:
        html = load_from_large_db(url_db, url)
        if html:
            # 兼容 bytes 与 str 两种缓存格式
            if isinstance(html, bytes):
                html = html.decode("utf-8", errors="replace")
            print("从缓存中读取网页内容")
            return BeautifulSoup(html, "html.parser")

        # 缓存为空或解析失败，走网络
        print("send request to fetch new content")
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code == 200:
            html = response.text
            delete_data_by_question_id(url_db, url)
            save_to_large_db(url_db, url, html)
            return BeautifulSoup(html, "html.parser")
            # return BeautifulSoup(html, "lxml")
        else:
            print(f"Cannot access {url}: HTTP {response.status_code}")
            return None

    except Exception as exc:
        print(f"Connect error for {url}: {exc}")
        return fetch_web_content_test(url, timeout)  # 递归调用，直到成功或超时
def extract_paragraphs(soup):
    """
    从HTML中提取段落。若没有<p>标签，则退而求其次按换行切分。
    """
    paragraphs = []
    p_tags = soup.find_all('p')
    if p_tags:
        for p in p_tags:
            text = p.get_text().strip()
            # 简单去除多余的空行
            text = re.sub(r'\s+', ' ', text)
            if text:
                paragraphs.append(text)
    else:
        # 如果没有<p>标签，则将整个页面文本按换行进行简单分割
        text = soup.get_text(separator="\n")
        text = re.sub(r'\n+', '\n', text).strip()
        for seg in text.split('\n'):
            seg = seg.strip()
            if seg:
                paragraphs.append(seg)
    return paragraphs

def get_top_k_paragraphs(question, paragraphs,SBert, k):
    """
    对给定段落进行相似度计算，返回最相近的k段。
    """
    if not paragraphs:
        return []
    question_embedding = SBert.encode(question, convert_to_tensor=True)
    paragraph_embeddings = SBert.encode(paragraphs, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(question_embedding, paragraph_embeddings).squeeze(0)
    top_results = similarity_scores.argsort(descending=True)[:k]
    top_paragraphs = []
    for idx in top_results:
        top_paragraphs.append(paragraphs[idx])
    return top_paragraphs

# def process_search_results(question, search_text_final):
#     """
#     用于对search_text_final中的链接做进一步处理，
#     获取网页内容并输出最相近的三段话。
#     """
#     search_text_final = list(set(search_text_final))
#     url_to_top3 = {}
#     for link in search_text_final:
#         # print(f"正在处理链接: {link}")
#         soup = fetch_web_content(link)
#         if not soup:
#             continue
#         paragraphs = extract_paragraphs(soup)
#         if not paragraphs:
#             continue
#         top3 = get_top_k_paragraphs(question, paragraphs, k=3)
#         url_to_top3[link] = top3
#     return url_to_top3
import os
import multiprocessing as mp
import concurrent.futures
from functools import partial
# ------------------------------------------------------------------
#  Configure the safe start‑method **once** at import time
# ------------------------------------------------------------------
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# ------------------------------------------------------------------
#  Worker‑initialiser: build the embedder once per child
# ------------------------------------------------------------------
import threading
from FlagEmbedding import FlagModel    
_embedder = None           # global inside each process
_LOCK = threading.Lock()
def get_embedder(text_emb_name: str = "bge-bi"):
    global _embedder
    # global _EMBEDDER
    if _embedder is None:
        with _LOCK:
            # if text_emb_name == "minilm":
            from sentence_transformers import CrossEncoder
            print('loading rank model minilm...')
            _embedder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, cache_folder = "emb_model/MiniLM",model_kwargs={"low_cpu_mem_usage": False})
    return _embedder


def scores_rank(scores,texts):
    items=[]
    for i in range(len(scores)):
        item={}
        item['score']=float(scores[i])
        item['text']=texts[i]
        items.append(item)

    sorted_data = sorted(items, key=lambda x: x['score'], reverse=True)

    return sorted_data
def crossencoder_similarity(question, texts, embedding_model_name, emb_model):
    assert embedding_model_name == 'minilm' or 'bge-ce'
    if len(texts) == 1 :
        l = [[question, texts[0] +' ']]
    elif len(texts) == 0:
        return []
    else:
        l = [[question, text] for text in texts]
    if embedding_model_name == 'bge-ce':
        scores = emb_model.compute_score(l)
    else:
        scores = emb_model.predict(l)


    return scores
def biencoder_similarity(question, passages, embedding_model_name, emb_model):
    assert embedding_model_name == 'bge-bi'
    questions = [question] * len(passages)
    q_embeddings = emb_model.encode_queries(questions)
    p_embeddings = emb_model.encode(passages)
    scores = q_embeddings @ p_embeddings.T
    return scores[0]

def s2p_relevance_scores(texts, question, embedding_model_name, emb_model):
    # print(embedding_model_name)
    if embedding_model_name == 'bge-bi':
        scores = biencoder_similarity(question, texts, embedding_model_name, emb_model)
        return scores
    elif embedding_model_name == 'minilm' or embedding_model_name == 'bge-ce':
        scores = crossencoder_similarity(question, texts, embedding_model_name, emb_model)
        return scores
    elif embedding_model_name == 'bm25':
        scores = emb_model(question, texts)
        return scores
    elif embedding_model_name == 'colbert':
        q_embeddings = emb_model.encode([question], return_dense=False, return_sparse=False, return_colbert_vecs=True)
        p_embeddings = emb_model.encode(texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
        scores = []
        for i in range(len(texts)):
            scores.append(emb_model.colbert_score(q_embeddings['colbert_vecs'][0], p_embeddings['colbert_vecs'][i]))
        return scores
    else:
        raise Exception('Unknown embedding model')

from blingfire import text_to_sentences_and_offsets


def split_sentences_windows(text, window_size=2, step_size=1):
    all_sentences = []
    if len(text) > 0:
        offsets = text_to_sentences_and_offsets(text)[1]
        for ofs in offsets:
            sentence = text[ofs[0]: ofs[1]]
            all_sentences.append(
                sentence
            )
    else:
        all_sentences.append("")

    if window_size > 1 and len(all_sentences) >= (window_size + step_size):
        all_windows = []

        for i in range(0, len(all_sentences) - window_size + 1, step_size):
            window = all_sentences[i:i + window_size]
            combined_sentence = ' '.join(window)
            all_windows.append(combined_sentence)
        all_sentences = all_windows
    return all_sentences

import concurrent.futures
def pages_embedding_search(question,related_passage, embedding_model_name, emb_model,top_k=3):
    content=related_passage

    if content!='Not Found!':
        splited_paragraphs = extract_paragraphs(content)
        if len(splited_paragraphs) < 1:
            return '',[]
        scores = s2p_relevance_scores(splited_paragraphs, question, embedding_model_name, emb_model)
        sorted_splited_paragraphs = scores_rank(scores, splited_paragraphs)

        if len(sorted_splited_paragraphs) >= 3:
            paragraph = sorted_splited_paragraphs[0]['text'] + sorted_splited_paragraphs[1]['text'] + sorted_splited_paragraphs[2]['text']
        else:
            paragraph = ''.join([p['text'] for p in sorted_splited_paragraphs])

        splited_sentences=split_sentences_windows(paragraph)

        scores = s2p_relevance_scores(splited_sentences, question, embedding_model_name, emb_model)
        sorted_sentences = scores_rank(scores, splited_sentences)

        return paragraph,sorted_sentences[0:top_k]
    else:
        return '',[]

from concurrent.futures import ThreadPoolExecutor, as_completed

# 根据机器 I/O 与 GPU 能力自行调节
_THREAD_POOL = ThreadPoolExecutor(max_workers=32)



import torch

def _process_link(link: str,
                  question: str,
                  top_k: int,
                  timeout: int,
                  emb_name: str):
    soup = fetch_web_content(link, timeout)
    if not soup:
        return link, None

    embedder = get_embedder(emb_name)

    # ========= 编码网页文本 =========
    with torch.inference_mode():                 # 等价于 no_grad
        paragraph, ranked = pages_embedding_search(
            question,
            soup,
            embedding_model_name=emb_name,
            emb_model=embedder,
            top_k=10
        )

    # 若无有效段落，直接返回
    if not ranked:
        return link, None

    torch.cuda.empty_cache()

    return link, [s["text"] for s in ranked[:min(top_k, len(ranked))]]


import os, concurrent.futures, functools
def process_search_results(question: str,
                           urls: list[str],
                           emb_name: str,
                            unused_emb_model: str,
                           timeout: int = 6,
                           k: int = 3):
    urls = list(dict.fromkeys(urls))          # 去重
    if not urls:
        return {}

    get_embedder(emb_name)

    fn = functools.partial(_process_link,
                           question=question,
                           top_k=k,
                           timeout=timeout,
                           emb_name=emb_name)

    futures = { _THREAD_POOL.submit(fn, u): u for u in urls }

    url_to_top = {}
    for fut in as_completed(futures):
        link, top = fut.result()
        if top:
            url_to_top[link] = top
    return url_to_top



from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_similar_entities_bert(entity_dict, entity_name_list, top_k=3):
    # Nothing to compare – return early
    if not entity_dict or not entity_name_list:
        return []

    # Prepare corpus
    entity_ids   = list(entity_dict.keys())
    entity_names = list(entity_dict.values())
    corpus       = entity_names + entity_name_list

    # Vectorise
    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Split into targets and candidates
    target_mat   = tfidf_matrix[-len(entity_name_list):]
    source_mat   = tfidf_matrix[:-len(entity_name_list)]

    # Another safety check
    if source_mat.shape[0] == 0:
        return []

    # Similarity
    sim = cosine_similarity(target_mat, source_mat)

    # Collect results
    results = []
    for i, name in enumerate(entity_name_list):
        idxs = sim[i].argsort()[-top_k:][::-1]
        for j in idxs:
            # ent_id = entity_ids[j]
            # results.append((ent_id, entity_dict[ent_id], sim[i, j]))
            ent_id = entity_ids[j]
            similar_name = entity_dict[ent_id]
            original_name = entity_name_list[i]  # The original name comes from entity_name_list
            if sim[i, j] > 0.8:
                results.append((ent_id, similar_name, sim[i, j], original_name))

    return results





















