from tqdm import tqdm
import argparse
import random
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

def initialize_large_database(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subgraphs (
            question_id TEXT,
            chunk_index INTEGER,
            data BLOB,
            PRIMARY KEY (question_id, chunk_index)
        )
        ''')
        conn.commit()

def retry_operation(func):
    """Decorator to retry database operations."""
    def wrapper(*args, **kwargs):
        attempts = 0
        max_retries = kwargs.pop('max_retries', 300)
        wait_time = kwargs.pop('wait_time', 6)
        while attempts < max_retries:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    print(f"Database is locked, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    attempts += 1
                else:
                    print("An error occurred:", e)
                    break
        print("Failed after several attempts.")
        return None
    return wrapper

@retry_operation
def delete_data_by_question_id(db_path, question_id):
    with sqlite3.connect(db_path) as conn:
        start = time.time()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM subgraphs WHERE question_id = ?', (question_id,))
        conn.commit()
        print(f"Data associated with question_id {question_id} has been deleted.")
        print(f"Time taken to delete data: {time.time() - start} seconds.")

@retry_operation
def save_to_large_db(db_path, question_id, data_dict, chunk_size=256 * 1024 * 1024):
    with sqlite3.connect(db_path) as conn:
        start = time.time()
        cursor = conn.cursor()
        data_blob = pickle.dumps(data_dict)
        for i in range(0, len(data_blob), chunk_size):
            chunk = data_blob[i:i+chunk_size]
            cursor.execute('INSERT INTO subgraphs (question_id, data, chunk_index) VALUES (?, ?, ?)',
                           (question_id, chunk, i // chunk_size))
            conn.commit()
        print("Data saved to database.")
        print(f"Time taken to save data: {time.time() - start} seconds.")
@retry_operation
def load_from_large_db(db_path, question_id):
    with sqlite3.connect(db_path) as conn:
        start = time.time()
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM subgraphs WHERE question_id = ? ORDER BY chunk_index', (question_id,))
        data_blob = b''.join(row[0] for row in cursor if row[0] is not None)
        
        if not data_blob:
            # print("No data found or data is empty.")
            return None
        
        try:
            result = pickle.loads(data_blob)
            return result
        except EOFError as e:
            print(f"EOFError when unpickling data: {e}")
            return None
