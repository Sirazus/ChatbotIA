from flask import Flask, request, send_file, abort, jsonify, url_for, render_template, Response
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Iterator
from collections import deque
import os
import logging
import atexit
from threading import Thread, Lock
import time
from datetime import datetime
from uuid import uuid4 as generate_uuid
import csv as csv_lib
import functools
import json
import re
import subprocess
import sys
import sqlite3
import io

from dotenv import load_dotenv

# Load environment variables from .env file AT THE VERY TOP
load_dotenv()

# Import RAG system and Fallback LLM from llm_handling AFTER load_dotenv
# MODIFIED: Imported new functions and prompts
from llm_handling import (
    initialize_and_get_rag_system,
    KnowledgeRAG,
    groq_bot_instance,
    RAG_SOURCES_DIR,
    RAG_STORAGE_PARENT_DIR,
    RAG_CHUNKED_SOURCES_FILENAME,
    get_answer_from_context
)
from system_prompts import QA_FORMATTER_PROMPT


# Setup logging (remains global for the app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_hybrid_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Main app logger

# --- Application Constants and Configuration ---
ADMIN_USERNAME = os.getenv('FLASK_ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('FLASK_ADMIN_PASSWORD', 'admin')
FLASK_APP_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_APP_PORT = int(os.getenv("FLASK_PORT", "7860"))
FLASK_DEBUG_MODE = os.getenv("FLASK_DEBUG", "True").lower() == "true"
_APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_EXTRACTIONS_DIR = os.path.join(_APP_BASE_DIR, 'text_extractions')
RELATED_QUESTIONS_TO_SHOW = 10
QUESTIONS_TO_SEND_TO_GROQ_QA = 3
# MODIFIED: Replaced separate confidence values with a single configurable one for the LLM formatter.
LLM_FORMATTER_CONFIDENCE_THRESHOLD = int(os.getenv("LLM_FORMATTER_CONFIDENCE_THRESHOLD", "95"))
HIGH_CONFIDENCE_THRESHOLD = 90 # For greetings, which are answered directly without LLM formatting.
# MODIFIED: Made CHAT_HISTORY_TO_SEND configurable via environment variable
CHAT_HISTORY_TO_SEND = int(os.getenv("CHAT_HISTORY_TO_SEND", "5")) # Defines how many *pairs* of (user, assistant) messages to send
CHAT_LOG_FILE = os.path.join(_APP_BASE_DIR, 'chat_history.csv')

rag_system: Optional[KnowledgeRAG] = None

# --- Persistent Chat History Management using SQLite ---
class ChatHistoryManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock = Lock()
        self._create_table()
        logger.info(f"SQLite chat history manager initialized at: {self.db_path}")

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        return conn

    def _create_table(self):
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_histories (
                        session_id TEXT PRIMARY KEY,
                        history TEXT NOT NULL
                    )
                """)
                conn.commit()

    def get_history(self, session_id: str, limit_turns: int = 5) -> list:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT history FROM chat_histories WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    history_list = json.loads(row[0])
                    return history_list[-(limit_turns * 2):]
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching history for session {session_id}: {e}", exc_info=True)
            return []

    def update_history(self, session_id: str, query: str, answer: str):
        with self.lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT history FROM chat_histories WHERE session_id = ?", (session_id,))
                    row = cursor.fetchone()
                    
                    history = json.loads(row[0]) if row else []
                    
                    history.append({'role': 'user', 'content': query})
                    history.append({'role': 'assistant', 'content': answer})

                    updated_history_json = json.dumps(history)

                    cursor.execute("""
                        INSERT OR REPLACE INTO chat_histories (session_id, history)
                        VALUES (?, ?)
                    """, (session_id, updated_history_json))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error updating history for session {session_id}: {e}", exc_info=True)
                
    def clear_history(self, session_id: str):
        with self.lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO chat_histories (session_id, history)
                        VALUES (?, ?)
                    """, (session_id, json.dumps([])))
                    conn.commit()
                logger.info(f"Chat history cleared for session: {session_id}")
            except Exception as e:
                 logger.error(f"Error clearing history for session {session_id}: {e}", exc_info=True)


# --- EmbeddingManager for CSV QA (remains in app.py) ---
@dataclass
class QAEmbeddings:
    questions: List[str]
    question_map: List[int]
    embeddings: torch.Tensor
    df_qa: pd.DataFrame
    original_questions: List[str]

class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {
            'general': None,
            'personal': None,
            'greetings': None
        }
        logger.info(f"EmbeddingManager initialized with model: {model_name}")

    def _process_questions(self, df: pd.DataFrame) -> Tuple[List[str], List[int], List[str]]:
        questions = []
        question_map = []
        original_questions = []

        if 'Question' not in df.columns:
            logger.warning(f"DataFrame for EmbeddingManager is missing 'Question' column. Cannot process questions from it.")
            return questions, question_map, original_questions

        for idx, question_text_raw in enumerate(df['Question']):
            if pd.isna(question_text_raw):
                continue
            question_text_cleaned = str(question_text_raw).strip()
            if not question_text_cleaned or question_text_cleaned.lower() == "nan":
                continue

            questions.append(question_text_cleaned)
            question_map.append(idx)
            original_questions.append(question_text_cleaned)

        return questions, question_map, original_questions

    def update_embeddings(self, general_qa: pd.DataFrame, personal_qa: pd.DataFrame, greetings_qa: pd.DataFrame):
        gen_questions, gen_question_map, gen_original_questions = self._process_questions(general_qa)
        gen_embeddings = self.model.encode(gen_questions, convert_to_tensor=True, show_progress_bar=False) if gen_questions else None

        pers_questions, pers_question_map, pers_original_questions = self._process_questions(personal_qa)
        pers_embeddings = self.model.encode(pers_questions, convert_to_tensor=True, show_progress_bar=False) if pers_questions else None

        greet_questions, greet_question_map, greet_original_questions = self._process_questions(greetings_qa)
        greet_embeddings = self.model.encode(greet_questions, convert_to_tensor=True, show_progress_bar=False) if greet_questions else None

        self.embeddings['general'] = QAEmbeddings(
            questions=gen_questions, question_map=gen_question_map, embeddings=gen_embeddings,
            df_qa=general_qa, original_questions=gen_original_questions
        )
        self.embeddings['personal'] = QAEmbeddings(
            questions=pers_questions, question_map=pers_question_map, embeddings=pers_embeddings,
            df_qa=personal_qa, original_questions=pers_original_questions
        )
        self.embeddings['greetings'] = QAEmbeddings(
            questions=greet_questions, question_map=greet_question_map, embeddings=greet_embeddings,
            df_qa=greetings_qa, original_questions=greet_original_questions
        )
        logger.info("CSV QA embeddings updated in EmbeddingManager.")

    def find_best_answers(self, user_query: str, qa_type: str, top_n: int = 5) -> Tuple[List[float], List[str], List[str], List[str], List[int]]:
        qa_data = self.embeddings[qa_type]
        if qa_data is None or qa_data.embeddings is None or len(qa_data.embeddings) == 0:
            return [], [], [], [], []

        query_embedding_tensor = self.model.encode([user_query], convert_to_tensor=True, show_progress_bar=False)
        if not isinstance(qa_data.embeddings, torch.Tensor):
             qa_data.embeddings = torch.tensor(qa_data.embeddings) # Safeguard

        cos_scores = util.cos_sim(query_embedding_tensor, qa_data.embeddings)[0]

        top_k = min(top_n, len(cos_scores))
        if top_k == 0:
            return [], [], [], [], []

        top_scores_tensor, indices_tensor = torch.topk(cos_scores, k=top_k)

        top_confidences = [score.item() * 100 for score in top_scores_tensor]
        top_indices_mapped = []
        top_questions = []

        for idx_tensor in indices_tensor:
            item_idx = idx_tensor.item()
            if item_idx < len(qa_data.question_map) and item_idx < len(qa_data.original_questions):
                 original_df_idx = qa_data.question_map[item_idx]
                 if original_df_idx < len(qa_data.df_qa):
                    top_indices_mapped.append(original_df_idx)
                    top_questions.append(qa_data.original_questions[item_idx])
                 else:
                    logger.warning(f"Index out of bounds: original_df_idx {original_df_idx} for df_qa length {len(qa_data.df_qa)}")
            else:
                logger.warning(f"Index out of bounds: item_idx {item_idx} for question_map/original_questions")

        valid_count = len(top_indices_mapped)
        top_confidences = top_confidences[:valid_count]
        top_questions = top_questions[:valid_count]
        
        # MODIFIED: Changed Answer to Respuesta to match new loading logic for xlsx
        answer_col = 'Respuesta' if 'Respuesta' in qa_data.df_qa.columns else 'Answer'
        top_answers = [str(qa_data.df_qa[answer_col].iloc[i]) for i in top_indices_mapped]
        top_images = [str(qa_data.df_qa['Image'].iloc[i]) if 'Image' in qa_data.df_qa.columns and pd.notna(qa_data.df_qa['Image'].iloc[i]) else None for i in top_indices_mapped]

        return top_confidences, top_questions, top_answers, top_images, top_indices_mapped

# --- DatabaseMonitor for personal_qa.csv placeholders (remains in app.py) ---
class DatabaseMonitor:
    def __init__(self, database_path):
        self.logger = logging.getLogger(__name__ + ".DatabaseMonitor")
        self.database_path = database_path
        self.last_modified = None
        self.last_size = None
        self.df = None
        self.lock = Lock()
        self.running = True
        self._load_database()
        self.monitor_thread = Thread(target=self._monitor_database, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"DatabaseMonitor initialized for: {database_path}")

    def _load_database(self):
        try:
            if not os.path.exists(self.database_path):
                self.logger.warning(f"Personal data file not found: {self.database_path}.")
                self.df = None
                return
            with self.lock:
                self.df = pd.read_csv(self.database_path, encoding='cp1252')
                self.last_modified = os.path.getmtime(self.database_path)
                self.last_size = os.path.getsize(self.database_path)
                self.logger.info(f"Personal data file reloaded: {self.database_path}")
        except Exception as e:
            self.logger.error(f"Error loading personal data file '{self.database_path}': {e}", exc_info=True)
            self.df = None

    def _monitor_database(self):
        while self.running:
            try:
                if not os.path.exists(self.database_path):
                    if self.df is not None:
                        self.logger.warning(f"Personal data file disappeared: {self.database_path}")
                        self.df = None; self.last_modified = None; self.last_size = None
                    time.sleep(5)
                    continue
                current_modified = os.path.getmtime(self.database_path); current_size = os.path.getsize(self.database_path)
                if (self.last_modified is None or current_modified != self.last_modified or
                    self.last_size is None or current_size != self.last_size):
                    self.logger.info("Personal data file change detected.")
                    self._load_database()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error monitoring personal data file: {e}", exc_info=True)
                time.sleep(5)

    def get_data(self, user_id):
        with self.lock:
            if self.df is not None and user_id:
                try:
                    if 'id' not in self.df.columns:
                        self.logger.warning("'id' column not found in personal_data.csv")
                        return None
                    id_col_type = self.df['id'].dtype
                    target_user_id = user_id
                    if pd.api.types.is_numeric_dtype(id_col_type):
                        try:
                            if user_id is None: return None
                            valid_ids = self.df['id'].dropna()
                            if not valid_ids.empty:
                                target_user_id = type(valid_ids.iloc[0])(user_id)
                            else:
                                target_user_id = int(user_id)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert user_id '{user_id}' to numeric type {id_col_type}")
                            return None
                    user_data = self.df[self.df['id'] == target_user_id]
                    if not user_data.empty: return user_data.iloc[0].to_dict()
                except Exception as e:
                    self.logger.error(f"Error retrieving data for user_id {user_id}: {e}", exc_info=True)
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("DatabaseMonitor stopped.")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- Initialize Managers ---
embedding_manager = EmbeddingManager()
history_manager = ChatHistoryManager('chat_history.db')
database_csv_path = os.path.join(RAG_SOURCES_DIR, 'database.csv')
personal_data_monitor = DatabaseMonitor(database_csv_path)

# --- Helper Functions (App specific) ---
def clean_html_from_text(text: str) -> str:
    """Removes HTML tags from a string using a simple regex."""
    if not isinstance(text, str):
        return text
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text.strip()

def normalize_text(text):
    if isinstance(text, str):
        replacements = {
            '\x91': "'", '\x92': "'", '\x93': '"', '\x94': '"',
            '\x96': '-', '\x97': '-', '\x85': '...', '\x95': '-',
            '"': '"', '"': '"', '‘': "'", '’': "'",
            '–': '-', '—': '-', '…': '...', '•': '-',
        }
        for old, new in replacements.items(): text = text.replace(old, new)
    return text

def require_admin_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USERNAME or auth.password != ADMIN_PASSWORD:
            return Response('Admin auth failed.', 401, {'WWW-Authenticate': 'Basic realm="Admin Login Required"'})
        return f(*args, **kwargs)
    return decorated

def initialize_chat_log():
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv_lib.writer(f)
            writer.writerow(['sl', 'date_time', 'session_id', 'user_id', 'query', 'answer'])

def store_chat_history(sid: str, uid: Optional[str], query: str, resp: Dict[str, Any]):
    try:
        # This now gets the final response key, which is 'answer' in the old logic
        answer = str(resp.get('answer', ''))
        history_manager.update_history(sid, query, answer)

        initialize_chat_log()
        next_sl = 1
        try:
            if os.path.exists(CHAT_LOG_FILE) and os.path.getsize(CHAT_LOG_FILE) > 0:
                df_log = pd.read_csv(CHAT_LOG_FILE, on_bad_lines='skip')
                if not df_log.empty and 'sl' in df_log.columns and pd.api.types.is_numeric_dtype(df_log['sl'].dropna()):
                    if not df_log['sl'].dropna().empty:
                        next_sl = int(df_log['sl'].dropna().max()) + 1
        except Exception as e:
            logger.error(f"Error reading SL from {CHAT_LOG_FILE}: {e}", exc_info=True)

        with open(CHAT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            csv_lib.writer(f).writerow([next_sl, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sid, uid or "N/A", query, answer])

    except Exception as e:
        logger.error(f"Error in store_chat_history for session {sid}: {e}", exc_info=True)

def get_formatted_chat_history(session_id: str) -> List[Dict[str, str]]:
    if not session_id:
        return []
    return history_manager.get_history(session_id, limit_turns=CHAT_HISTORY_TO_SEND)

def get_qa_context_for_groq(all_questions: List[Dict]) -> str:
    valid_qa_pairs = []
    non_greeting_questions = [q for q in all_questions if q.get('source_type') != 'greetings']
    sorted_questions = sorted(non_greeting_questions, key=lambda x: x.get('confidence', 0), reverse=True)

    for qa in sorted_questions[:QUESTIONS_TO_SEND_TO_GROQ_QA]:
        answer = qa.get('answer')
        if (not pd.isna(answer) and isinstance(answer, str) and answer.strip() and
            "not available" not in answer.lower()):
            valid_qa_pairs.append(f"Q: {qa.get('question')}\nA: {answer}")
    return '\n'.join(valid_qa_pairs)

def replace_placeholders_in_answer(answer, db_data):
    if pd.isna(answer) or str(answer).strip() == '':
        return "Sorry, this information is not available yet"
    answer_str = str(answer)
    placeholders = re.findall(r'\{(\w+)\}', answer_str)
    if not placeholders: return answer_str
    if db_data is None:
        return "To get this specific information, please ensure you are logged in or have provided your user ID."
    missing_count = 0; replacements_made = 0
    for placeholder in set(placeholders):
        key = placeholder.strip()
        value = db_data.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == '':
            answer_str = answer_str.replace(f'{{{key}}}', "not available")
            missing_count += 1
        else:
            answer_str = answer_str.replace(f'{{{key}}}', str(value))
            replacements_made +=1
    if missing_count == len(placeholders) and len(placeholders) > 0 :
        return "Sorry, some specific details for you are not available at the moment."
    if "not available" in answer_str.lower() and replacements_made < len(placeholders):
         if answer_str == "not available" and len(placeholders) == 1:
             return "Sorry, this information is not available yet."
    if re.search(r'\{(\w+)\}', answer_str):
        logger.warning(f"Unresolved placeholders remain after replacement attempt: {answer_str}")
        answer_str = re.sub(r'\{(\w+)\}', "a specific detail", answer_str)
        if "a specific detail" in answer_str and not "Sorry" in answer_str:
            return "Sorry, I couldn't retrieve all the specific details for this answer. " + answer_str
        return "Sorry, I couldn't retrieve all the specific details for this answer. Some information has been generalized."
    return answer_str

# --- Non-Streaming Logic (Preserved from original) ---
def get_hybrid_response_logic_non_streaming(user_query: str, session_id: str, user_id: Optional[str], chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    global rag_system

    if not user_query: return {'error': 'No query provided'}
    if not session_id: return {'error': 'session_id is required'}

    personal_db_data = personal_data_monitor.get_data(user_id) if user_id else None

    # MODIFIED: Capture indices from the search
    conf_greet, q_greet, a_greet, img_greet, idx_greet = embedding_manager.find_best_answers(user_query, 'greetings', top_n=1)
    conf_pers, q_pers, a_pers, img_pers, idx_pers = embedding_manager.find_best_answers(user_query, 'personal', top_n=RELATED_QUESTIONS_TO_SHOW)
    conf_gen, q_gen, a_gen, img_gen, idx_gen = embedding_manager.find_best_answers(user_query, 'general', top_n=RELATED_QUESTIONS_TO_SHOW)

    all_csv_candidate_answers = []
    if conf_greet and conf_greet[0] >= HIGH_CONFIDENCE_THRESHOLD:
        all_csv_candidate_answers.append({'question': q_greet[0], 'answer': a_greet[0], 'image': img_greet[0] if img_greet else None, 'confidence': conf_greet[0], 'source_type': 'greetings', 'original_index': idx_greet[0]})
    if conf_pers:
        # MODIFIED: Add original_index to candidates
        for c, q, a, img, idx in zip(conf_pers, q_pers, a_pers, img_pers, idx_pers):
            processed_a = replace_placeholders_in_answer(a, personal_db_data)
            if not ("Sorry, this information is not available yet" in processed_a or "To get this specific information" in processed_a):
                all_csv_candidate_answers.append({'question': q, 'answer': processed_a, 'image': img, 'confidence': c, 'source_type': 'personal', 'original_index': idx})
    if conf_gen:
        # MODIFIED: Add original_index to candidates
        for c, q, a, img, idx in zip(conf_gen, q_gen, a_gen, img_gen, idx_gen):
            if not (pd.isna(a) or str(a).strip() == '' or str(a).lower() == 'nan'):
                all_csv_candidate_answers.append({'question': q, 'answer': str(a), 'image': img, 'confidence': c, 'source_type': 'general', 'original_index': idx})

    all_csv_candidate_answers.sort(key=lambda x: x['confidence'], reverse=True)

    related_questions_list = []

    if all_csv_candidate_answers:
        best_csv_match = all_csv_candidate_answers[0]
        is_direct_csv_answer = False
        source_name = ""
        
        # MODIFIED: Use new configurable threshold for LLM formatting
        best_source_type = best_csv_match['source_type']
        best_confidence = best_csv_match['confidence']

        if best_source_type == 'greetings' and best_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            is_direct_csv_answer = True
            source_name = 'greetings_qa'
        elif best_source_type in ['personal', 'general'] and best_confidence >= LLM_FORMATTER_CONFIDENCE_THRESHOLD:
            is_direct_csv_answer = True
            source_name = f"{best_source_type}_qa"

        if is_direct_csv_answer:
            # MODIFICATION START: Reroute high-confidence matches to the LLM for formatting
            best_match_source = best_csv_match['source_type']
            
            # For greetings, we still provide a direct answer without LLM formatting
            if best_match_source == 'greetings':
                response_data = {'query': user_query, 'answer': best_csv_match['answer'], 'confidence': best_csv_match['confidence'], 'original_question': best_csv_match['question'], 'source': source_name}
                if best_csv_match.get('image'):
                     response_data['image_url'] = url_for('static', filename=best_csv_match['image'], _external=True)
            else:
                # For 'personal' and 'general', use the LLM to format the answer from the full row
                best_match_index = best_csv_match['original_index']
                
                # Retrieve the full row from the original dataframe stored in the embedding manager
                original_df = embedding_manager.embeddings[best_match_source].df_qa
                matched_row_data = original_df.iloc[best_match_index]
                
                # Format the row data as a string context for the LLM
                # We drop the 'Question' column as it's a duplicate of 'Pregunta' and not needed in the context
                context_dict = matched_row_data.drop('Question', errors='ignore').to_dict()
                context_str = "\n".join([f"'{key}': '{value}'" for key, value in context_dict.items() if pd.notna(value) and str(value).strip() != ''])

                # Call the LLM to generate a conversational answer based on the row data
                final_answer = get_answer_from_context(
                    question=user_query,
                    context=context_str,
                    system_prompt=QA_FORMATTER_PROMPT
                )
                
                response_data = {
                    'query': user_query, 
                    'answer': final_answer, 
                    'confidence': best_csv_match['confidence'], 
                    'original_question': best_csv_match['question'], 
                    'source': f'{source_name}_llm_formatted'
                }
                if best_csv_match.get('image'):
                     response_data['image_url'] = url_for('static', filename=best_csv_match['image'], _external=True)

            # MODIFICATION END

            for i, cand_q in enumerate(all_csv_candidate_answers):
                if i == 0: continue
                if cand_q['source_type'] != 'greetings':
                     related_questions_list.append({'question': cand_q['question'], 'answer': cand_q['answer'], 'match': cand_q['confidence']})
                if len(related_questions_list) >= RELATED_QUESTIONS_TO_SHOW: break
            response_data['related_questions'] = related_questions_list
            store_chat_history(session_id, user_id, user_query, response_data)
            return response_data

    if rag_system and rag_system.retriever:
        logger.info(f"Attempting FAISS RAG query for: {user_query[:50]}...")
        rag_result = rag_system.invoke(user_query) # Use invoke for non-streaming
        rag_answer = rag_result.get("answer")
        
        if rag_answer and "the provided bibliography does not contain specific information" not in rag_answer.lower():
            logger.info(f"FAISS RAG system provided a valid answer: {rag_answer[:100]}...")
            response_data = {
                'query': user_query, 'answer': rag_answer, 'confidence': 85,
                'source': 'document_rag_faiss', 'related_questions': [],
                'document_sources_details': rag_result.get("cited_source_details")
            }
            store_chat_history(session_id, user_id, user_query, response_data)
            return response_data

    logger.info(f"No high-confidence answer. Using Groq fallback.")
    chat_history_messages_for_groq = chat_history if chat_history is not None else get_formatted_chat_history(session_id)
    groq_context = {'current_query': user_query, 'chat_history': chat_history_messages_for_groq, 'qa_related_info': ""}
    groq_stream = groq_bot_instance.stream_response(groq_context)
    groq_answer = "".join([chunk for chunk in groq_stream])

    response_data = {'query': user_query, 'answer': groq_answer, 'confidence': 75, 'source': 'groq_general_fallback', 'related_questions': []}
    store_chat_history(session_id, user_id, user_query, response_data)
    return response_data

# --- Streaming Logic ---
def generate_streaming_response(user_query: str, session_id: str, user_id: Optional[str], chat_history: Optional[List[Dict]] = None) -> Iterator[str]:
    """
    Handles the logic for generating a response and yields chunks of the response as a stream.
    """
    global rag_system
    
    personal_db_data = personal_data_monitor.get_data(user_id) if user_id else None
    conf_greet, _, a_greet, _, idx_greet = embedding_manager.find_best_answers(user_query, 'greetings', top_n=1)
    conf_pers, _, a_pers, _, idx_pers = embedding_manager.find_best_answers(user_query, 'personal', top_n=1)
    conf_gen, _, a_gen, _, idx_gen = embedding_manager.find_best_answers(user_query, 'general', top_n=1)

    # MODIFIED: Use new configurable threshold and logic for picking best candidate
    candidates = []
    # Greetings have their own threshold for a direct, non-LLM answer
    if conf_greet and conf_greet[0] >= HIGH_CONFIDENCE_THRESHOLD:
        candidates.append({'answer': a_greet[0], 'confidence': conf_greet[0], 'source': 'greetings', 'index': idx_greet[0]})
    
    # Personal and General QA have a stricter threshold to be sent to the LLM formatter
    if conf_pers and conf_pers[0] >= LLM_FORMATTER_CONFIDENCE_THRESHOLD:
        processed_a = replace_placeholders_in_answer(a_pers[0], personal_db_data)
        # Only add candidate if placeholder replacement was successful
        if not ("Sorry, this information is not available yet" in processed_a or "To get this specific information" in processed_a):
            candidates.append({'answer': processed_a, 'confidence': conf_pers[0], 'source': 'personal', 'index': idx_pers[0]})

    if conf_gen and conf_gen[0] >= LLM_FORMATTER_CONFIDENCE_THRESHOLD:
        # Filter out empty/invalid answers
        if not (pd.isna(a_gen[0]) or str(a_gen[0]).strip() == '' or str(a_gen[0]).lower() == 'nan'):
            candidates.append({'answer': a_gen[0], 'confidence': conf_gen[0], 'source': 'general', 'index': idx_gen[0]})
    
    if candidates:
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        best_source_type = best_candidate['source']
        logger.info(f"High-confidence match from CSV source: {best_source_type}")

        # If the best match is a greeting, yield it directly
        if best_source_type == 'greetings':
             yield best_candidate['answer']
             return
        
        # Otherwise, the best match is 'personal' or 'general' and needs LLM formatting
        original_df = embedding_manager.embeddings[best_source_type].df_qa
        matched_row_data = original_df.iloc[best_candidate['index']]
        context_dict = matched_row_data.drop('Question', errors='ignore').to_dict()
        context_str = "\n".join([f"'{key}': '{value}'" for key, value in context_dict.items() if pd.notna(value) and str(value).strip() != ''])
        
        final_answer = get_answer_from_context(
            question=user_query,
            context=context_str,
            system_prompt=QA_FORMATTER_PROMPT
        )
        yield final_answer
        return

    if rag_system and rag_system.retriever:
        logger.info(f"Attempting to stream from FAISS RAG for: {user_query[:50]}...")
        rag_stream = rag_system.stream(user_query)
        first_chunk = next(rag_stream, None)
        
        if first_chunk and "the provided bibliography does not contain specific information" not in first_chunk.lower():
            logger.info("FAISS RAG streaming valid answer...")
            yield first_chunk
            yield from rag_stream
            return
    
    logger.info(f"No high-confidence CSV or RAG answer. Streaming from Groq fallback.")
    chat_history_messages_for_groq = chat_history if chat_history is not None else get_formatted_chat_history(session_id)
    groq_context = {'current_query': user_query, 'chat_history': chat_history_messages_for_groq, 'qa_related_info': ""}
    yield from groq_bot_instance.stream_response(groq_context)

def stream_formatter(logic_generator: Iterator[str], session_id: str, user_id: Optional[str], query: str) -> Iterator[str]:
    """
    Wraps raw text chunks into the Server-Sent Events (SSE) format and logs the full response at the end.
    """
    chunk_id = f"chatcmpl-{str(generate_uuid())}"
    model_name = "MedicalAssisstantBot/v1"
    full_response_chunks = []

    for chunk in logic_generator:
        if not chunk: continue
        full_response_chunks.append(chunk)
        response_json = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(response_json)}\n\n"

    final_json = {
        "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_json)}\n\n"
    yield "data: [DONE]\n\n"
    
    # After streaming is complete, log the full conversation to the database
    full_response = "".join(full_response_chunks)
    
    # MODIFIED: Added print statement for full streamed response
    print(f"\n--- STREAMED FULL RESPONSE ---")
    print(full_response)
    print(f"------------------------------\n")

    history_manager.update_history(session_id, query, full_response)

# --- Original Chat Endpoint (Preserved) ---
@app.route('/chat-bot', methods=['POST'])
def get_answer_hybrid():
    data = request.json
    user_query = data.get('query', '')
    user_query = clean_html_from_text(user_query) # ADDED
    user_id = data.get('user_id')
    session_id = data.get('session_id')
    
    if not user_query or not session_id:
        return jsonify({'error': 'query and session_id are required'}), 400

    response_data = get_hybrid_response_logic_non_streaming(user_query, session_id, user_id, None)
    return jsonify(response_data)

# --- OpenAI Compatible Endpoints (Added) ---
@app.route('/v1/models', methods=['GET'])
def list_models():
    model_data = {
        "object": "list",
        "data": [{"id": "MedicalAssisstantBot/v1", "object": "model", "created": int(time.time()), "owned_by": "user"}]
    }
    return jsonify(model_data)

@app.route('/v1/chat/completions', methods=['POST'])
def openai_compatible_chat_endpoint():
    data = request.json
    is_streaming = data.get("stream", False)
    
    messages = data.get("messages", [])
    if not messages: return jsonify({"error": "No messages provided"}), 400
        
    user_query = messages[-1].get("content", "")
    user_query = clean_html_from_text(user_query) # ADDED
    chat_history = messages[:-1]
    session_id = data.get("conversation_id", f"webui-session-{str(generate_uuid())}")
    user_id = None

    if is_streaming:
        logic_generator = generate_streaming_response(user_query, session_id, user_id, chat_history)
        return Response(stream_formatter(logic_generator, session_id, user_id, user_query), mimetype='text/event-stream')
    else:
        full_response_dict = get_hybrid_response_logic_non_streaming(user_query, session_id, user_id, chat_history)
        response_content = full_response_dict.get("answer", "Sorry, an error occurred.")
        
        openai_response = {
            "id": f"chatcmpl-{str(generate_uuid())}", "object": "chat.completion", "created": int(time.time()),
            "model": "MedicalAssisstantBot/v1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        history_manager.update_history(session_id, user_query, response_content)
        return jsonify(openai_response)


# --- Admin and Utility Routes (Unchanged) ---
@app.route('/')
def index_route():
    template_to_render = 'chat-bot.html'
    if not os.path.exists(os.path.join(app.root_path, 'templates', template_to_render)):
        logger.warning(f"Template '{template_to_render}' not found. Serving basic message.")
        return "Chatbot interface not found. Please ensure 'templates/chat-bot.html' exists.", 404
    return render_template(template_to_render)

@app.route('/admin/faiss_rag_status', methods=['GET'])
@require_admin_auth
def get_faiss_rag_status():
    global rag_system
    if not rag_system:
        return jsonify({"error": "FAISS RAG system not initialized."}), 500
    try:
        status = {
            "status": "Initialized" if rag_system.retriever else "Initialized (Retriever not ready)",
            "index_storage_dir": rag_system.index_storage_dir,
            "embedding_model": rag_system.embedding_model_name,
            "groq_model": rag_system.groq_model_name,
            "retriever_k": rag_system.retriever.k if rag_system.retriever else "N/A",
            "processed_source_files": rag_system.processed_source_files,
            "index_type": "FAISS",
            "index_loaded_or_built": rag_system.vector_store is not None
        }
        if rag_system.vector_store and hasattr(rag_system.vector_store, 'index') and rag_system.vector_store.index:
            try:
                status["num_vectors_in_index"] = rag_system.vector_store.index.ntotal
            except Exception:
                status["num_vectors_in_index"] = "N/A (Could not get count)"
        else:
            status["num_vectors_in_index"] = "N/A (Vector store or index not available)"
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting FAISS RAG status: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# NEW FUNCTION: Endpoint to download the combined QA databases as an Excel file
@app.route('/admin/download_qa_database', methods=['GET'])
@require_admin_auth
def download_qa_database():
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Safely access the dataframes from the embedding manager
            if embedding_manager.embeddings['general'] and embedding_manager.embeddings['general'].df_qa is not None:
                embedding_manager.embeddings['general'].df_qa.to_excel(writer, sheet_name='General_QA', index=False)
            
            if embedding_manager.embeddings['personal'] and embedding_manager.embeddings['personal'].df_qa is not None:
                embedding_manager.embeddings['personal'].df_qa.to_excel(writer, sheet_name='Personal_QA', index=False)

            if embedding_manager.embeddings['greetings'] and embedding_manager.embeddings['greetings'].df_qa is not None:
                embedding_manager.embeddings['g greetings'].df_qa.to_excel(writer, sheet_name='Greetings', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'qa_database_{datetime.now().strftime("%Y%m%d")}.xlsx'
        )
    except Exception as e:
        logger.error(f"Error generating QA database file: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate QA database file'}), 500

@app.route('/admin/rebuild_faiss_index', methods=['POST'])
@require_admin_auth
def rebuild_faiss_index_route():
    global rag_system
    logger.info("Admin request to rebuild FAISS RAG index received. Starting two-step process.")

    logger.info("Step 1: Running chunker.py to pre-process source documents.")
    chunker_script_path = os.path.join(_APP_BASE_DIR, 'chunker.py')
    chunked_json_output_path = os.path.join(RAG_STORAGE_PARENT_DIR, RAG_CHUNKED_SOURCES_FILENAME)

    os.makedirs(TEXT_EXTRACTIONS_DIR, exist_ok=True)

    if not os.path.exists(chunker_script_path):
        logger.error(f"Chunker script not found at '{chunker_script_path}'. Aborting rebuild.")
        return jsonify({"error": f"chunker.py not found. Cannot proceed with rebuild."}), 500

    command = [
        sys.executable,
        chunker_script_path,
        '--sources-dir', RAG_SOURCES_DIR,
        '--output-file', chunked_json_output_path,
        '--text-output-dir', TEXT_EXTRACTIONS_DIR
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info("Chunker script executed successfully.")
        logger.info(f"Chunker stdout:\n{process.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Chunker script failed with exit code {e.returncode}.")
        logger.error(f"Chunker stderr:\n{e.stderr}")
        return jsonify({"error": "Step 1 (Chunking) failed.", "details": e.stderr}), 500
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the chunker script: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during the chunking step: {str(e)}"}), 500

    logger.info("Step 2: Rebuilding FAISS index from the newly generated chunks.")
    try:
        new_rag_system_instance = initialize_and_get_rag_system(force_rebuild=True)

        if new_rag_system_instance and new_rag_system_instance.vector_store:
            rag_system = new_rag_system_instance
            logger.info("FAISS RAG index rebuild completed and new RAG system instance is active.")
            updated_status_response = get_faiss_rag_status()
            return jsonify({"message": "FAISS RAG index rebuild completed.", "status": updated_status_response.get_json()}), 200
        else:
            logger.error("FAISS RAG index rebuild failed during the indexing phase.")
            return jsonify({"error": "Step 2 (Indexing) failed. Check logs."}), 500

    except Exception as e:
        logger.error(f"Error during admin FAISS index rebuild (indexing phase): {e}", exc_info=True)
        return jsonify({"error": f"Failed to rebuild index during indexing phase: {str(e)}"}), 500

@app.route('/db/status', methods=['GET'])
@require_admin_auth
def get_personal_db_status():
    try:
        status_info = {
            'personal_data_csv_monitor_status': 'running',
            'file_exists': os.path.exists(personal_data_monitor.database_path),
            'data_loaded': personal_data_monitor.df is not None, 'last_update': None
        }
        if status_info['file_exists'] and os.path.getmtime(personal_data_monitor.database_path) is not None:
            status_info['last_update'] = datetime.fromtimestamp(os.path.getmtime(personal_data_monitor.database_path)).isoformat()
        return jsonify(status_info)
    except Exception as e: return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/report', methods=['GET'])
@require_admin_auth
def download_report():
    try:
        if not os.path.exists(CHAT_LOG_FILE) or os.path.getsize(CHAT_LOG_FILE) == 0:
            return jsonify({'error': 'No chat history available.'}), 404
        return send_file(CHAT_LOG_FILE, mimetype='text/csv', as_attachment=True, download_name=f'chat_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except Exception as e:
        logger.error(f"Error downloading report: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate report'}), 500

@app.route('/create-session', methods=['POST'])
def create_session_route():
    try:
        session_id = str(generate_uuid())
        logger.info(f"New session created: {session_id}")
        return jsonify({'status': 'success', 'session_id': session_id}), 200
    except Exception as e:
        logger.error(f"Session creation error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/version', methods=['GET'])
def get_version_route():
    return jsonify({'version': '3.9.8-Env-Chat-History'}), 200 # Updated version

@app.route('/clear-history', methods=['POST'])
def clear_session_history_route():
    data = request.json
    session_id = data.get('session_id')
    if not session_id: 
        return jsonify({'status': 'error', 'message': 'session_id is required'}), 400
    history_manager.clear_history(session_id)
    return jsonify({'status': 'success', 'message': f'History cleared for session {session_id}'})

# --- App Cleanup and Startup ---
def cleanup_application():
    if personal_data_monitor: personal_data_monitor.stop()
    logger.info("Application cleanup finished.")
atexit.register(cleanup_application)

def load_qa_data_on_startup():
    global embedding_manager
    # MODIFIED: Added print statement
    print("\n--- Loading QA Source Files ---")
    try:
        general_qa_path = os.path.join(RAG_SOURCES_DIR, 'general_qa.csv')
        personal_qa_path = os.path.join(RAG_SOURCES_DIR, 'personal_qa.csv')
        greetings_qa_path = os.path.join(RAG_SOURCES_DIR, 'greetings.csv')

        general_qa_df = pd.DataFrame(columns=['Question', 'Answer', 'Image'])
        personal_qa_df = pd.DataFrame(columns=['Question', 'Answer', 'Image'])
        greetings_qa_df = pd.DataFrame(columns=['Question', 'Answer', 'Image'])

        if os.path.exists(general_qa_path):
            try: 
                general_qa_df = pd.read_csv(general_qa_path, encoding='cp1252')
                print(f"- Loaded: {os.path.basename(general_qa_path)}")
            except Exception as e_csv: logger.error(f"Error reading general_qa.csv: {e_csv}")
        else:
            logger.warning(f"Optional file 'general_qa.csv' not found in '{RAG_SOURCES_DIR}'.")

        if os.path.exists(personal_qa_path):
            try: 
                personal_qa_df = pd.read_csv(personal_qa_path, encoding='cp1252')
                print(f"- Loaded: {os.path.basename(personal_qa_path)}")
            except Exception as e_csv: logger.error(f"Error reading personal_qa.csv: {e_csv}")
        else:
            logger.warning(f"Optional file 'personal_qa.csv' not found in '{RAG_SOURCES_DIR}'.")

        if os.path.exists(greetings_qa_path):
            try: 
                greetings_qa_df = pd.read_csv(greetings_qa_path, encoding='cp1252')
                print(f"- Loaded: {os.path.basename(greetings_qa_path)}")
            except Exception as e_csv: logger.error(f"Error reading greetings.csv: {e_csv}")
        else:
            logger.warning(f"Optional file 'greetings.csv' not found in '{RAG_SOURCES_DIR}'.")

        logger.info(f"Scanning for additional QA sources (.xlsx) in '{RAG_SOURCES_DIR}'...")
        if os.path.isdir(RAG_SOURCES_DIR):
            xlsx_files_found = [f for f in os.listdir(RAG_SOURCES_DIR) if f.endswith('.xlsx') and os.path.isfile(os.path.join(RAG_SOURCES_DIR, f))]
            
            if xlsx_files_found:
                all_general_dfs = [general_qa_df] if not general_qa_df.empty else []
                for xlsx_file in xlsx_files_found:
                    try:
                        xlsx_path = os.path.join(RAG_SOURCES_DIR, xlsx_file)
                        logger.info(f"Processing XLSX source file: {xlsx_file}")
                        df_excel = pd.read_excel(xlsx_path)

                        # MODIFIED: New logic to preserve all columns and handle dynamic headers
                        if 'Pregunta' in df_excel.columns and 'Respuesta' in df_excel.columns:
                            logger.info(f"Found 'Pregunta' and 'Respuesta' in {xlsx_file}. Preserving all columns.")
                            # The 'Question' column is required by the EmbeddingManager for semantic search.
                            # We create it from 'Pregunta' but keep all original columns.
                            df_excel['Question'] = df_excel['Pregunta']
                            all_general_dfs.append(df_excel)
                            print(f"- Loaded and processing: {xlsx_file}")
                        else:
                            logger.warning(f"Skipping XLSX file '{xlsx_file}' as it lacks the required 'Pregunta' and 'Respuesta' columns.")
                    except Exception as e_xlsx:
                        logger.error(f"Error processing XLSX file '{xlsx_file}': {e_xlsx}")
                
                if len(all_general_dfs) > 0:
                    general_qa_df = pd.concat(all_general_dfs, ignore_index=True)
                    logger.info(f"Successfully merged data from {len(xlsx_files_found)} XLSX file(s) into the general QA set.")
        else:
            logger.warning(f"Sources directory '{RAG_SOURCES_DIR}' not found. Cannot scan for additional QA files.")

        dataframes_to_process = {
            "general": general_qa_df,
            "personal": personal_qa_df,
            "greetings": greetings_qa_df
        }

        for df_name, df_val in dataframes_to_process.items():
            if df_val.empty: continue

            # Normalize text in all columns to prevent issues
            for col in df_val.columns:
                 if not df_val[col].isnull().all():
                     df_val[col] = df_val[col].astype(str).apply(normalize_text)

            # Ensure 'Question' column exists for embedding manager compatibility
            if 'Question' not in df_val.columns:
                # For CSVs that might not have 'Pregunta' but have 'Question'
                if 'Question' in df_val.columns:
                     pass # Already exists
                else:
                    df_val['Question'] = None
                    logger.warning(f"'Question' column missing in {df_name} data. Added empty column.")

        embedding_manager.update_embeddings(
            dataframes_to_process["general"],
            dataframes_to_process["personal"],
            dataframes_to_process["greetings"]
        )
        logger.info("CSV & XLSX QA data loaded and embeddings initialized.")

    except Exception as e:
        logger.critical(f"CRITICAL: Error loading or processing QA data: {e}. Semantic QA may not function.", exc_info=True)
    # MODIFIED: Added print statement
    print("-----------------------------\n")

if __name__ == '__main__':
    for folder_path in [os.path.join(_APP_BASE_DIR, 'templates'),
                        os.path.join(_APP_BASE_DIR, 'static'),
                        TEXT_EXTRACTIONS_DIR]:
        os.makedirs(folder_path, exist_ok=True)

    load_qa_data_on_startup()
    initialize_chat_log()

    logger.info("Attempting to initialize RAG system from llm_handling module...")
    rag_system = initialize_and_get_rag_system()
    if rag_system:
        logger.info("RAG system initialized successfully via llm_handling module.")
    else:
        logger.warning("RAG system failed to initialize. Document RAG functionality will be unavailable.")

    logger.info(f"Flask application starting with Hybrid RAG on {FLASK_APP_HOST}:{FLASK_APP_PORT} Debug: {FLASK_DEBUG_MODE}...")
    if not FLASK_DEBUG_MODE:
        werkzeug_log = logging.getLogger('werkzeug')
        werkzeug_log.setLevel(logging.ERROR)

    app.run(host=FLASK_APP_HOST, port=FLASK_APP_PORT, debug=FLASK_DEBUG_MODE, use_reloader=False)