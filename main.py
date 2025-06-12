import os
import json
import re
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calculate
from sentence_transformers import SentenceTransformer, util as sbert_util
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# ===== CONFIGURATION =====
load_dotenv()

AZURE_AI_API_KEY = os.getenv('AZURE_AI_API_KEY')
AZURE_AI_ENDPOINT = os.getenv('AZURE_AI_ENDPOINT')

LLM_CONFIGURATIONS = [
    {
        "name": "Llama-3.2-3B-Instruct", 
        "type": "hf_local",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct", 
        "requires_hf_login": True
    },
    {
        "name": "GPT-4.1",
        "type": "azure_ai_inference",
        "azure_ai_endpoint": AZURE_AI_ENDPOINT,
        "azure_ai_model_name": "gpt-4.1",
        "azure_api_key": AZURE_AI_API_KEY
    },
    {
        "name": "DeepSeek-R1",
        "type": "azure_ai_inference",
        "azure_ai_endpoint": AZURE_AI_ENDPOINT,
        "azure_ai_model_name": "deepseek-r1",
        "azure_api_key": AZURE_AI_API_KEY,
        "hide_reasoning": True
    }
]

JUDGE_LLM_CONFIG = {
    "type": "azure_ai_inference",
    "hf_model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "azure_ai_endpoint": AZURE_AI_ENDPOINT,
    "azure_ai_model_name": "gpt-4.1",
    "azure_api_key": AZURE_AI_API_KEY
}

MAX_TEXTS_TO_PROCESS = 500
SQUAD_JSON_FILE_PATH = './data/train-v2.0.json'  # Original SQuAD format
CUSTOM_JSON_FILE_PATH = './data/squad_500.json'  # New custom format
PLOTS_OUTPUT_DIR = 'rephrasing_plots_azure_ai_inf'

# Choose which data file to use
USE_CUSTOM_FORMAT = True  # Set to True for new format, False for original SQuAD
DATA_FILE_PATH = CUSTOM_JSON_FILE_PATH if USE_CUSTOM_FORMAT else SQUAD_JSON_FILE_PATH

# Global variables
current_hf_model, current_hf_tokenizer, current_hf_device = None, None, None
qa_validation_pipeline = None
sbert_model = None
azure_ai_inference_clients = {}

# ===== UTILITY FUNCTIONS =====
def ensure_dir(dir_path):
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path)
        print(f"Created dir: {dir_path}")

def download_nltk_punkt_once():
    try: 
        nltk.data.find('tokenizers/punkt')
    except: 
        print("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded.")

def load_sbert_model():
    global sbert_model
    if sbert_model is None:
        print("Loading SBERT model...")
        try: 
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SBERT model loaded.")
        except Exception as e: 
            print(f"Error loading SBERT: {e}")
            sbert_model = None

def load_qa_pipeline(device):
    global qa_validation_pipeline
    if qa_validation_pipeline is None:
        print("Loading auxiliary QA pipeline...")
        try:
            device_idx = 0 if device.type == 'cuda' else -1
            qa_validation_pipeline = hf_pipeline(
                "question-answering", 
                model="distilbert-base-cased-distilled-squad", 
                device=device_idx
            )
            print("QA pipeline loaded.")
        except Exception as e: 
            print(f"Error loading QA pipeline: {e}")
            qa_validation_pipeline = None

# ===== NEW: JACCARD SIMILARITY SCORE CALCULATION =====
def calculate_jaccard_score(original_text, rephrased_text, mode='token'):
    """
    Calculate Jaccard similarity between original and rephrased text.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Args:
        original_text: Original text
        rephrased_text: Rephrased text  
        mode: 'token' for word-level, 'char' for character-level, 'ngram' for n-gram level
    
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    try:
        if not original_text or not rephrased_text:
            return 0.0
        
        if mode == 'token':
            # Word-level Jaccard similarity
            orig_tokens = set(original_text.lower().split())
            repr_tokens = set(rephrased_text.lower().split())
            
        elif mode == 'char':
            # Character-level Jaccard similarity
            orig_tokens = set(original_text.lower())
            repr_tokens = set(rephrased_text.lower())
            
        elif mode == 'ngram':
            # Bigram-level Jaccard similarity
            def get_ngrams(text, n=2):
                tokens = text.lower().split()
                return set(zip(tokens[i:], tokens[i+1:]) for i in range(len(tokens)-n+1))
            
            orig_tokens = get_ngrams(original_text)
            repr_tokens = get_ngrams(rephrased_text)
            
        else:
            # Default to token-level
            orig_tokens = set(original_text.lower().split())
            repr_tokens = set(rephrased_text.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(orig_tokens.intersection(repr_tokens))
        union = len(orig_tokens.union(repr_tokens))
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        return jaccard_score
        
    except Exception as e:
        print(f"Error calculating Jaccard score: {e}")
        return 0.0

def calculate_comprehensive_jaccard_scores(original_text, rephrased_text):
    """
    Calculate multiple Jaccard similarity scores for comprehensive analysis.
    
    Returns:
        Dict with different Jaccard similarity measurements
    """
    scores = {}
    
    # Token-level Jaccard similarity
    scores['jaccard_token'] = calculate_jaccard_score(original_text, rephrased_text, mode='token')
    
    # Character-level Jaccard similarity  
    scores['jaccard_char'] = calculate_jaccard_score(original_text, rephrased_text, mode='char')
    
    # N-gram level Jaccard similarity
    scores['jaccard_bigram'] = calculate_jaccard_score(original_text, rephrased_text, mode='ngram')
    
    # Average Jaccard score across methods
    jaccard_values = [scores['jaccard_token'], scores['jaccard_char'], scores['jaccard_bigram']]
    scores['jaccard_average'] = np.mean(jaccard_values)
    
    return scores

# ===== NEW: CONFUSION MATRIX FUNCTIONS =====
def create_rejection_confusion_matrix(results, output_dir):
    """
    Create confusion matrix for rejection analysis.
    True labels: 1 = should be answerable (kept), 0 = should be unanswerable (omitted)
    Predicted labels: 1 = model can answer, 0 = model cannot answer
    """
    print("Creating rejection confusion matrix...")
    
    try:
        all_true_labels = []
        all_pred_labels = []
        model_data = {}
        
        for result in results:
            if 'llm_qa_results' not in result or 'validation_scores' not in result:
                continue
                
            model_name = result['llm_name']
            if model_name not in model_data:
                model_data[model_name] = {'true_labels': [], 'pred_labels': []}
            
            # Get kept questions results
            kept_qa_results = result['llm_qa_results']
            kept_answerability = kept_qa_results.get('answerability', [])
            
            # Get omitted questions results (from validation scores)
            validation_scores = result['validation_scores']
            num_omitted = validation_scores.get('num_omitted_questions', 0)
            num_omitted_unanswerable = validation_scores.get('num_omitted_unanswerable', 0)
            
            # Add kept questions (should be answerable = 1)
            for is_answerable in kept_answerability:
                true_label = 1  # Should be answerable (kept)
                pred_label = 1 if is_answerable else 0  # Model's prediction
                
                all_true_labels.append(true_label)
                all_pred_labels.append(pred_label)
                model_data[model_name]['true_labels'].append(true_label)
                model_data[model_name]['pred_labels'].append(pred_label)
            
            # Add omitted questions (should be unanswerable = 0)
            if num_omitted > 0:
                # Add unanswerable omitted questions
                for _ in range(num_omitted_unanswerable):
                    true_label = 0  # Should be unanswerable (omitted)
                    pred_label = 0  # Model correctly couldn't answer
                    
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_label)
                    model_data[model_name]['true_labels'].append(true_label)
                    model_data[model_name]['pred_labels'].append(pred_label)
                
                # Add answerable omitted questions (model failed to reject)
                num_omitted_answerable = num_omitted - num_omitted_unanswerable
                for _ in range(num_omitted_answerable):
                    true_label = 0  # Should be unanswerable (omitted)
                    pred_label = 1  # Model incorrectly could answer
                    
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_label)
                    model_data[model_name]['true_labels'].append(true_label)
                    model_data[model_name]['pred_labels'].append(pred_label)
        
        if not all_true_labels:
            print("No data available for confusion matrix")
            return
        
        # Create overall confusion matrix
        cm_overall = confusion_matrix(all_true_labels, all_pred_labels)
        
        # Plot overall confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cannot Answer', 'Can Answer'],
                   yticklabels=['Should be Unanswerable (Omitted)', 'Should be Answerable (Kept)'])
        plt.title('Overall Rejection Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Model Prediction', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rejection_confusion_matrix_overall.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create per-model confusion matrices
        fig, axes = plt.subplots(1, len(model_data), figsize=(6*len(model_data), 5))
        if len(model_data) == 1:
            axes = [axes]
        
        for idx, (model_name, data) in enumerate(model_data.items()):
            if len(data['true_labels']) == 0:
                continue
                
            cm_model = confusion_matrix(data['true_labels'], data['pred_labels'])
            
            sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Cannot Answer', 'Can Answer'],
                       yticklabels=['Should be Unanswerable', 'Should be Answerable'])
            axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Model Prediction')
            if idx == 0:
                axes[idx].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rejection_confusion_matrix_by_model.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print classification report
        print("\n=== REJECTION CLASSIFICATION REPORT ===")
        report = classification_report(all_true_labels, all_pred_labels, 
                                     target_names=['Should be Unanswerable (Omitted)', 'Should be Answerable (Kept)'])
        print(report)
        
        # Save detailed metrics
        with open(os.path.join(output_dir, 'rejection_classification_report.txt'), 'w') as f:
            f.write("REJECTION CLASSIFICATION REPORT\n")
            f.write("================================\n\n")
            f.write("True labels: 1 = should be answerable (kept), 0 = should be unanswerable (omitted)\n")
            f.write("Predicted labels: 1 = model can answer, 0 = model cannot answer\n\n")
            f.write(report)
        
        print("Confusion matrices saved to output directory")
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")

# ===== AZURE AI FUNCTIONS =====
def get_azure_ai_inference_client(llm_config):
    endpoint = llm_config.get("azure_ai_endpoint")
    api_key = llm_config.get("azure_api_key")
    
    if not endpoint or not api_key:
        return None
    
    if endpoint in azure_ai_inference_clients:
        return azure_ai_inference_clients[endpoint]
    
    try:
        client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        azure_ai_inference_clients[endpoint] = client
        return client
    except Exception as e:
        print(f"Error creating Azure client: {e}")
        return None

def clean_deepseek_response(response_text, llm_config):
    if not llm_config.get("hide_reasoning", False):
        return response_text
    
    cleaned = response_text
    
    # Remove thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    
    # Remove reasoning patterns
    reasoning_patterns = [
        r'\*\*Reasoning:\*\*.*?(?=\n\n|\*\*|$)',
        r'\*\*Analysis:\*\*.*?(?=\n\n|\*\*|$)',
        r'Let me analyze.*?(?=\n\n|\.)',
        r'I need to.*?(?=\n\n|\.)',
        r'\[Internal reasoning.*?\]',
    ]
    
    for pattern in reasoning_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    
    return cleaned if len(cleaned.strip()) >= 20 else response_text

# ===== HUGGING FACE FUNCTIONS =====
def load_hf_model_and_tokenizer(model_id, device, for_judge=False):
    global current_hf_model, current_hf_tokenizer, current_hf_device
    print(f"Loading HF model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
        
        if not for_judge: 
            current_hf_model, current_hf_tokenizer, current_hf_device = model, tokenizer, device
        
        print(f"Successfully loaded {model_id}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading HF model {model_id}: {e}")
        if not for_judge: 
            current_hf_model = None
            current_hf_tokenizer = None
        raise

def release_hf_model(model_to_release=None, tokenizer_to_release=None, is_global_rephraser=True):
    global current_hf_model, current_hf_tokenizer, current_hf_device
    
    m, t = (current_hf_model, current_hf_tokenizer) if is_global_rephraser else (model_to_release, tokenizer_to_release)
    
    if m: 
        del m
    if t: 
        del t
    
    if is_global_rephraser: 
        current_hf_model, current_hf_tokenizer, current_hf_device = None, None, None
    
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    
    print("HF model released.")

def _generate_hf_response(model, tokenizer, prompt_text, max_new_tokens, temperature=0.6):
    device = next(model.parameters()).device
    max_len = getattr(model.config, 'max_position_embeddings', 4096)
    max_input_len = max_len - max_new_tokens - 100
    
    inputs = tokenizer(
        prompt_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_input_len
    ).to(device)
    
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.pad_token_id, 
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ===== DATA LOADING =====
def load_official_squad_json(file_path):
    """
    Load JSON data with the new simplified format:
    [
        {
            "title": "...",
            "full_context": "...", 
            "questions_details": [
                {
                    "id": "...",
                    "question": "...",
                    "answers_text": [...],
                    "is_impossible": false
                }
            ]
        }
    ]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f: 
            json_data = json.load(f)
        
        processed_data = []
        
        # Handle both list format and single object format
        if isinstance(json_data, dict):
            # If it's a single object, wrap it in a list
            json_data = [json_data] 
        elif isinstance(json_data, dict) and 'data' in json_data:
            # If it's the old SQuAD format, handle it
            return load_original_squad_format(json_data)
        
        for entry_idx, entry in enumerate(json_data):
            title = entry.get('title', f'Unknown_Title_{entry_idx}')
            context = entry.get('full_context', '')
            questions_list = entry.get('questions_details', [])
            
            if not context or not questions_list:
                continue
            
            # Filter for answerable questions only
            answerable_questions = []
            for q_idx, qa in enumerate(questions_list):
                if not qa.get('is_impossible', False) and qa.get('question') and qa.get('answers_text'):
                    answerable_questions.append({
                        'question': qa['question'], 
                        'original_answers': qa.get('answers_text', []), 
                        'id': qa.get('id', f"{title}_q{q_idx}")
                    })
            
            if answerable_questions:
                processed_data.append({
                    "entry_id": f"{title}_{entry_idx}", 
                    "title": title, 
                    "context": context, 
                    "answerable_question_objects": answerable_questions
                })
        
        print(f"Loaded {len(processed_data)} entries from JSON file")
        print(f"Sample entry: {processed_data[0]['title'] if processed_data else 'None'}")
        return processed_data
        
    except Exception as e: 
        print(f"Error loading JSON data: {e}")
        return None

def create_sample_custom_json(output_path='./data/sample_custom_format.json'):
    """
    Create a sample JSON file in the new custom format to help users understand the structure.
    """
    sample_data = [
        {
            "title": "Sample_Topic_1",
            "full_context": "This is a sample context paragraph containing information about a topic. It includes various facts and details that can be used to answer questions. The paragraph should be substantial enough to contain information for multiple questions.",
            "questions_details": [
                {
                    "id": "sample_q1",
                    "question": "What is the main topic of this paragraph?",
                    "answers_text": ["sample topic", "topic"],
                    "is_impossible": False
                },
                {
                    "id": "sample_q2", 
                    "question": "What type of information does the paragraph contain?",
                    "answers_text": ["facts and details", "various facts"],
                    "is_impossible": False
                },
                {
                    "id": "sample_q3_impossible",
                    "question": "What color is the paragraph?",
                    "answers_text": [],
                    "is_impossible": True
                }
            ]
        },
        {
            "title": "Sample_Topic_2", 
            "full_context": "Another example paragraph with different content. This paragraph discusses a completely different subject matter and provides unique information that would require different questions to extract the key details.",
            "questions_details": [
                {
                    "id": "sample_q4",
                    "question": "What does this paragraph discuss?",
                    "answers_text": ["different subject matter", "completely different subject"],
                    "is_impossible": False
                },
                {
                    "id": "sample_q5",
                    "question": "What type of information does this paragraph provide?",
                    "answers_text": ["unique information"],
                    "is_impossible": False
                }
            ]
        }
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample custom format JSON created at: {output_path}")
    return sample_data

def load_original_squad_format(squad_data):
    """
    Fallback function to handle original SQuAD format for backward compatibility.
    """
    try:
        processed_data = []
        
        for topic_data in squad_data.get('data', []):
            title = topic_data.get('title', 'Unknown_Title')
            
            for para_idx, paragraph in enumerate(topic_data.get('paragraphs', [])):
                context = paragraph.get('context')
                qas_list = paragraph.get('qas', [])
                
                answerable_questions = [
                    {
                        'question': qa['question'], 
                        'original_answers': [ans['text'] for ans in qa.get('answers', [])], 
                        'id': qa.get('id', f"{title}_p{para_idx}_q{i}")
                    }
                    for i, qa in enumerate(qas_list) 
                    if not qa.get('is_impossible', False) and qa.get('question') and qa.get('answers')
                ]
                
                if context and answerable_questions:
                    processed_data.append({
                        "entry_id": f"{title}_p{para_idx}", 
                        "title": title, 
                        "context": context, 
                        "answerable_question_objects": answerable_questions
                    })
        
        print(f"Loaded {len(processed_data)} entries from original SQuAD format")
        return processed_data
        
    except Exception as e: 
        print(f"Error loading original SQuAD data: {e}")
        return None
    """
    Fallback function to handle original SQuAD format for backward compatibility.
    """
    try:
        processed_data = []
        
        for topic_data in squad_data.get('data', []):
            title = topic_data.get('title', 'Unknown_Title')
            
            for para_idx, paragraph in enumerate(topic_data.get('paragraphs', [])):
                context = paragraph.get('context')
                qas_list = paragraph.get('qas', [])
                
                answerable_questions = [
                    {
                        'question': qa['question'], 
                        'original_answers': [ans['text'] for ans in qa.get('answers', [])], 
                        'id': qa.get('id', f"{title}_p{para_idx}_q{i}")
                    }
                    for i, qa in enumerate(qas_list) 
                    if not qa.get('is_impossible', False) and qa.get('question') and qa.get('answers')
                ]
                
                if context and answerable_questions:
                    processed_data.append({
                        "entry_id": f"{title}_p{para_idx}", 
                        "title": title, 
                        "context": context, 
                        "answerable_question_objects": answerable_questions
                    })
        
        print(f"Loaded {len(processed_data)} entries from original SQuAD format")
        return processed_data
        
    except Exception as e: 
        print(f"Error loading original SQuAD data: {e}")
        return None

# ===== TEXT GENERATION =====
def generate_rewrite_hf_local(original_passage, questions_to_keep, questions_to_omit, system_prompt, user_prompt_template):
    if not current_hf_model or not current_hf_tokenizer: 
        return "Error: Local HF model not loaded."
    
    formatted_qs_keep = "".join(f"- {q}\n" for q in questions_to_keep) if questions_to_keep else "N/A\n"
    formatted_qs_omit = "".join(f"- {q}\n" for q in questions_to_omit) if questions_to_omit else "N/A\n"
    
    user_msg = user_prompt_template.format(
        original_passage=original_passage, 
        formatted_questions_to_keep=formatted_qs_keep, 
        formatted_questions_to_omit=formatted_qs_omit
    )
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_msg}
    ]
    
    try:
        prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt_text = f"System: {system_prompt}\nUser: {user_msg}\nAssistant:"

    try:
        max_new_tokens = min(int(len(current_hf_tokenizer.encode(original_passage)) * 1.3) + 100, 1024)
        rewritten = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens)
        
        if rewritten and rewritten.lower() != original_passage.lower():
            return rewritten
        else:
            return "Model did not provide a distinct rewrite."
            
    except Exception as e: 
        return f"Error in HF rephrasing: {e}"

def generate_rewrite_azure_ai_inference(original_passage, questions_to_keep, questions_to_omit, llm_config, system_prompt, user_prompt_template):
    client = get_azure_ai_inference_client(llm_config)
    if not client: 
        return f"Error: Azure client not configured for {llm_config['name']}"
    
    model_name = llm_config.get("azure_ai_model_name")
    if not model_name: 
        return f"Error: Model name missing for {llm_config['name']}"

    formatted_qs_keep = "".join(f"- {q}\n" for q in questions_to_keep) if questions_to_keep else "N/A\n"
    formatted_qs_omit = "".join(f"- {q}\n" for q in questions_to_omit) if questions_to_omit else "N/A\n"
    
    # Enhanced system prompt for DeepSeek
    enhanced_system_prompt = system_prompt
    if llm_config.get("hide_reasoning", False):
        enhanced_system_prompt += (
            "\n\nIMPORTANT: Provide ONLY the final rewritten passage. "
            "No reasoning, analysis, explanations, or thinking process. "
            "Just the clean, rewritten text."
        )
    
    user_msg = user_prompt_template.format(
        original_passage=original_passage, 
        formatted_questions_to_keep=formatted_qs_keep, 
        formatted_questions_to_omit=formatted_qs_omit
    )
    
    messages = [
        SystemMessage(content=enhanced_system_prompt),
        UserMessage(content=user_msg)
    ]

    try:
        response = client.complete(
            messages=messages,
            model=model_name,
            max_tokens=1536,
            temperature=0.2
        )
        
        if hasattr(response, 'choices') and response.choices:
            rewritten = response.choices[0].message.content.strip()
            rewritten = clean_deepseek_response(rewritten, llm_config)
            
            if rewritten and rewritten.lower() != original_passage.lower():
                return rewritten
            else:
                return "Model did not provide a distinct rewrite"
        else:
            return "Error: No response content from Azure AI"
            
    except Exception as e:
        return f"Error in Azure AI call: {e}"

def get_llm_answers_from_rephrased_text(rephrased_text, questions_to_ask, llm_config):
    """
    Modified to handle answerability detection for rejection accuracy calculation.
    Returns both answers and answerability predictions.
    """
    if not rephrased_text or rephrased_text.startswith("Error:") or "Model did not provide" in rephrased_text:
        return {
            'answers': ["Error: Invalid rephrased text"] * len(questions_to_ask),
            'answerability': [False] * len(questions_to_ask)
        }
    
    qa_system_prompt = (
        "Based ONLY on the provided Context, answer the Question. "
        "If the Context does not contain the information needed to answer the question, "
        "respond with 'UNANSWERABLE' and nothing else. "
        "Only provide answers that can be directly supported by the given context."
    )
    
    llm_answers = []
    answerability_predictions = []
    
    try:
        llm_type = llm_config["type"]
        
        if llm_type == "hf_local":
            if not current_hf_model or not current_hf_tokenizer: 
                return {
                    'answers': ["Error: HF model not available"] * len(questions_to_ask),
                    'answerability': [False] * len(questions_to_ask)
                }
                
            for question in questions_to_ask:
                qa_prompt = f"Context: \"{rephrased_text}\"\n\nQuestion: \"{question}\"\n\nAnswer:"
                messages = [{"role": "system", "content": qa_system_prompt}, {"role": "user", "content": qa_prompt}]
                prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                answer = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=100, temperature=0.2)
                
                # Check if model indicated unanswerable
                is_answerable = not (
                    answer and (
                        answer.upper() == "UNANSWERABLE" or
                        "unanswerable" in answer.lower() or
                        "cannot be answered" in answer.lower() or
                        "information not found" in answer.lower() or
                        "not in context" in answer.lower()
                    )
                )
                
                llm_answers.append(answer or "No answer provided")
                answerability_predictions.append(is_answerable)
        
        elif llm_type == "azure_ai_inference":
            client = get_azure_ai_inference_client(llm_config)
            if not client: 
                return {
                    'answers': [f"Error: Azure client not available"] * len(questions_to_ask),
                    'answerability': [False] * len(questions_to_ask)
                }
                
            model_name = llm_config["azure_ai_model_name"]
            
            for question in questions_to_ask:
                enhanced_qa_prompt = qa_system_prompt
                if llm_config.get("hide_reasoning", False):
                    enhanced_qa_prompt += " Provide ONLY the direct answer or 'UNANSWERABLE'. No reasoning or explanations."
                
                qa_content = f"Context:\n\"{rephrased_text}\"\n\nQuestion:\n\"{question}\"\n\nAnswer:"
                messages = [
                    SystemMessage(content=enhanced_qa_prompt), 
                    UserMessage(content=qa_content)
                ]
                
                response = client.complete(
                    messages=messages, 
                    model=model_name, 
                    max_tokens=100, 
                    temperature=0.2
                )
                
                if hasattr(response, 'choices') and response.choices:
                    answer = response.choices[0].message.content.strip()
                    answer = clean_deepseek_response(answer, llm_config)
                    
                    # Check if model indicated unanswerable
                    is_answerable = not (
                        answer and (
                            answer.upper() == "UNANSWERABLE" or
                            "unanswerable" in answer.lower() or
                            "cannot be answered" in answer.lower() or
                            "information not found" in answer.lower() or
                            "not in context" in answer.lower()
                        )
                    )
                    
                    llm_answers.append(answer or "No answer provided")
                    answerability_predictions.append(is_answerable)
                else:
                    llm_answers.append("Error: No response from Azure")
                    answerability_predictions.append(False)
        else:
            llm_answers = [f"Error: Unknown LLM type '{llm_type}'"] * len(questions_to_ask)
            answerability_predictions = [False] * len(questions_to_ask)
            
    except Exception as e:
        print(f"Exception during LLM QA: {e}")
        remaining = len(questions_to_ask) - len(llm_answers)
        llm_answers.extend([f"Error: {e}"] * remaining)
        answerability_predictions.extend([False] * remaining)
        
    return {
        'answers': llm_answers,
        'answerability': answerability_predictions
    }

# ===== EVALUATION FUNCTIONS =====
def get_llm_judge_score_from_text(score_text):
    if not isinstance(score_text, str): 
        return 0.0
    
    # Extract numbers
    numbers = [int(s) for s in score_text.split() if s.isdigit()]
    if numbers and 1 <= numbers[0] <= 5: 
        return float(numbers[0])
    
    # Look for pattern like "3/5" or "3 out of 5"
    match = re.search(r"(\b[1-5]\b)(?:/5| out of 5)?", score_text)
    if match: 
        return float(match.group(1))
    
    return 0.0

def evaluate_with_llm_judge(original_text, rephrased_text, questions_to_keep_objs, questions_to_omit_strings, llm_qa_results, judge_config):
    """
    Modified to handle the new QA results structure with answerability.
    """
    judge_scores = {"judge_rephrase_quality_score": 0.0, "judge_kept_q_answer_correctness_score": 0.0}
    
    if not rephrased_text or rephrased_text.startswith("Error:"):
        return judge_scores

    # Extract answers from the new structure
    llm_answers_for_kept_qs = llm_qa_results.get('answers', [])
    kept_q_list = [q_obj['question'] for q_obj in questions_to_keep_objs]
    
    rephrase_sys_prompt = "Rate (1-5, 5=best): Rephrased Text vs Original. Consider inclusion of keep questions, omission of omit questions, coherence. ONLY provide the score."
    rephrase_user_prompt = f"Original:\n{original_text}...\n\nRephrased:\n{rephrased_text}...\n\nKeep Info For:\n{chr(10).join(f'- {q}...' for q in kept_q_list)}\n\nOmit Info For:\n{chr(10).join(f'- {q}...' for q in questions_to_omit_strings)}\n\nScore (1-5):"
    
    ans_sys_prompt = "Rate LLM's answer vs SQuAD answer (1-5, 5=best). Consider correctness and equivalence. ONLY provide the score."
    
    temp_judge_model, temp_judge_tokenizer = None, None

    try:
        judge_type = judge_config.get("type")
        
        if judge_type == "hf_local":
            model_id = judge_config.get("hf_model_id")
            if not model_id: 
                return judge_scores
                
            judge_device = current_hf_device if current_hf_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            temp_judge_model, temp_judge_tokenizer = load_hf_model_and_tokenizer(model_id, judge_device, for_judge=True)
            
            # Judge rephrase quality
            messages = [{"role":"system","content":rephrase_sys_prompt}, {"role":"user","content":rephrase_user_prompt}]
            prompt = temp_judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rephrase_score_text = _generate_hf_response(temp_judge_model, temp_judge_tokenizer, prompt, max_new_tokens=10, temperature=0.1)

            # Judge answer correctness
            ans_scores = []
            for i, q_obj in enumerate(questions_to_keep_objs):
                if i < len(llm_answers_for_kept_qs) and not llm_answers_for_kept_qs[i].startswith("Error:"):
                    llm_ans = llm_answers_for_kept_qs[i]
                    squad_ans = q_obj['original_answers'][0] if q_obj['original_answers'] else 'N/A'
                    
                    ans_user_prompt = f"Question:\n\"{q_obj['question']}...\"\nSQuAD Answer:\n\"{squad_ans}...\"\nLLM Answer:\n\"{llm_ans}...\"\nScore (1-5):"
                    ans_messages = [{"role":"system","content":ans_sys_prompt}, {"role":"user","content":ans_user_prompt}]
                    ans_prompt = temp_judge_tokenizer.apply_chat_template(ans_messages, tokenize=False, add_generation_prompt=True)
                    ans_score_text = _generate_hf_response(temp_judge_model, temp_judge_tokenizer, ans_prompt, max_new_tokens=10, temperature=0.1)
                    
                    score_val = get_llm_judge_score_from_text(ans_score_text)
                    if score_val > 0:
                        ans_scores.append(score_val)
            
            avg_ans_score = np.mean(ans_scores) if ans_scores else 0.0
        
        elif judge_type == "azure_ai_inference":
            judge_client = get_azure_ai_inference_client(judge_config)
            judge_model = judge_config.get("azure_ai_model_name")
            
            if not judge_client or not judge_model: 
                return judge_scores

            # Judge rephrase quality
            messages = [SystemMessage(content=rephrase_sys_prompt), UserMessage(content=rephrase_user_prompt)]
            response = judge_client.complete(messages=messages, model=judge_model, max_tokens=10, temperature=0.1)
            
            rephrase_score_text = ""
            if hasattr(response, 'choices') and response.choices:
                rephrase_score_text = response.choices[0].message.content.strip()
            
            # Judge answer correctness
            ans_scores = []
            for i, q_obj in enumerate(questions_to_keep_objs):
                if i < len(llm_answers_for_kept_qs) and not llm_answers_for_kept_qs[i].startswith("Error:"):
                    llm_ans = llm_answers_for_kept_qs[i]
                    squad_ans = q_obj['original_answers'][0] if q_obj['original_answers'] else 'N/A'
                    
                    ans_user_prompt = f"Question:\n\"{q_obj['question']}...\"\nSQuAD Answer:\n\"{squad_ans}...\"\nLLM Answer:\n\"{llm_ans}...\"\nScore (1-5):"
                    ans_messages = [SystemMessage(content=ans_sys_prompt), UserMessage(content=ans_user_prompt)]
                    ans_response = judge_client.complete(messages=ans_messages, model=judge_model, max_tokens=10, temperature=0.1)
                    
                    ans_score_text = ""
                    if hasattr(ans_response, 'choices') and ans_response.choices:
                        ans_score_text = ans_response.choices[0].message.content.strip()
                    
                    score_val = get_llm_judge_score_from_text(ans_score_text)
                    if score_val > 0:
                        ans_scores.append(score_val)
            
            avg_ans_score = np.mean(ans_scores) if ans_scores else 0.0
        else:
            rephrase_score_text = ""
            avg_ans_score = 0.0

    except Exception as e: 
        print(f"Error in LLM judge: {e}")
        rephrase_score_text = ""
        avg_ans_score = 0.0
    finally:
        if temp_judge_model:
            release_hf_model(model_to_release=temp_judge_model, tokenizer_to_release=temp_judge_tokenizer, is_global_rephraser=False)

    judge_scores["judge_rephrase_quality_score"] = get_llm_judge_score_from_text(rephrase_score_text)
    judge_scores["judge_kept_q_answer_correctness_score"] = avg_ans_score
    return judge_scores

def calculate_comprehensive_validation_scores(original_text, rephrased_text, q_keep_objects, q_omit_strings, 
                                            llm_qa_results, llm_judge_eval_scores, llm_config):
    """
    Updated with corrected rejection accuracy calculation and Jaccard similarity scores.
    """
    scores = {}
    
    # Handle error cases
    if not rephrased_text or rephrased_text.startswith("Error:") or "Model did not provide" in rephrased_text:
        error_metrics = ['rouge1_f', 'rougeL_f', 'sbert_similarity_context', 'bert_score_rephrased_vs_original_f1', 
                        'qa_inclusion_rate_by_aux_qa', 'llm_avg_answer_sbert_similarity_kept_q',
                        'avg_qa_omission_success_rate', 'rejection_accuracy', 'jaccard_token', 'jaccard_char', 
                        'jaccard_bigram', 'jaccard_average', 'len_o', 'len_r', 'len_ratio',
                        'judge_rephrase_quality_score', 'judge_kept_q_answer_correctness_score']
        for k in error_metrics: 
            scores[k] = 0.0
        scores['error_message'] = rephrased_text if isinstance(rephrased_text, str) else "Generation error"
        return scores

    device = current_hf_device if current_hf_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract answers and answerability predictions from new structure
    llm_answers_for_kept_qs = llm_qa_results.get('answers', [])
    kept_answerability_predictions = llm_qa_results.get('answerability', [])
    
    # Calculate Jaccard Similarity Scores
    jaccard_scores = calculate_comprehensive_jaccard_scores(original_text, rephrased_text)
    scores.update(jaccard_scores)
    
    # ROUGE scores
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rs = scorer.score(original_text, rephrased_text)
        scores.update({'rouge1_f': rs['rouge1'].fmeasure, 'rougeL_f': rs['rougeL'].fmeasure})
    except: 
        scores.update({'rouge1_f': 0.0, 'rougeL_f': 0.0})
    
    # BERTScore
    try:
        _, _, F1_bert = bert_score_calculate([rephrased_text], [original_text], lang="en", verbose=False, device=str(device))
        scores['bert_score_rephrased_vs_original_f1'] = F1_bert.mean().item()
    except: 
        scores['bert_score_rephrased_vs_original_f1'] = 0.0
    
    # SBERT similarity
    if sbert_model:
        try:
            emb_orig = sbert_model.encode(original_text, convert_to_tensor=True, device=device)
            emb_repr = sbert_model.encode(rephrased_text, convert_to_tensor=True, device=device)
            scores['sbert_similarity_context'] = sbert_util.pytorch_cos_sim(emb_orig, emb_repr).item()
        except: 
            scores['sbert_similarity_context'] = 0.0
    else: 
        scores['sbert_similarity_context'] = 0.0

    # QA inclusion rate for kept questions
    if qa_validation_pipeline and q_keep_objects:
        kept_count = 0
        for q_obj in q_keep_objects:
            try:
                result = qa_validation_pipeline(question=q_obj["question"], context=rephrased_text)
                if result and result['score'] > 0.1 and result['answer'].strip(): 
                    kept_count += 1
            except: 
                pass
        scores['qa_inclusion_rate_by_aux_qa'] = kept_count / len(q_keep_objects) if q_keep_objects else 0.0
    else: 
        scores['qa_inclusion_rate_by_aux_qa'] = 0.0

    # LLM answer similarity for kept questions
    if sbert_model and q_keep_objects and llm_answers_for_kept_qs:
        similarities = []
        for i, q_obj in enumerate(q_keep_objects):
            if i < len(llm_answers_for_kept_qs):
                llm_ans = llm_answers_for_kept_qs[i]
                squad_answers = q_obj["original_answers"]
                
                if squad_answers and llm_ans and not llm_ans.startswith("Error:") and "unanswerable" not in llm_ans.lower():
                    try:
                        emb_llm = sbert_model.encode(llm_ans, convert_to_tensor=True, device=device)
                        max_sim = 0.0
                        for squad_ans in squad_answers:
                            emb_squad = sbert_model.encode(squad_ans, convert_to_tensor=True, device=device)
                            sim = sbert_util.pytorch_cos_sim(emb_llm, emb_squad).item()
                            max_sim = max(max_sim, sim)
                        similarities.append(max_sim)
                    except: 
                        pass
        scores['llm_avg_answer_sbert_similarity_kept_q'] = np.mean(similarities) if similarities else 0.0
    else: 
        scores['llm_avg_answer_sbert_similarity_kept_q'] = 0.0
    
    # ===== CORRECTED REJECTION ACCURACY CALCULATION USING LLM =====
    if q_omit_strings:
        # Get LLM answers for omitted questions to test rejection
        llm_omit_results = get_llm_answers_from_rephrased_text(rephrased_text, q_omit_strings, llm_config)
        omit_answerability = llm_omit_results.get('answerability', [])
        
        # Count how many omitted questions became unanswerable
        omitted_unanswerable_count = sum(1 for is_answerable in omit_answerability if not is_answerable)
        
        # Calculate rejection accuracy: (unanswerable omitted) / (total omitted)
        total_omitted = len(q_omit_strings)
        rejection_accuracy = omitted_unanswerable_count / total_omitted if total_omitted > 0 else 0.0
        
        # Store both metrics
        scores['rejection_accuracy'] = rejection_accuracy
        scores['avg_qa_omission_success_rate'] = rejection_accuracy  # These are now the same
        
        # Add metadata for transparency
        scores['num_omitted_questions'] = total_omitted
        scores['num_omitted_unanswerable'] = omitted_unanswerable_count
        scores['rejection_accuracy_calculation_method'] = 'llm_based_omitted_unanswerable_divided_by_total_omitted'
        
    else:
        # No omitted questions - rejection accuracy is undefined
        scores['rejection_accuracy'] = 0.0
        scores['avg_qa_omission_success_rate'] = 0.0
        scores['num_omitted_questions'] = 0
        scores['num_omitted_unanswerable'] = 0
        scores['rejection_accuracy_calculation_method'] = 'no_omitted_questions'

    # Judge scores
    scores['judge_rephrase_quality_score'] = llm_judge_eval_scores.get("judge_rephrase_quality_score", 0.0)
    scores['judge_kept_q_answer_correctness_score'] = llm_judge_eval_scores.get("judge_kept_q_answer_correctness_score", 0.0)
    
    # Length metrics
    len_orig = len(original_text.split())
    len_repr = len(rephrased_text.split())
    scores.update({
        'len_o': len_orig, 
        'len_r': len_repr, 
        'len_ratio': len_repr / len_orig if len_orig > 0 else 0.0
    })
    
    return scores

# ===== PLOTTING FUNCTIONS =====
def generate_summary_plots(results_file_path, output_dir):
    print(f"\nGenerating plots from: {results_file_path}")
    ensure_dir(output_dir)
    
    try:
        with open(results_file_path, 'r', encoding='utf-8') as f: 
            results = json.load(f)
        
        df = pd.DataFrame(results)
        if 'validation_scores' not in df.columns: 
            print("Error: 'validation_scores' missing.")
            return
        
        # Create confusion matrix
        create_rejection_confusion_matrix(results, output_dir)
        
        # Normalize validation scores
        scores_df = pd.json_normalize(df['validation_scores'].apply(lambda x: x if isinstance(x, dict) else {}))
        df = pd.concat([df.drop(columns=['validation_scores']), scores_df], axis=1)
        
        # Convert to numeric and fill NaN with 0
        score_columns = [
            'rouge1_f', 'rougeL_f', 'bert_score_rephrased_vs_original_f1', 'sbert_similarity_context',
            'qa_inclusion_rate_by_aux_qa', 'llm_avg_answer_sbert_similarity_kept_q',
            'avg_qa_omission_success_rate', 'rejection_accuracy', 'jaccard_token', 'jaccard_char',
            'jaccard_bigram', 'jaccard_average',
            'judge_rephrase_quality_score', 'judge_kept_q_answer_correctness_score', 'len_ratio'
        ]
        
        for col in score_columns:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        if df.empty:
             print("No valid data for plotting.")
             return
             
    except Exception as e: 
        print(f"Error loading results: {e}")
        return

    metrics_labels = {
        'bert_score_rephrased_vs_original_f1': 'BERTScore F1',
        'sbert_similarity_context': 'SBERT Similarity',
        'llm_avg_answer_sbert_similarity_kept_q': "LLM Answer Similarity",
        'rejection_accuracy': 'Rejection Accuracy',
        'avg_qa_omission_success_rate': 'Omission Success Rate',
        'jaccard_token': 'Jaccard Token Similarity',
        'jaccard_char': 'Jaccard Character Similarity',
        'jaccard_bigram': 'Jaccard Bigram Similarity',
        'jaccard_average': 'Jaccard Average Similarity',
        'judge_rephrase_quality_score': 'Judge: Rephrase Quality',
        'judge_kept_q_answer_correctness_score': "Judge: Answer Correctness"
    }
    
    sns.set_theme(style="whitegrid")
    has_strategies = 'omission_strategy' in df.columns and df['omission_strategy'].nunique() > 1
    has_rephrase_type = 'rephrase_type' in df.columns and df['rephrase_type'].nunique() > 1
    
    print(f"Data: {len(df)} entries, LLMs: {df['llm_name'].unique().tolist()}")
    if has_strategies:
        print(f"Strategies: {df['omission_strategy'].unique().tolist()}")
    if has_rephrase_type:
        print(f"Rephrase Types: {df['rephrase_type'].unique().tolist()}")

    # Print rejection accuracy summary
    print("\n=== CORRECTED REJECTION ACCURACY SUMMARY ===")
    for llm_name in df['llm_name'].unique():
        llm_data = df[df['llm_name'] == llm_name]
        if 'rejection_accuracy' in llm_data.columns:
            avg_rejection = llm_data['rejection_accuracy'].mean()
            print(f"{llm_name}: {avg_rejection:.3f} avg rejection accuracy")

    # Print Jaccard Score summary
    print("\n=== JACCARD SIMILARITY SUMMARY ===")
    for llm_name in df['llm_name'].unique():
        llm_data = df[df['llm_name'] == llm_name]
        if 'jaccard_average' in llm_data.columns:
            avg_jaccard = llm_data['jaccard_average'].mean()
            print(f"{llm_name}: {avg_jaccard:.3f} avg Jaccard similarity")
            
            # Print detailed Jaccard scores
            if 'jaccard_token' in llm_data.columns:
                token_jaccard = llm_data['jaccard_token'].mean()
                char_jaccard = llm_data['jaccard_char'].mean()
                bigram_jaccard = llm_data['jaccard_bigram'].mean()
                print(f"  Token: {token_jaccard:.3f}, Char: {char_jaccard:.3f}, Bigram: {bigram_jaccard:.3f}")

    # LLM Performance Aggregation
    print("Generating LLM performance plots...")
    for metric, label in metrics_labels.items():
        if metric not in df.columns:
            continue
        
        # Aggregate by LLM
        llm_stats = df.groupby('llm_name')[metric].agg(['mean', 'std', 'count']).reset_index()
        llm_stats = llm_stats.sort_values('mean', ascending=False)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            llm_stats['llm_name'], 
            llm_stats['mean'], 
            yerr=llm_stats['std'], 
            capsize=5, 
            color=sns.color_palette("viridis", len(llm_stats)),
            alpha=0.8, 
            edgecolor='black'
        )
        
        # Add value labels
        for bar, mean_val, count in zip(bars, llm_stats['mean'], llm_stats['count']):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01,
                f'{mean_val:.3f}\n(n={count})', 
                ha='center', va='bottom', fontweight='bold'
            )
        
        plt.title(f'{label} - LLM Performance', fontsize=16, fontweight='bold')
        plt.ylabel(f'{label} (Mean ± Std)', fontsize=12)
        plt.xlabel('LLM Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'llm_performance_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Rephrase Type Performance (if applicable)
    if has_rephrase_type:
        print("Generating rephrase type performance plots...")
        for metric, label in metrics_labels.items():
            if metric not in df.columns:
                continue
            
            rephrase_stats = df.groupby('rephrase_type')[metric].agg(['mean', 'std', 'count']).reset_index()
            rephrase_stats = rephrase_stats.sort_values('mean', ascending=False)
            
            plt.figure(figsize=(10, 8))
            bars = plt.bar(
                rephrase_stats['rephrase_type'], 
                rephrase_stats['mean'], 
                yerr=rephrase_stats['std'], 
                capsize=5, 
                color=sns.color_palette("Set2", len(rephrase_stats)),
                alpha=0.8, 
                edgecolor='black'
            )
            
            for bar, mean_val, count in zip(bars, rephrase_stats['mean'], rephrase_stats['count']):
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01,
                    f'{mean_val:.3f}\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold'
                )
            
            plt.title(f'{label} - Rephrase Type Performance', fontsize=16, fontweight='bold')
            plt.ylabel(f'{label} (Mean ± Std)', fontsize=12)
            plt.xlabel('Rephrase Type', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'rephrase_type_performance_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Questions Removed Performance Analysis
    if has_strategies:
        print("Generating questions removed performance plots...")
        for metric, label in requested_metrics.items():
            if metric not in df.columns:
                continue
            
            # Create data for plotting by questions removed
            plot_data = []
            for _, row in df.iterrows():
                strategy = row.get('omission_strategy', 'unknown')
                questions_removed = "1 Question" if "omit_1" in strategy else "2 Questions" if "omit_2" in strategy else "Unknown"
                plot_data.append({
                    'Questions_Removed': questions_removed,
                    'Model': row['llm_name'],
                    'Rephrase_Type': row.get('rephrase_type', 'original'),
                    'Score': row[metric]
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            plt.figure(figsize=(12, 8))
            
            if has_rephrase_type:
                # Group by both questions removed and rephrase type
                sns.barplot(data=plot_df, x='Questions_Removed', y='Score', hue='Rephrase_Type', 
                           ci='sd', capsize=0.1, palette='Set2')
                plt.title(f'{label} by Questions Removed and Rephrase Type', fontsize=16, fontweight='bold')
                plt.legend(title='Rephrase Type')
            else:
                # Group by questions removed only
                sns.barplot(data=plot_df, x='Questions_Removed', y='Score', ci='sd', capsize=0.1, 
                           palette='viridis')
                plt.title(f'{label} by Questions Removed', fontsize=16, fontweight='bold')
            
            plt.ylabel(f'{label} (Mean ± Std)', fontsize=12)
            plt.xlabel('Questions Removed', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'questions_removed_performance_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Create enhanced summary tables
    print("Generating enhanced validation summary...")
    table_metrics = {
        'rejection_accuracy': 'Rejection Accuracy',  # Proportion of omitted questions that become unanswerable
        'jaccard_token': 'Jaccard Token',
        'bert_score_rephrased_vs_original_f1': 'BERTScore',
        'sbert_similarity_context': 'SBERT Similarity', 
        'judge_rephrase_quality_score': 'LLM Judge'
    }
    
    # Create the specific requested table with key metrics including Jaccard Token
    print("Generating requested metrics table...")
    requested_metrics = {
        'bert_score_rephrased_vs_original_f1': 'BERTScore F1',
        'sbert_similarity_context': 'SBERT Similarity',
        'llm_avg_answer_sbert_similarity_kept_q': 'LLM Answer Similarity',
        'rejection_accuracy': 'Rejection Accuracy',  # This is now the correct metric
        'judge_rephrase_quality_score': 'Judge: Rephrase Quality',
        'judge_kept_q_answer_correctness_score': 'Judge: Answer Correctness',
        'jaccard_token': 'Jaccard Token'
    }
    
    # Create comprehensive table by model and omission strategy (questions removed)
    comprehensive_table_data = []
    
    print("\n=== COMPREHENSIVE RESULTS BY MODEL AND QUESTIONS REMOVED ===")
    
    for llm_name in df['llm_name'].unique():
        llm_data = df[df['llm_name'] == llm_name]
        
        if has_strategies:
            for strategy in df['omission_strategy'].unique():
                strategy_data = llm_data[llm_data['omission_strategy'] == strategy]
                if not strategy_data.empty:
                    # Extract number of questions removed from strategy name
                    questions_removed = "1" if "omit_1" in strategy else "2" if "omit_2" in strategy else "Unknown"
                    
                    if has_rephrase_type:
                        for rephrase_type in df['rephrase_type'].unique():
                            rephrase_data = strategy_data[strategy_data['rephrase_type'] == rephrase_type]
                            if not rephrase_data.empty:
                                row = {
                                    'Model': llm_name, 
                                    'Questions_Removed': questions_removed,
                                    'Rephrase_Type': rephrase_type,
                                    'Count': len(rephrase_data)
                                }
                                for metric, label in requested_metrics.items():
                                    if metric in rephrase_data.columns:
                                        row[label] = f"{rephrase_data[metric].mean():.3f}"
                                    else:
                                        row[label] = "N/A"
                                comprehensive_table_data.append(row)
                    else:
                        row = {
                            'Model': llm_name, 
                            'Questions_Removed': questions_removed,
                            'Count': len(strategy_data)
                        }
                        for metric, label in requested_metrics.items():
                            if metric in strategy_data.columns:
                                row[label] = f"{strategy_data[metric].mean():.3f}"
                            else:
                                row[label] = "N/A"
                        comprehensive_table_data.append(row)
        else:
            # No strategies, just by model
            row = {'Model': llm_name, 'Count': len(llm_data)}
            for metric, label in requested_metrics.items():
                if metric in llm_data.columns:
                    row[label] = f"{llm_data[metric].mean():.3f}"
                else:
                    row[label] = "N/A"
            comprehensive_table_data.append(row)
    
    if comprehensive_table_data:
        comprehensive_df = pd.DataFrame(comprehensive_table_data)
        print(comprehensive_df.to_string(index=False))
        comprehensive_df.to_csv(os.path.join(output_dir, 'comprehensive_results_by_questions_removed.csv'), index=False)
        print(f"\nComprehensive table saved to: comprehensive_results_by_questions_removed.csv")
    
    # Create summary by questions removed
    if has_strategies:
        print("\n=== RESULTS BY NUMBER OF QUESTIONS REMOVED ===")
        questions_removed_summary = []
        
        for strategy in df['omission_strategy'].unique():
            strategy_data = df[df['omission_strategy'] == strategy]
            questions_removed = "1" if "omit_1" in strategy else "2" if "omit_2" in strategy else "Unknown"
            
            summary_row = {
                'Questions_Removed': questions_removed,
                'Strategy': strategy,
                'Total_Entries': len(strategy_data)
            }
            
            for metric, label in requested_metrics.items():
                if metric in strategy_data.columns:
                    summary_row[label] = f"{strategy_data[metric].mean():.3f} ± {strategy_data[metric].std():.3f}"
                else:
                    summary_row[label] = "N/A"
            
            questions_removed_summary.append(summary_row)
        
        if questions_removed_summary:
            questions_removed_df = pd.DataFrame(questions_removed_summary)
            print(questions_removed_df.to_string(index=False))
            questions_removed_df.to_csv(os.path.join(output_dir, 'results_by_questions_removed.csv'), index=False)
            print(f"\nQuestions removed summary saved to: results_by_questions_removed.csv")
    
    # Create summary table
    table_data = []
    for llm_name in df['llm_name'].unique():
        llm_data = df[df['llm_name'] == llm_name]
        
        if has_rephrase_type:
            for rephrase_type in df['rephrase_type'].unique():
                rephrase_data = llm_data[llm_data['rephrase_type'] == rephrase_type]
                if not rephrase_data.empty:
                    row = {'Model': llm_name, 'Rephrase_Type': rephrase_type}
                    for metric, label in table_metrics.items():
                        if metric in rephrase_data.columns:
                            row[label] = f"{rephrase_data[metric].mean():.3f}"
                        else:
                            row[label] = "N/A"
                    table_data.append(row)
        elif has_strategies:
            for strategy in df['omission_strategy'].unique():
                strategy_data = llm_data[llm_data['omission_strategy'] == strategy]
                if not strategy_data.empty:
                    row = {'Model': llm_name, 'Strategy': strategy}
                    for metric, label in table_metrics.items():
                        if metric in strategy_data.columns:
                            row[label] = f"{strategy_data[metric].mean():.3f}"
                        else:
                            row[label] = "N/A"
                    table_data.append(row)
        else:
            row = {'Model': llm_name}
            for metric, label in table_metrics.items():
                if metric in llm_data.columns:
                    row[label] = f"{llm_data[metric].mean():.3f}"
                else:
                    row[label] = "N/A"
            table_data.append(row)
    
    if table_data:
        table_df = pd.DataFrame(table_data)
        print("\n=== ENHANCED VALIDATION RESULTS ===")
        print(table_df.to_string(index=False))
        table_df.to_csv(os.path.join(output_dir, 'enhanced_validation_results.csv'), index=False)
    
    # Create model comparison table for requested metrics
    print("\n=== MODEL COMPARISON - REQUESTED METRICS ===")
    model_comparison_data = []
    
    for llm_name in df['llm_name'].unique():
        llm_data = df[df['llm_name'] == llm_name]
        row = {'Model': llm_name, 'Total_Entries': len(llm_data)}
        
        for metric, label in requested_metrics.items():
            if metric in llm_data.columns:
                mean_score = llm_data[metric].mean()
                std_score = llm_data[metric].std()
                row[label] = f"{mean_score:.3f} ± {std_score:.3f}"
            else:
                row[label] = "N/A"
        
        model_comparison_data.append(row)
    
    if model_comparison_data:
        model_comparison_df = pd.DataFrame(model_comparison_data)
        print(model_comparison_df.to_string(index=False))
        model_comparison_df.to_csv(os.path.join(output_dir, 'model_comparison_requested_metrics.csv'), index=False)
        print(f"\nModel comparison saved to: model_comparison_requested_metrics.csv")

    print(f"\nPlots and confusion matrix saved to: {output_dir}")

# ===== MAIN FUNCTION =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup
    ensure_dir(PLOTS_OUTPUT_DIR)
    ensure_dir('./data')  # Ensure data directory exists
    download_nltk_punkt_once()
    load_qa_pipeline(device)
    load_sbert_model()
    
    # Create sample custom format file for reference
    if USE_CUSTOM_FORMAT and not os.path.exists(DATA_FILE_PATH):
        print(f"Custom format file not found. Creating sample file...")
        create_sample_custom_json('./data/sample_custom_format.json')
        print(f"Please replace the sample data with your actual data in: {DATA_FILE_PATH}")
        return

    # Test Azure connections
    print("\n=== Testing Azure AI Connections ===")
    for config in LLM_CONFIGURATIONS:
        if config["type"] == "azure_ai_inference":
            print(f"Testing {config['name']}...")
            client = get_azure_ai_inference_client(config)
            if client:
                try:
                    response = client.complete(
                        messages=[
                            SystemMessage(content="You are helpful."), 
                            UserMessage(content="Say 'Hello!'")
                        ],
                        model=config["azure_ai_model_name"],
                        max_tokens=10
                    )
                    if hasattr(response, 'choices') and response.choices:
                        print(f"  ✅ Connection successful: {response.choices[0].message.content}")
                    else:
                        print(f"  ❌ No response content")
                except Exception as e:
                    print(f"  ❌ Connection failed: {e}")
            else:
                print(f"  ❌ Client creation failed")
    print("=== Connection Tests Complete ===\n")

    system_prompt = """You are an expert text editor specializing in selective information preservation and omission.
                       Your task is to rewrite a given passage with the following objectives:
                       1. PRESERVE: Maintain all information necessary to answer the "questions to keep"
                       2. OMIT: Remove or obscure information that would help answer the "questions to omit" 
                       3. COHERENCE: Ensure the rewritten passage remains coherent, natural, and readable
                       4. ACCURACY: Keep factual information intact for the preserved questions
                       Guidelines:
                        - The rewritten passage should be comprehensive enough to answer all "keep" questions
                        - Information relevant to "omit" questions should be removed, obscured, or replaced with vague references
                        - Maintain the general topic and context of the original passage
                        - Ensure smooth transitions and logical flow in the rewritten text
                        - Do not explicitly mention that information has been omitted
                        - If the answer to the omitted question is a name, replace it with a generic term like \"the person\" or \"the entity\""""

    # Flexible user prompt template
    user_prompt_template = """Original Passage:
                              "{original_passage}"
                              Questions to KEEP (preserve information for these):
                              {formatted_questions_to_keep}
                              Questions to OMIT (remove/obscure information for these):
                              {formatted_questions_to_omit}
                              Please provide the rewritten passage:"""

    # Load data
    print(f"Loading data from: {DATA_FILE_PATH}")
    if USE_CUSTOM_FORMAT:
        print("Using custom JSON format (Zelda-style)")
    else:
        print("Using original SQuAD format")
        
    json_data = load_official_squad_json(DATA_FILE_PATH)
    if not json_data:
        print("Failed to load data. Exiting.")
        return
    
    entries = json_data[:MAX_TEXTS_TO_PROCESS]
    print(f"Processing {len(entries)} entries with {len(LLM_CONFIGURATIONS)} models\n")

    # Processing
    results = []
    omission_strategies = {
        "omit_1_question": {"num_to_omit": 1, "min_questions": 3},
        "omit_2_questions": {"num_to_omit": 2, "min_questions": 3}
    }
    
    models_processed = set()

    for llm_config in LLM_CONFIGURATIONS:
        llm_name = llm_config["name"]
        llm_type = llm_config["type"]
        
        print(f"\n--- Processing with {llm_name} ---")
        
        # Load model
        model_ready = False
        if llm_type == "hf_local":
            try: 
                load_hf_model_and_tokenizer(llm_config['model_id'], device)
                model_ready = True
                print(f"✅ HF model loaded")
            except Exception as e: 
                print(f"❌ Failed to load HF model: {e}")
                continue
        elif llm_type == "azure_ai_inference":
            if not all([llm_config.get("azure_ai_endpoint"), llm_config.get("azure_api_key"), llm_config.get("azure_ai_model_name")]):
                print(f"❌ Missing Azure configuration")
                continue
            client = get_azure_ai_inference_client(llm_config)
            if client:
                model_ready = True
                print(f"✅ Azure client ready")
            else:
                print(f"❌ Failed to create Azure client")
                continue
        
        if not model_ready:
            continue
        
        models_processed.add(llm_name)
        model_results = 0
        
        # Process strategies
        for strategy_name, strategy_config in omission_strategies.items():
            print(f"  Strategy: {strategy_name}")
            strategy_results = 0
            
            # Process entries
            for i, entry in enumerate(entries):
                context = entry["context"]
                title = entry["title"]
                questions = entry["answerable_question_objects"]
                
                # Check if enough questions
                min_needed = strategy_config["min_questions"]
                if len(questions) < min_needed:
                    continue
                
                print(f"    Entry {i+1}/{len(entries)}: {title}...")
                
                # Select questions
                random.shuffle(questions)
                num_omit = strategy_config["num_to_omit"]
                omit_questions = questions[:num_omit]
                keep_questions = questions[num_omit:num_omit+random.randint(1,2)]
                
                if not keep_questions:
                    continue
                
                omit_q_strings = [q["question"] for q in omit_questions]
                keep_q_strings = [q["question"] for q in keep_questions]
                
                # ===== ORIGINAL REPHRASING =====
                print(f"      🔄 Original rephrasing (keep {len(keep_questions)}, omit {len(omit_questions)})")
                if llm_type == "hf_local":
                    rewritten = generate_rewrite_hf_local(context, keep_q_strings, omit_q_strings, system_prompt, user_prompt_template)
                else:
                    rewritten = generate_rewrite_azure_ai_inference(context, keep_q_strings, omit_q_strings, llm_config, system_prompt, user_prompt_template)
                
                if rewritten.startswith("Error:"):
                    print(f"        ❌ Original rewrite failed: {rewritten}")
                    continue
                else:
                    print(f"        ✅ Original rewrite successful")
                
                # Get LLM answers with answerability detection
                llm_qa_results = get_llm_answers_from_rephrased_text(rewritten, keep_q_strings, llm_config)
                
                # Judge evaluation
                judge_scores = evaluate_with_llm_judge(context, rewritten, keep_questions, omit_q_strings, llm_qa_results, JUDGE_LLM_CONFIG)
                
                # Validation scores with corrected rejection accuracy and Jaccard score
                validation_scores = calculate_comprehensive_validation_scores(context, rewritten, keep_questions, omit_q_strings, llm_qa_results, judge_scores, llm_config)
                
                # Store original rephrasing result
                result_original = {
                    "llm_name": llm_name, 
                    "omission_strategy": strategy_name,
                    "rephrase_type": "original",  # NEW: Track rephrasing type
                    "original_title": title, 
                    "original_context_snippet": context,
                    "questions_to_keep": keep_q_strings,
                    "questions_to_omit": omit_q_strings,
                    "llm_qa_results": llm_qa_results,
                    "original_answers_for_kept_qs": {q['question']: q['original_answers'] for q in keep_questions},
                    "rewritten_context": rewritten, 
                    "validation_scores": validation_scores,
                    "entry_id": entry.get("entry_id", "N/A"),
                    "status": "SUCCESS"
                }
                results.append(result_original)
                strategy_results += 1
                model_results += 1
                
                # ===== REVERSE REPHRASING =====
                print(f"      🔄 Reverse rephrasing (keep {len(omit_questions)}, omit {len(keep_questions)})")
                
                # Swap the questions: originally omitted become kept, originally kept become omitted
                reverse_keep_q_strings = omit_q_strings  # Previously omitted questions
                reverse_omit_q_strings = keep_q_strings  # Previously kept questions
                reverse_keep_questions = omit_questions  # Question objects for previously omitted
                reverse_omit_questions = keep_questions  # Question objects for previously kept
                
                if llm_type == "hf_local":
                    reverse_rewritten = generate_rewrite_hf_local(context, reverse_keep_q_strings, reverse_omit_q_strings, system_prompt, user_prompt_template)
                else:
                    reverse_rewritten = generate_rewrite_azure_ai_inference(context, reverse_keep_q_strings, reverse_omit_q_strings, llm_config, system_prompt, user_prompt_template)
                
                if reverse_rewritten.startswith("Error:"):
                    print(f"        ❌ Reverse rewrite failed: {reverse_rewritten}")
                else:
                    print(f"        ✅ Reverse rewrite successful")
                    
                    # Get LLM answers for reverse rephrasing
                    reverse_llm_qa_results = get_llm_answers_from_rephrased_text(reverse_rewritten, reverse_keep_q_strings, llm_config)
                    
                    # Judge evaluation for reverse rephrasing
                    reverse_judge_scores = evaluate_with_llm_judge(context, reverse_rewritten, reverse_keep_questions, reverse_omit_q_strings, reverse_llm_qa_results, JUDGE_LLM_CONFIG)
                    
                    # Validation scores for reverse rephrasing
                    reverse_validation_scores = calculate_comprehensive_validation_scores(context, reverse_rewritten, reverse_keep_questions, reverse_omit_q_strings, reverse_llm_qa_results, reverse_judge_scores, llm_config)
                    
                    # Store reverse rephrasing result
                    result_reverse = {
                        "llm_name": llm_name, 
                        "omission_strategy": strategy_name,
                        "rephrase_type": "reverse",  # NEW: Track rephrasing type
                        "original_title": title, 
                        "original_context_snippet": context,
                        "questions_to_keep": reverse_keep_q_strings,
                        "questions_to_omit": reverse_omit_q_strings,
                        "llm_qa_results": reverse_llm_qa_results,
                        "original_answers_for_kept_qs": {q['question']: q['original_answers'] for q in reverse_keep_questions},
                        "rewritten_context": reverse_rewritten, 
                        "validation_scores": reverse_validation_scores,
                        "entry_id": entry.get("entry_id", "N/A") + "_reverse",
                        "status": "SUCCESS"
                    }
                    results.append(result_reverse)
                    strategy_results += 1
                    model_results += 1
                
                # Save intermediate results
                if len(results) % 10 == 0:
                    with open('temp_results_enhanced.json', 'w', encoding='utf-8') as f: 
                        json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"    Strategy completed: {strategy_results} entries")
        
        print(f"  Model completed: {model_results} total entries")
        
        if llm_type == "hf_local": 
            release_hf_model()

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Models processed: {models_processed}")
    print(f"Total results: {len(results)}")
    
    if results:
        by_model = {}
        by_rephrase_type = {}
        for result in results:
            model = result['llm_name']
            rephrase_type = result.get('rephrase_type', 'unknown')
            by_model[model] = by_model.get(model, 0) + 1
            by_rephrase_type[rephrase_type] = by_rephrase_type.get(rephrase_type, 0) + 1
        
        print("Results by model:")
        for model, count in by_model.items():
            print(f"  {model}: {count} entries")
        
        print("Results by rephrase type:")
        for rephrase_type, count in by_rephrase_type.items():
            print(f"  {rephrase_type}: {count} entries")
        
        # Print enhanced summaries
        print("\n=== REJECTION ACCURACY SUMMARY ===")
        for model_name in by_model.keys():
            model_results = [r for r in results if r['llm_name'] == model_name]
            rejection_accuracies = [r['validation_scores'].get('rejection_accuracy', 0.0) for r in model_results]
            avg_rejection_accuracy = np.mean(rejection_accuracies)
            print(f"{model_name}: {avg_rejection_accuracy:.3f} avg rejection accuracy")
        
        print("\n=== JACCARD SIMILARITY SUMMARY ===")
        for model_name in by_model.keys():
            model_results = [r for r in results if r['llm_name'] == model_name]
            jaccard_scores = [r['validation_scores'].get('jaccard_average', 0.0) for r in model_results]
            avg_jaccard_score = np.mean(jaccard_scores)
            print(f"{model_name}: {avg_jaccard_score:.3f} avg Jaccard similarity")

    # Save final results
    output_file = f'enhanced_results_{MAX_TEXTS_TO_PROCESS}_texts.json'
    with open(output_file, 'w', encoding='utf-8') as f: 
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    
    # Generate plots and confusion matrix
    if results:
        generate_summary_plots(output_file, PLOTS_OUTPUT_DIR)
    else:
        print("No results to plot")

    print("\n--- Completed ---")

if __name__ == '__main__':
    main()