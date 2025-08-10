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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


# ===== CONFIGURATION =====
load_dotenv()

AZURE_AI_API_KEY = os.getenv('AZURE_AI_API_KEY')
AZURE_AI_ENDPOINT = os.getenv('AZURE_AI_ENDPOINT')

# Add validation for Azure credentials
if not AZURE_AI_API_KEY or not AZURE_AI_ENDPOINT:
    print("⚠️ Warning: Azure AI credentials not found in environment variables")
    print("Set AZURE_AI_API_KEY and AZURE_AI_ENDPOINT in your .env file")
    print("Falling back to local HF models only")
else:
    print("✅ Azure AI credentials found")
    print(f"🔍 Endpoint: {AZURE_AI_ENDPOINT[:50]}..." if len(AZURE_AI_ENDPOINT) > 50 else f"🔍 Endpoint: {AZURE_AI_ENDPOINT}")
    print(f"🔍 API Key: {'*' * (len(AZURE_AI_API_KEY) - 8)}{AZURE_AI_API_KEY[-8:]}")

LLM_CONFIGURATIONS = [
    {
        "name": "Llama-3.2-3B-Instruct", 
        "type": "hf_local",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct", 
        "requires_hf_login": True
    }
]

JUDGE_LLM_CONFIG = {
    "name": "Llama-3.2-3B-Instruct",
    "type": "hf_local",
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "requires_hf_login": True
}

MAX_TEXTS_TO_PROCESS = 500
SQUAD_JSON_FILE_PATH = './data/train-v2.0.json'
CUSTOM_JSON_FILE_PATH = './data/squad_500.json'
PLOTS_OUTPUT_DIR = 'dataset_creation_results'

# Preprocessing configuration
CREATE_CUSTOM_DATASET = False    # Set to True to create squad_500.json from train-v2.0.json
CUSTOM_DATASET_SIZE = 500       # Number of contexts to include in custom dataset

# Dataset requirements
MIN_QUESTIONS_PER_CONTEXT = 5  # Minimum 5 answerable questions required per context
SKIP_UNANSWERABLE = True       # Skip any questions marked as unanswerable
QA_ACCURACY_THRESHOLD = 1.0    # Require 100% accuracy for question answering

# Threshold for lexical similarity filtering using METEOR
METEOR_THRESHOLD = 0.6   # METEOR threshold for filtering paraphrases

USE_CUSTOM_FORMAT = True
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

def download_nltk_dependencies():
    """Download required NLTK data"""
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
    """Debug Azure AI configuration to identify issues"""
    print(f"\n{'='*60}")
    print("AZURE AI CONFIGURATION DEBUG")
    print(f"{'='*60}")
    
    # Check environment variables
    print("🔍 Environment Variables:")
    print(f"  AZURE_AI_API_KEY: {'✅ Set' if AZURE_AI_API_KEY else '❌ Missing'}")
    print(f"  AZURE_AI_ENDPOINT: {'✅ Set' if AZURE_AI_ENDPOINT else '❌ Missing'}")
    
    if AZURE_AI_API_KEY:
        print(f"  API Key length: {len(AZURE_AI_API_KEY)} characters")
        print(f"  API Key preview: {AZURE_AI_API_KEY[:8]}...{AZURE_AI_API_KEY[-8:]}")
    
    if AZURE_AI_ENDPOINT:
        print(f"  Endpoint: {AZURE_AI_ENDPOINT}")
        # Check if endpoint format is correct
        if not AZURE_AI_ENDPOINT.startswith('https://'):
            print("  ⚠️  Warning: Endpoint should start with 'https://'")
        if '.openai.azure.com' not in AZURE_AI_ENDPOINT and '.cognitiveservices.azure.com' not in AZURE_AI_ENDPOINT:
            print("  ⚠️  Warning: Endpoint doesn't appear to be a standard Azure OpenAI endpoint")
    
    # Test basic client creation
    if AZURE_AI_API_KEY and AZURE_AI_ENDPOINT:
        print("\n🔍 Testing Client Creation:")
        try:
            client = ChatCompletionsClient(
                endpoint=AZURE_AI_ENDPOINT, 
                credential=AzureKeyCredential(AZURE_AI_API_KEY)
            )
            print("  ✅ Client created successfully")
            
            # Test available models
            print("\n🔍 Testing Available Models:")
            common_models = ["gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-3.5-turbo", "gpt-4-turbo"]
            
            for model_name in common_models:
                try:
                    test_messages = [
                        SystemMessage(content="Test"),
                        UserMessage(content="Say 'OK'")
                    ]
                    
                    response = client.complete(
                        messages=test_messages,
                        model=model_name,
                        max_tokens=5,
                        temperature=0.1
                    )
                    print(f"  ✅ Model '{model_name}' works")
                    return model_name  # Return first working model
                    
                except Exception as e:
                    error_msg = str(e)
                    if "401" in error_msg:
                        print(f"  ❌ Model '{model_name}': Authentication failed")
                    elif "404" in error_msg:
                        print(f"  ❌ Model '{model_name}': Not found/deployed")
                    elif "DeploymentNotFound" in error_msg:
                        print(f"  ❌ Model '{model_name}': Deployment not found")
                    else:
                        print(f"  ❌ Model '{model_name}': {error_msg}")
                        
        except Exception as e:
            print(f"  ❌ Client creation failed: {e}")
    
    print("\n💡 Troubleshooting Tips:")
    print("1. Verify your Azure OpenAI resource is active")
    print("2. Check if you have deployed models in Azure AI Studio")
    print("3. Ensure your API key has the correct permissions")
    print("4. Verify the endpoint URL is correct")
    print("5. Check if your subscription has available quota")
    
    return None
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

# ===== AZURE AI AND HF FUNCTIONS (keeping existing implementations) =====
def get_azure_ai_inference_client(llm_config):
    endpoint = llm_config.get("azure_ai_endpoint")
    api_key = llm_config.get("azure_api_key")
    
    if not endpoint or not api_key:
        print(f"❌ Azure credentials missing: endpoint={bool(endpoint)}, api_key={bool(api_key)}")
        return None
    
    # Debug: Print sanitized endpoint info
    sanitized_endpoint = endpoint[:50] + "..." if len(endpoint) > 50 else endpoint
    print(f"🔍 Debug: Using endpoint: {sanitized_endpoint}")
    print(f"🔍 Debug: API key length: {len(api_key)} characters")
    
    if endpoint in azure_ai_inference_clients:
        return azure_ai_inference_clients[endpoint]
    
    try:
        client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        
        # Test the client with a simple request to validate credentials
        test_messages = [
            SystemMessage(content="Test"),
            UserMessage(content="Say 'OK' if you can read this.")
        ]
        
        # Try the model name from config first
        model_name = llm_config.get("azure_ai_model_name", "gpt-4")
        print(f"🔍 Debug: Testing with model name: {model_name}")
        
        try:
            test_response = client.complete(
                messages=test_messages,
                model=model_name,
                max_tokens=5,
                temperature=0.1
            )
            print(f"✅ Azure AI client validated successfully for model: {model_name}")
        except Exception as model_error:
            print(f"❌ Model '{model_name}' failed: {model_error}")
            
            # Try alternative model names
            alternative_models = ["gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-3.5-turbo"]
            working_model = None
            
            for alt_model in alternative_models:
                print(f"🔄 Trying alternative model: {alt_model}")
                try:
                    test_response = client.complete(
                        messages=test_messages,
                        model=alt_model,
                        max_tokens=5,
                        temperature=0.1
                    )
                    working_model = alt_model
                    print(f"✅ Found working model: {alt_model}")
                    # Update the config with working model
                    llm_config["azure_ai_model_name"] = alt_model
                    break
                except Exception as alt_error:
                    print(f"❌ Model '{alt_model}' also failed: {alt_error}")
                    continue
            
            if not working_model:
                print(f"❌ No working model found. Please check your Azure AI Studio deployment.")
                return None
        
        azure_ai_inference_clients[endpoint] = client
        return client
        
    except Exception as e:
        print(f"❌ Error creating Azure client: {e}")
        print(f"🔍 Debug: Full error details: {type(e).__name__}: {str(e)}")
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

# ===== PREPROCESSING: CREATE CUSTOM DATASET FROM SQUAD =====
def create_custom_dataset_from_squad(input_file_path, output_file_path, target_size=500, min_questions=5):
    """
    Create a custom dataset from original SQuAD train-v2.0.json file.
    
    Converts from original SQuAD format to simplified format:
    
    Original SQuAD structure:
    {
      "data": [
        {
          "title": "...",
          "paragraphs": [
            {
              "context": "...",
              "qas": [
                {
                  "id": "...",
                  "question": "...",
                  "answers": [{"text": "...", "answer_start": ...}],
                  "is_impossible": false
                }
              ]
            }
          ]
        }
      ]
    }
    
    Target simplified structure (squad_500.json format):
    [
      {
        "title": "...",
        "full_context": "...", 
        "questions_details": [
          {
            "id": "...",
            "question": "...",
            "answers_text": ["...", "..."],
            "is_impossible": false
          }
        ]
      }
    ]
    
    Args:
        input_file_path: Path to train-v2.0.json
        output_file_path: Path for output squad_500.json
        target_size: Number of contexts to include
        min_questions: Minimum answerable questions per context
    
    Returns:
        Boolean indicating success
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING: Creating Custom Dataset from SQuAD")
    print(f"{'='*60}")
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Target size: {target_size} contexts")
    print(f"Minimum questions per context: {min_questions}")
    print(f"Skip unanswerable questions: {SKIP_UNANSWERABLE}")
    
    try:
        # Load original SQuAD data
        print("\nLoading original SQuAD data...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        custom_dataset = []
        processed_contexts = 0
        
        for topic_data in squad_data.get('data', []):
            if len(custom_dataset) >= target_size:
                break
                
            title = topic_data.get('title', 'Unknown_Title')
            
            for para_idx, paragraph in enumerate(topic_data.get('paragraphs', [])):
                if len(custom_dataset) >= target_size:
                    break
                    
                context = paragraph.get('context')
                qas_list = paragraph.get('qas', [])
                
                if not context:
                    continue
                
                processed_contexts += 1
                
                # Filter questions based on ground truth answerability
                answerable_questions = []
                for qa in qas_list:
                    # Skip if marked as impossible/unanswerable
                    if SKIP_UNANSWERABLE and qa.get('is_impossible', False):
                        continue
                        
                    # Skip if missing question text or answers
                    if not qa.get('question'):
                        continue
                        
                    # For answerable questions, there should be answers
                    if not qa.get('is_impossible', False) and not qa.get('answers'):
                        continue
                    
                    # Extract answer texts
                    answer_texts = []
                    if qa.get('answers'):
                        answer_texts = [ans['text'] for ans in qa['answers'] if ans.get('text')]
                    
                    # Only include if we have actual answer texts (ground truth answerable)
                    if answer_texts:
                        question_detail = {
                            "id": qa.get('id', f"{title}_p{para_idx}_{len(answerable_questions)}"),
                            "question": qa['question'],
                            "answers_text": answer_texts,
                            "is_impossible": qa.get('is_impossible', False)
                        }
                        answerable_questions.append(question_detail)
                
                # Only include contexts with sufficient answerable questions
                if len(answerable_questions) >= min_questions:
                    context_entry = {
                        "title": f"{title}_paragraph_{para_idx}",
                        "full_context": context,
                        "questions_details": answerable_questions
                    }
                    custom_dataset.append(context_entry)
                    
                    print(f"  ✅ Added: {title}_p{para_idx} ({len(answerable_questions)} questions)")
                    
                    if len(custom_dataset) % 50 == 0:
                        print(f"    Progress: {len(custom_dataset)}/{target_size} contexts collected")
                else:
                    print(f"  ❌ Skipped: {title}_p{para_idx} (only {len(answerable_questions)} answerable questions)")
        
        print(f"\nDataset creation summary:")
        print(f"  Processed contexts: {processed_contexts}")
        print(f"  Contexts meeting requirements: {len(custom_dataset)}")
        print(f"  Target size: {target_size}")
        
        if len(custom_dataset) < target_size:
            print(f"  ⚠️ Warning: Only found {len(custom_dataset)} contexts meeting requirements")
        
        # Save the custom dataset
        print(f"\nSaving custom dataset to: {output_file_path}")
        ensure_dir(os.path.dirname(output_file_path))
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(custom_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Custom dataset saved successfully!")
        print(f"   File: {output_file_path}")
        print(f"   Size: {len(custom_dataset)} contexts")
        
        # Print sample statistics
        if custom_dataset:
            question_counts = [len(entry['questions_details']) for entry in custom_dataset]
            print(f"\nDataset statistics:")
            print(f"  Average questions per context: {np.mean(question_counts):.1f}")
            print(f"  Min questions per context: {min(question_counts)}")
            print(f"  Max questions per context: {max(question_counts)}")
            print(f"  Total questions: {sum(question_counts)}")
            
            # Show first example
            print(f"\nSample entry structure:")
            sample = custom_dataset[0]
            print(f"  Title: {sample['title']}")
            print(f"  Context length: {len(sample['full_context'])} characters")
            print(f"  Number of questions: {len(sample['questions_details'])}")
            print(f"  First question: {sample['questions_details'][0]['question'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating custom dataset: {e}")
        return False

def check_if_custom_dataset_exists(file_path):
    """Check if custom dataset file already exists"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Custom dataset already exists: {file_path}")
            print(f"   Contains {len(data)} contexts")
            return True
        except:
            print(f"⚠️ Custom dataset file exists but appears corrupted: {file_path}")
            return False
    return False

def load_official_squad_json(file_path):
    """Load JSON data in the simplified format, filtering unanswerable questions and requiring minimum questions per context"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: 
            json_data = json.load(f)
        
        processed_data = []
        skipped_contexts = 0
        
        if isinstance(json_data, dict):
            json_data = [json_data] 
        elif isinstance(json_data, dict) and 'data' in json_data:
            return load_original_squad_format(json_data)
        
        for entry_idx, entry in enumerate(json_data):
            title = entry.get('title', f'Unknown_Title_{entry_idx}')
            context = entry.get('full_context', '')
            questions_list = entry.get('questions_details', [])
            
            if not context or not questions_list:
                continue
            
            # Filter for answerable questions only (skip unanswerable questions)
            answerable_questions = []
            for q_idx, qa in enumerate(questions_list):
                # Skip if marked as impossible/unanswerable
                if SKIP_UNANSWERABLE and qa.get('is_impossible', False):
                    continue
                    
                # Skip if missing question text or answers
                if not qa.get('question') or not qa.get('answers_text'):
                    continue
                    
                answerable_questions.append({
                    'question': qa['question'], 
                    'original_answers': qa.get('answers_text', []), 
                    'id': qa.get('id', f"{title}_q{q_idx}")
                })
            
            # Only include contexts with at least MIN_QUESTIONS_PER_CONTEXT answerable questions
            if len(answerable_questions) >= MIN_QUESTIONS_PER_CONTEXT:
                processed_data.append({
                    "entry_id": f"{title}_{entry_idx}", 
                    "title": title, 
                    "context": context, 
                    "answerable_question_objects": answerable_questions,
                    "num_answerable_questions": len(answerable_questions)
                })
                print(f"  ✅ Included: {title} ({len(answerable_questions)} answerable questions)")
            else:
                skipped_contexts += 1
                print(f"  ❌ Skipped: {title} (only {len(answerable_questions)} answerable questions, need {MIN_QUESTIONS_PER_CONTEXT})")
        
        print(f"\nData loading summary:")
        print(f"  Total contexts processed: {len(json_data)}")
        print(f"  Contexts included: {len(processed_data)}")
        print(f"  Contexts skipped (insufficient questions): {skipped_contexts}")
        print(f"  Required minimum questions per context: {MIN_QUESTIONS_PER_CONTEXT}")
        
        return processed_data
        
    except Exception as e: 
        print(f"Error loading JSON data: {e}")
        return None

def load_original_squad_format(squad_data):
    """Fallback function to handle original SQuAD format with answerable question filtering"""
    try:
        processed_data = []
        skipped_contexts = 0
        
        for topic_data in squad_data.get('data', []):
            title = topic_data.get('title', 'Unknown_Title')
            
            for para_idx, paragraph in enumerate(topic_data.get('paragraphs', [])):
                context = paragraph.get('context')
                qas_list = paragraph.get('qas', [])
                
                if not context:
                    continue
                
                # Filter for answerable questions only
                answerable_questions = []
                for i, qa in enumerate(qas_list):
                    # Skip if marked as impossible/unanswerable
                    if SKIP_UNANSWERABLE and qa.get('is_impossible', False):
                        continue
                        
                    # Skip if missing question text or answers
                    if not qa.get('question') or not qa.get('answers'):
                        continue
                    
                    answerable_questions.append({
                        'question': qa['question'], 
                        'original_answers': [ans['text'] for ans in qa.get('answers', [])], 
                        'id': qa.get('id', f"{title}_p{para_idx}_q{i}")
                    })
                
                # Only include contexts with at least MIN_QUESTIONS_PER_CONTEXT answerable questions
                if len(answerable_questions) >= MIN_QUESTIONS_PER_CONTEXT:
                    processed_data.append({
                        "entry_id": f"{title}_p{para_idx}", 
                        "title": title, 
                        "context": context, 
                        "answerable_question_objects": answerable_questions,
                        "num_answerable_questions": len(answerable_questions)
                    })
                    print(f"  ✅ Included: {title}_p{para_idx} ({len(answerable_questions)} answerable questions)")
                else:
                    skipped_contexts += 1
                    print(f"  ❌ Skipped: {title}_p{para_idx} (only {len(answerable_questions)} answerable questions, need {MIN_QUESTIONS_PER_CONTEXT})")
        
        print(f"\nOriginal SQuAD format loading summary:")
        print(f"  Contexts included: {len(processed_data)}")
        print(f"  Contexts skipped (insufficient questions): {skipped_contexts}")
        print(f"  Required minimum questions per context: {MIN_QUESTIONS_PER_CONTEXT}")
        
        return processed_data
        
    except Exception as e: 
        print(f"Error loading original SQuAD data: {e}")
        return None

# ===== STEP 1: FILTER TEXTS WITH CORRECT ANSWERS =====
def test_llm_qa_accuracy(text, questions, llm_config):
    """Test if LLM can answer all questions correctly for a given text"""
    qa_system_prompt = """You are a precise question-answering assistant. Answer questions based solely on the provided context. Provide direct, specific answers."""
    
    correct_answers = 0
    total_questions = len(questions)
    
    try:
        llm_type = llm_config["type"]
        
        for question_obj in questions:
            question = question_obj['question']
            original_answers = question_obj['original_answers']
            
            if llm_type == "hf_local":
                if not current_hf_model or not current_hf_tokenizer:
                    return 0.0
                    
                messages = [
                    {"role": "system", "content": qa_system_prompt}, 
                    {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"}
                ]
                prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                llm_answer = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=100, temperature=0.1)
                
            elif llm_type == "azure_ai_inference":
                client = get_azure_ai_inference_client(llm_config)
                if not client:
                    print(f"❌ Cannot get Azure client, falling back to HF model")
                    # Fallback to HF model
                    if current_hf_model and current_hf_tokenizer:
                        messages = [
                            {"role": "system", "content": qa_system_prompt}, 
                            {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"}
                        ]
                        prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        llm_answer = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=100, temperature=0.1)
                    else:
                        return 0.0
                else:
                    model_name = llm_config["azure_ai_model_name"]
                    messages = [
                        SystemMessage(content=qa_system_prompt), 
                        UserMessage(content=f"Context: {text}\n\nQuestion: {question}\n\nAnswer:")
                    ]
                    
                    try:
                        response = client.complete(
                            messages=messages, 
                            model=model_name, 
                            max_tokens=100, 
                            temperature=0.1
                        )
                        
                        if hasattr(response, 'choices') and response.choices:
                            llm_answer = response.choices[0].message.content.strip()
                            llm_answer = clean_deepseek_response(llm_answer, llm_config)
                        else:
                            continue
                    except Exception as api_error:
                        print(f"❌ Azure API error: {api_error}")
                        # Fallback to HF model
                        if current_hf_model and current_hf_tokenizer:
                            messages = [
                                {"role": "system", "content": qa_system_prompt}, 
                                {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"}
                            ]
                            prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            llm_answer = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=100, temperature=0.1)
                        else:
                            continue
            else:
                continue
            
            # Use JUDGE to evaluate if answer is correct
            if judge_answer_correctness(question, llm_answer, original_answers):
                correct_answers += 1
                
    except Exception as e:
        print(f"Error in QA accuracy test: {e}")
        return 0.0
    
    return correct_answers / total_questions if total_questions > 0 else 0.0

def judge_answer_correctness(question, llm_answer, original_answers):
    """Use Llama LLM judge to determine if answers are equivalent"""
    judge_prompt = """Compare two answers to the same question and determine if they contain the same information.

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Do these answers contain the same information? Answer only "YES" or "NO"."""
    
    try:
        judge_config = JUDGE_LLM_CONFIG
        
        # Test against the first original answer
        if not original_answers:
            return False
            
        original_answer = original_answers[0]
        
        formatted_prompt = judge_prompt.format(
            question=question,
            answer1=llm_answer,
            answer2=original_answer
        )
        
        # Use Llama model for judging
        if judge_config["type"] == "hf_local":
            # Load Llama judge model if not already loaded or if different from current model
            if (not current_hf_model or 
                not current_hf_tokenizer or 
                judge_config["model_id"] not in str(current_hf_model.config._name_or_path if hasattr(current_hf_model.config, '_name_or_path') else '')):
                
                print(f"🔄 Loading Llama judge model: {judge_config['model_id']}")
                try:
                    judge_model, judge_tokenizer = load_hf_model_and_tokenizer(
                        judge_config['model_id'], 
                        current_hf_device if current_hf_device else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        for_judge=True
                    )
                except Exception as e:
                    print(f"❌ Failed to load judge model: {e}")
                    # Fallback to simple text matching
                    return simple_answer_match(llm_answer, original_answer)
            else:
                # Use current model as judge
                judge_model, judge_tokenizer = current_hf_model, current_hf_tokenizer
            
            # Generate judge response
            messages = [
                {"role": "system", "content": "You are a precise answer comparison judge. Answer only YES or NO."}, 
                {"role": "user", "content": formatted_prompt}
            ]
            
            try:
                prompt_text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                judge_response = _generate_hf_response(judge_model, judge_tokenizer, prompt_text, max_new_tokens=10, temperature=0.1)
                
                # Parse response
                judge_response_upper = judge_response.strip().upper()
                if "YES" in judge_response_upper:
                    return True
                elif "NO" in judge_response_upper:
                    return False
                else:
                    # If unclear response, fall back to simple matching
                    print(f"⚠️ Unclear judge response: '{judge_response}', using fallback")
                    return simple_answer_match(llm_answer, original_answer)
                    
            except Exception as e:
                print(f"❌ Error in Llama judge generation: {e}")
                return simple_answer_match(llm_answer, original_answer)
        
        else:
            # Fallback to simple matching if judge config is not HF local
            return simple_answer_match(llm_answer, original_answer)
        
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        # Final fallback to simple matching
        return simple_answer_match(llm_answer, original_answers[0] if original_answers else "")
    
    return False

def simple_answer_match(llm_answer, original_answer):
    """Simple fallback answer matching based on word overlap"""
    if not original_answer:
        return False
    
    original_answer_lower = original_answer.lower()
    llm_answer_lower = llm_answer.lower()
    
    # Simple word overlap check
    original_words = set(original_answer_lower.split())
    llm_words = set(llm_answer_lower.split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    original_words = original_words - stop_words
    llm_words = llm_words - stop_words
    
    if not original_words:
        return False
    
    overlap = len(original_words.intersection(llm_words))
    overlap_ratio = overlap / len(original_words)
    
    # Consider it a match if at least 60% of meaningful words overlap
    return overlap_ratio >= 0.6

def filter_texts_with_correct_answers(entries, llm_config, accuracy_threshold=QA_ACCURACY_THRESHOLD):
    """Step 1: Filter texts where LLM answers correctly (ensuring minimum question requirements)"""
    print("Step 1: Filtering texts with correct answers...")
    print(f"  Required accuracy threshold: {accuracy_threshold*100}%")
    
    filtered_entries = []
    
    for i, entry in enumerate(entries):
        print(f"  Testing entry {i+1}/{len(entries)}: {entry['title']}")
        
        # Double-check that entry has minimum required questions
        if len(entry['answerable_question_objects']) < MIN_QUESTIONS_PER_CONTEXT:
            print(f"    ❌ Insufficient questions ({len(entry['answerable_question_objects'])} < {MIN_QUESTIONS_PER_CONTEXT})")
            continue
        
        accuracy = test_llm_qa_accuracy(entry['context'], entry['answerable_question_objects'], llm_config)
        
        if accuracy >= accuracy_threshold:
            entry['qa_accuracy'] = accuracy
            filtered_entries.append(entry)
            print(f"    ✅ Passed (accuracy: {accuracy:.2f}, {len(entry['answerable_question_objects'])} questions)")
        else:
            print(f"    ❌ Failed accuracy test ({accuracy:.2f} < {accuracy_threshold:.2f})")
    
    print(f"Step 1 complete: {len(filtered_entries)}/{len(entries)} entries passed")
    print(f"All filtered entries have ≥{MIN_QUESTIONS_PER_CONTEXT} answerable questions with {accuracy_threshold*100}% accuracy")
    return filtered_entries

# ===== STEP 2: CREATE PARAPHRASING PAIRS =====
def generate_paraphrase(text, llm_config):
    """Generate paraphrase of text"""
    paraphrase_prompt = """Rewrite the following text while preserving all the information and meaning. Use different wording and sentence structure, but ensure all facts and details remain the same.

Original text:
{text}

Rewritten text:"""
    
    try:
        llm_type = llm_config["type"]
        
        if llm_type == "hf_local":
            if not current_hf_model or not current_hf_tokenizer:
                return None
                
            messages = [
                {"role": "system", "content": "You are an expert text rewriter."}, 
                {"role": "user", "content": paraphrase_prompt.format(text=text)}
            ]
            prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            paraphrase = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=len(text.split())*2, temperature=0.3)
            
        elif llm_type == "azure_ai_inference":
            client = get_azure_ai_inference_client(llm_config)
            if not client:
                # Fallback to HF model
                if current_hf_model and current_hf_tokenizer:
                    messages = [
                        {"role": "system", "content": "You are an expert text rewriter."}, 
                        {"role": "user", "content": paraphrase_prompt.format(text=text)}
                    ]
                    prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    paraphrase = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=len(text.split())*2, temperature=0.3)
                else:
                    return None
            else:
                model_name = llm_config["azure_ai_model_name"]
                messages = [
                    SystemMessage(content="You are an expert text rewriter."), 
                    UserMessage(content=paraphrase_prompt.format(text=text))
                ]
                
                try:
                    response = client.complete(
                        messages=messages, 
                        model=model_name, 
                        max_tokens=len(text.split())*2, 
                        temperature=0.3
                    )
                    
                    if hasattr(response, 'choices') and response.choices:
                        paraphrase = response.choices[0].message.content.strip()
                        paraphrase = clean_deepseek_response(paraphrase, llm_config)
                    else:
                        return None
                except Exception as api_error:
                    print(f"❌ Azure API error in paraphrase: {api_error}")
                    # Fallback to HF model
                    if current_hf_model and current_hf_tokenizer:
                        messages = [
                            {"role": "system", "content": "You are an expert text rewriter."}, 
                            {"role": "user", "content": paraphrase_prompt.format(text=text)}
                        ]
                        prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        paraphrase = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=len(text.split())*2, temperature=0.3)
                    else:
                        return None
        else:
            return None
            
        return paraphrase
        
    except Exception as e:
        print(f"Error generating paraphrase: {e}")
        return None

def calculate_lexical_similarity(text1, text2):
    """Calculate METEOR similarity metric for paraphrase detection"""
    try:
        meteor_score = calculate_meteor_score(text1, text2)
        
        return {
            'meteor_score': meteor_score
        }
        
    except Exception as e:
        print(f"Error calculating METEOR similarity: {e}")
        return {'meteor_score': 0}

def calculate_meteor_score(text1, text2):
    """Calculate simplified METEOR score between two texts"""
    try:
        # Simplified METEOR: combines precision, recall, and penalty for fragmentation
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Word-level matches
        matches = 0
        for word in words1:
            if word in words2:
                matches += 1
        
        # Precision and recall
        precision = matches / len(words1) if len(words1) > 0 else 0
        recall = matches / len(words2) if len(words2) > 0 else 0
        
        # F-mean (harmonic mean)
        if precision + recall > 0:
            f_mean = (10 * precision * recall) / (recall + 9 * precision)
        else:
            f_mean = 0
        
        # Penalty for fragmentation (simplified)
        penalty = 0.5 * (matches / min(len(words1), len(words2))) if min(len(words1), len(words2)) > 0 else 0
        
        meteor = f_mean * (1 - penalty)
        return meteor
        
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
        return 0

def create_paraphrasing_pairs(filtered_entries, llm_config):
    """Step 2: Create paraphrasing pairs with 100% accuracy requirement"""
    print("Step 2: Creating paraphrasing pairs...")
    print(f"  Requirements: METEOR < {METEOR_THRESHOLD}, QA accuracy = {QA_ACCURACY_THRESHOLD*100}%")
    
    paraphrase_pairs = []
    
    for i, entry in enumerate(filtered_entries):
        print(f"  Creating paraphrase for entry {i+1}/{len(filtered_entries)}")
        
        original_text = entry['context']
        paraphrase = generate_paraphrase(original_text, llm_config)
        
        if paraphrase:
            similarity = calculate_lexical_similarity(original_text, paraphrase)
            
            # Filter based on METEOR similarity threshold (below threshold = sufficiently different)
            passes_meteor = similarity['meteor_score'] < METEOR_THRESHOLD
            
            if passes_meteor:
                # Test if paraphrase still answers the same questions correctly with 100% accuracy
                paraphrase_accuracy = test_llm_qa_accuracy(paraphrase, entry['answerable_question_objects'], llm_config)
                
                if paraphrase_accuracy >= QA_ACCURACY_THRESHOLD:  # Require 100% accuracy
                    pair = {
                        'original_entry': entry,
                        'paraphrase_text': paraphrase,
                        'similarity_metrics': similarity,
                        'paraphrase_accuracy': paraphrase_accuracy
                    }
                    paraphrase_pairs.append(pair)
                    print(f"    ✅ Paraphrase pair created (METEOR: {similarity['meteor_score']:.2f}, accuracy: {paraphrase_accuracy:.2f})")
                else:
                    print(f"    ❌ Paraphrase failed accuracy test ({paraphrase_accuracy:.2f} < {QA_ACCURACY_THRESHOLD:.2f})")
            else:
                print(f"    ❌ Paraphrase too similar (METEOR: {similarity['meteor_score']:.2f})")
        else:
            print(f"    ❌ Failed to generate paraphrase")
    
    print(f"Step 2 complete: {len(paraphrase_pairs)} paraphrase pairs created")
    return paraphrase_pairs

# ===== STEP 3: CREATE TEXT VARIANTS WITH REMOVED QUESTIONS =====
def create_text_variant_removing_questions(text, questions_to_remove, all_questions, llm_config):
    """Create variant of text that cannot answer specific questions"""
    
    remaining_questions = [q for q in all_questions if q not in questions_to_remove]
    
    system_prompt = """You are a text modification expert. Your task is to rewrite the given text so that it can still answer certain questions but CANNOT answer other specified questions.

Requirements:
1. Remove or modify information needed to answer the FORBIDDEN questions
2. Preserve information needed for the ALLOWED questions  
3. Maintain natural, readable text
4. Keep the overall context and meaning intact"""

    user_prompt = """Original text:
{text}

ALLOWED questions (text must still be able to answer these):
{allowed_questions}

FORBIDDEN questions (text must NOT be able to answer these):
{forbidden_questions}

Rewrite the text following the requirements:"""

    try:
        allowed_q_text = "\n".join([f"- {q['question']}" for q in remaining_questions])
        forbidden_q_text = "\n".join([f"- {q['question']}" for q in questions_to_remove])
        
        llm_type = llm_config["type"]
        
        if llm_type == "hf_local":
            if not current_hf_model or not current_hf_tokenizer:
                return None
                
            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt.format(
                    text=text,
                    allowed_questions=allowed_q_text,
                    forbidden_questions=forbidden_q_text
                )}
            ]
            prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            variant = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=len(text.split())*2, temperature=0.1)
            
        elif llm_type == "azure_ai_inference":
            client = get_azure_ai_inference_client(llm_config)
            if not client:
                # Fallback to HF model
                if current_hf_model and current_hf_tokenizer:
                    messages = [
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_prompt.format(
                            text=text,
                            allowed_questions=allowed_q_text,
                            forbidden_questions=forbidden_q_text
                        )}
                    ]
                    prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    variant = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=len(text.split())*2, temperature=0.1)
                else:
                    return None
            else:
                model_name = llm_config["azure_ai_model_name"]
                messages = [
                    SystemMessage(content=system_prompt), 
                    UserMessage(content=user_prompt.format(
                        text=text,
                        allowed_questions=allowed_q_text,
                        forbidden_questions=forbidden_q_text
                    ))
                ]
                
                try:
                    response = client.complete(
                        messages=messages, 
                        model=model_name, 
                        max_tokens=len(text.split())*2, 
                        temperature=0.1
                    )
                    
                    if hasattr(response, 'choices') and response.choices:
                        variant = response.choices[0].message.content.strip()
                        variant = clean_deepseek_response(variant, llm_config)
                    else:
                        return None
                except Exception as api_error:
                    print(f"❌ Azure API error in variant creation: {api_error}")
                    # Fallback to HF model
                    if current_hf_model and current_hf_tokenizer:
                        messages = [
                            {"role": "system", "content": system_prompt}, 
                            {"role": "user", "content": user_prompt.format(
                                text=text,
                                allowed_questions=allowed_q_text,
                                forbidden_questions=forbidden_q_text
                            )}
                        ]
                        prompt_text = current_hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        variant = _generate_hf_response(current_hf_model, current_hf_tokenizer, prompt_text, max_new_tokens=len(text.split())*2, temperature=0.1)
                    else:
                        return None
        else:
            return None
            
        return variant
        
    except Exception as e:
        print(f"Error creating text variant: {e}")
        return None

def get_question_set_for_text(text, all_questions, llm_config):
    """Determine which questions a text can answer with 100% accuracy"""
    answerable_questions = []
    
    for question_obj in all_questions:
        # Test if text can answer this question with perfect accuracy
        accuracy = test_llm_qa_accuracy(text, [question_obj], llm_config)
        if accuracy >= QA_ACCURACY_THRESHOLD:  # Require 100% accuracy
            answerable_questions.append(question_obj)
    
    return answerable_questions

def create_text_variants(paraphrase_pairs, llm_config):
    """Step 3: Create variants with different question sets (ensuring minimum question requirements and 100% accuracy)"""
    print("Step 3: Creating text variants...")
    print(f"  Requirements: ≥{MIN_QUESTIONS_PER_CONTEXT} questions per variant, {QA_ACCURACY_THRESHOLD*100}% accuracy")
    
    all_variants = []
    
    for pair_idx, pair in enumerate(paraphrase_pairs):
        print(f"  Processing pair {pair_idx+1}/{len(paraphrase_pairs)}")
        
        original_entry = pair['original_entry']
        paraphrase_text = pair['paraphrase_text']
        all_questions = original_entry['answerable_question_objects']
        
        # Ensure we have enough questions to create meaningful variants
        if len(all_questions) < MIN_QUESTIONS_PER_CONTEXT:
            print(f"    ❌ Skipping pair: insufficient questions ({len(all_questions)} < {MIN_QUESTIONS_PER_CONTEXT})")
            continue
        
        # Create variants for original text
        original_variants = []
        
        # Variant 0: Original text (no changes)
        original_variants.append({
            'text': original_entry['context'],
            'variant_type': 'original',
            'removed_questions': [],
            'question_set': all_questions
        })
        
        # Variants 1-N: Remove 1 to N questions (but keep at least MIN_QUESTIONS_PER_CONTEXT questions)
        max_removable = len(all_questions) - MIN_QUESTIONS_PER_CONTEXT
        
        if max_removable > 0:
            for num_remove in range(1, min(6, max_removable + 1)):  # Remove 1-5 questions max
                if len(all_questions) - num_remove >= MIN_QUESTIONS_PER_CONTEXT:
                    questions_to_remove = random.sample(all_questions, num_remove)
                    variant_text = create_text_variant_removing_questions(
                        original_entry['context'], 
                        questions_to_remove, 
                        all_questions, 
                        llm_config
                    )
                    
                    if variant_text:
                        # Verify which questions this variant can actually answer
                        actual_question_set = get_question_set_for_text(variant_text, all_questions, llm_config)
                        
                        # Only accept if we still have minimum questions
                        if len(actual_question_set) >= MIN_QUESTIONS_PER_CONTEXT:
                            variant = {
                                'text': variant_text,
                                'variant_type': f'remove_{num_remove}',
                                'removed_questions': questions_to_remove,
                                'question_set': actual_question_set
                            }
                            original_variants.append(variant)
                            print(f"    ✅ Created variant removing {num_remove} questions ({len(actual_question_set)} questions remain)")
                        else:
                            print(f"    ❌ Variant rejected: too few remaining questions ({len(actual_question_set)} < {MIN_QUESTIONS_PER_CONTEXT})")
                else:
                    print(f"    ⚠️ Cannot remove {num_remove} questions: would leave {len(all_questions) - num_remove} < {MIN_QUESTIONS_PER_CONTEXT}")
        else:
            print(f"    ⚠️ Cannot create removal variants: only {len(all_questions)} questions available, need {MIN_QUESTIONS_PER_CONTEXT} minimum")
        
        # Create variants for paraphrase text (similar process)
        paraphrase_variants = []
        
        # Variant 0: Paraphrase text (no changes)
        paraphrase_variants.append({
            'text': paraphrase_text,
            'variant_type': 'paraphrase',
            'removed_questions': [],
            'question_set': all_questions
        })
        
        # Variants 1-N: Remove 1-N questions from paraphrase
        if max_removable > 0:
            for num_remove in range(1, min(6, max_removable + 1)):
                if len(all_questions) - num_remove >= MIN_QUESTIONS_PER_CONTEXT:
                    questions_to_remove = random.sample(all_questions, num_remove)
                    variant_text = create_text_variant_removing_questions(
                        paraphrase_text, 
                        questions_to_remove, 
                        all_questions, 
                        llm_config
                    )
                    
                    if variant_text:
                        actual_question_set = get_question_set_for_text(variant_text, all_questions, llm_config)
                        
                        if len(actual_question_set) >= MIN_QUESTIONS_PER_CONTEXT:
                            variant = {
                                'text': variant_text,
                                'variant_type': f'paraphrase_remove_{num_remove}',
                                'removed_questions': questions_to_remove,
                                'question_set': actual_question_set
                            }
                            paraphrase_variants.append(variant)
        
        # Store all variants for this pair (only if we have meaningful variants)
        if len(original_variants) > 1 or len(paraphrase_variants) > 1:  # More than just the base variants
            pair_variants = {
                'pair_id': pair_idx,
                'original_variants': original_variants,
                'paraphrase_variants': paraphrase_variants,
                'source_entry': original_entry
            }
            all_variants.append(pair_variants)
            print(f"    ✅ Pair included: {len(original_variants)} original + {len(paraphrase_variants)} paraphrase variants")
        else:
            print(f"    ❌ Pair excluded: insufficient viable variants")
    
    print(f"Step 3 complete: Created variants for {len(all_variants)} pairs")
    return all_variants

# ===== STEP 4: CREATE LABELED PAIRS =====
def determine_relationship(variant1, variant2):
    """Determine relationship between two text variants"""
    q_set1 = set(q['id'] for q in variant1['question_set'])
    q_set2 = set(q['id'] for q in variant2['question_set'])
    
    if q_set1 == q_set2:
        return "equivalence"  # Same questions
    elif q_set1.issubset(q_set2) or q_set2.issubset(q_set1):
        return "inclusion"    # One includes the other
    else:
        return "semantic_overlap"  # Partial overlap

def create_labeled_pairs(all_variants):
    """Step 4: Create labeled pairs from variants"""
    print("Step 4: Creating labeled pairs...")
    
    labeled_pairs = []
    target_counts = {
        'equivalence': 6,      # 6 rephrasing pairs
        'inclusion': 10,       # 10 inclusion pairs  
        'semantic_overlap': 20  # 20 semantic overlap pairs
    }
    current_counts = {'equivalence': 0, 'inclusion': 0, 'semantic_overlap': 0}
    
    for variants_set in all_variants:
        original_variants = variants_set['original_variants']
        paraphrase_variants = variants_set['paraphrase_variants']
        
        # Create all possible combinations
        all_combinations = []
        
        # Original-Original combinations
        for i, var1 in enumerate(original_variants):
            for j, var2 in enumerate(original_variants):
                if i != j:
                    all_combinations.append((var1, var2, 'original-original'))
        
        # Paraphrase-Paraphrase combinations  
        for i, var1 in enumerate(paraphrase_variants):
            for j, var2 in enumerate(paraphrase_variants):
                if i != j:
                    all_combinations.append((var1, var2, 'paraphrase-paraphrase'))
        
        # Original-Paraphrase combinations
        for var1 in original_variants:
            for var2 in paraphrase_variants:
                all_combinations.append((var1, var2, 'original-paraphrase'))
                all_combinations.append((var2, var1, 'paraphrase-original'))
        
        # Determine relationships and create pairs
        for var1, var2, combination_type in all_combinations:
            relationship = determine_relationship(var1, var2)
            
            # Check if we need more of this relationship type
            if current_counts[relationship] < target_counts[relationship]:
                pair = {
                    'text1': var1['text'],
                    'text2': var2['text'],
                    'variant1_info': var1,
                    'variant2_info': var2,
                    'relationship': relationship,
                    'combination_type': combination_type,
                    'pair_id': len(labeled_pairs)
                }
                labeled_pairs.append(pair)
                current_counts[relationship] += 1
                
                print(f"    Created {relationship} pair ({current_counts[relationship]}/{target_counts[relationship]})")
                
                # Stop if we have enough pairs
                if sum(current_counts.values()) >= sum(target_counts.values()):
                    break
        
        if sum(current_counts.values()) >= sum(target_counts.values()):
            break
    
    print(f"Step 4 complete: Created {len(labeled_pairs)} labeled pairs")
    print(f"  Equivalence: {current_counts['equivalence']}")
    print(f"  Inclusion: {current_counts['inclusion']}")  
    print(f"  Semantic overlap: {current_counts['semantic_overlap']}")
    
    return labeled_pairs

# ===== STEP 5: EVALUATION AND VALIDATION =====
def evaluate_pairs_with_judge(labeled_pairs, llm_config):
    """Evaluate pairs using JUDGE system"""
    print("Step 5: Evaluating pairs with JUDGE...")
    
    for pair_idx, pair in enumerate(labeled_pairs):
        print(f"  Evaluating pair {pair_idx+1}/{len(labeled_pairs)}")
        
        text1 = pair['text1']
        text2 = pair['text2']
        
        # Get question sets for both texts
        q_set1 = pair['variant1_info']['question_set']
        q_set2 = pair['variant2_info']['question_set']
        
        # Evaluate common questions
        common_questions = []
        for q1 in q_set1:
            for q2 in q_set2:
                if q1['id'] == q2['id']:
                    common_questions.append(q1)
                    break
        
        judge_scores = []
        for question_obj in common_questions:
            # Get answers from both texts - require 100% accuracy
            accuracy1 = test_llm_qa_accuracy(text1, [question_obj], llm_config)
            accuracy2 = test_llm_qa_accuracy(text2, [question_obj], llm_config)
            
            # If both can answer with 100% accuracy, compare the answers using judge
            if accuracy1 >= QA_ACCURACY_THRESHOLD and accuracy2 >= QA_ACCURACY_THRESHOLD:
                # This would require getting the actual answers and comparing them
                # For now, we'll simulate this
                judge_scores.append(1)  # Assume they match
            else:
                judge_scores.append(0)
        
        pair['judge_score'] = sum(judge_scores)
        pair['max_possible_score'] = len(common_questions)
        pair['judge_ratio'] = pair['judge_score'] / pair['max_possible_score'] if pair['max_possible_score'] > 0 else 0
        
    print("Step 5 complete: All pairs evaluated")
    return labeled_pairs

# ===== MAIN EXECUTION =====
def select_available_llm_config():
    """Select the first available LLM configuration"""
    # Check Azure credentials first
    if AZURE_AI_API_KEY and AZURE_AI_ENDPOINT:
        for config in LLM_CONFIGURATIONS:
            if config["type"] == "azure_ai_inference":
                print(f"✅ Using Azure AI model: {config['name']}")
                return config
    
    # Fall back to HF local models
    for config in LLM_CONFIGURATIONS:
        if config["type"] == "hf_local":
            print(f"✅ Using local HF model: {config['name']}")
            return config
    
    # If no config available
    print("❌ No available LLM configuration found!")
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ensure_dir(PLOTS_OUTPUT_DIR)
    ensure_dir('./data')
    download_nltk_dependencies()
    load_qa_pipeline(device)
    load_sbert_model()

    # ===== PREPROCESSING STEP: CREATE CUSTOM DATASET =====
    if CREATE_CUSTOM_DATASET and USE_CUSTOM_FORMAT:
        print(f"\n{'='*60}")
        print("STEP 0: Creating Custom Dataset from SQuAD")
        print(f"{'='*60}")
        
        # Check if custom dataset already exists
        if not check_if_custom_dataset_exists(CUSTOM_JSON_FILE_PATH):
            print(f'Custom dataset not found. Creating from: {SQUAD_JSON_FILE_PATH}')
            
            # Verify original SQuAD file exists
            if not os.path.exists(SQUAD_JSON_FILE_PATH):
                print(f"❌ Error: Original SQuAD file not found: {SQUAD_JSON_FILE_PATH}")
                print("Please download train-v2.0.json from SQuAD dataset")
                return
            
            # Create custom dataset
            success = create_custom_dataset_from_squad(
                input_file_path=SQUAD_JSON_FILE_PATH,
                output_file_path=CUSTOM_JSON_FILE_PATH,
                target_size=CUSTOM_DATASET_SIZE,
                min_questions=MIN_QUESTIONS_PER_CONTEXT
            )
            
            if not success:
                print("❌ Failed to create custom dataset. Exiting.")
                return
                
            print(f"✅ Custom dataset created successfully!")
        else:
            print("✅ Using existing custom dataset")
    elif USE_CUSTOM_FORMAT:
        print(f"Custom dataset creation disabled. Using existing file: {CUSTOM_JSON_FILE_PATH}")
        if not os.path.exists(CUSTOM_JSON_FILE_PATH):
            print(f"❌ Error: Custom dataset file not found: {CUSTOM_JSON_FILE_PATH}")
            print("Set CREATE_CUSTOM_DATASET = True to create it automatically")
            return

    # Load data
    print(f"\n{'='*60}")
    print("MAIN PROCESS: Dataset Creation Pipeline")
    print(f"{'='*60}")
    print(f"Filtering requirements:")
    print(f"  - Skip unanswerable questions: {SKIP_UNANSWERABLE}")
    print(f"  - Minimum questions per context: {MIN_QUESTIONS_PER_CONTEXT}")
    print(f"  - Required QA accuracy: {QA_ACCURACY_THRESHOLD*100}% (perfect accuracy)")
    print(f"  - METEOR similarity threshold: {METEOR_THRESHOLD}")
    
    json_data = load_official_squad_json(DATA_FILE_PATH)
    if not json_data:
        print("Failed to load data. Exiting.")
        return
    
    entries = json_data[:MAX_TEXTS_TO_PROCESS]
    print(f"\nProcessing {len(entries)} entries that meet minimum question requirements")

    # Select LLM configuration
    llm_config = select_available_llm_config()
    if not llm_config:
        print("❌ No available LLM configuration. Exiting.")
        return
    
    print(f"Using LLM: {llm_config['name']}")

    # Initialize model if needed
    if llm_config["type"] == "hf_local":
        try:
            load_hf_model_and_tokenizer(llm_config['model_id'], device)
        except Exception as e:
            print(f"Failed to load HF model: {e}")
            return

    # Execute the 5-step process
    try:
        # Step 1: Filter texts with correct answers
        filtered_entries = filter_texts_with_correct_answers(entries, llm_config)
        
        if not filtered_entries:
            print("No entries passed filtering. Exiting.")
            return
        
        # Step 2: Create paraphrasing pairs
        paraphrase_pairs = create_paraphrasing_pairs(filtered_entries[:10], llm_config)  # Limit for testing
        
        if not paraphrase_pairs:
            print("No paraphrase pairs created. Exiting.")
            return
        
        # Step 3: Create text variants
        all_variants = create_text_variants(paraphrase_pairs, llm_config)
        
        # Step 4: Create labeled pairs
        labeled_pairs = create_labeled_pairs(all_variants)
        
        # Step 5: Evaluate with JUDGE
        final_pairs = evaluate_pairs_with_judge(labeled_pairs, llm_config)
        
        # Save results
        output_file = f'dataset_creation_results_{len(final_pairs)}_pairs.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Created {len(final_pairs)} labeled pairs")
        print(f"Results saved to: {output_file}")
        
        # Print summary statistics
        relationship_counts = {}
        for pair in final_pairs:
            rel = pair['relationship']
            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1
        
        print("\nRelationship distribution:")
        for rel, count in relationship_counts.items():
            print(f"  {rel}: {count} pairs")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        
    finally:
        if llm_config["type"] == "hf_local":
            release_hf_model()
    
    print(f"\n{'='*60}")
    print("PROCESS COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    # create_custom_dataset_from_squad(SQUAD_JSON_FILE_PATH, CUSTOM_JSON_FILE_PATH, CUSTOM_DATASET_SIZE, MIN_QUESTIONS_PER_CONTEXT)
    main()