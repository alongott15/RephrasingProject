import os
import json
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
import torch.nn as nn
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===== AZURE AI CLIENT =====
class AzureAIClient:
    def __init__(self, endpoint: str = None, api_key: str = None, model_name: str = "gpt-4.1"):
        self.endpoint = endpoint or os.getenv("AZURE_AI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_AI_API_KEY")
        self.model_name = model_name
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure AI endpoint and API key must be provided")
        
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
    
    def chat_completion(self, system_message: str, user_message: str, temperature: float = 0.0) -> str:
        try:
            response = self.client.complete(
                messages=[
                    SystemMessage(content=system_message),
                    UserMessage(content=user_message)
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Azure AI completion failed: {e}")
            raise

# ===== CONFIGURATION =====
load_dotenv()

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Classification labels (based on Hebrew document requirements)
LABEL_MAPPING = {
    'equivalence': 0,           # יחס שקילות - same questions answered exactly
    'inclusion': 1,             # יחס הכלה - inclusion between questions they answer  
    'semantic_overlap': 2       # חפיפה סימנטית - semantic mutual overlapping
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# BERT model configurations
BERT_MODELS = {
    'BERT-base': 'bert-base-uncased',
    'RoBERTa-base': 'roberta-base', 
    'Longformer-base': 'allenai/longformer-base-4096'
}

# ===== DATASET LOADING AND PREPROCESSING =====
class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=512):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = self.labels[idx]
        
        # Use [CLS] Document A [SEP] Document B [SEP] format
        encoding = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(dataset_path):
    """Load the generated dataset from JSON file"""
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata if available
    if 'metadata' in data:
        metadata = data['metadata']
        pairs = data['pairs']
        print(f"📊 Dataset Metadata:")
        print(f"   Creation date: {metadata.get('creation_timestamp', 'N/A')}")
        print(f"   Paraphrase pairs: {metadata.get('paraphrase_pairs_from_step2', 'N/A')}")
        print(f"   Total labeled pairs: {metadata.get('total_labeled_pairs', len(pairs))}")
        print(f"   Average per paraphrase: {metadata.get('average_pairs_per_paraphrase', 'N/A')}")
    else:
        pairs = data
        print(f"   Total pairs in dataset: {len(pairs)}")
    
    # Convert to DataFrame for easier processing
    df_data = []
    for pair in pairs:
        # Map relationship types to Hebrew document categories
        relationship = pair['relationship'].lower()
        
        # Map to the 3-class system from Hebrew document
        if relationship in ['equivalence', 'paraphrasing', 'paraphrase', 'rephrasing']:
            mapped_relationship = 'equivalence'  # יחס שקילות
        elif relationship in ['inclusion', 'a_includes_b', 'b_includes_a', 'a includes b', 'b includes a']:
            mapped_relationship = 'inclusion'     # יחס הכלה
        else:
            mapped_relationship = 'semantic_overlap'  # חפיפה סימנטית
        
        df_data.append({
            'text1': pair['text1'],
            'text2': pair['text2'],
            'relationship': mapped_relationship,
            'original_relationship': pair['relationship'],  # Keep original for reference
            'combination_type': pair.get('combination_type', 'unknown'),
            'pair_id': pair.get('pair_id', 0),
            'paraphrase_pair_id': pair.get('paraphrase_pair_id', 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Calculate expected numbers based on Hebrew document methodology
    # Each paraphrase pair generates 36 new pairs: 6 equivalence + 10 inclusion + 20 semantic overlap
    total_pairs = len(df)
    num_paraphrase_sources = total_pairs // 36 if total_pairs >= 36 else 1
    
    print(f"\n📈 Dataset Statistics (Hebrew Document Categories):")
    print(f"   Total pairs: {len(df)}")
    print(f"   Estimated paraphrase sources: ~{num_paraphrase_sources}")
    print(f"   Target distribution per 36 pairs (from each paraphrase source):")
    print(f"     - Equivalence (שקילות): 6 pairs (16.7%)")
    print(f"     - Inclusion (הכלה): 10 pairs (27.8%)")
    print(f"     - Semantic Overlap (חפיפה סימנטית): 20 pairs (55.6%)")
    print(f"\n   Expected total distribution for {total_pairs} pairs:")
    expected_equiv = int((6/36) * total_pairs)
    expected_incl = int((10/36) * total_pairs)
    expected_overlap = int((20/36) * total_pairs)
    print(f"     - Equivalence: ~{expected_equiv} pairs ({6/36*100:.1f}%)")
    print(f"     - Inclusion: ~{expected_incl} pairs ({10/36*100:.1f}%)")
    print(f"     - Semantic Overlap: ~{expected_overlap} pairs ({20/36*100:.1f}%)")
    print(f"\n   Actual distribution:")
    for rel, count in df['relationship'].value_counts().items():
        print(f"     {rel}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def create_train_test_split(df, test_size=0.2, stratify=True):
    """Split dataset into train and test sets"""
    print(f"\n🔄 Creating train/test split (test_size={test_size})")
    
    if stratify and len(df['relationship'].unique()) > 1:
        # Stratified split to maintain label distribution
        X = df[['text1', 'text2']].values
        y = df['relationship'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        train_df = pd.DataFrame({
            'text1': X_train[:, 0],
            'text2': X_train[:, 1], 
            'relationship': y_train
        })
        
        test_df = pd.DataFrame({
            'text1': X_test[:, 0],
            'text2': X_test[:, 1],
            'relationship': y_test
        })
    else:
        # Random split if stratification is not possible
        train_df = df.sample(frac=1-test_size, random_state=RANDOM_SEED)
        test_df = df.drop(train_df.index)
    
    print(f"   Train set: {len(train_df)} pairs")
    print(f"   Test set: {len(test_df)} pairs")
    print(f"   Train distribution: {dict(train_df['relationship'].value_counts())}")
    print(f"   Test distribution: {dict(test_df['relationship'].value_counts())}")
    
    return train_df, test_df

# ===== GPT-4 CLASSIFIER (Adapted for Hebrew document requirements) =====
class GPTClassifier:
    def __init__(self, azure_client, mode='zero_shot'):
        self.client = azure_client
        self.mode = mode  # 'zero_shot' or 'few_shot'
        
        # Zero-shot prompt adapted for 3-class system
        self.zero_shot_prompt = """You are a language expert tasked with identifying the semantic relationship between two texts based on the questions they can answer. The possible relationships are:

1. EQUIVALENCE (שקילות) - Both texts answer exactly the same set of questions
2. INCLUSION (הכלה) - One text can answer all questions of the other text plus additional questions
3. SEMANTIC_OVERLAP (חפיפה סימנטית) - The texts have partial semantic overlap but neither fully includes the other

Text A:
"{TEXT_A}"

Text B:
"{TEXT_B}"

What is the semantic relationship between Text A and Text B?
Answer with one of: "EQUIVALENCE", "INCLUSION", or "SEMANTIC_OVERLAP"."""

        # Few-shot prompt adapted for 3-class system
        self.few_shot_prompt = """You are a language expert tasked with identifying the semantic relationship between two texts based on the questions they can answer.

The possible relationships are:
1. EQUIVALENCE - Both texts answer exactly the same questions
2. INCLUSION - One text answers all questions of the other plus more
3. SEMANTIC_OVERLAP - Partial overlap but no complete inclusion

Example 1:
Text A: "The Eiffel Tower is located in Paris and attracts millions of tourists every year."
Text B: "Many tourists visit the Eiffel Tower in Paris annually."
Answer: INCLUSION

Example 2:
Text A: "Photosynthesis occurs in plant leaves using sunlight, water, and carbon dioxide."
Text B: "The process of photosynthesis in plants uses water, CO₂, and sunlight in leaves."
Answer: EQUIVALENCE

Example 3:
Text A: "The collapse of mortgage-backed securities triggered the 2008 financial crisis."
Text B: "The Great Depression was caused by a stock market crash in 1929."
Answer: SEMANTIC_OVERLAP

Now, determine the relationship:
Text A:
"{TEXT_A}"

Text B:
"{TEXT_B}"

Answer:"""
    
    def predict_single(self, text1, text2):
        """Classify a single pair of texts"""
        if self.mode == 'zero_shot':
            user_message = self.zero_shot_prompt.format(TEXT_A=text1, TEXT_B=text2)
        else:  # few_shot
            user_message = self.few_shot_prompt.format(TEXT_A=text1, TEXT_B=text2)
        
        try:
            response = self.client.chat_completion(
                "", 
                user_message, 
                temperature=0.0
            ).strip()
            
            # Map response to our labels
            response_lower = response.lower()
            if 'equivalence' in response_lower:
                return 'equivalence'
            elif 'inclusion' in response_lower:
                return 'inclusion'
            elif 'semantic_overlap' in response_lower or 'overlap' in response_lower:
                return 'semantic_overlap'
            else:
                print(f"⚠️ Unexpected GPT response: {response}")
                return 'semantic_overlap'  # Default fallback
                
        except Exception as e:
            print(f"❌ GPT classification error: {e}")
            return 'semantic_overlap'  # Default fallback
    
    def predict(self, texts1, texts2):
        """Classify multiple pairs of texts"""
        predictions = []
        total = len(texts1)
        
        print(f"🤖 GPT-4 {self.mode} classifying {total} pairs...")
        
        for i, (text1, text2) in enumerate(zip(texts1, texts2)):
            pred = self.predict_single(text1, text2)
            predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{total}")
        
        return predictions

# ===== BERT-BASED CLASSIFIERS =====
class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        print(f"Initializing BERT model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
        # Handle special cases for different models
        if 'longformer' in model_name.lower():
            self.max_length = 4096  # Longformer can handle longer sequences
        else:
            self.max_length = 512
        
    def prepare_model(self, num_labels=3):  # 3 classes for Hebrew document requirements
        """Prepare the model for fine-tuning"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(DEVICE)
    
    def train(self, train_texts1, train_texts2, train_labels, 
              val_texts1=None, val_texts2=None, val_labels=None):
        """Fine-tune BERT on the 3-class dataset"""
        print(f"🔧 Fine-tuning {self.model_name} classifier for 3-class Hebrew document system...")
        
        # Prepare model
        self.prepare_model()
        
        # Convert labels to numeric
        label_encoder = LabelEncoder()
        train_labels_numeric = label_encoder.fit_transform(train_labels)
        self.label_encoder = label_encoder
        
        # Create datasets
        train_dataset = TextPairDataset(
            train_texts1, train_texts2, train_labels_numeric, 
            self.tokenizer, max_length=self.max_length
        )
        
        val_dataset = None
        if val_texts1 is not None:
            val_labels_numeric = label_encoder.transform(val_labels)
            val_dataset = TextPairDataset(
                val_texts1, val_texts2, val_labels_numeric, 
                self.tokenizer, max_length=self.max_length
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./bert_classifier_{self.model_name.replace("/", "_")}',
            num_train_epochs=3,
            per_device_train_batch_size=16 if 'longformer' not in self.model_name.lower() else 8,
            per_device_eval_batch_size=16 if 'longformer' not in self.model_name.lower() else 8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model='eval_loss' if val_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        self.trainer.train()
        print(f"✅ {self.model_name} fine-tuning completed")
    
    def predict(self, texts1, texts2):
        """Make predictions using fine-tuned BERT"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        predictions = []
        
        print(f"🤖 {self.model_name} classifying {len(texts1)} pairs...")
        
        with torch.no_grad():
            for i, (text1, text2) in enumerate(zip(texts1, texts2)):
                inputs = self.tokenizer(
                    text1, text2,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(DEVICE)
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
                predictions.append(predicted_label)
                
                if (i + 1) % 50 == 0:
                    print(f"   Progress: {i + 1}/{len(texts1)}")
        
        return predictions

# ===== EVALUATION AND METRICS =====
def evaluate_classifier(y_true, y_pred, classifier_name):
    """Evaluate classifier performance"""
    print(f"\n📊 Evaluating {classifier_name}")
    print("="*50)
    
    # Primary metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAPPING.keys())))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_MAPPING.keys()))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=['Equivalence\n(שקילות)', 'Inclusion\n(הכלה)', 'Semantic Overlap\n(חפיפה סימנטית)'],
                yticklabels=['Equivalence\n(שקילות)', 'Inclusion\n(הכלה)', 'Semantic Overlap\n(חפיפה סימנטית)'],
                cmap='Blues')
    plt.title(f'Confusion Matrix - {classifier_name}\n(Hebrew Document 3-Class System)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{classifier_name.lower().replace(" ", "_").replace("-", "_")}.png', dpi=300)
    plt.show()
    
    return accuracy, macro_f1

def analyze_class_distribution(predictions_dict, true_labels):
    """Analyze how well models match expected distribution from Hebrew document"""
    print(f"\n📊 CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Expected distribution from Hebrew document (per every 36 pairs from each paraphrase source)
    expected_dist = {
        'equivalence': 6/36,      # ~16.7%
        'inclusion': 10/36,       # ~27.8%
        'semantic_overlap': 20/36  # ~55.6%
    }
    
    total_test_pairs = len(true_labels)
    estimated_paraphrase_sources = total_test_pairs // 36 if total_test_pairs >= 36 else 1
    
    print("Expected distribution per paraphrase source (Hebrew document methodology):")
    print("  Each paraphrase pair → 36 new pairs with distribution:")
    for class_name, percentage in expected_dist.items():
        pairs_per_36 = int(percentage * 36)
        total_expected = int(percentage * total_test_pairs)
        print(f"    {class_name}: {pairs_per_36}/36 per source → ~{total_expected} total ({percentage:.1%})")
    
    print(f"\nActual distribution in test set ({total_test_pairs} pairs):")
    true_dist = pd.Series(true_labels).value_counts(normalize=True)
    true_counts = pd.Series(true_labels).value_counts()
    for class_name in LABEL_MAPPING.keys():
        percentage = true_dist.get(class_name, 0)
        count = true_counts.get(class_name, 0)
        print(f"  {class_name}: {count} pairs ({percentage:.1%})")
    
    print(f"\nModel predictions distribution:")
    for model_name, predictions in predictions_dict.items():
        pred_dist = pd.Series(predictions).value_counts(normalize=True)
        pred_counts = pd.Series(predictions).value_counts()
        print(f"\n{model_name}:")
        for class_name in LABEL_MAPPING.keys():
            percentage = pred_dist.get(class_name, 0)
            count = pred_counts.get(class_name, 0)
            print(f"  {class_name}: {count} pairs ({percentage:.1%})")

def compare_classifiers(results_dict, test_labels):
    """Compare performance of all classifiers"""
    print(f"\n🏆 CLASSIFIER COMPARISON (Hebrew Document System)")
    print("="*60)
    
    comparison_data = []
    
    for name, predictions in results_dict.items():
        accuracy = accuracy_score(test_labels, predictions)
        macro_f1 = f1_score(test_labels, predictions, average='macro')
        
        comparison_data.append({
            'Classifier': name,
            'Accuracy': accuracy,
            'Macro-F1': macro_f1,
            'Correct': sum(1 for t, p in zip(test_labels, predictions) if t == p),
            'Total': len(test_labels)
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(comparison_df['Classifier'], comparison_df['Accuracy'], 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(comparison_df)])
    ax1.set_title('Classifier Accuracy Comparison\n(Hebrew Document 3-Class System)')
    ax1.set_xlabel('Classifier')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Macro-F1 comparison
    bars2 = ax2.bar(comparison_df['Classifier'], comparison_df['Macro-F1'], 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(comparison_df)])
    ax2.set_title('Classifier Macro-F1 Comparison\n(Hebrew Document 3-Class System)')
    ax2.set_xlabel('Classifier')
    ax2.set_ylabel('Macro-F1 Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('classifier_comparison_hebrew_system.png', dpi=300)
    plt.show()
    
    return comparison_df

# ===== MAIN CLASSIFICATION PIPELINE =====
def main_hebrew_document_classification_pipeline(dataset_path):
    """Main pipeline for classification according to Hebrew document requirements"""
    print(f"🚀 Starting Classification Pipeline (Hebrew Document 3-Class System)")
    print(f"📋 Classes: Equivalence (שקילות), Inclusion (הכלה), Semantic Overlap (חפיפה סימנטית)")
    print(f"{'='*80}")
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df, test_size=0.2)
    
    # Extract data
    train_texts1 = train_df['text1'].tolist()
    train_texts2 = train_df['text2'].tolist()
    train_labels = train_df['relationship'].tolist()
    
    test_texts1 = test_df['text1'].tolist()
    test_texts2 = test_df['text2'].tolist() 
    test_labels = test_df['relationship'].tolist()
    
    # Initialize results storage
    results = {}
    evaluation_metrics = {}
    
    # List all classifiers to run
    classifiers_to_run = [
        ('GPT-4 Zero-Shot', 'generative'),
        ('GPT-4 Few-Shot', 'generative'),
        ('BERT-base', 'bert'),
        ('RoBERTa-base', 'bert'),
        ('Longformer-base', 'bert')
    ]
    
    print(f"\n🎯 Will run {len(classifiers_to_run)} classifiers:")
    for name, type_ in classifiers_to_run:
        print(f"   • {name} ({type_})")
    print()
    
    # Initialize Azure client once for all GPT classifiers
    azure_client = None
    try:
        azure_client = AzureAIClient()
        print("✅ Azure AI client initialized successfully")
    except Exception as e:
        print(f"⚠️ Azure AI client initialization failed: {e}")
        print("   GPT classifiers will be skipped")
    
    # 1. GPT-4 Zero-Shot Classifier
    print(f"\n{'='*70}")
    print("1/5. GPT-4 ZERO-SHOT CLASSIFIER")
    print(f"{'='*70}")
    
    if azure_client:
        try:
            print("🤖 Running GPT-4 Zero-Shot classifier...")
            gpt_zs_classifier = GPTClassifier(azure_client, mode='zero_shot')
            gpt_zs_predictions = gpt_zs_classifier.predict(test_texts1, test_texts2)
            results['GPT-4 Zero-Shot'] = gpt_zs_predictions
            
            acc, f1 = evaluate_classifier(test_labels, gpt_zs_predictions, 'GPT-4 Zero-Shot')
            evaluation_metrics['GPT-4 Zero-Shot'] = {'accuracy': acc, 'macro_f1': f1}
            print("✅ GPT-4 Zero-Shot completed successfully")
            
        except Exception as e:
            print(f"❌ GPT-4 Zero-Shot classifier failed: {e}")
            results['GPT-4 Zero-Shot'] = ['semantic_overlap'] * len(test_labels)
            evaluation_metrics['GPT-4 Zero-Shot'] = {'accuracy': 0.0, 'macro_f1': 0.0}
            print("⚠️ Using fallback predictions for GPT-4 Zero-Shot")
    else:
        print("⏭️ Skipping GPT-4 Zero-Shot (Azure client not available)")
        results['GPT-4 Zero-Shot'] = ['semantic_overlap'] * len(test_labels)
        evaluation_metrics['GPT-4 Zero-Shot'] = {'accuracy': 0.0, 'macro_f1': 0.0}
    
    # 2. GPT-4 Few-Shot Classifier
    print(f"\n{'='*70}")
    print("2/5. GPT-4 FEW-SHOT CLASSIFIER")
    print(f"{'='*70}")
    
    if azure_client:
        try:
            print("🤖 Running GPT-4 Few-Shot classifier...")
            gpt_fs_classifier = GPTClassifier(azure_client, mode='few_shot')
            gpt_fs_predictions = gpt_fs_classifier.predict(test_texts1, test_texts2)
            results['GPT-4 Few-Shot'] = gpt_fs_predictions
            
            acc, f1 = evaluate_classifier(test_labels, gpt_fs_predictions, 'GPT-4 Few-Shot')
            evaluation_metrics['GPT-4 Few-Shot'] = {'accuracy': acc, 'macro_f1': f1}
            print("✅ GPT-4 Few-Shot completed successfully")
            
        except Exception as e:
            print(f"❌ GPT-4 Few-Shot classifier failed: {e}")
            results['GPT-4 Few-Shot'] = ['semantic_overlap'] * len(test_labels)
            evaluation_metrics['GPT-4 Few-Shot'] = {'accuracy': 0.0, 'macro_f1': 0.0}
            print("⚠️ Using fallback predictions for GPT-4 Few-Shot")
    else:
        print("⏭️ Skipping GPT-4 Few-Shot (Azure client not available)")
        results['GPT-4 Few-Shot'] = ['semantic_overlap'] * len(test_labels)
        evaluation_metrics['GPT-4 Few-Shot'] = {'accuracy': 0.0, 'macro_f1': 0.0}
    
    # 3-5. BERT-based Classifiers
    classifier_count = 3
    for model_name, model_path in BERT_MODELS.items():
        print(f"\n{'='*70}")
        print(f"{classifier_count}/5. {model_name.upper()} CLASSIFIER")
        print(f"{'='*70}")
        
        try:
            print(f"🤖 Running {model_name} classifier...")
            print(f"   Model path: {model_path}")
            
            # Initialize classifier
            bert_classifier = BERTClassifier(model_path)
            
            # Train classifier
            print(f"   Training {model_name}...")
            bert_classifier.train(train_texts1, train_texts2, train_labels)
            
            # Make predictions
            print(f"   Making predictions with {model_name}...")
            bert_predictions = bert_classifier.predict(test_texts1, test_texts2)
            results[model_name] = bert_predictions
            
            # Evaluate
            acc, f1 = evaluate_classifier(test_labels, bert_predictions, model_name)
            evaluation_metrics[model_name] = {'accuracy': acc, 'macro_f1': f1}
            print(f"✅ {model_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {model_name} classifier failed: {e}")
            import traceback
            print(f"   Error details: {traceback.format_exc()}")
            results[model_name] = ['semantic_overlap'] * len(test_labels)
            evaluation_metrics[model_name] = {'accuracy': 0.0, 'macro_f1': 0.0}
            print(f"⚠️ Using fallback predictions for {model_name}")
        
        classifier_count += 1
    
    # Summary of completed classifiers
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETION SUMMARY")
    print(f"{'='*70}")
    
    successful_classifiers = []
    failed_classifiers = []
    
    for name in results.keys():
        if evaluation_metrics[name]['accuracy'] > 0:
            successful_classifiers.append(name)
        else:
            failed_classifiers.append(name)
    
    print(f"✅ Successfully completed: {len(successful_classifiers)}/{len(classifiers_to_run)} classifiers")
    for name in successful_classifiers:
        acc = evaluation_metrics[name]['accuracy']
        f1 = evaluation_metrics[name]['macro_f1']
        print(f"   • {name}: Accuracy={acc:.3f}, Macro-F1={f1:.3f}")
    
    if failed_classifiers:
        print(f"\n❌ Failed classifiers: {len(failed_classifiers)}")
        for name in failed_classifiers:
            print(f"   • {name} (using fallback predictions)")
    
    # Analyze class distributions
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    analyze_class_distribution(results, test_labels)
    
    # Compare all classifiers
    print(f"\n{'='*70}")
    print("CLASSIFIER COMPARISON")
    print(f"{'='*70}")
    comparison_df = compare_classifiers(results, test_labels)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'hebrew_document_classification_results_{timestamp}.json'
    
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'dataset_path': dataset_path,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'random_seed': RANDOM_SEED,
            'classification_system': 'Hebrew Document 3-Class System',
            'label_mapping': LABEL_MAPPING,
            'hebrew_document_methodology': {
                'description': 'Each paraphrase pair generates 36 new pairs',
                'distribution_per_36_pairs': {
                    'equivalence': '6 pairs (16.7%)',
                    'inclusion': '10 pairs (27.8%)', 
                    'semantic_overlap': '20 pairs (55.6%)'
                },
                'total_pairs': len(df),
                'estimated_paraphrase_sources': len(df) // 36 if len(df) >= 36 else 1,
                'expected_total_distribution': {
                    'equivalence_pairs': int((6/36) * len(df)),
                    'inclusion_pairs': int((10/36) * len(df)),
                    'semantic_overlap_pairs': int((20/36) * len(df))
                }
            },
            'pipeline_summary': {
                'total_classifiers': len(classifiers_to_run),
                'successful_classifiers': len(successful_classifiers),
                'failed_classifiers': len(failed_classifiers),
                'successful_list': successful_classifiers,
                'failed_list': failed_classifiers
            },
            'models_tested': list(results.keys())
        },
        'test_data': {
            'texts1': test_texts1,
            'texts2': test_texts2,
            'true_labels': test_labels
        },
        'predictions': results,
        'evaluation_metrics': evaluation_metrics,
        'comparison': comparison_df.to_dict('records')
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print final summary
    print(f"\n🎯 FINAL SUMMARY (Hebrew Document 3-Class System)")
    print("="*60)
    print("Hebrew Document Methodology:")
    print("  • Each paraphrase pair → 36 new labeled pairs")
    print("  • Distribution per 36 pairs: 6 equivalence + 10 inclusion + 20 semantic overlap")
    print(f"  • Your dataset: {len(df)} total pairs from ~{len(df)//36 if len(df)>=36 else 1} paraphrase sources")
    print(f"\nPipeline Results:")
    print(f"  • Successfully ran: {len(successful_classifiers)}/{len(classifiers_to_run)} classifiers")
    print("\nClasses:")
    print("  • Equivalence (שקילות): Same questions answered exactly")
    print("  • Inclusion (הכלה): One includes the other's questions")  
    print("  • Semantic Overlap (חפיפה סימנטית): Partial overlap")
    
    if successful_classifiers:
        print("\nBest performing models:")
        # Filter comparison_df to only include successful classifiers
        successful_df = comparison_df[comparison_df['Classifier'].isin(successful_classifiers)]
        top_3 = successful_df.head(3)
        for _, row in top_3.iterrows():
            print(f"  {row['Classifier']}: Accuracy={row['Accuracy']:.3f}, Macro-F1={row['Macro-F1']:.3f}")
    else:
        print("\n⚠️ No classifiers completed successfully - check error messages above")
    
    return results, evaluation_metrics, comparison_df

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset_creation_results_2952_pairs_20250815_092116.json"  # Update with your dataset path
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please provide the correct path to your generated dataset")
    else:
        results, metrics, comparison = main_hebrew_document_classification_pipeline(dataset_path)
        
        print(f"\n🎉 Hebrew Document Classification pipeline completed!")
        print(f"Results follow the 3-class system specified in your Hebrew document.")
        print(f"Methodology: Each paraphrase pair → 36 new pairs (6 equivalence + 10 inclusion + 20 semantic overlap)")