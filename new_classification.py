import os
import json
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# Classification labels
LABEL_MAPPING = {
    'equivalence': 0,      # Rephrasing/equivalence pairs
    'inclusion': 1,        # Inclusion relationship
    'semantic_overlap': 2  # Semantic overlap
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Model configurations
BERT_MODELS = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2', 
    'microsoft/DialoGPT-medium',
    'distilbert-base-uncased'
]

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
        
        # Tokenize the pair of texts
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
        df_data.append({
            'text1': pair['text1'],
            'text2': pair['text2'],
            'relationship': pair['relationship'],
            'combination_type': pair.get('combination_type', 'unknown'),
            'pair_id': pair.get('pair_id', 0),
            'paraphrase_pair_id': pair.get('paraphrase_pair_id', 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Print dataset statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total pairs: {len(df)}")
    print(f"   Relationship distribution:")
    for rel, count in df['relationship'].value_counts().items():
        print(f"     {rel}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def create_train_test_split(df, test_size=0.2, stratify=True):
    """Split dataset into train and test sets"""
    print(f"\n🔄 Creating train/test split (test_size={test_size})")
    
    if stratify:
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
        # Random split
        train_df = df.sample(frac=1-test_size, random_state=RANDOM_SEED)
        test_df = df.drop(train_df.index)
    
    print(f"   Train set: {len(train_df)} pairs")
    print(f"   Test set: {len(test_df)} pairs")
    print(f"   Train distribution: {dict(train_df['relationship'].value_counts())}")
    print(f"   Test distribution: {dict(test_df['relationship'].value_counts())}")
    
    return train_df, test_df

# ===== GPT-4.1 CLASSIFIER =====
class GPT41Classifier:
    def __init__(self, azure_client):
        self.client = azure_client
        self.system_prompt = """You are an expert text relationship classifier. Given two texts, classify their relationship into exactly one of these three categories:

1. EQUIVALENCE: The texts are paraphrases/rephrasing of each other - same content, different wording
2. INCLUSION: One text's content is included within the other - one is a subset of the other  
3. SEMANTIC_OVERLAP: The texts have partial semantic overlap but neither includes the other completely

Respond with only the category name: EQUIVALENCE, INCLUSION, or SEMANTIC_OVERLAP"""
    
    def predict_single(self, text1, text2):
        """Classify a single pair of texts"""
        user_message = f"""Text 1: {text1}

Text 2: {text2}

Classification:"""
        
        try:
            response = self.client.chat_completion(
                self.system_prompt, 
                user_message, 
                temperature=0.1
            ).strip().upper()
            
            # Map response to our labels
            if 'EQUIVALENCE' in response:
                return 'equivalence'
            elif 'INCLUSION' in response:
                return 'inclusion'
            elif 'SEMANTIC_OVERLAP' in response or 'OVERLAP' in response:
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
        
        print(f"🤖 GPT-4.1 classifying {total} pairs...")
        
        for i, (text1, text2) in enumerate(zip(texts1, texts2)):
            pred = self.predict_single(text1, text2)
            predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{total}")
        
        return predictions

# ===== BERT-BASED CLASSIFIERS =====
class SentenceBERTClassifier:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.classifier = None
        
    def extract_features(self, texts1, texts2):
        """Extract sentence embeddings and compute similarity features"""
        print("🔍 Extracting Sentence-BERT features...")
        
        # Get embeddings for both text sets
        embeddings1 = self.model.encode(texts1, convert_to_tensor=True)
        embeddings2 = self.model.encode(texts2, convert_to_tensor=True)
        
        # Compute various similarity features
        features = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            # Cosine similarity
            cos_sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            
            # Element-wise features
            diff = emb1 - emb2
            abs_diff = torch.abs(diff)
            
            feature_vector = [
                cos_sim,
                torch.mean(abs_diff).item(),
                torch.max(abs_diff).item(),
                torch.std(diff).item(),
                torch.norm(emb1).item(),
                torch.norm(emb2).item(),
                torch.dot(emb1, emb2).item()
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, train_texts1, train_texts2, train_labels):
        """Train a simple classifier on extracted features"""
        from sklearn.ensemble import RandomForestClassifier
        
        print(f"🔧 Training Sentence-BERT classifier...")
        
        # Extract features
        X_train = self.extract_features(train_texts1, train_texts2)
        
        # Convert labels to numeric
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_labels)
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        self.classifier.fit(X_train, y_train)
        self.label_encoder = label_encoder
        
        print("✅ Sentence-BERT classifier trained")
    
    def predict(self, texts1, texts2):
        """Make predictions on new data"""
        if self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        X_test = self.extract_features(texts1, texts2)
        y_pred_numeric = self.classifier.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_numeric)
        
        return y_pred.tolist()

class FineTunedBERTClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        print(f"Initializing Fine-tuned BERT model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
    def prepare_model(self, num_labels=3):
        """Prepare the model for fine-tuning"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(DEVICE)
    
    def train(self, train_texts1, train_texts2, train_labels, 
              val_texts1=None, val_texts2=None, val_labels=None):
        """Fine-tune BERT on the dataset"""
        print("🔧 Fine-tuning BERT classifier...")
        
        # Prepare model
        self.prepare_model()
        
        # Convert labels to numeric
        label_encoder = LabelEncoder()
        train_labels_numeric = label_encoder.fit_transform(train_labels)
        self.label_encoder = label_encoder
        
        # Create datasets
        train_dataset = TextPairDataset(
            train_texts1, train_texts2, train_labels_numeric, self.tokenizer
        )
        
        val_dataset = None
        if val_texts1 is not None:
            val_labels_numeric = label_encoder.transform(val_labels)
            val_dataset = TextPairDataset(
                val_texts1, val_texts2, val_labels_numeric, self.tokenizer
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./bert_classifier',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
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
        print("✅ BERT fine-tuning completed")
    
    def predict(self, texts1, texts2):
        """Make predictions using fine-tuned BERT"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        predictions = []
        
        print(f"🤖 Fine-tuned BERT classifying {len(texts1)} pairs...")
        
        with torch.no_grad():
            for i, (text1, text2) in enumerate(zip(texts1, texts2)):
                inputs = self.tokenizer(
                    text1, text2,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
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

# ===== EVALUATION AND COMPARISON =====
def evaluate_classifier(y_true, y_pred, classifier_name):
    """Evaluate classifier performance"""
    print(f"\n📊 Evaluating {classifier_name}")
    print("="*50)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAPPING.keys())))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_MAPPING.keys()))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=list(LABEL_MAPPING.keys()),
                yticklabels=list(LABEL_MAPPING.keys()),
                cmap='Blues')
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{classifier_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()
    
    return accuracy

def compare_classifiers(results_dict, test_labels):
    """Compare performance of all classifiers"""
    print(f"\n🏆 CLASSIFIER COMPARISON")
    print("="*60)
    
    comparison_data = []
    
    for name, predictions in results_dict.items():
        accuracy = accuracy_score(test_labels, predictions)
        comparison_data.append({
            'Classifier': name,
            'Accuracy': accuracy,
            'Correct': sum(1 for t, p in zip(test_labels, predictions) if t == p),
            'Total': len(test_labels)
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_df['Classifier'], comparison_df['Accuracy'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(comparison_df)])
    
    plt.title('Classifier Performance Comparison')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('classifier_comparison.png', dpi=300)
    plt.show()
    
    return comparison_df

# ===== MAIN CLASSIFICATION PIPELINE =====
def main_classification_pipeline(dataset_path):
    """Main pipeline for training and evaluating classifiers"""
    print(f"🚀 Starting Text Pair Classification Pipeline")
    print(f"{'='*60}")
    
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
    
    # Initialize classifiers
    results = {}
    
    # 1. GPT-4.1 Classifier
    print(f"\n{'='*60}")
    print("1. GPT-4.1 CLASSIFIER")
    print(f"{'='*60}")
    
    try:
        azure_client = AzureAIClient()
        gpt_classifier = GPT41Classifier(azure_client)
        gpt_predictions = gpt_classifier.predict(test_texts1, test_texts2)
        results['GPT-4.1'] = gpt_predictions
        
        gpt_accuracy = evaluate_classifier(test_labels, gpt_predictions, 'GPT-4.1')
        
    except Exception as e:
        print(f"❌ GPT-4.1 classifier failed: {e}")
        results['GPT-4.1'] = ['semantic_overlap'] * len(test_labels)  # Fallback
    
    # 2. Sentence-BERT Classifier  
    print(f"\n{'='*60}")
    print("2. SENTENCE-BERT CLASSIFIER")
    print(f"{'='*60}")
    
    try:
        sbert_classifier = SentenceBERTClassifier()
        sbert_classifier.train(train_texts1, train_texts2, train_labels)
        sbert_predictions = sbert_classifier.predict(test_texts1, test_texts2)
        results['Sentence-BERT'] = sbert_predictions
        
        sbert_accuracy = evaluate_classifier(test_labels, sbert_predictions, 'Sentence-BERT')
        
    except Exception as e:
        print(f"❌ Sentence-BERT classifier failed: {e}")
        results['Sentence-BERT'] = ['semantic_overlap'] * len(test_labels)  # Fallback
    
    # 3. Fine-tuned BERT Classifier
    print(f"\n{'='*60}")
    print("3. FINE-TUNED BERT CLASSIFIER")
    print(f"{'='*60}")
    
    try:
        bert_classifier = FineTunedBERTClassifier()
        bert_classifier.train(train_texts1, train_texts2, train_labels)
        bert_predictions = bert_classifier.predict(test_texts1, test_texts2)
        results['Fine-tuned BERT'] = bert_predictions
        
        bert_accuracy = evaluate_classifier(test_labels, bert_predictions, 'Fine-tuned BERT')
        
    except Exception as e:
        print(f"❌ Fine-tuned BERT classifier failed: {e}")
        results['Fine-tuned BERT'] = ['semantic_overlap'] * len(test_labels)  # Fallback
    
    # Compare all classifiers
    comparison_df = compare_classifiers(results, test_labels)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'classification_results_{timestamp}.json'
    
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'dataset_path': dataset_path,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'random_seed': RANDOM_SEED
        },
        'test_data': {
            'texts1': test_texts1,
            'texts2': test_texts2,
            'true_labels': test_labels
        },
        'predictions': results,
        'comparison': comparison_df.to_dict('records')
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results, comparison_df

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset_creation_results_180_pairs_20241213_143022.json"  # Update with your dataset path
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please provide the correct path to your generated dataset")
    else:
        results, comparison = main_classification_pipeline(dataset_path)
        
        print(f"\n🎉 Classification pipeline completed!")
        print(f"Check the generated plots and results file for detailed analysis.")