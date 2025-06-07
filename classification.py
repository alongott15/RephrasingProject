import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

class SCRDatasetFromRephrasing:
    """Dataset class for SCR using rephrasing results from main.py"""
    
    def __init__(self, json_file_path):
        self.data = self.load_and_process_data(json_file_path)
        
        # Define label mappings
        self.label_to_id = {
            'Paraphrasing': 0,
            'A includes B': 1, 
            'B includes A': 2,
            'Mutual Exclusion': 3
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_and_process_data(self, json_file_path):
        """Load and process the rephrasing results JSON file"""
        print(f"Loading data from {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            rephrasing_results = json.load(f)
        
        processed_data = []
        
        for result in rephrasing_results:
            if result.get('status') != 'SUCCESS':
                continue
                
            original_context = result.get('original_context_snippet', '')
            rewritten_context = result.get('rewritten_context', '')
            questions_to_keep = result.get('questions_to_keep', [])
            questions_to_omit = result.get('questions_to_omit', [])
            omission_strategy = result.get('omission_strategy', '')
            
            # Skip if essential data is missing
            if not original_context or not rewritten_context or not questions_to_keep:
                continue
                
            # Create different relationship types based on the rephrasing results
            processed_data.extend(self.create_scr_pairs(
                original_context, rewritten_context, 
                questions_to_keep, questions_to_omit, 
                omission_strategy
            ))
        
        print(f"Created {len(processed_data)} SCR pairs from {len(rephrasing_results)} rephrasing results")
        return processed_data
    
    def create_scr_pairs(self, original_context, rewritten_context, 
                        questions_to_keep, questions_to_omit, omission_strategy):
        """Create SCR pairs from rephrasing results"""
        pairs = []
        
        # 1. PARAPHRASING CASES
        # When no questions were omitted, treat as paraphrases
        if not questions_to_omit or len(questions_to_omit) == 0:
            pairs.append({
                'text_a': original_context,
                'text_b': rewritten_context,
                'label': 'Paraphrasing',
                'source': 'no_omission'
            })
            # Add reverse order for robustness
            pairs.append({
                'text_a': rewritten_context,
                'text_b': original_context,
                'label': 'Paraphrasing',
                'source': 'no_omission_reverse'
            })
        
        # 2. INCLUSION CASES (A includes B)
        # Original context includes rewritten context (information was omitted)
        if questions_to_omit and len(questions_to_omit) > 0:
            pairs.append({
                'text_a': original_context,  # Full text
                'text_b': rewritten_context,  # Reduced text (info omitted)
                'label': 'A includes B',
                'source': f'omission_{omission_strategy}'
            })
        
        # 3. INCLUSION CASES (B includes A)
        # Rewritten context includes original context
        # We can create this by sometimes reversing the positions
        if questions_to_omit and len(questions_to_omit) > 0:
            # Randomly create some B includes A cases for balance
            if random.random() < 0.3:  # 30% chance to create reverse inclusion
                pairs.append({
                    'text_a': rewritten_context,  # Treat as reduced text
                    'text_b': original_context,   # Treat as full text
                    'label': 'B includes A',
                    'source': f'reverse_omission_{omission_strategy}'
                })
        
        # 4. MUTUAL EXCLUSION CASES
        # Create mutual exclusion by pairing with unrelated contexts
        # We'll handle this in a separate method to avoid circular references
        
        return pairs
    
    def add_mutual_exclusion_pairs(self, ratio=0.25):
        """Add mutual exclusion pairs by combining unrelated contexts"""
        print("Adding mutual exclusion pairs...")
        
        # Get all contexts for random pairing
        all_contexts = []
        for item in self.data:
            all_contexts.append(item['text_a'])
            all_contexts.append(item['text_b'])
        
        # Remove duplicates
        unique_contexts = list(set(all_contexts))
        
        # Calculate number of exclusion pairs to add
        current_size = len(self.data)
        target_exclusion_pairs = int(current_size * ratio)
        
        exclusion_pairs = []
        for _ in range(target_exclusion_pairs):
            # Pick two random unrelated contexts
            context_a, context_b = random.sample(unique_contexts, 2)
            
            # Ensure they're sufficiently different (basic check)
            if len(set(context_a.split()) & set(context_b.split())) < len(context_a.split()) * 0.3:
                exclusion_pairs.append({
                    'text_a': context_a,
                    'text_b': context_b,
                    'label': 'Mutual Exclusion',
                    'source': 'random_pairing'
                })
        
        self.data.extend(exclusion_pairs)
        print(f"Added {len(exclusion_pairs)} mutual exclusion pairs")
    
    def balance_dataset(self):
        """Balance the dataset across all four categories"""
        print("Balancing dataset...")
        
        # Count current distribution
        label_counts = {}
        for item in self.data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Current distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        # Find the target size (smallest class size * 4)
        min_count = min(label_counts.values()) if label_counts else 0
        target_per_class = max(min_count, 50)  # At least 50 per class
        
        print(f"Target per class: {target_per_class}")
        
        # Sample from each class to create balanced dataset
        balanced_data = []
        
        for target_label in self.label_to_id.keys():
            class_items = [item for item in self.data if item['label'] == target_label]
            
            if len(class_items) >= target_per_class:
                # Sample down
                sampled_items = random.sample(class_items, target_per_class)
            else:
                # Sample up with replacement
                sampled_items = random.choices(class_items, k=target_per_class)
            
            balanced_data.extend(sampled_items)
        
        self.data = balanced_data
        
        # Print final distribution
        final_counts = {}
        for item in self.data:
            label = item['label']
            final_counts[label] = final_counts.get(label, 0) + 1
        
        print("Final balanced distribution:")
        for label, count in final_counts.items():
            print(f"  {label}: {count}")
        
        print(f"Total dataset size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'text_a': self.data[idx]['text_a'],
            'text_b': self.data[idx]['text_b'],
            'label': self.data[idx]['label']
        }
    
    def get_data_summary(self):
        """Get summary statistics of the dataset"""
        label_counts = {}
        source_counts = {}
        
        for item in self.data:
            label = item['label']
            source = item.get('source', 'unknown')
            
            label_counts[label] = label_counts.get(label, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_size': len(self.data),
            'label_distribution': label_counts,
            'source_distribution': source_counts
        }

class GenerativeSCRClassifier:
    """Generative model classifier for SCR using Azure AI or local models"""
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.client = None
        
        if model_config["type"] == "azure_ai_inference":
            self.client = ChatCompletionsClient(
                endpoint=model_config["azure_ai_endpoint"],
                credential=AzureKeyCredential(model_config["azure_api_key"])
            )
    
    def get_zero_shot_prompt(self, text_a, text_b):
        """Generate zero-shot prompt for SCR classification"""
        prompt = f"""You are a language expert tasked with identifying the semantic relationship between two texts. The possible relationships are:

1. Paraphrasing – both texts express the same information.
2. A includes B – Text A contains all the information in B, plus additional content.
3. B includes A – Text B contains all information in A plus more.
4. Mutual Exclusion – the texts have unrelated or incompatible information.

Text A:
"{text_a[:500]}..."

Text B:
"{text_b[:500]}..."

What is the semantic relationship between Text A and Text B?
Answer with one of: "Paraphrasing", "A includes B", "B includes A", or "Mutual Exclusion"."""
        return prompt
    
    def get_few_shot_prompt(self, text_a, text_b):
        """Generate few-shot prompt for SCR classification"""
        prompt = f"""You are a language expert tasked with identifying the semantic relationship between two texts. 
The possible relationships are:
1. Paraphrasing
2. A includes B
3. B includes A
4. Mutual Exclusion

Example 1:
Text A: "The Eiffel Tower is located in Paris and attracts millions of tourists every year."
Text B: "Many tourists visit the Eiffel Tower in Paris annually."
Answer: A includes B

Example 2:
Text A: "Photosynthesis occurs in plant leaves using sunlight, water, and carbon dioxide."
Text B: "The process of photosynthesis in plants uses water, CO₂, and sunlight in leaves."
Answer: Paraphrasing

Example 3:
Text A: "The collapse of mortgage-backed securities triggered the 2008 financial crisis."
Text B: "The Great Depression was caused by a stock market crash in 1929."
Answer: Mutual Exclusion

Now, determine the relationship in the following example:
Text A:
"{text_a[:500]}..."

Text B:
"{text_b[:500]}..."

Answer:"""
        return prompt
    
    def classify_pair(self, text_a, text_b, few_shot=False):
        """Classify a single text pair"""
        if few_shot:
            prompt = self.get_few_shot_prompt(text_a, text_b)
        else:
            prompt = self.get_zero_shot_prompt(text_a, text_b)
        
        try:
            if self.model_config["type"] == "azure_ai_inference":
                messages = [UserMessage(content=prompt)]
                response = self.client.complete(
                    messages=messages,
                    model=self.model_config["azure_ai_model_name"],
                    max_tokens=50,
                    temperature=0.0
                )
                
                if hasattr(response, 'choices') and response.choices:
                    prediction = response.choices[0].message.content.strip()
                    return self.extract_label(prediction)
                else:
                    return "Mutual Exclusion"  # Default fallback
            else:
                # For local HF models - would need implementation
                return "Mutual Exclusion"  # Placeholder
                
        except Exception as e:
            print(f"Error in classification: {e}")
            return "Mutual Exclusion"  # Default fallback
    
    def extract_label(self, response_text):
        """Extract SCR label from model response"""
        response_lower = response_text.lower()
        
        # Check for exact matches first
        labels = ["Paraphrasing", "A includes B", "B includes A", "Mutual Exclusion"]
        for label in labels:
            if label.lower() in response_lower:
                return label
        
        # Check for partial matches
        if "paraphras" in response_lower:
            return "Paraphrasing"
        elif "a includes b" in response_lower or "text a includes" in response_lower:
            return "A includes B"
        elif "b includes a" in response_lower or "text b includes" in response_lower:
            return "B includes A"
        else:
            return "Mutual Exclusion"
    
    def evaluate(self, dataset, few_shot=False, sample_size=None):
        """Evaluate model on dataset"""
        predictions = []
        true_labels = []
        
        # Sample dataset if too large
        eval_indices = list(range(len(dataset)))
        if sample_size and len(eval_indices) > sample_size:
            eval_indices = random.sample(eval_indices, sample_size)
        
        print(f"Evaluating {'few-shot' if few_shot else 'zero-shot'} classification on {len(eval_indices)} examples...")
        
        for i, idx in enumerate(eval_indices):
            item = dataset[idx]
            pred = self.classify_pair(item['text_a'], item['text_b'], few_shot=few_shot)
            predictions.append(pred)
            true_labels.append(item['label'])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(eval_indices)} examples")
        
        return self.compute_metrics(true_labels, predictions)
    
    def compute_metrics(self, true_labels, predictions):
        """Compute classification metrics"""
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        
        # Compute confusion matrix
        labels = ["Paraphrasing", "A includes B", "B includes A", "Mutual Exclusion"]
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        # Check directional consistency
        consistency = self.check_directional_consistency(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'consistency': consistency,
            'classification_report': classification_report(true_labels, predictions, labels=labels)
        }
    
    def check_directional_consistency(self, true_labels, predictions):
        """Check consistency of directional predictions"""
        # For inclusion relationships, check if model is consistent with directionality
        inclusion_correct = 0
        inclusion_total = 0
        
        for true_label, pred_label in zip(true_labels, predictions):
            if true_label in ['A includes B', 'B includes A']:
                inclusion_total += 1
                if pred_label in ['A includes B', 'B includes A']:
                    inclusion_correct += 1
        
        return inclusion_correct / inclusion_total if inclusion_total > 0 else 0.0

class BERTSCRClassifier:
    """BERT-based classifier for SCR"""
    
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_to_id = {
            'Paraphrasing': 0,
            'A includes B': 1,
            'B includes A': 2, 
            'Mutual Exclusion': 3
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def prepare_dataset(self, dataset):
        """Prepare dataset for BERT training"""
        texts_a = []
        texts_b = []
        labels = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            texts_a.append(item['text_a'])
            texts_b.append(item['text_b'])
            labels.append(self.label_to_id[item['label']])
        
        # Tokenize the text pairs
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        encodings = self.tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return hf_dataset
    
    def train(self, train_dataset, val_dataset, output_dir="./scr_bert_model"):
        """Train BERT classifier"""
        print(f"Training BERT classifier with {self.model_name}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_id)
        )
        
        # Prepare datasets
        train_hf = self.prepare_dataset(train_dataset)
        val_hf = self.prepare_dataset(val_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # Reduced for memory
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None  # Disable wandb logging
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics_for_trainer
        )
        
        # Train
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def compute_metrics_for_trainer(self, eval_pred):
        """Compute metrics for trainer"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1
        }
    
    def evaluate(self, test_dataset):
        """Evaluate trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        print(f"Evaluating on {len(test_dataset)} test samples...")
        
        test_hf = self.prepare_dataset(test_dataset)
        
        # Get predictions
        self.model.eval()
        predictions = []
        true_labels = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        with torch.no_grad():
            for i in range(len(test_hf)):
                inputs = {
                    'input_ids': test_hf[i]['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': test_hf[i]['attention_mask'].unsqueeze(0).to(device)
                }
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(self.id_to_label[pred])
                true_labels.append(test_dataset[i]['label'])
        
        return self.compute_metrics(true_labels, predictions)
    
    def compute_metrics(self, true_labels, predictions):
        """Compute classification metrics"""
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        
        labels = ["Paraphrasing", "A includes B", "B includes A", "Mutual Exclusion"]
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(true_labels, predictions, labels=labels)
        }

def plot_confusion_matrix(cm, labels, title, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run SCR classification experiments using rephrasing results"""
    
    # Configuration for Azure AI models
    AZURE_CONFIG = {
        "type": "azure_ai_inference",
        "azure_ai_endpoint": os.getenv('AZURE_AI_ENDPOINT'),
        "azure_ai_model_name": "gpt-4.1",
        "azure_api_key": os.getenv('AZURE_AI_API_KEY')
    }
    
    # Load dataset from main.py JSON output
    JSON_FILE_PATH = 'final_results_500_texts.json'  # Adjust path as needed
    
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: JSON file {JSON_FILE_PATH} not found.")
        print("Please run main.py first to generate the rephrasing results.")
        return
    
    # Create SCR dataset from rephrasing results
    print("="*60)
    print("LOADING SCR DATASET FROM REPHRASING RESULTS")
    print("="*60)
    
    random.seed(42)  # For reproducibility
    
    scr_dataset = SCRDatasetFromRephrasing(JSON_FILE_PATH)
    
    # Add mutual exclusion pairs and balance dataset
    scr_dataset.add_mutual_exclusion_pairs(ratio=0.25)
    scr_dataset.balance_dataset()
    
    # Print dataset summary
    summary = scr_dataset.get_data_summary()
    print("\nDataset Summary:")
    print(f"Total size: {summary['total_size']}")
    print("Label distribution:")
    for label, count in summary['label_distribution'].items():
        print(f"  {label}: {count}")
    print("Source distribution:")
    for source, count in summary['source_distribution'].items():
        print(f"  {source}: {count}")
    
    # Split dataset
    indices = list(range(len(scr_dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, 
        stratify=[scr_dataset[i]['label'] for i in indices]
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42,
        stratify=[scr_dataset[i]['label'] for i in train_idx]
    )
    
    # Create split datasets
    def create_subset(dataset, indices):
        class SubsetDataset:
            def __init__(self, parent_dataset, indices):
                self.parent = parent_dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.parent[self.indices[idx]]
        
        return SubsetDataset(dataset, indices)
    
    train_dataset = create_subset(scr_dataset, train_idx)
    val_dataset = create_subset(scr_dataset, val_idx)
    test_dataset = create_subset(scr_dataset, test_idx)
    
    print(f"\nDataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    results = {}
    
    # 1. Generative Model Evaluation
    print("\n" + "="*60)
    print("GENERATIVE MODEL EVALUATION")
    print("="*60)
    
    if AZURE_CONFIG["azure_ai_endpoint"] and AZURE_CONFIG["azure_api_key"]:
        generative_classifier = GenerativeSCRClassifier(AZURE_CONFIG)
        
        # Zero-shot evaluation (sample for cost control)
        print("\nEvaluating GPT-4.1 Zero-shot...")
        zs_results = generative_classifier.evaluate(test_dataset, few_shot=False, sample_size=50)
        results['GPT_4.1_ZS'] = zs_results
        
        print(f"Zero-shot Accuracy: {zs_results['accuracy']:.3f}")
        print(f"Zero-shot Macro-F1: {zs_results['macro_f1']:.3f}")
        print(f"Zero-shot Consistency: {zs_results['consistency']:.3f}")
        
        # Few-shot evaluation
        print("\nEvaluating GPT-4.1 Few-shot...")
        fs_results = generative_classifier.evaluate(test_dataset, few_shot=True, sample_size=50)
        results['GPT_4.1_FS'] = fs_results
        
        print(f"Few-shot Accuracy: {fs_results['accuracy']:.3f}")
        print(f"Few-shot Macro-F1: {fs_results['macro_f1']:.3f}")
        print(f"Few-shot Consistency: {fs_results['consistency']:.3f}")
        
        # Plot confusion matrices
        labels = ["Paraphrasing", "A includes B", "B includes A", "Mutual Exclusion"]
        plot_confusion_matrix(zs_results['confusion_matrix'], labels, 
                            "GPT-4.1 Zero-shot Confusion Matrix")
        plot_confusion_matrix(fs_results['confusion_matrix'], labels,
                            "GPT-4.1 Few-shot Confusion Matrix")
    else:
        print("Azure AI configuration not found. Skipping generative model evaluation.")
    
    # 2. BERT-based Model Training and Evaluation
    print("\n" + "="*60)
    print("BERT-BASED MODEL EVALUATION")
    print("="*60)
    
    # BERT-base
    print("\nTraining BERT-base classifier...")
    bert_classifier = BERTSCRClassifier("bert-base-uncased")
    bert_classifier.train(train_dataset, val_dataset)
    bert_results = bert_classifier.evaluate(test_dataset)
    results['BERT_base'] = bert_results
    
    print(f"BERT-base Accuracy: {bert_results['accuracy']:.3f}")
    print(f"BERT-base Macro-F1: {bert_results['macro_f1']:.3f}")
    
    # RoBERTa-base
    print("\nTraining RoBERTa-base classifier...")
    roberta_classifier = BERTSCRClassifier("roberta-base")
    roberta_classifier.train(train_dataset, val_dataset, "./scr_roberta_model")
    roberta_results = roberta_classifier.evaluate(test_dataset)
    results['RoBERTa_base'] = roberta_results
    
    print(f"RoBERTa-base Accuracy: {roberta_results['accuracy']:.3f}")
    print(f"RoBERTa-base Macro-F1: {roberta_results['macro_f1']:.3f}")
    
    # Plot BERT confusion matrices
    labels = ["Paraphrasing", "A includes B", "B includes A", "Mutual Exclusion"]
    plot_confusion_matrix(bert_results['confusion_matrix'], labels,
                        "BERT-base Confusion Matrix")
    plot_confusion_matrix(roberta_results['confusion_matrix'], labels,
                        "RoBERTa-base Confusion Matrix")
    
    # 3. Results Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.3f}",
            'Macro-F1': f"{result['macro_f1']:.3f}",
            'Consistency': f"{result.get('consistency', 0.0):.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    with open('scr_classification_results_from_rephrasing.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model, result in results.items():
            json_results[model] = {
                'accuracy': float(result['accuracy']),
                'macro_f1': float(result['macro_f1']),
                'consistency': float(result.get('consistency', 0.0)),
                'confusion_matrix': result['confusion_matrix'].tolist(),
                'classification_report': result['classification_report']
            }
        json.dump(json_results, f, indent=2)
    
    # Save dataset summary
    with open('scr_dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to scr_classification_results_from_rephrasing.json")
    print(f"Dataset summary saved to scr_dataset_summary.json")
    
    # Additional analysis: Show examples from each category
    print("\n" + "="*60)
    print("SAMPLE EXAMPLES FROM EACH CATEGORY")
    print("="*60)
    
    examples_per_category = {}
    for i in range(min(len(test_dataset), 100)):  # Check first 100 test examples
        item = test_dataset[i]
        label = item['label']
        if label not in examples_per_category:
            examples_per_category[label] = []
        if len(examples_per_category[label]) < 2:  # Store 2 examples per category
            examples_per_category[label].append({
                'text_a': item['text_a'][:200] + "..." if len(item['text_a']) > 200 else item['text_a'],
                'text_b': item['text_b'][:200] + "..." if len(item['text_b']) > 200 else item['text_b']
            })
    
    for label, examples in examples_per_category.items():
        print(f"\n{label.upper()} EXAMPLES:")
        for i, example in enumerate(examples, 1):
            print(f"  Example {i}:")
            print(f"    Text A: {example['text_a']}")
            print(f"    Text B: {example['text_b']}")
            print()
    
    print("Classification evaluation complete!")

if __name__ == "__main__":
    main()