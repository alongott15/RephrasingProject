# Semantic Coverage Relationship Modeling

This project implements a comprehensive framework for studying **Semantic Coverage Relationships (SCR)** between texts through question answerability analysis. The implementation follows a systematic methodology that creates synthetic datasets to train and evaluate models for classifying semantic relationships between documents.

## Project Overview

The core methodology operationalizes semantic relationships by analyzing which questions each text can answer. This project implements a complete pipeline that:

1. **`new_main.py` - Dataset Creation Pipeline:** Creates a synthetic dataset by systematically generating text variants and labeled pairs following a specific 5-step methodology
2. **`new_classification.py` - SCR Classification:** Trains and evaluates multiple model architectures to classify semantic relationships using the generated dataset

## Methodology (Based on Hebrew Document Requirements)

The framework follows a rigorous 5-step process to create exactly 36 labeled pairs from each successful paraphrase pair:

### Step 1: Text Filtering with QA Accuracy
- Filter texts where the LLM answers **ALL questions correctly** (100% accuracy requirement)
- Uses JUDGE system (binary evaluation) to verify answer correctness
- Note: No comparison to SQUAD Ground Truth as specified
- Requires minimum 5 answerable questions per context
- Skips unanswerable questions marked in the dataset

### Step 2: Paraphrase Creation and Filtering
- Generate paraphrases that preserve all information for answering the same questions
- Filter using METEOR similarity threshold to ensure sufficient lexical differences
- Verify paraphrases maintain 100% QA accuracy on all original questions
- Enhanced prompting ensures paraphrases can answer specific question sets

### Step 3: Synchronized Text Variant Generation
- **Step 3a:** Create exactly 6 variants per text (original + versions removing 1-5 questions)
- **Step 3b:** Filter using JUDGE to verify variants answer expected questions correctly
- **Step 3c:** Generate synchronized variants for both original and paraphrase texts using identical removal patterns

### Step 4: Systematic Labeled Pair Creation
From each paraphrase pair, create exactly 36 labeled pairs with specific distribution:
- **6 Equivalence pairs** (שקילות): Texts answering exactly the same questions
- **10 Inclusion pairs** (הכלה): One text answers all questions of the other plus additional ones  
- **20 Semantic Overlap pairs** (חפיפה סימנטית): Texts with partial semantic overlap but no complete inclusion

### Step 5: JUDGE-Based Evaluation
- Evaluate pairs using JUDGE system that compares answer equivalence (binary: same information or not)
- Score based on common questions both texts can answer
- Maximum score of 5 for pairs sharing 5 questions

## Three-Class Semantic Coverage System

The project implements a Hebrew document-specified 3-class system:

### 1. **Equivalence (שקילות)**
Both texts can answer exactly the same set of questions, representing paraphrasing relationships where information content is preserved but expression differs.

### 2. **Inclusion (הכלה)** 
One text contains all information of another plus additional details. If Text A can answer all questions that Text B can answer (and more), then A includes B.

### 3. **Semantic Overlap (חפיפה סימנטית)**
Texts have partial semantic overlap but neither fully includes the other. They share some answerable questions but each has unique information.

## Technical Implementation

### Dataset Creation (`new_main.py`)

**Supported Models:**
- **Azure AI Integration:** GPT-4.1 via Azure AI Inference (primary)
- **HuggingFace Models:** Llama-3.2-3B-Instruct (fallback)
- **JUDGE System:** Azure GPT-4.1 for answer evaluation

**Key Features:**
- Automatic preprocessing from SQuAD v2.0 to simplified format
- Custom dataset creation with configurable size (e.g., 500 contexts)
- METEOR-based lexical similarity filtering (threshold: 0.6)
- Synchronized variant generation ensuring proper pairing
- Comprehensive evaluation metrics and validation

**Configuration:**
```python
# Core requirements from Hebrew document
MIN_QUESTIONS_PER_CONTEXT = 5      # Minimum answerable questions
MIN_QUESTIONS_FOR_VARIANTS = 5     # Need at least 5 for variant creation  
QA_ACCURACY_THRESHOLD = 1.0        # Require 100% accuracy
METEOR_THRESHOLD = 0.6             # METEOR similarity threshold
SKIP_UNANSWERABLE = True           # Skip unanswerable questions
```

**Expected Output:**
- JSON file with metadata tracking paraphrase sources and pair distributions
- Exactly 36 pairs per successful paraphrase pair
- Comprehensive metrics for each step of the process

### Classification (`new_classification.py`)

**Model Architectures Tested:**

1. **Discriminative Models (Fine-tuned):**
   - BERT-base-uncased
   - RoBERTa-base  
   - Longformer-base-4096

2. **Generative Models (Prompted):**
   - GPT-4.1 Zero-Shot
   - GPT-4.1 Few-Shot

**Evaluation Metrics:**
- Accuracy and Macro-F1 scores
- Confusion matrices with Hebrew labels
- Class distribution analysis comparing to expected distributions
- Cross-model performance comparison

**Expected Distribution Analysis:**
The system analyzes how well models match the expected Hebrew document distribution:
- 16.7% Equivalence (6/36 per paraphrase source)
- 27.8% Inclusion (10/36 per paraphrase source)  
- 55.6% Semantic Overlap (20/36 per paraphrase source)

## Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

Create `.env` file with Azure credentials:
```
AZURE_AI_API_KEY=your_api_key_here
AZURE_AI_ENDPOINT=your_endpoint_here
```

### 2. Data Preparation
Place SQuAD v2.0 data in `./data/`:
```
./data/train-v2.0.json
```

### 3. Dataset Creation
```bash
python new_main.py
```

**Configuration Options:**
- Set `CREATE_CUSTOM_DATASET = True` to generate custom dataset from SQuAD
- Set `CUSTOM_DATASET_SIZE = 500` for dataset size
- Adjust `MAX_TEXTS_TO_PROCESS` for processing limits

**Expected Output:**
```
dataset_creation_results_XXXX_pairs_TIMESTAMP.json
```

### 4. Model Training and Evaluation
```bash
python new_classification.py
```

Update the dataset path in the script:
```python
dataset_path = "dataset_creation_results_XXXX_pairs_TIMESTAMP.json"
```

## Results and Performance

### Dataset Creation Performance
The pipeline successfully creates structured datasets following the Hebrew document methodology:

- **Step 1-2:** Filters texts and creates paraphrase pairs using METEOR similarity
- **Step 3-4:** Generates exactly 36 labeled pairs per paraphrase pair
- **Step 5:** Validates using JUDGE system without Ground Truth comparison

**Key Success Metrics:**
- Number of successful paraphrase pairs from Step 2
- Achievement of exactly 36 pairs per paraphrase pair
- Distribution adherence: 6 equivalence + 10 inclusion + 20 semantic overlap
- 100% QA accuracy maintenance throughout the process

### Classification Performance
The framework benchmarks multiple model architectures on the 3-class SCR task:

**Experimental Results:**

| Classifier | Accuracy | Macro-F1 | Status |
|------------|----------|----------|--------|
| **RoBERTa-base** | **0.614** | **0.446** | ✅ |
| **DistilBERT** | **0.606** | **0.441** | ✅ |
| **Logistic Regression** | **0.604** | **0.478** | ✅ |
| **Random Forest** | **0.591** | **0.529** | ✅ |
| **Longformer-base** | 0.555 | 0.238 | ✅ |
| **GPT Zero-Shot** | 0.339 | 0.341 | ✅ |
| **GPT Few-Shot** | 0.313 | 0.307 | ✅ |

**Key Performance Insights:**

- **Best Overall Performance:** RoBERTa-base achieved the highest accuracy (61.4%) and strong macro-F1 (44.6%)
- **Discriminative vs. Generative:** Fine-tuned discriminative models significantly outperformed prompted generative models
- **Traditional ML Success:** Logistic Regression achieved competitive accuracy (60.4%) with the highest macro-F1 (47.8%)
- **Class Balance:** Random Forest showed excellent macro-F1 (52.9%), indicating better handling of class imbalance
- **GPT Limitations:** Both zero-shot and few-shot GPT approaches struggled with the task (≤34% accuracy)

**Analysis:**
- The 3-class Hebrew document system (Equivalence/Inclusion/Semantic Overlap) benefits from discriminative fine-tuning
- Traditional machine learning approaches remain competitive for this structured classification task
- Longer context models (Longformer) showed mixed results, suggesting task complexity beyond sequence length
- Prompted LLMs may require task-specific fine-tuning for optimal SCR classification performance

**Evaluation Framework:**
- Confusion matrices with Hebrew semantic relationship labels
- Distribution analysis comparing actual vs. expected ratios (16.7%/27.8%/55.6%)
- Cross-model performance on the systematic 3-class system

## File Structure

```
project/
├── new_main.py                 # Main dataset creation pipeline
├── new_classification.py       # Classification training and evaluation
├── data/
│   ├── train-v2.0.json        # Original SQuAD v2.0 dataset
│   └── squad_500.json         # Custom preprocessed dataset (auto-generated)
├── .env                       # Azure AI credentials
├── requirements.txt           # Package dependencies
└── README.md                  # This file
```

## Key Features

### Hebrew Document Compliance
- Implements exact methodology from Hebrew document specifications
- Creates systematic 36-pair distributions per paraphrase source
- Uses JUDGE system without Ground Truth comparison
- Supports Hebrew semantic relationship labels in outputs

### Robust Technical Implementation
- Azure AI integration with HuggingFace fallback
- Comprehensive error handling and validation
- Detailed logging and progress tracking
- Configurable parameters for research flexibility

### Research-Grade Evaluation
- Multiple model architecture comparison
- Statistical analysis of class distributions
- Reproducible results with fixed random seeds
- Comprehensive metadata tracking

## Dependencies

Core packages required:
```
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
azure-ai-inference>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
nltk>=3.7
python-dotenv>=0.19.0
```

## Research Applications

This framework enables research in:
- **Multi-document Summarization:** Understanding information overlap and inclusion relationships
- **Information Retrieval:** Semantic relationship classification for better search results  
- **Fact-checking:** Analyzing semantic coverage between claims and evidence
- **Educational Technology:** Assessing question answerability across different text presentations
- **Knowledge Base Construction:** Identifying semantic relationships for knowledge graph creation

## Citation

If you use this framework in your research, please cite the methodology paper and acknowledge the Hebrew document specifications that guided the implementation.

## Contact and Support

For questions about the implementation or methodology, please refer to the detailed documentation in the source code files and the original Hebrew document specifications.
