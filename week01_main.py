import time
import torch
from transformers import AutoTokenizer
from src.accelerator import get_device
from src.data_loader import load_financial_dataset

def run_benchmark():
    """
    Executes a performance benchmark for financial data tensor conversion.
    """
    # 1. Setup Device
    device = get_device()
    
    # 2. Load Dataset (Sujet-ai Financial RAG dataset)
    dataset_name = "sujet-ai/Sujet-Financial-RAG-EN-Dataset"
    dataset = load_financial_dataset(dataset_name)
    
    # 3. Initialize Tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 4. Prepare Sample Data (First 100 rows for benchmarking)
    sample_texts = dataset['train']['context'][:100]
    
    # 5. Measure Latency for Tensor Conversion
    start_time = time.perf_counter()
    
    tokens = tokenizer(
        sample_texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # 6. Print Results
    print("-" * 30)
    print(f"Benchmark Results")
    print(f"Target Hardware: {device}")
    print(f"Processed Rows: {len(sample_texts)}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark()