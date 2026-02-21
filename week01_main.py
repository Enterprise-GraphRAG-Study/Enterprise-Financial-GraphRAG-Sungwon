import time
from transformers import AutoTokenizer

from src.accelerator import get_device
from src.data_loader import load_financial_dataset


def run_benchmark() -> None:
    """Executes a performance benchmark for financial data tensor conversion."""
    device = get_device()

    dataset_name = "sujet-ai/Sujet-Financial-RAG-EN-Dataset"
    dataset = load_financial_dataset(dataset_name)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample_texts = dataset["train"]["context"][:100]

    start_time = time.perf_counter()

    tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    print("-" * 30)
    print("Benchmark Results")
    print(f"Target Hardware: {device}")
    print(f"Processed Rows: {len(sample_texts)}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("-" * 30)


if __name__ == "__main__":
    run_benchmark()
