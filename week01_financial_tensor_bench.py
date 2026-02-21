import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

def week1_mission():
    # 1. M4 Max GPU(MPS) 가속 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Device: {device} (M4 Max 128GB RAM ready)")

    # 2. 금융 데이터셋 로드 (SEC 10-K 기반)
    print("📥 Loading Sujet Financial RAG Dataset...")
    try:
        dataset = load_dataset("sujet-ai/Sujet-Financial-RAG-EN-Dataset", split="train")
        sample_context = dataset[0]['context'] 
        print(f" Data Sample Loaded (Length: {len(sample_context)})")
    except Exception as e:
        print(f" 데이터 로드 실패: {e}")
        return

    # 3. 토크나이저 및 모델 로드
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # 4. 성능 측정: 텍스트를 숫자로 변환 (Embedding)
    inputs = tokenizer(sample_context[:1000], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    print("\n M4 Max MPS Embedding Speed Test...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    end_time = time.time()
    
    # 5. 결과 출력
    last_hidden_state = outputs.last_hidden_state
    print(f" Tensor Shape: {last_hidden_state.shape}") # (Batch, Seq, Hidden)
    print(f" Processing Time: {(end_time - start_time)*1000:.2f} ms")
    print("\n 1주차 미션 성공! 데이터가 텐서로 완벽히 변환되었습니다.")

if __name__ == "__main__":
    week1_mission()