import os

import numpy as np
from openai import OpenAI


def cosine_similarity(vector_a, vector_b):
    """두 벡터가 얼마나 비슷한 방향을 가리키는지 계산한다."""
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    # 내적을 두 벡터 길이의 곱으로 나누면 코사인 유사도가 된다.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    # 키 값은 출력하거나 코드에 작성하지 않는다.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정해주세요.")

    # OpenAI()가 OPENAI_API_KEY 환경변수를 자동으로 읽는다.
    client = OpenAI()

    question = "남은 휴가가 며칠인지 어디서 확인해요?"

    documents = [
        "잔여 연차 일수는 인사시스템에서 조회할 수 있습니다.",
        "휴가 신청은 인사시스템에서 상신하고 팀장 승인을 받아야 합니다.",
        "급여명세서는 매월 급여 지급일에 조회할 수 있습니다.",
        "금융 거래 배치 작업은 매일 새벽에 실행됩니다.",
    ]

    # 질문과 비교할 문서들을 같은 임베딩 모델로 변환한다.
    texts = [question] + documents

    # 여러 문장을 한 번의 API 요청으로 보낸다.
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        encoding_format="float",
    )

    # 응답의 index 순서에 맞춰 각 문장의 벡터를 꺼낸다.
    embeddings = [
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]

    # 첫 번째 벡터는 질문, 나머지 벡터는 문서에 해당한다.
    question_vector = embeddings[0]
    document_vectors = embeddings[1:]

    print("질문:", question)
    print("질문 벡터의 차원 수:", len(question_vector))
    print("질문 벡터의 앞 5개 숫자:", question_vector[:5])

    # 자기 자신과 비교하면 약 1이 나오는지 확인한다.
    self_score = cosine_similarity(question_vector, question_vector)
    print(f"질문과 자기 자신의 유사도: {self_score:.4f}")

    results = []

    for document, vector in zip(documents, document_vectors):
        score = cosine_similarity(question_vector, vector)
        results.append((score, document))

    # 유사도가 높은 문서부터 정렬한다.
    results.sort(key=lambda item: item[0], reverse=True)

    print("\n[질문과 문서의 코사인 유사도]")
    for rank, (score, document) in enumerate(results, start=1):
        print(f"{rank}위 | {score:.4f} | {document}")


if __name__ == "__main__":
    main()