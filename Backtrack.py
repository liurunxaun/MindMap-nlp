import utils.llm_api as llm


def extract(question):
    response = llm.spark_4_0(question)
    return response


def main(question):
    answer = extract(question)
    return answer


if __name__ == "__main__":
    question = "machine translation 领域有哪些数据集？"
    answer = main(question)
    print(answer)