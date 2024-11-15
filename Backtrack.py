import utils.llm_api as llm


def extract(question):

    response = llm.spark_4_0(question)
    return response


def main(question):
    # 1. 从问题中提取条件实体、目的实体、实体类型
    condition_entity, condition_label, aim_entity, condition_label = extract(question)


    answer = ""
    return answer


if __name__ == "__main__":
    question = "machine translation 领域有哪些数据集？"
    answer = main(question)
    print(answer)