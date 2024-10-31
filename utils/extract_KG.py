import time
import itertools
import re
import csv
import json

input_csv_file = "/media/MindMap-main/output/chatdoctor5k/output_gpt4_comparation_disease.csv"  # Input CSV file path
output_json_file = "/media/MindMap-main/data/chatdoctor5k/test.json"  # Output JSON file path
input_KG_all = "/media/MindMap-main/data/chatdoctor5k/NER_chatgpt.json"  # KG data file path

# 函数：从 qustion_output 中提取实体
def extract_entities(qustion_output):
    if '<SEP>' in qustion_output and '<EOS>' in qustion_output:
        start = qustion_output.find('<SEP>') + len('<SEP>')
        end = qustion_output.find('<EOS>')
        extracted_entities = qustion_output[start:end].strip()

        # 提取实体并去掉"The extracted entities are"部分
        if "The extracted entities are" in extracted_entities:
            extracted_entities = extracted_entities.replace("The extracted entities are", "").strip()

        # 将实体按逗号分隔并去除空格
        entities_list = [entity.strip() for entity in extracted_entities.split(",")]
        return ", ".join(entities_list)  # 返回实体字符串
    return ""

# 读取 input_KG_all 文件，解析成字典
# 逐行读取 input_KG_all 文件
kg_data = []
with open(input_KG_all, "r", encoding="utf-8") as f_kg:
    for line in f_kg:
        kg_data.append(json.loads(line.strip()))  # 逐行解析 JSON，并将其加入 kg_data 列表


# 读取 CSV 并写入到 JSON 文件
with open(output_json_file, "a+", encoding="utf-8") as f_out:
    with open(input_csv_file, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)  # Use DictReader to read CSV file
        for row in reader:
            question = row["Question"]  # Extract Question column
            print(f"Processing question: {question}")
            label = row["Label"]  # Extract Label column

            # 在 input_KG_all 中找到对应的 qustion_output
            input_kg = ""
            for item in kg_data:
                if question in item.get("qustion_output", ""):
                    x = item["qustion_output"]
                    print(f"=======x=======\n{x}")
                    input_kg = extract_entities(item["qustion_output"])  # 提取实体
                    break  # 找到后停止搜索

            # 如果没有找到匹配的实体，则将 input_kg 设为 "feifa"
            if not input_kg:
                input_kg = "feifa"

            # 构造输出数据结构
            output_data = {
                "input": question,
                "input_KG": input_kg,
                "output": label,  # Replace with actual model output if needed
                "output_KG": ""  # Assuming label represents the output KG
            }

            # 写入到 JSON 文件
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            print(f"Successfully wrote data for question: {question}")