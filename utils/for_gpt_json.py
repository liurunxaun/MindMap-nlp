import csv
import json

input_csv_file = "/media/MindMap-main/output/chatdoctor5k/output_gpt4_comparation_disease.csv"  # 输入 CSV 文件路径
output_json_file = "/media/MindMap-main/data/chatdoctor5k/chatdoctor5k_for_gpt3.5_and_gpt4.0.json"  # 输出 JSON 文件路径

with open(output_json_file, "w", encoding="utf-8") as f_out:
    with open(input_csv_file, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)  # 使用 DictReader 读取 CSV 文件

        id_counter = 1  # 初始化 ID 计数器
        for row in reader:
            question = row["Question"]  # 提取 Question 列

            # 生成格式化的输出
            output_data = {
                "query": "You are a professional doctor. Please respond to the patient's following question: " + question,
                "id": {"id": id_counter}
            }

            # 写入文件
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            id_counter += 1  # 增加 ID 计数器

print(f"Output written to {output_json_file}")
