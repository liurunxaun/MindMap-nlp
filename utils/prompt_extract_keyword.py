"""
@Time: 2024/9/29 15:43
@Author: yanzx
@Desc: 
"""
import time
import itertools
from sparkai.llm.llm import ChatSparkLLM
import re
import csv
import json
from sparkai.core.messages import ChatMessage
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from sparkai.errors import SparkAIConnectionError

# Configuration for the Spark AI model
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
SPARKAI_APP_ID = '4f6b5020'
SPARKAI_API_SECRET = 'MGI3MGQ5MjJhMjAzMWY2ODFkNDMxYjI1'
SPARKAI_API_KEY = 'f299adff306173af4588cdd14dca6788'
SPARKAI_DOMAIN = '4.0Ultra'
# 调整正则表达式以匹配“提取的实体”后的内容
# 第一个正则表达式
re1 = r'The extracted entities are (.*)'
# 第二个正则表达式
re2 = r'The extracted entities are\s*### Output:\s*(.*)'
spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)


def prompt_extract_keyword(input_text):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I've been feeling a sharp pain in my lower back after lifting heavy objects. What could be causing this pain and what should I do?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I've been feeling a sharp pain in my lower back after lifting heavy objects. What could be causing this pain and what should I do?<SEP>The extracted entities are Lower back pain, Heavy lifting, Possible injury<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Based on your description, it sounds like you may have strained a muscle or perhaps affected a disc. Rest, ice, and gentle stretching may help, but it's important to consult a healthcare professional if the pain persists.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Based on your description, it sounds like you may have strained a muscle or perhaps affected a disc. Rest, ice, and gentle stretching may help, but it's important to consult a healthcare professional if the pain persists.<SEP>The extracted entities are Muscle strain, Disc issue, Rest, Ice, Stretching<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    # Format the message content
    message_content = template.format(input=input_text)
    messages = [ChatMessage(role="user", content=message_content)]
    handler = ChunkPrintHandler()

    try:
        time.sleep(5)
        response = spark.generate([messages], callbacks=[handler])
        print(f"response:{response.generations[0][0].text.strip()}")
        # Check response content
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            question_kg = re.findall(re1, response.generations[0][0].text.strip())
            time.sleep(10)
            if question_kg == []:
                question_kg = re.findall(re2, response.generations[0][0].text.strip())
            print(f"question_kg[0]:{question_kg[0]}")
            return question_kg[0] if question_kg else ""  # Return first match or empty string
        else:
            time.sleep(10)
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "feifa"  # Return error flag for illegal request
        else:
            print(f"发生连接错误: {str(e)}")

    except Exception as e:
        print(f"发生错误: {str(e)}")

    return []  # Default return an empty list


input_csv_file = "/media/MindMap-main/output/chatdoctor5k/output_gpt4_comparation_disease.csv"  # Input CSV file path
output_json_file = "/media/MindMap-main/data/chatdoctor5k/test1.json"  # Output JSON file path


with open(output_json_file, "a+", encoding="utf-8") as f_out:
    with open(input_csv_file, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)  # Use DictReader to read CSV file
        for row in itertools.islice(reader, 268, None):
            question = row["Question"]  # Extract Question column
            print(f"question:{question}")
            label = row["Label"]  # Extract Label column

            # Get the keyword extraction
            input_kg = prompt_extract_keyword(question)
            if input_kg == "feifa":
                output_data = {
                    "input": question,
                    "input_KG": "",
                    "output": label,  # Replace with actual model output if needed
                    "output_KG": ""  # Assuming label represents the output KG
                }
            # Construct the output structure
            else:
                output_data = {
                    "input": question,
                    "input_KG": input_kg,
                    "output": label,  # Replace with actual model output if needed
                    "output_KG": ""  # Assuming label represents the output KG
                }
            # Write to JSON file
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            print(f"写入成功")

print(f"Output written to {output_json_file}")
