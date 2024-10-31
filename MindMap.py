from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from sparkai.errors import SparkAIConnectionError
import time
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import openai
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep


def chat_35(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ])
    return completion.choices[0].message.content


def chat_4(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ])
    return completion.choices[0].message.content


def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    exist_entity = {}
    # print(f"start_entity_name, end_entity_name: {start_entity_name, end_entity_name}")
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)

            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_", " ")

                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_", " ")
                    path_str += "->" + relations[i] + "->"

            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}

        if len(paths) > 5:
            paths = sorted(paths, key=len)[:5]

        return paths, exist_entity


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def get_entity_neighbors(entity_name: str, disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]

        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]

        if "disease" in rel_type.replace("_", " "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_", " "), rel_type.replace("_", " "),
                                  ','.join([x.replace("_", " ") for x in neighbors])
                                  ])

    return neighbor_list, disease


def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph paths. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    # 构建要发送的消息
    message_content = template.format(Path=path_input)
    messages = [ChatMessage(role="user", content=message_content)]

    handler = ChunkPrintHandler()
    # print(f"response正常：{response}")
    time.sleep(10)
    try:
        response = spark.generate([messages], callbacks=[handler])
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "feifa"  # 返回空列表
        else:
            print(f"发生其他错误: {str(e)}")

    return []  # 默认返回空列表



def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    # 构建要发送的消息
    message_content = template.format(neighbor=neighbor)
    messages = [ChatMessage(role="user", content=message_content)]
    handler = ChunkPrintHandler()
    time.sleep(10)
    try:
        response = spark.generate([messages], callbacks=[handler])
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "feifa"  # 返回空列表
        else:
            print(f"发生其他错误: {str(e)}")

    return []  # 默认返回空列表


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def is_unable_to_answer(response):
    analysis = openai.Completion.create(
        engine="text-davinci-002",
        prompt=response,
        max_tokens=1,
        temperature=0.0,
        n=1,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
    if not score.isdigit():
        return True
    threshold = 0.6
    if float(score) > threshold:
        return False
    else:
        return True


def autowrap_text(text, font, max_width):
    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines


def final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor):
    template = """
       You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.
       \nPatient input: {input_text}\n
       You have some medical knowledge information in the following:\n
       ### {response_of_KG_list_path}\n
       ### {response_of_KG_neighbor}\n
       What disease does the patient have? What tests should the patient take to confirm the diagnosis? What recommended medications can cure the disease? Think step by step.\n
       Output1: The answer includes disease and tests and recommended medications.\n
       Output2: Show me the inference process as a string about extracting what knowledge from which Path-based Evidence or Neighbor-based Evidence, and in the end infer what result. Transport the inference process into the following format:\n
       Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...).\n
       Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n
       There is a sample:\n
       Output 1:
       Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.\n
       Output 2:
       Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').\n
       Output 3:
       Patient(Path-based Evidence 1)
       └── has been experiencing(Path-based Evidence 1)
           └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
               └── could be caused by(Path-based Evidence 2)
                   └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                       ├── requires(Neighbor-based Evidence 1)
                       │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                       │       └── may include(Neighbor-based Evidence 2)
                       │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                       ├── can be treated with(Path-based Evidence 3)
                       │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                       └── should be accompanied by(Neighbor-based Evidence 3)
                           └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
       """

    # 使用模板格式化
    message_content = template.format(
        input_text=input_text,
        response_of_KG_list_path=response_of_KG_list_path,
        response_of_KG_neighbor=response_of_KG_neighbor
    )
    messages = [ChatMessage(role="user", content=message_content)]
    handler = ChunkPrintHandler()
    time.sleep(20)  # 等待1秒
    try:
        response = spark.generate([messages], callbacks=[handler])

        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
            return ""  # 返回空列表

    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return 'feifa'  # 返回空列表
        else:
            print(f"发生连接错误: {str(e)}")
            return ""  # 返回空列表

    return ""  # 默认返回空列表


def prompt_document(question, instruction):
    # 构建用户输入的消息内容
    prompt = f"""
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.

    Patient input:
    {question}

    You have some medical knowledge information in the following:
    {instruction}

    What disease does the patient have? What tests should the patient take to confirm the diagnosis? What recommended medications can cure the disease?
    """

    # 发送请求到星火大模型
    messages = [
        ChatMessage(role="system", content="You are an expert in medical diagnosis."),
        ChatMessage(role="user", content=prompt)
    ]

    handler = ChunkPrintHandler()
    time.sleep(10)
    try:
        response = spark.generate([messages], callbacks=[handler])
        # print(response)
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "feifa"  # 返回非法标识
        else:
            print(f"发生连接错误: {str(e)}")

    return []  # 默认返回空列表


if __name__ == "__main__":
    # 星火认知大模型的配置
    SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
    SPARKAI_APP_ID = 'c651a376'
    SPARKAI_API_SECRET = 'Y2RlYzRlNWU3ZWVhYWM3NTU5N2Q4NjIy'
    SPARKAI_API_KEY = 'cde9573970818d4d8e08e99598569174'
    SPARKAI_DOMAIN = 'general'

    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )

    # 1. build neo4j knowledge graph datasets
    uri = "bolt://3.222.113.125:7687"
    username = "neo4j"
    password = "speed-steeple-staff"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    ##############################build KG

    # session.run("MATCH (n) DETACH DELETE n")  # clean all
    print(f"连接成功")
    # read triples
    # df = pd.read_csv('./data/chatdoctor5k/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
    #
    # print(f"正在读train")
    # for index, row in df.iloc[:4000].iterrows():
    #     head_name = row['head']
    #     tail_name = row['tail']
    #     relation_name = row['relation']
    #
    #     query = (
    #             "MERGE (h:Entity { name: $head_name }) "
    #             "MERGE (t:Entity { name: $tail_name }) "
    #             "MERGE (h)-[r:`" + relation_name + "`]->(t)"
    #     )
    #     session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
    print("读取完成，运行")
    # # 2. OpenAI API based keyword extraction and match entities

    # OPENAI_API_KEY = YOUR_OPENAI_KEY
    # chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    # with open('/media/MindMap-main/new_data1_test_output_all.csv', 'a+', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(
    #         ['Question', 'Label', 'MindMap', "MindMap_part_2", "TOT", 'GPT3.5', 'BM25_retrieval', 'Embedding_retrieval',
    #          'KG_retrieval', 'GPT4'])

    with open('./data/chatdoctor5k/entity_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open('/media/MindMap-main/data/chatdoctor5k/keyword_embeddings.pkl', 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)

    with open("/media/MindMap-main/data/chatdoctor5k/test.json", "r") as f:
        for line in itertools.islice(f, 633, None):
            data = json.loads(line)  # 逐行读取并解析JSON
            input_text = data.get("input", "")

            if input_text == [] or input_text == "feifa":
                continue
            print('Question:\n', input_text)

            output_text = data.get("output", "")
            print(f"\noutput_text:{output_text}")

            question_kg = data.get("input_KG", "")
            print(f"question_kg:{question_kg}")
            if len(question_kg) == 0:
                print("<Warning> no entities found")
                # continue
            # question_kg = question_kg[0].replace("<END>", "").replace("<EOS>", "")
            # question_kg = question_kg.replace("\n", "")
            question_kg = question_kg.split(", ")
            print("question_kg", question_kg)

            # answer_kg = data.get("output_KG", "")
            # if len(answer_kg) == 0:
            #     print("<Warning> no entities found")
            #     continue
            # answer_kg = answer_kg[0].replace("<END>", "").replace("<EOS>", "")
            # answer_kg = answer_kg.replace("\n", "")
            # answer_kg = answer_kg.split(", ")
            # print(f"answer_kg:{answer_kg}")

            # 函数计算所有实体嵌入与当前关键词嵌入之间的余弦相似度。
            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])

            for kg_entity in question_kg:
                try:
                    keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                    kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])
                except ValueError:
                    # 如果 kg_entity 不在 keywords 列表中，跳过该循环
                    continue
                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()

                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ", "_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ", "_"))
            print('match_kg', match_kg, "\n")

            # # 4. neo4j knowledge graph path finding
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                print(f"candidate_entity:{candidate_entity}")
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    print("==========4开始找自己的路径===========")
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
                        # print(f"paths,exist_entity:{paths, exist_entity}")
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                # print(f"path_list:{path_list}")
                                paths_list.append(path_list)

                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    # print(f"paths_list:{paths_list}\n")
                    result_path = combine_lists(*paths_list)
                    print(f"=====完成单个路径合并======：\n")

                    if result_path != []:
                        result_path_list.extend(result_path)
                    if flag == 1:
                        continue
                    else:
                        break
                    # 这段代码的主要目的是从多个可能的路径中根据起始节点进行筛选，
                    # 确保每个起始节点尽量得到相等数量的路径（最多五条），
                    # 并处理一些特殊情况（如空路径和只有一个起始节点）。最终输出的结果是有效路径的集合。
                start_tmp = []
                for path_new in result_path_list:

                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])

                if len(start_tmp) == 0:
                    result_path = {}
                    single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                        # 如果起始节点数量大于或等于5，从result_path_list中选取路径，直到收集到五条有效路径。
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        # 起始节点少于5，则计算每个起始节点最多应获得的路径数量，并尽量均匀分配路径
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break
                    try:
                        single_path = result_path_list[0]
                        print(f"single_path:{single_path}\n")
                    except:
                        single_path = result_path_list

            else:
                result_path = {}
                single_path = {}

            # print('自己的kg路径result_path：', result_path, "\n")

            print("==========5开始找邻居路径===========")
            # # 5. neo4j knowledge graph neighbor entities
            neighbor_list = []
            neighbor_list_disease = []
            for match_entity in match_kg:
                disease_flag = 0
                neighbors, disease = get_entity_neighbors(match_entity, disease_flag)
                # print(f"\nmatch_entity,neighbors,disease:{match_entity}****{neighbors}*******{disease}\n")
                neighbor_list.extend(neighbors)

                while disease != []:
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            neighbors, disease = get_entity_neighbors(disease_entity, disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors, disease = get_entity_neighbors(disease_entity, disease_flag)
                            neighbor_list_disease.extend(neighbors)
            if len(neighbor_list) <= 5:
                neighbor_list.extend(neighbor_list_disease)

            # print("\nneighbor_list", neighbor_list, "\n")

            print("==========6开始找自己的证据路径===========")
            # 6. knowledge gragh path based prompt generation
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    # print(f"result_new_path:{result_new_path},\n")
                    path = "\n".join(result_new_path)
                    response_of_KG_list_path = prompt_path_finding(path)
                    if response_of_KG_list_path == "feifa":
                        print(f"\n!!!!!!!!模型无法回答！！！！！\n")
                        response_of_KG_list_path = ""

                    # print(f"\n==自己的==\npath :{path}\n\nresponse_of_KG_list_path:\n{response_of_KG_list_path}")
                    # if is_unable_to_answer(response_of_KG_list_path):
                    #     response_of_KG_list_path = prompt_path_finding(path)
                    # print("response_of_KG_list_path",response_of_KG_list_path)
            else:
                response_of_KG_list_path = '{}'
            response_single_path = prompt_path_finding(single_path)
            if response_single_path == 'feifa':
                print(f"\n!!!!!!!!模型无法回答！！！！！\n")
                response_single_path = ""
            print(f"response_single_path 完成")
            # if is_unable_to_answer(response_single_path):
            #     response_single_path = prompt_path_finding(single_path)

            print("============7开始找邻居证据路径=========")
            # # 7. knowledge gragh neighbor entities based prompt generation   
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)
            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            print(f"\n===邻居路径===：\n{neighbor_input}\n")
            response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            if response_of_KG_neighbor == "feifa":
                response_of_KG_list_path = ""
            # if is_unable_to_answer(response_of_KG_neighbor):
            #     response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            print("response_of_KG_neighbor\n", response_of_KG_neighbor)

            print("=======8总结答案=======")
            # # 8. prompt-based medical diaglogue answer generation
            output_all = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)
            print(f"output_all：\n{output_all}")
            if output_all == "feifa":
                print(f"\n!!!!!!!!模型无法回答！！！！！\n")
            # if is_unable_to_answer(output_all):
            #     output_all = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            # 查找 Output 1 和 Output 2\3 的位置
            if output_all != "feifa":
                # 使用正则表达式提取输出内容
                pattern = r"Output\s*1:\s*(.*?)\s*(?:Output\s*2:|Output2:|Output\s*3:|Output3:|$)"
                match1 = re.search(pattern, output_all, re.DOTALL)

                if match1:
                    output1 = match1.group(1).strip()
                    print(f"output1: {output1}")
                else:
                    print("1没有抽取")
                    output1 = None

                # 提取 Output 2
                pattern2 = r"Output\s*2:\s*(.*?)\s*(?:Output\s*3:|Output3:|$)"
                match2 = re.search(pattern2, output_all, re.DOTALL)

                if match2:
                    output2 = match2.group(1).strip()
                    print(f"output2: {output2}")
                else:
                    print("2没有抽取")
                    output2 = None

                # 提取 Output 3
                pattern3 = r"Output\s*3:\s*(.*)"
                match3 = re.search(pattern3, output_all, re.DOTALL)

                if match3:
                    output3 = match3.group(1).strip()
                    print(f"output3: {output3}")
                else:
                    print("3没有抽取")
                    output3 = None
            if output_all == "feifa":
                output1 = ""
                output2 = ""
                output3 = ""
                print(f"赋值三个答案均为空")
            if output_all != "feifa" and output1 == None and output2 == None and output3 == None:
                output1 = output_all
                print(f"没有匹配到，全部赋值给output1：{output1}")
            # 定义正则表达式
            # re4 = r"Output 1:\s*(.*?)\s*Output 2:"
            # re5 = r"Output 2:\s*(.*?)\s*Output 3:"
            #
            # # 提取 Output 1
            # output1_matches = re.findall(re4, output_all, flags=re.DOTALL)
            # if output1_matches:
            #     output1 = output1_matches[0].strip()  # 去除多余空格
            #     print(f"找到了：{output1}")
            # else:
            #     output1 = None  # 如果没有找到，设置为 None
            #     print(f"没有找到")
            #     continue

            # output2 = re.findall(re5, output_all, flags=re.DOTALL)
            # if len(output2) > 0:
            #     output2 = output2[0]
            # else:
            #     continue

            # output3_index = output_all.find("Output 3:")
            # if output3_index != -1:
            #     output3 = output_all[output3_index + len("Output 3:"):].strip()

            ## 9. Experiment 1: chatgpt，暂时不做
            # try:
            #     chatgpt_result = chat_35(str(input_text[0]))
            # except:
            #     sleep(40)
            #     chatgpt_result = chat_35(str(input_text[0]))
            # print('\nGPT-3.5:',chatgpt_result)

            ### 10. Experiment 2: document retrieval + bm25
            print(f"实验2 bm25")
            document_dir = "/media/MindMap-main/data/chatdoctor5k/document"
            document_paths = [os.path.join(document_dir, f) for f in os.listdir(document_dir)]

            corpus = []
            for path in document_paths:
                with open(path, "r", encoding="utf-8") as f:
                    corpus.append(f.read().lower().split())

            dictionary = corpora.Dictionary(corpus)
            bm25_model = BM25Okapi(corpus)

            bm25_corpus = [bm25_model.get_scores(doc) for doc in corpus]
            bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))

            query = input_text
            query_tokens = query.lower().split()
            tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
            tfidf_query = tfidf_model[dictionary.doc2bow(query_tokens)]
            best_document_index, best_similarity = 0, 0

            bm25_scores = bm25_index[tfidf_query]
            for i, score in enumerate(bm25_scores):
                if score > best_similarity:
                    best_similarity = score
                    best_document_index = i

            with open(document_paths[best_document_index], "r", encoding="utf-8") as f:
                best_document_content = f.read()

            document_bm25_result = prompt_document(input_text, best_document_content)
            # if is_unable_to_answer(document_bm25_result):
            #     document_bm25_result = prompt_document(input_text[0],best_document_content)
            if document_bm25_result == "feifa":
                print(f"\n!!!!!!!!模型无法回答！！！！！\n")
                document_bm25_result = ""
            print('\nBM25_retrieval:\n', document_bm25_result)

            print("===========实验3 文本嵌入检索==============")
            ### 11. Experiment 3: document + embedding retrieval
            model = Word2Vec.load("/media/MindMap-main/data/chatdoctor5k/data1_test_word2vec.model")
            ques_vec = np.mean([model.wv[token] for token in input_text.split()], axis=0)
            similarities = []
            for doc in docs:
                doc_vec = np.mean([model.wv[token] for token in doc.split()], axis=0)
                similarity = cosine_similarity([ques_vec], [doc_vec])[0][0]
                similarities.append(similarity)

            max_index = np.argmax(similarities)
            most_similar_doc = docs[max_index]

            document_embedding_result = prompt_document(input_text, most_similar_doc)
            if document_embedding_result == "feifa":
                document_embedding_result = ""
            # if is_unable_to_answer(document_embedding_result):
            #     document_embedding_result = prompt_document(input_text[0],most_similar_doc)
            print('\nEmbedding retrieval:\n', document_embedding_result)

            ### 12. Experiment 4: kg retrieval
            print(f"========实验4 kg=========")
            kg_retrieval = prompt_document(input_text, response_single_path)
            if kg_retrieval == "feifa":
                kg_retrieval = ""
            # if is_unable_to_answer(kg_retrieval):
            #     kg_retrieval = prompt_document(input_text[0],response_single_path)
            print('\nKG_retrieval:\n', kg_retrieval)

            ### 13. Experimet 5: gpt4
            # try:
            #     gpt4_result = chat_4(str(input_text[0]))
            # except:
            #     gpt4_result = chat_4(str(input_text[0]))
            # print('\nGPT4:\n',gpt4_result)

            # ### save the final result
            with open('/media/MindMap-main/new_data1_test_output_all.csv', 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow(
                    [input_text, output_text, output1, output2, output3, "chatgpt_result", document_bm25_result,
                     document_embedding_result, kg_retrieval, "gpt4_result"])
                f6.flush()
