import itertools
import json
import os
import pickle
import re
import time
from typing import List

import numpy as np
import pandas as pd
from flask import Flask, request
from neo4j import GraphDatabase
from sparkai.core.messages import ChatMessage
from sparkai.errors import SparkAIConnectionError
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler

app = Flask(__name__)

# 星火认知大模型的配置
SPARKAI_URL_4_0 = 'wss://spark-api.xf-yun.com/v4.0/chat'

SPARKAI_APP_ID = '4f6b5020'
SPARKAI_API_SECRET = 'MGI3MGQ5MjJhMjAzMWY2ODFkNDMxYjI1'
SPARKAI_API_KEY = 'f299adff306173af4588cdd14dca6788'
SPARKAI_DOMAIN_4_0 = '4.0Ultra'

spark4_0 = ChatSparkLLM(
    spark_api_url=SPARKAI_URL_4_0,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN_4_0,
    streaming=False,
)


def spark_4_0(text):
    text = "You are a professional doctor. Please respond to the patient's following question:" + text
    messages = [ChatMessage(role="user", content=text)]
    handler = ChunkPrintHandler()
    time.sleep(10)
    try:
        response = spark4_0.generate([messages], callbacks=[handler])
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            print(f"response4.0:{response}")
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "illegal"  # 返回空列表
        else:
            print(f"发生其他错误: {str(e)}")

    return []  # 默认返回空列表


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


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
    # time.sleep(10)
    try:
        response = spark4_0.generate([messages], callbacks=[handler])
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "illegal"  # 返回空列表
        else:
            print(f"发生其他错误: {str(e)}")

    return []  # 默认返回空列表


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
    time.sleep(3)  # 等待1秒
    try:
        response = spark4_0.generate([messages], callbacks=[handler])
        print(f"response:{response}")
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
            return ""  # 返回空列表

    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return 'illegal'  # 返回空列表
        else:
            print(f"发生连接错误: {str(e)}")
            return ""  # 返回空列表

    return ""  # 默认返回空列表


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
    time.sleep(3)
    try:
        response = spark4_0.generate([messages], callbacks=[handler])
        # 检查响应内容
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            return response.generations[0][0].text.strip()
        else:
            print("未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "illegal"  # 返回空列表
        else:
            print(f"发生其他错误: {str(e)}")

    return []  # 默认返回空列表


# 从prompt_extract_keyword文件中复制来的
def prompt_extract_keyword(input_text, re2, re4):
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
        response = spark4_0.generate([messages], callbacks=[handler])
        print(f"大模型提取问题实体的response:{response.generations[0][0].text.strip()}")
        # Check response content
        if response and len(response.generations) > 0 and len(response.generations[0]) > 0:
            question_kg = re.findall(re4, response.generations[0][0].text.strip())
            # 如果有多项，用逗号分割并去掉空格
            if question_kg:
                question_kg = [item.strip() for item in re.split(r',\s*', question_kg[0])]
            time.sleep(10)
            if question_kg == []:
                question_kg = re.findall(re2, response.generations[0][0].text.strip())
            print(f"question_kg:{question_kg}")
            return question_kg if question_kg else ""  # Return first match or empty string
        else:
            print("大模型未生成有效的响应内容。")
    except SparkAIConnectionError as e:
        if '10013' in str(e):
            print("非法请求，返回空列表。")
            return "illegal"  # Return error flag for illegal request
        else:
            print(f"发生连接错误: {str(e)}")

    except Exception as e:
        print(f"发生错误: {str(e)}")

    return []  # Default return an empty list


def process(input_text):
    # 0. 从java后端接收问题输入
    print("==========1接收问题===========")

    # input_text = "Doctor, I have been experiencing sudden and frequent panic attacks. I don't know what to do."
    if input_text == [] or input_text == "illegal":
        print("输入的问题为空或有问题")
        exit()
    print('Question:\n', input_text)

    # 2 提取问题中的实体
    print("==========2提取问题中的实体===========")

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"
    re4 = r'The extracted entities are (.*?)<EOS>'

    question_kg = prompt_extract_keyword(input_text, re2, re4)
    if len(question_kg) == 0:
        print("<Warning> no entities found")
    print(f"question_kg:{question_kg}")

    # 3 函数计算所有实体嵌入与当前关键词嵌入之间的余弦相似度。
    print("==========3在知识图谱中匹配实体===========")

    with open('./data/chatdoctor5k/entity_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open('./data/chatdoctor5k/keyword_embeddings.pkl', 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)

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
    print('match_kg:', match_kg, "\n")

    # 4. neo4j knowledge graph path finding
    print("==========4路径子图探索===========")
    if len(match_kg) != 1 or 0:
        start_entity = match_kg[0]
        candidate_entity = match_kg[1:]
        print(f"candidate_entity:{candidate_entity}")
        result_path_list = []
        while 1:
            flag = 0
            paths_list = []

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
            print(f"=====完成单个路径合并======")

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
        print('没找到路径子图\n')

    print("==========5邻居子图探索===========")
    # # 5. neo4j knowledge graph neighbor entities
    neighbor_list = []
    neighbor_list_disease = []
    for match_entity in match_kg:
        disease_flag = 0
        neighbors, disease = get_entity_neighbors(match_entity, disease_flag)
        print(f"match_entity,neighbors,disease:{match_entity}****{neighbors}*******{disease}\n")
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
    print("neighbor_list", neighbor_list, "\n")

    print("==========6路径子图融合===========")
    # 6. knowledge graph path based prompt generation
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
            if response_of_KG_list_path == "illegal":
                response_of_KG_list_path = ""
                print(f"\n!!!!!!!!模型无法回答！！！！！\n")
    else:
        response_of_KG_list_path = '{}'

    response_single_path = prompt_path_finding(single_path)
    print("response_single_path:\n", response_single_path)

    if response_single_path == 'illegal':
        response_single_path = ""
        print(f"\n!!!!!!!!模型无法回答！！！！！\n")

    print(f"路径子图融合完成\n")

    # 7. knowledge graph neighbor entities based prompt generation
    print("============7邻居子图融合=========")
    response_of_KG_list_neighbor = []
    neighbor_new_list = []
    for neighbor_i in neighbor_list:
        neighbor = "->".join(neighbor_i)
        neighbor_new_list.append(neighbor)
    if len(neighbor_new_list) > 5:
        neighbor_input = "\n".join(neighbor_new_list[:5])
    else:
        neighbor_input = "\n".join(neighbor_new_list)
    response_of_KG_neighbor = prompt_neighbor(neighbor_input)
    if response_of_KG_neighbor == "illegal":
        response_of_KG_list_path = ""
    print("response_of_KG_neighbor:\n", response_of_KG_neighbor)

    # 8. prompt-based medical dialogues answer generation
    print("=======8大模型生成答案=======")
    output_all = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)
    if output_all == "illegal":
        print(f"\n!!!!!!!!模型无法回答！！！！！\n")
        return 0
    else:
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

        if output_all == "illegal":
            output1 = ""
            output2 = ""
            output3 = ""
            print(f"赋值三个答案均为空")

        if output_all != "illegal" and output1 == None and output2 == None and output3 == None:
            output1 = output_all
            print(f"没有匹配到，全部赋值给output1：{output1}")

        return [output1, output2, output3]


@app.route('/process-data', methods=['POST'])
def process_data():
    # 接收 Java 发来的数据
    data = request.json
    question = data.get('question')
    # print(question)

    answer = process(question)
    # answer = ["Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.",
    #           "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').",
    #           """Patient(Path-based Evidence 1)
    #                 └── has been experiencing(Path-based Evidence 1)
    #                     └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
    #                         └── could be caused by(Path-based Evidence 2)
    #                             └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
    #                                 ├── requires(Neighbor-based Evidence 1)
    #                                 │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
    #                                 │       └── may include(Neighbor-based Evidence 2)
    #                                 │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
    #                                 ├── can be treated with(Path-based Evidence 3)
    #                                 │   └── anti-inflammatory drugs and Steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
    #                                 └── should be accompanied by(Neighbor-based Evidence 3)
    #                                     └── resting the voice and avoiding irritants(Neighbor-based Evidence 3) """]

    # 处理数据（假设处理逻辑）
    result = {"answer": answer}
    print(result)

    json_result = json.dumps(result, ensure_ascii=False)  # 确保中文正常显示

    # 将处理结果返回给 Java
    return json_result


if __name__ == "__main__":
    # 1. build neo4j knowledge graph datasets
    uri = "bolt://44.202.86.31:7687"
    username = "neo4j"
    password = "bin-ramp-seals"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    print(f"连接成功")

    app.run(host='0.0.0.0', port=5002)
    process_data()
