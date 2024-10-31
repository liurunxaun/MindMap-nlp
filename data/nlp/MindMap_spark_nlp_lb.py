from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from sparkai.errors import SparkAIConnectionError
import time
from websocket import create_connection
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
import os
from PIL import Image, ImageDraw, ImageFont
import csv
# from gensim import corpora
# from gensim.models import TfidfModel
# from gensim.similarities import SparseMatrixSimilarity
# from rank_bm25 import BM25Okapi
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec
import sys
from time import sleep
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# 星火认知大模型的配置
SPARKAI_URL_4_0 = 'wss://spark-api.xf-yun.com/v4.0/chat'
SPARKAI_URL_3_5 = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_URL_1_1 = 'wss://spark-api.xf-yun.com/v1.1/chat'
SPARKAI_APP_ID = 'c651a376'
SPARKAI_API_SECRET = 'Y2RlYzRlNWU3ZWVhYWM3NTU5N2Q4NjIy'
SPARKAI_API_KEY = 'cde9573970818d4d8e08e99598569174'
SPARKAI_DOMAIN_4_0 = '4.0Ultra'
SPARKAI_DOMAIN_3_5 = 'generalv3.5'
SPARKAI_DOMAIN_1_1 = 'general'
spark1_1 = ChatSparkLLM(
    spark_api_url=SPARKAI_URL_1_1,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN_1_1,
    streaming=False,
)
spark4_0 = ChatSparkLLM(
    spark_api_url=SPARKAI_URL_4_0,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN_4_0,
    streaming=False,
)
spark3_5 = ChatSparkLLM(
    spark_api_url=SPARKAI_URL_3_5,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN_3_5,
    streaming=False,
)


# s首次响应时间
def spark_4_0(query, ip_port='10.43.108.62:8678'):
    """
    封装的函数，用于通过 WebSocket 查询信息，并统计首字符的响应时间

    参数:
        query (str): 查询的内容，例如 "北京市今天天气怎么样？"
        ip_port (str): WebSocket服务器的IP和端口号，默认为 '10.43.108.62:8678'

    返回:
        tuple: 查询的完整结果 (str)，首字符的响应时间 (float)
    """
    url = 'ws://{}/get_res/'.format(ip_port)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
        'Cookie': 'miid=46129921999119747;bid=9;_uab_collina=155930245139965167817273;'
    }
    # 模板字符串
    template = """
       You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.
       \nPatient input: {input_text}\n
       The answer includes disease and tests and recommended medications. Don't break into new lines, give me the text in one paragraph\n
       There is a sample:\n
       Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.\n
       """
    text = template.format(input_text=query)
    # 准备数据
    data = {
        'query': text,
        'user_id': 'yjyang18',
    }

    # 打开 WebSocket 连接并发送数据
    wss = create_connection(url, header=headers, timeout=10)
    wss.send(json.dumps(data, ensure_ascii=False))

    final_response = ""  # 用于累积所有的response
    recv = True
    first_response_time = None  # 记录首字符响应时间
    start_time = time.time()  # 请求发送的时间

    while recv:
        data = wss.recv()
        # 检查是否收到空数据
        if not data:
            print("收到空数据，继续接收下一条数据...")
            continue  # 跳过本次循环，继续接收
        t = json.loads(data)
        response = t["response"]

        # 如果是第一次接收到数据，记录首字符响应时间
        if first_response_time is None:
            first_response_time = time.time() - start_time

        final_response += response  # 累积每次收到的response
        # 如果收到 '<end>'，则停止接收
        if response == '<end>':
            recv = False
            wss.close()

    # 返回完整的累积答案，去掉最后的 <end> 标记，以及首字符的响应时间
    return final_response.replace('<end>', ''), first_response_time


def spark_3_5(text):
    text = "You are a professional doctor. Please respond to the patient's following question:" + text
    messages = [ChatMessage(role="user", content=text)]
    handler = ChunkPrintHandler()
    # time.sleep(10)
    try:
        response = spark3_5.generate([messages], callbacks=[handler])
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


# def cosine_similarity_manual(x, y):
#     dot_product = np.dot(x, y.T)
#     norm_x = np.linalg.norm(x, axis=-1)
#     norm_y = np.linalg.norm(y, axis=-1)
#     sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
#     return sim
def cosine_similarity_manual(x, y):
    print(x.shape[0])
    sim = y @ x.T
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
    Use the knowledge graph information. Try to convert them to natural language in English, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    # 构建要发送的消息
    message_content = template.format(neighbor=neighbor)
    messages = [ChatMessage(role="user", content=message_content)]
    handler = ChunkPrintHandler()
    # time.sleep(10)
    try:
        response = spark1_1.generate([messages], callbacks=[handler])
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


def final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor, ip_port='10.43.108.62:8678'):
    # 模板字符串
    template = """
        您是一位机器学习领域的专家AI，能够根据对话中的详细信息诊断技术问题，并推荐最佳的解决方法或模型。
        用户输入: {input_text}
        您可以访问以下相关知识资源，但请根据问题内容进行选择性参考，仅在必要时使用相关信息！！以确保回答的准确性和专业性：
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}
        请用中文回答，不换行。
        """

    # 使用模板格式化
    query = template.format(
        input_text=input_text,
        response_of_KG_list_path=response_of_KG_list_path,
        response_of_KG_neighbor=response_of_KG_neighbor
    )
    # print(f"\nquery:{query}")
    url = 'ws://{}/get_res/'.format(ip_port)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
        'Cookie': 'miid=46129921999119747;bid=9;_uab_collina=155930245139965167817273;'
    }
    # 准备数据
    data = {
        'query': query,
        'user_id': 'yjyang18',
    }

    # 打开 WebSocket 连接并发送数据
    wss = create_connection(url, header=headers, timeout=30)
    wss.send(json.dumps(data, ensure_ascii=False))

    final_response = ""  # 用于累积所有的response
    recv = True
    first_response_time = None  # 用于记录首字符响应时间
    start_time = time.time()  # 请求发送的时间

    while recv:
        data = wss.recv()
        # print(f"data{data}")
        # 检查是否收到空数据

        if not data:
            print("收到空数据，继续接收下一条数据...")
            continue  # 跳过本次循环，继续接收
        t = json.loads(data)
        response = t["response"]

        # 如果是第一次接收到数据，记录首字符响应时间
        if first_response_time is None:
            first_response_time = time.time() - start_time

        final_response += response  # 累积每次收到的response
        # 如果收到 '<end>'，则停止接收
        if response == '<end>':
            recv = False
            wss.close()

    # 返回完整的累积答案，去掉最后的 <end> 标记，以及首字符的响应时间
    return final_response.replace('<end>', ''), first_response_time


def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph paths. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language in English, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    # 构建要发送的消息
    message_content = template.format(Path=path_input)
    messages = [ChatMessage(role="user", content=message_content)]

    handler = ChunkPrintHandler()
    # print(f"response正常：{response}")
    # time.sleep(3)
    try:
        response = spark1_1.generate([messages], callbacks=[handler])
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


def get_most_relevant_paths(x, keywords, paths, top_k=5, min_similarity=0.1):
    """
    计算提取的关键词列表与每条路径的相似度，选择最相关的 top_k 条路径。

    参数:
    - keywords: 用户输入中提取的关键词列表
    - paths: 知识图谱中的路径列表
    - top_k: 返回的最相关路径数量，默认为 5
    - min_similarity: 返回的路径的最小相似度阈值，默认为 0.1

    返回:
    - 最相关的 top_k 条路径及其相似度得分（满足最小相似度条件）
    """
    # 将关键词列表转换为字符串
    keyword_string = ' '.join(keywords)

    # 将每条路径转换为字符串
    processed_paths = [' '.join(path) for path in paths]

    # 将关键词字符串与路径合并为一个语料库
    corpus = [keyword_string] + processed_paths

    # 使用 TF-IDF 向量化语料库
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 计算关键词与每条路径的余弦相似度
    keyword_vector = tfidf_matrix[0]  # 关键词的向量
    path_vectors = tfidf_matrix[1:]  # 路径的向量
    similarities = cosine_similarity(keyword_vector, path_vectors).flatten()
    # print(f"similarities:{similarities}")
    if x == 0:
        # 获取满足相似度条件的路径索引并排序
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
        sorted_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
    else:
        # 不考虑相似度，只根据 top_k 获取最高相似度的路径
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

        # 提取符合条件的路径及其相似度
    top_paths = [paths[i] for i in sorted_indices]
    top_similarities = [similarities[i] for i in sorted_indices]

    print(f"\nTop {len(top_paths)} paths selected:\n{top_similarities}\n")
    return top_paths


def find_own_path(match_kg, question_kg):
    """
    查找并生成自身路径的函数。

    参数:
    - match_kg: 匹配的实体列表
    - question_kg: 提取的关键词实体列表

    返回:
    - response_single_path: 单一路径生成的响应
    """
    result_path_list = []

    # 处理当 match_kg 中的实体数量大于 1 的情况
    if len(match_kg) > 1:
        start_entity = match_kg[0]
        candidate_entity = match_kg[1:]
        # 迭代查找路径
        while True:
            flag = 0
            paths_list = []

            while candidate_entity:
                end_entity = candidate_entity[0]
                candidate_entity.remove(end_entity)

                # 查找从 start_entity 到 end_entity 的最短路径
                paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
                print(f"paths, exist_entity:{paths}---{exist_entity}")
                # 如果找不到路径，更新起始节点并退出内层循环
                if not paths or paths == ['']:
                    flag = 1
                    if not candidate_entity:
                        flag = 0
                    else:
                        start_entity = candidate_entity[0]
                        candidate_entity.remove(start_entity)
                    break
                # 将找到的路径转换为列表格式并存储
                path_list = [p.split('->') for p in paths]
                if path_list:
                    paths_list.append(path_list)
                if exist_entity:
                    try:
                        candidate_entity.remove(exist_entity)
                    except ValueError:
                        continue
                start_entity = end_entity

            # 合并所有找到的路径
            result_path = combine_lists(*paths_list)
            if result_path:
                result_path_list.extend(result_path)
                print(f"合并之后的个数：{len(result_path_list)}")
            if flag == 1:
                continue
            else:
                break

        # 根据起始节点筛选路径
    start_entities = list(set([path[0] for path in result_path_list if path]))
    # print(f"start_entities:{start_entities}")
    if not start_entities:
        result_path, single_path = {}, {}
    elif len(start_entities) == 1:
        result_path = result_path_list[:5]
        # result_path = get_most_relevant_paths(1, question_kg, result_path_list)
    else:
        # 根据起始节点的数量选择最多五条路径
        result_path = []
        if len(start_entities) >= 5:
            for path in result_path_list:
                if path[0] in start_entities:
                    result_path.append(path)
                    start_entities.remove(path[0])
                if len(result_path) == 5:
                    break
        else:
            count = 5 // len(start_entities)
            remind = 5 % len(start_entities)
            count_tmp = 0

            for path in result_path_list:
                if len(result_path) >= 5:
                    break
                if path[0] in start_entities:
                    if count_tmp < count:
                        result_path.append(path)
                        count_tmp += 1
                    else:
                        start_entities.remove(path[0])
                        count_tmp = 0
                        if path[0] in start_entities:
                            result_path.append(path)
                            count_tmp += 1

                    if len(start_entities) == 1:
                        count += remind

    # 生成最终的路径响应
    if result_path:
        result_new_path = ["->".join(p) for p in result_path]
        path = "\n".join(result_new_path)
        print(f"path:{path}")
        response_of_KG_list_path = prompt_path_finding(path)
        if response_of_KG_list_path == "feifa":
            response_of_KG_list_path = ""
    else:
        print(f"没有找到自己的子图")
        response_of_KG_list_path = '{}'
    # single_path = result_path_list[0] if result_path_list else {}
    # # 生成单一路径的响应
    # response_single_path = prompt_path_finding(single_path) if single_path else ""
    # if response_single_path == 'feifa':
    #     print(f"\n!!!!!!!!模型无法回答！！！！！\n")
    #     response_single_path = ""
    #
    # print(f"response_single_path 完成")
    return response_of_KG_list_path


def find_neighbor_path(match_kg, question_kg):
    """
    查找知识图谱中与匹配实体相关的邻居路径，并生成邻居路径的响应。

    参数:
    - match_kg: 匹配的实体列表
    - question_kg: 提取的关键词实体列表

    返回:
    - response_of_KG_neighbor: 邻居路径生成的响应
    """
    neighbor_list = []
    neighbor_list_disease = []

    # 获取与匹配实体相关的邻居实体
    for match_entity in match_kg:
        neighbors, disease = get_entity_neighbors(match_entity, disease_flag=0)
        neighbor_list.extend(neighbors)

        # 处理与疾病相关的邻居
        while disease:
            # 查找在 match_kg 中存在的疾病
            new_disease = [d for d in disease if d in match_kg]

            if new_disease:
                # 如果找到在 match_kg 中的疾病，获取其邻居
                for disease_entity in new_disease:
                    neighbors, disease = get_entity_neighbors(disease_entity, disease_flag=1)
                    neighbor_list_disease.extend(neighbors)
            else:
                # 处理所有其他疾病实体
                for disease_entity in disease:
                    neighbors, disease = get_entity_neighbors(disease_entity, disease_flag=1)
                    neighbor_list_disease.extend(neighbors)

    # 合并所有邻居实体
    neighbor_list.extend(neighbor_list_disease)
    print(f"检索到的邻居路径条数：{len(neighbor_list)}")

    # 如果检索到的邻居列表不为空，根据相似度筛选最相关的邻居路径
    if neighbor_list:
        neighbor_list = neighbor_list[:5]
        # neighbor_list = get_most_relevant_paths(1, question_kg, neighbor_list)
        print("选取相似度之后的neighbor_list", neighbor_list, "\n")
    else:
        print("邻居路径为空")
        return ""

    print("============7 开始生成邻居路径的证据 ===========")
    if len(neighbor_list) != 0:
        # 将筛选后的邻居路径转换为所需的输入格式
        neighbor_new_list = ["->".join(neighbor) for neighbor in neighbor_list]
        neighbor_input = "\n".join(neighbor_new_list)
        # 调用生成邻居路径的响应
        response_of_KG_neighbor = prompt_neighbor(neighbor_input)
        # 检查是否生成了无效响应
        if response_of_KG_neighbor == "feifa":
            response_of_KG_neighbor = ""

    else:
        response_of_KG_neighbor = ""
    print("response_of_KG_neighbor\n", response_of_KG_neighbor)

    return response_of_KG_neighbor


def contect_ne4j(df_path):
    # session.run("MATCH (n) DETACH DELETE n")  # clean all
    # read triples
    with open(df_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:  # Ensure there are exactly 3 parts per line: entity - relation - entity
                head_name, relation_name, tail_name = parts
                query = (
                        "MERGE (h:Entity { name: $head_name }) "
                        "MERGE (t:Entity { name: $tail_name }) "
                        "MERGE (h)-[r:`" + relation_name + "`]->(t)"
                )
                session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
    print("读取完成，运行")


def get_question_entity(input_text, ip_port='10.43.108.62:8678'):
    # print(f"\nquery:{query}")
    url = 'ws://{}/get_res/'.format(ip_port)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
        'Cookie': 'miid=46129921999119747;bid=9;_uab_collina=155930245139965167817273;'
    }
    PROMPT_DICT = {
        "prompt_question_input": (
                "根据以下输入提取相关的实体："
                "例子：在图像识别任务中，我们使用卷积神经网络（CNN）来处理输入数据，并结合迁移学习提升模型的泛化能力。使用的优化算法为Adam。"
                "### 输出:\n" + "图像识别, 卷积神经网络, CNN, 迁移学习, Adam"
                "\n\n### 输入:\n"+ input_text + "### 输出:\n"
        ),
    }

    re1 = r'<SEP>提取的实体是：(.*?)<EOS>'
    re2 = r'(.*?)<EOS>'
    re3 = r'提取的实体是：(.*?)<EOS>'
    re4 = r'提取的实体是：(.*)'
    re5 = r'"(.*?)"'
    re6 = r': (.*)'


    prompt_question_input = PROMPT_DICT["prompt_question_input"]
    prompt_question = prompt_question_input.format_map(input_text)

    # 准备数据
    data = {
        'query': prompt_question,
        'user_id': 'yjyang18',
    }

    # 打开 WebSocket 连接并发送数据
    wss = create_connection(url, header=headers, timeout=30)
    wss.send(json.dumps(data, ensure_ascii=False))

    final_response = ""  # 用于累积所有的response
    recv = True

    while recv:
        data = wss.recv()

        if not data:
            print("收到空数据，继续接收下一条数据...")
            continue  # 跳过本次循环，继续接收
        t = json.loads(data)
        response = t["response"]
        final_response += response  # 累积每次收到的response
        # 如果收到 '<end>'，则停止接收
        if response == '<end>':
            recv = False
            wss.close()
    x = final_response.replace('<end>', '')
    print(f"回复：{x}")

    question_kg = re.findall(re1, x)
    if question_kg == []:
        question_kg = re.findall(re3, x)
    if question_kg == []:
        question_kg = re.findall(re2, x)
    if question_kg == []:
        question_kg = re.findall(re4, x)
    if question_kg == []:
        question_kg = re.findall(re5, x)
    if question_kg == []:
        question_kg = re.findall(re6, x)
    if question_kg == []:
        print(x)
        question_kg = x
    # 返回完整的累积答案，去掉最后的 <end> 标记，以及首字符的响应时间
    return question_kg


def generate(input_text):
    df_path = "/work/beiluo/keyan_mindmap/data/sample.txt"
    model = BGEM3FlagModel('/work/beiluo/model/BAAI/bge-m3',
                           use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # contect_ne4j(df_path)
    with open('/work/beiluo/keyan_mindmap/data/bge_entity_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])

    if input_text == '':
        print("请输入问题")
        exit()
    print('Question:\n', input_text)

    ######获取query的实体#######
    start_time_0 = time.time()  # 请求发送的时间
    question_kg = get_question_entity(input_text)
    print(f"question_kg:{question_kg}")
    if len(question_kg) == 0:
        print("<Warning> no entities found")
        # continue
    if len(question_kg) > 1:
        question_kg = question_kg.split(", ")
    # 函数计算所有实体嵌入与当前关键词嵌入之间的余弦相似度。
    match_kg = []
    for kg_entity in question_kg:
        embeddings_1 = model.encode(kg_entity,
                                    batch_size=12,
                                    max_length=8192,
                                    # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs']
        cos_similarities = cosine_similarity_manual(entity_embeddings_emb, embeddings_1)
        # print(cos_similarities)
        max_index = cos_similarities.argmax()

        match_kg_i = entity_embeddings["entities"][max_index]
        while match_kg_i in match_kg:
            cos_similarities[max_index] = 0
            max_index = cos_similarities.argmax()
            match_kg_i = entity_embeddings["entities"][max_index]

        match_kg.append(match_kg_i)
    print('match_kg', match_kg, "\n")

    print("==========4开始找自己的路径===========")
    response_of_KG_list_path = find_own_path(match_kg, question_kg)

    print("==========5开始找邻居路径===========")
    response_of_KG_neighbor = find_neighbor_path(match_kg, question_kg)
    # 记录路径生成的总时间
    path_response_time = time.time() - start_time_0
    print(f"path_time:{path_response_time}\n")

    print("==========6生成Mindmap答案========")
    mindmap_res, mindmap_time = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)
    print(f"output_all：\n{mindmap_res}\n{mindmap_time}")
    if mindmap_res == None:
        print(f"\n!!!!!!!!模型无法回答！！！！！\n")


# 1. build neo4j knowledge graph datasets
uri = "bolt://54.210.193.164"
username = "neo4j"
password = "pages-procedures-offset"
driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()
##############################build KG
# session.run("MATCH (n) DETACH DELETE n")  # clean all
print(f"连接成功")

generate("Fast semantic parsing with well-typedness guarantees是谁写的")
