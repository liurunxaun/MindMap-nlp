import itertools
import json
import pickle
import re
import time
from typing import List
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from flask import Flask, request
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sparkai.core.messages import ChatMessage
from sparkai.errors import SparkAIConnectionError
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sympy.physics.units import length
from websocket import create_connection
from translate import Translator

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
       question: {input_text}
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


def build_ne4j(df_path):
    # clean all
    session.run("MATCH (n) DETACH DELETE n")
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
    print("构建完成")


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
                "### 输出:\n" + "图像识别,卷积神经网络,CNN,迁移学习,Adam"
                                "\n\n### 输入:\n" + input_text + "### 输出:\n"
        ),
    }

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
    question_entities = final_response.replace('<end>', '')

    question_entities = [term.strip() for term in question_entities.split(',')]
    print(f"提取到的问题实体：{question_entities}")

    return question_entities


def cosine_similarity_manual(x, y):
    # print(x.shape[0])
    sim = y @ x.T
    return sim


def get_related_title_relation(match_kg):
    """
    输入：知识图谱中匹配到的实体数组
    处理过程：

    输出：知识图谱匹配到的实体及其相关论文标题数组
    """
    titles = []
    return titles


def get_related_title_label(question_match_kg):
    """
    输入：知识图谱中匹配到的实体数组
    处理过程：
        对每个实体
            neo4j检索标签是否是标题。
                是：[实体，title]放入titles数组中。
                不是：neo4j中检索标签是标题的邻居，对每个tile，[实体，title]放入titles数组中
    输出：知识图谱匹配到的实体及其相关论文标题数组
    """
    print("\n==========4 获得所有相关论文标题===========")
    titles = []
    question_entity_titles = [] # 格式是[[问题实体，匹配实体，论文标题], [ , , ],...]

    for question_match_entity in question_match_kg:
        question_entity = question_match_entity[0]
        match_entity = question_match_entity[1]
        with driver.session() as session:
            result = session.run(
                "MATCH (n) "
                "WHERE n.name = $entity_name "
                "RETURN n.label AS label",
                entity_name=match_entity
            )
            label = result.single()
            label = label["label"] if label else None

            if label == "标题":
                title = match_entity
                titles.append([match_entity, title])
                question_entity_titles.append([question_entity, match_entity ,title])
            else:
                result = session.run(
                    "MATCH (n)-[r]-(m) "
                    "WHERE n.name = $entity_name AND m.label = '标题' "
                    "RETURN m.name AS name",
                    entity_name=match_entity
                )
                neighbors = [record["name"] for record in result]
                for title in neighbors:
                    titles.append([match_entity, title])
                    question_entity_titles.append([question_entity, match_entity, title])

    return titles, question_entity_titles


def get_titles_graph(titles):
    """
    输入：检索到的匹配到的实体的所有相关的论文标题的数组，标题是字符串类型的
    处理过程：对每个标题，neo4j查询它的所有邻居和关系
    输出：每个标题对应的子图（后面我要用来在vue前端可视化）
    """
    print("\n==========5 获得所有相关论文标题的知识图谱===========")
    graphs = {
                'nodes': [],
                'edges': []
            }

    with driver.session() as session:
        if len(titles) >= 3:
            graph_number = 3
        else:
            graph_number = len(titles)
        for i in range(graph_number):
            title = titles[i][1]
            print("title:" + title)
            result = session.run(
                "MATCH (n {name: $title})-[r]-(m) "
                "WHERE n.label = '标题' "
                "RETURN n, r, m",
                title=title
            )

            for record in result:
                node_n = record['n']
                node_m = record['m']
                relationship = record['r']

                # 添加节点
                graphs['nodes'].append({
                    'id': node_n.element_id,
                    'label': node_n['name'],
                    'type': node_n['label']
                })
                graphs['nodes'].append({
                    'id': node_m.element_id,
                    'label': node_m['name'],
                    'type': node_m['label']
                })

                # 添加边
                graphs['edges'].append({
                    'source': node_n.element_id,
                    'target': node_m.element_id,
                    'relationship': type(relationship).__name__
                })

    return [graphs]


def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    exist_entity = {}
    # print(f"start_entity_name, end_entity_name: {start_entity_name, end_entity_name}")
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]-(end_entity)) "
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


def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph paths. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name.\n\n

    Output:
    """

    # 构建要发送的消息
    message_content = template.format(Path=path_input)
    messages = [ChatMessage(role="user", content=message_content)]

    handler = ChunkPrintHandler()
    # print(f"response正常：{response}")
    # time.sleep(3)
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


def path_exploration_and_aggregation(match_kg, question_kg):
    """
    查找并生成自身路径的函数。

    参数:
    - match_kg: 匹配的实体列表
    - question_kg: 提取的关键词实体列表

    返回:
    - response_single_path: 单一路径生成的响应
    """
    print("\n==========6 路径子图探索===========")
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
    print("\n==========7 路径子图融合===========")
    if result_path:
        result_new_path = ["->".join(p) for p in result_path]
        path = "\n".join(result_new_path)
        print(f"path:{path}")
        response_of_KG_list_path = prompt_path_finding(path)
        if response_of_KG_list_path == "feifa":
            response_of_KG_list_path = ""
    else:
        print(f"路径子图融合结果为空")
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


def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. \n\n

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
            return "feifa"  # 返回空列表
        else:
            print(f"发生其他错误: {str(e)}")

    return []  # 默认返回空列表


def get_entity_neighbors(entity_name: str, disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]-(n)
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


def neighbor_exploration_and_aggregation(match_kg, question_kg):
    """
    查找知识图谱中与匹配实体相关的邻居路径，并生成邻居路径的响应。

    参数:
    - match_kg: 匹配的实体列表
    - question_kg: 提取的关键词实体列表

    返回:
    - response_of_KG_neighbor: 邻居路径生成的响应
    """
    print("\n==========8 邻居子图探索===========")
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
        print("没有找到邻居子图")
        return ""

    print("\n============9 邻居子图融合 ===========")
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
        print("邻居子图融合结果为空")
    print("邻居子图融合结果：\n", response_of_KG_neighbor)

    return response_of_KG_neighbor


def final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor, ip_port='10.43.108.62:8678'):

    # 模板字符串
    template = """
        您是一位机器学习领域的专家AI，能够根据对话中的详细信息诊断技术问题，并推荐最佳的解决方法或模型。
        用户输入: {input_text}
        您可以访问以下相关知识资源，但请根据问题内容进行选择性参考，仅在必要时使用相关信息！！以确保回答的准确性和专业性：
        ### \n{response_of_KG_list_path}
        ### \n{response_of_KG_neighbor}
        请用中文回答，不换行。对于论文题目和专业词汇不需要翻译。
        """
    # 使用模板格式化，填充内容
    query = template.format(
        input_text=input_text,
        response_of_KG_list_path=response_of_KG_list_path,
        response_of_KG_neighbor=response_of_KG_neighbor
    )
    print(f"\nquery:{query}")

    # 准备数据
    url = 'ws://{}/get_res/'.format(ip_port)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
        'Cookie': 'miid=46129921999119747;bid=9;_uab_collina=155930245139965167817273;'
    }
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


def generate(input_text):
    # 1 构建neo4j知识图谱数据库
    print("\n==========1 构建neo4j知识图谱数据库===========")
    df_path = "data/nlp/knowledgeGraph/relation.txt"
    # 如果对应neo4j中没有该知识图谱数据库，需要运行这段代码。如果已经有了，可以注释掉。
    # build_ne4j(df_path)


    # 2 提取问题中的实体
    print("\n==========2 提取问题中的实体===========")
    start_time_0 = time.time()  # 请求发送的时间
    question_kg = get_question_entity(input_text)
    if question_kg == []:
        print("没有提取到实体")
        answer = spark_4_0(input_text)
        return [answer]
    print(f"question_kg:{question_kg}")


    # 3 函数计算所有实体嵌入与当前关键词嵌入之间的余弦相似度
    print("\n==========3 在知识图谱中匹配实体===========")

    # 读取知识图谱中实体的embedding
    with open('data/nlp/embedding/entity_embeddings_tfidf.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    print(f"加载完毕，共有{len(entity_embeddings_emb)}")

    # 从加载的数据中提取向量化器
    vectorizer = entity_embeddings["vectorizer"]

    match_kg = []
    question_match_kg = [] # 其实kg就是实体们，在mindmap的叙事语言中抽象成了知识图谱

    for kg_entity in question_kg:
        # 使用加载的 TfidfVectorizer 对输入实体进行编码
        query_embedding = vectorizer.transform([kg_entity])
        cos_similarities = cosine_similarity(query_embedding, entity_embeddings["embeddings"]).flatten()
        max_index = cos_similarities.argmax()
        print(max_index, entity_embeddings["entities"][max_index])

        match_kg_i = entity_embeddings["entities"][max_index]
        while match_kg_i in match_kg:
            cos_similarities[max_index] = 0
            max_index = cos_similarities.argmax()
            match_kg_i = entity_embeddings["entities"][max_index]

        match_kg_i = match_kg_i.split(':')[0]
        match_kg.append(match_kg_i)
        question_match_kg.append([kg_entity, match_kg_i])
    print('question_match_kg', question_match_kg)


    # 4 获得所有相关论文标题
    titles, question_entity_titles = get_related_title_label(question_match_kg)
    print("问题实体匹配实体所有相关论文标题:")
    print(question_entity_titles)


    # 5 户的所有相关论文标题的知识图谱
    graphs = get_titles_graph(titles)
    print(graphs)


    # 6、7 路径子图探索和融合
    response_of_KG_list_path = path_exploration_and_aggregation(match_kg, question_kg)


    # 8、9 邻居子图探索和融合
    # TODO:不要筛选路径只剩下5条
    response_of_KG_neighbor = neighbor_exploration_and_aggregation(match_kg, question_kg)


    # 记录路径生成的总时间
    path_response_time = time.time() - start_time_0
    print(f"\npath_time:{path_response_time}")


    # 10 生成答案
    print("\n==========10 生成答案========")
    mindmap_res, mindmap_time = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)
    print(f"答案和耗时：\n{mindmap_res}\n{mindmap_time}")
    if mindmap_res == None:
        print(f"模型无法回答")
        return ["模型无法回答"]
    else:
        return [mindmap_res, question_entity_titles, graphs]


@app.route('/process-data', methods=['POST'])
def main():
    # 接收 Java 发来的数据
    data = request.json
    question = data.get('question')
    if question == '':
        print("\n输入的问题为空")
        return ['']
    print('\nQuestion:', question)

    # 初始化翻译器
    translator = Translator(from_lang="zh", to_lang="en")
    # 检查是否包含中文字符
    if re.search('[\u4e00-\u9fff]', question):
        # 如果包含中文字符，进行翻译
        question = translator.translate(question)
        print(f"翻译后的英文查询文本: {question}")
    else:
        # 如果是英文，不进行翻译
        question = question
        print("检测到英文查询文本，不进行翻译。")

    answers = generate(question)
    result = {"answer": answers}
    print()
    print(result)

    # 封装成json格式，并确保中文正常显示
    json_result = json.dumps(result, ensure_ascii=False)
    # 将处理结果返回给 Java
    return json_result


if __name__ == "__main__":
    # 连接neo4j
    uri = "bolt://127.0.0.1:7687"
    username = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    print("\n成功连接neo4j\n")

    # 监听接口，生成并返回答案
    app.run(host='0.0.0.0', port=5002)
    main()
