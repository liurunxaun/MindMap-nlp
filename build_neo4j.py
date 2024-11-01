from neo4j import GraphDatabase

# def build_ne4j(df_path):
#     # clean all
#     print("删除原有知识图谱数据库")
#     session.run("MATCH (n) DETACH DELETE n")
#     print("删除成功")
#     # read triples
#     with open(df_path, 'r', encoding='utf-8') as file:
#         print("开始构建")
#         i = 0
#         for line in file:
#             i += 1
#             if i % 10000 == 0:
#                 print(i)
#             parts = line.strip().split('\t')
#             if len(parts) == 3:  # Ensure there are exactly 3 parts per line: entity - relation - entity
#                 head_name, relation_name, tail_name = parts
#                 query = (
#                         "MERGE (h:Entity { name: $head_name }) "
#                         "MERGE (t:Entity { name: $tail_name }) "
#                         "MERGE (h)-[r:`" + relation_name + "`]->(t)"
#                 )
#                 session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
#     print("构建完成")


def build_ne4j_label(df_path):
    # clean all
    print("删除原有知识图谱数据库")
    # session.run("MATCH (n) DETACH DELETE n")
    print("删除成功")

    # read triples
    with open(df_path, 'r', encoding='utf-8') as file, open('./data/nlp/problem.txt', 'a', encoding='utf-8') as problem_file:
        print("开始构建")
        i = 0
        for line in file:
            i += 1
            if i % 10000 == 0:
                print(i)
            parts = line.strip().split('\t')
            if len(parts) == 3:  # 确保每行有3个部分：head:label - relation - tail:label
                head_part, relation_name, tail_part = parts

                # 提取实体和标签
                try:
                    head_name, head_label = head_part.split(':', 1)
                    tail_name, tail_label = tail_part.split(':', 1)
                except Exception as e:
                    print(f"第{i}条数据有问题: {e}")
                    # 将有问题的行写入文件
                    problem_file.write(line)

                # 创建带标签的节点
                query = (
                    "MERGE (h:Entity { name: $head_name }) "
                    "SET h.label = $head_label "
                    "MERGE (t:Entity { name: $tail_name }) "
                    "SET t.label = $tail_label "
                    "MERGE (h)-[r:`" + relation_name + "`]->(t)"
                )
                session.run(query, head_name=head_name, tail_name=tail_name, head_label=head_label,
                            tail_label=tail_label, relation_name=relation_name)

    print("构建完成")



# 连接neo4j
uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()
print("\n成功连接neo4j\n")

df_path = "./data/nlp/relation_label.txt"
# build_ne4j(df_path)
build_ne4j_label(df_path)