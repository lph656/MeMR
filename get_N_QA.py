import json
import os # 导入 os 模块用于检查文件是否存在

def get_lowest_score_qas_from_file(file_path, n):
    """
    从JSON文件中找出总分最低的N个问答的ID和question。

    Args:
        file_path (str): JSON文件的路径。
        n (int): 需要返回的最低分数问答的数量。

    Returns:
        list: 包含ID、question和总分的字典列表，按总分升序排列。
              如果文件不存在或解析失败，则返回空列表。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。请检查文件路径是否正确。")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：无法解析文件 '{file_path}' 中的JSON数据。请检查JSON格式是否正确。")
        return []
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return []

    if not isinstance(data, list):
        print(f"错误：JSON文件 '{file_path}' 的根元素不是一个列表。请确保文件包含一个JSON数组。")
        return []

    # 提取每个问答的ID、question和总分
    items_to_sort = []
    for item in data:
        # 确保项是字典，并且包含 'id', '总分', 'question' 这三个关键字段
        if isinstance(item, dict) and "id" in item and "总分" in item and "question" in item:
            items_to_sort.append({
                "id": item["id"],
                "总分": item["总分"],
                "question": item["question"]
            })
        else:
            print(f"警告：文件中存在不符合预期格式的项（缺少 'id'、'总分' 或 'question'，或不是字典）：{item}")


    # 按总分升序排序
    items_to_sort.sort(key=lambda x: x["总分"])

    # 返回前N个最低分数的问答（包含ID和question）
    return items_to_sort[:n]

if __name__ == "__main__":
    # JSON文件路径已硬编码
    json_file_path = "test_infer_RAG/generate_scores_DS_v1/zhongliuke/zhongliuke.json"

    try:
        N = int(input("请输入您想查看的最低分数问答的数量N："))
        if N <= 0:
            print("N必须是一个正整数。")
        else:
            lowest_qas = get_lowest_score_qas_from_file(json_file_path, N)

            if lowest_qas:
                print(f"\n总分最低的{N}个问答的ID和question如下：")
                for qa in lowest_qas:
                    print(f"ID: {qa['id']}, Question: {qa['question']}")
            else:
                print("没有找到符合条件的数据，或者文件解析失败。")
    except ValueError:
        print("无效的输入。N必须是一个整数。")

