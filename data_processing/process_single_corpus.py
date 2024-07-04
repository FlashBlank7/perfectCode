# 导入pickle模块用于序列化和反序列化数据
import pickle
# 导入Counter类用于计算每个元素的频率
from collections import Counter

# 定义一个函数用于加载pickle文件
def load_pickle(filename):
    # 使用with语句打开文件，确保文件会被正确关闭
    with open(filename, 'rb') as f:
        # 使用pickle.load()方法加载pickle文件中的数据
        # 并指定编码为iso-8859-1，这可能是因为文件是在此编码下保存的
        data = pickle.load(f, encoding='iso-8859-1')
    # 返回加载的数据
    return data

# 定义一个函数用于根据qids分割数据
def split_data(total_data, qids):
    # 创建一个Counter对象来计算qids的频率
    result = Counter(qids)
    # 初始化两个列表来存储单选题和多选题的数据
    total_data_single = []
    total_data_multiple = []
    # 遍历总数据
    for data in total_data:
        # 如果qid在结果中只出现一次，则将其添加到单选题列表中
        if result[data[0][0]] == 1:
            total_data_single.append(data)
        # 否则，添加到多选题列表中
        else:
            total_data_multiple.append(data)
    # 返回单选题和多选题的数据列表
    return total_data_single, total_data_multiple

# 定义一个函数用于处理STAQC数据
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    # 打开文件并读取数据
    with open(filepath, 'r') as f:
        total_data = eval(f.read())
    # 从数据中提取qids
    qids = [data[0][0] for data in total_data]
    # 分割数据
    total_data_single, total_data_multiple = split_data(total_data, qids)
    # 保存单选题和多选题的数据
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))

# 定义一个函数用于处理大量数据
def data_large_processing(filepath, save_single_path, save_multiple_path):
    # 加载pickle文件中的数据
    total_data = load_pickle(filepath)
    # 从数据中提取qids
    qids = [data[0][0] for data in total_data]
    # 分割数据
    total_data_single, total_data_multiple = split_data(total_data, qids)
    # 保存单选题和多选题的数据，这次使用pickle格式
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)

# 定义一个函数用于将单选题未标记数据转换为标记数据
def single_unlabeled_to_labeled(input_path, output_path):
    # 加载pickle文件中的数据
    total_data = load_pickle(input_path)
    # 创建一个列表，其中每个元素是一个包含问题和标签的列表
    labels = [[data[0], 1] for data in total_data]
    # 对标签列表进行排序，首先按问题，然后按标签
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    # 保存标记后的数据
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))

        
# 如果此脚本作为主程序运行
if __name__ == "__main__":
    # 定义STAQC Python数据处理的相关路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    # 调用data_staqc_processing函数处理STAQC Python数据
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    # 定义STAQC SQL数据处理的相关路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    # 调用data_staqc_processing函数处理STAQC SQL数据
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 定义大型Python数据处理的相关路径
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    # 调用data_large_processing函数处理大型Python数据
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 定义大型SQL数据处理的相关路径
    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    # 调用data_large_processing函数处理大型SQL数据
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    # 定义大型SQL单选题数据转换为标记数据的路径
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    # 调用single_unlabeled_to_labeled函数将大型SQL单选题数据转换为标记数据
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    # 调用single_unlabeled_to_labeled函数将大型Python单选题数据转换为标记数据
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)

