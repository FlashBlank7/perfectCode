# 导入所需的模块
import pickle
import multiprocessing
from python_structured import *  # 导入python_structured模块中的函数
from sqlang_structured import *  # 导入sqlang_structured模块中的函数

# 定义用于并行处理Python和SQL查询、代码和上下文的函数
def multipro_python_query(data_list):
    # 并行处理Python查询，返回解析后的查询结果列表
    return [python_query_parse(line) for line in data_list]

def multipro_python_code(data_list):
    # 并行处理Python代码，返回解析后的代码结果列表
    return [python_code_parse(line) for line in data_list]

def multipro_python_context(data_list):
    # 并行处理Python上下文，返回解析后的上下文结果列表
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])  # 如果遇到特殊标记'-10000'，则直接添加该标记
        else:
            result.append(python_context_parse(line))  # 否则解析上下文
    return result

def multipro_sqlang_query(data_list):
    # 并行处理SQL查询，返回解析后的查询结果列表
    return [sqlang_query_parse(line) for line in data_list]

def multipro_sqlang_code(data_list):
    # 并行处理SQL代码，返回解析后的代码结果列表
    return [sqlang_code_parse(line) for line in data_list]

def multipro_sqlang_context(data_list):
    # 并行处理SQL上下文，返回解析后的上下文结果列表
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])  # 如果遇到特殊标记'-10000'，则直接添加该标记
        else:
            result.append(sqlang_context_parse(line))  # 否则解析上下文
    return result

# 定义解析函数，用于处理数据并返回上下文、查询和代码数据
def parse(data_list, split_num, context_func, query_func, code_func):
    # 创建一个进程池
    pool = multiprocessing.Pool()
    # 将数据列表分割成多个子列表，每个子列表分配给一个进程处理
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    # 并行执行上下文、查询和代码的解析函数，收集结果
    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]  # 合并上下文数据
    print(f'context条数：{len(context_data)}')
    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]  # 合并查询数据
    print(f'query条数：{len(query_data)}')
    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]  # 合并代码数据
    print(f'code条数：{len(code_data)}')
    # 关闭进程池
    pool.close()
    pool.join()
    # 返回解析后的上下文、查询和代码数据
    return context_data, query_data, code_data

# 定义主函数，用于处理数据并保存结果
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    # 以二进制读取模式打开源数据文件
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)  # 加载数据
    # 调用解析函数处理数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query



if __name__ == '__main__':

    # 定义STAQC数据集中的Python代码的路径和保存路径
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    # 定义STAQC数据集中的SQL代码的路径和保存路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 调用主函数，处理STAQC数据集中的Python代码，参数包括语言类型、分割数量、路径和保存路径
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)

    # 调用主函数，处理STAQC数据集中的SQL代码，参数包括语言类型、分割数量、路径和保存路径
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    # 定义大型数据集中的Python代码的路径和保存路径
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    # 定义大型数据集中的SQL代码的路径和保存路径
    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    # 调用主函数，处理大型数据集中的Python代码，参数包括语言类型、分割数量、路径和保存路径
    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)

    # 调用主函数，处理大型数据集中的SQL代码，参数包括语言类型、分割数量、路径和保存路径
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)
