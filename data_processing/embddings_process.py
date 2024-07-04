# 导入所需的库
import pickle
import numpy as np
from gensim.models import KeyedVectors

# 将词向量文件保存为二进制文件
def trans_bin(path1, path2):
    # 加载文本格式的词向量
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 初始化相似度计算，并保存为二进制文件
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)

# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())
    
    # 定义词典中的特殊词汇
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # PAD_ID, SOS_ID, EOS_ID, UNK_ID
    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]
    
    # 加载词向量
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            fail_word.append(word)
    
    # 转换为NumPy数组
    word_vectors = np.array(word_vectors)
    # 反转词典键值对
    word_dict = dict(map(reversed, enumerate(word_dict)))
    
    # 保存词向量和词典
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)
    print("完成")


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    # 用于存储词在词典中的位置
    location = []

    # 如果类型是'code'
    if type == 'code':
        # 在位置列表中添加一个1
        location.append(1)
        # 获取文本的长度
        len_c = len(text)
        # 如果文本长度加1小于350
        if len_c + 1 < 350:
            # 如果文本长度为1且第一个词是特殊词-1000
            if len_c == 1 and text[0] == '-1000':
                # 在位置列表中添加一个2
                location.append(2)
            else:
                # 遍历文本
                for i in range(0, len_c):
                    # 获取词在词典中的索引，如果不在词典中则使用'UNK'的索引
                    index = word_dict.get(text[i], word_dict['UNK'])
                    # 将索引添加到位置列表中
                    location.append(index)
                # 在位置列表中添加一个2，表示代码的结束
                location.append(2)
        else:
            # 如果文本长度超过348，则只取前348个词进行处理
            for i in range(0, 348):
                # 获取词在词典中的索引，如果不在词典中则使用'UNK'的索引
                index = word_dict.get(text[i], word_dict['UNK'])
                # 将索引添加到位置列表中
                location.append(index)
            # 在位置列表中添加一个2，表示代码的结束
            location.append(2)
    else:
        # 处理文本类型的文本
        if len(text) == 0:
            # 如果文本为空，则在位置列表中添加一个0
            location.append(0)
        elif text[0] == '-10000':
            # 如果文本以特殊词-10000开头，则在位置列表中添加一个0
            location.append(0)
        else:
            # 遍历文本
            for i in range(0, len(text)):
                # 获取词在词典中的索引，如果不在词典中则使用'UNK'的索引
                index = word_dict.get(text[i], word_dict['UNK'])
                # 将索引添加到位置列表中
                location.append(index)
    # 返回位置列表
    return location



# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    # 以二进制读取方式打开词典文件
    with open(word_dict_path, 'rb') as f:
        # 加载词典
        word_dict = pickle.load(f)
    # 以文本读取方式打开类型文件
    with open(type_path, 'r') as f:
        # 读取并评估文件内容，得到语料库
        corpus = eval(f.read())
    
    # 初始化总数据列表
    total_data = []
    # 遍历语料库
    for i in range(len(corpus)):
        # 获取每条数据的query id
        qid = corpus[i][0]
        # 获取Si的词列表位置
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        # 获取Si1的词列表位置
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        # 获取代码的词列表位置
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)
        # 获取查询的词列表位置
        query_word_list = get_index('text', corpus[i][3], word_dict)
        # 定义块的长度为4
        block_length = 4
        # 定义标签为0
        label = 0
        
        # 如果Si的词列表长度超过100，则截断至100，否则补充0至100
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        # 如果Si1的词列表长度超过100，则截断至100，否则补充0至100
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (100 - len(Si1_word_list))
        # 如果代码的词列表长度不足350，则补充0至350
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        # 如果查询的词列表长度超过25，则截断至25，否则补充0至25
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (25 - len(query_word_list))
        
        # 构建一个数据条目，包括query id、Si和Si1的词列表、代码、查询词列表、块长度和标签
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        # 将数据条目添加到总数据列表
        total_data.append(one_data)
    
    # 以二进制写入方式打开最终类型文件
    with open(final_type_path, 'wb') as file:
        # 将总数据列表序列化并写入文件
        pickle.dump(total_data, file)


if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # ==========================最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================

    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)
