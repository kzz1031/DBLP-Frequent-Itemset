from typing import List, Tuple, Dict, Set
import os
from datetime import datetime
from performance import PerformanceMonitor

root_path='/Users/Zhuanz/Documents/Junior/Data Mining'
result_path='/Users/Zhuanz/Documents/Junior/Data Mining/Result'

def Apriori(dataset: List[List[str]], min_sup: int) -> Tuple[List[List[str]], Dict]:
    """
    Apriori算法实现
    :param dataset: 事务数据集
    :param min_sup: 最小支持度计数
    :return: 频繁项集列表和性能监控结果
    """
    monitor = PerformanceMonitor()
    monitor.start()

    # 创建项的支持度计数字典
    item_count = {}
    for transaction in dataset:
        for item in transaction:
            item_count[item] = item_count.get(item, 0) + 1
    
    # 生成频繁1项集
    L1 = [[item] for item, count in item_count.items() if count >= min_sup]
    L1.sort()
    
    # 存储所有频繁项集
    L = [L1]
    k = 1
    candidate_count = 0
    
    while L[k-1]:
        Ck = apriori_gen(L[k-1], k)  # 生成候选项集
        candidate_count += len(Ck)
        monitor.add_metric('candidate_count', len(Ck))
        monitor.record_memory()
        
        # 计算候选项集的支持度
        item_count = {}
        for transaction in dataset:
            for candidate in Ck:
                if all(item in transaction for item in candidate):
                    item_count[tuple(candidate)] = item_count.get(tuple(candidate), 0) + 1
        
        # 筛选频繁项集
        Lk = [list(candidate) for candidate, count in item_count.items() if count >= min_sup]
        Lk.sort()
        
        if Lk:
            L.append(Lk)
            k += 1
        else:
            break
    
    # 合并所有频繁项集
    monitor.end()
    return [item for sublist in L for item in sublist], monitor.get_results()

def apriori_gen(Lk_1: List[List[str]], k: int) -> List[List[str]]:
    """
    生成候选k项集
    :param Lk_1: k-1项频繁项集列表
    :param k: 当前项集大小
    :return: 候选k项集列表
    """
    candidates = []
    for i in range(len(Lk_1)):
        for j in range(i+1, len(Lk_1)):
            # 比较前k-2个元素是否相同
            if Lk_1[i][:k-2] == Lk_1[j][:k-2]:
                candidate = sorted(list(set(Lk_1[i] + Lk_1[j])))
                if len(candidate) == k:
                    # 检查所有k-1子集是否都是频繁的
                    is_valid = True
                    for m in range(len(candidate)):
                        subset = candidate[:m] + candidate[m+1:]
                        if subset not in Lk_1:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.append(candidate)
    return candidates

if __name__ == '__main__':
    print('Loading the dataset...')
    with open(root_path+'/authors_encoded.txt','r') as f:
        f_lines = list(line for line in (l.strip() for l in f) if line)
        dataset = []
        for line in f_lines:
            # 将每行文本分割成项目列表
            items = line.split()
            if items:  # 确保不是空行
                dataset.append(items)

    # 取前10000条数据作为样本
    sample = dataset[:10000]
    print(f'Sample size: {len(sample)} transactions')

    # 设置最小支持度计数（这里设置为5）
    min_support_count = 5
    print(f'Minimum support count: {min_support_count}')

    print('Start mining frequent itemsets...')
    frequent_itemsets, performance = Apriori(sample, min_support_count)
    
    output_file = os.path.join(result_path, f'apriory_result.txt')
    
    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Sample size: {len(sample)} transactions\n')
        f.write(f'Minimum support count: {min_support_count}\n\n')
        f.write('Frequent itemsets found:\n')
        
        for itemset in frequent_itemsets:
            support_count = sum(1 for transaction in sample if all(item in transaction for item in itemset))
            support = support_count / len(sample)
            f.write(f'Itemset: {itemset}, Support: {support:.3f}, Count: {support_count}\n')
        
        f.write(f'\nTotal number of frequent itemsets: {len(frequent_itemsets)}')
        f.write('\nPerformance Metrics:\n')
        f.write(f'Execution Time: {performance["execution_time"]:.8f} seconds\n')
        f.write(f'Peak Memory Usage: {performance["peak_memory"]:.8f} MB\n')
        f.write(f'Total Candidate Count: {performance["metrics"]["candidate_count"]}\n')
    
    print(f'Results have been saved to: {output_file}')
