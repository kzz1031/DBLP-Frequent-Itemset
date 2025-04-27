from typing import List, Set, Dict, Tuple
import os
from performance import PerformanceMonitor

root_path='/Users/Zhuanz/Documents/Junior/Data Mining'
result_path='/Users/Zhuanz/Documents/Junior/Data Mining/Result'

def ECLAT(dataset: List[List[str]], min_sup: int) -> Tuple[List[List[str]], Dict]:
    """
    ECLAT算法实现
    :param dataset: 事务数据集
    :param min_sup: 最小支持度计数
    :return: 频繁项集列表和性能监控结果
    """
    monitor = PerformanceMonitor()
    monitor.start()
    
    # 构建垂直数据格式
    tid_sets = {}
    for tid, transaction in enumerate(dataset):
        for item in transaction:
            if item not in tid_sets:
                tid_sets[item] = set()
            tid_sets[item].add(tid)
    
    # 筛选频繁1项集
    frequent_items = {item: tids for item, tids in tid_sets.items() 
                     if len(tids) >= min_sup}
    
    # 初始化结果列表
    result = [[item] for item in sorted(frequent_items.keys())]
    
    # 记录位图内存使用
    tid_sets_size = sum(len(tids) * 8 for tids in tid_sets.values()) / 1024 / 1024  # MB
    monitor.add_metric('bitmap_memory', tid_sets_size)
    monitor.record_memory()
    
    # 递归挖掘频繁项集
    eclat_recursive(frequent_items, min_sup, [], result)
    
    monitor.end()
    return result, monitor.get_results()

def eclat_recursive(tid_sets: Dict[str, Set[int]], min_sup: int, 
                   prefix: List[str], result: List[List[str]]):
    """递归挖掘频繁项集"""
    items = sorted(tid_sets.keys())
    
    for i, item_i in enumerate(items):
        # 构建新的前缀
        new_prefix = prefix + [item_i]
        i_tids = tid_sets[item_i]
        
        # 构建新的前缀的TID集
        new_tid_sets = {}
        for j in range(i + 1, len(items)):
            item_j = items[j]
            j_tids = tid_sets[item_j]
            # 计算交集
            intersection = i_tids & j_tids
            if len(intersection) >= min_sup:
                new_tid_sets[item_j] = intersection
        
        # 如果有新的频繁项集，继续递归
        if new_tid_sets:
            result.append(new_prefix)
            eclat_recursive(new_tid_sets, min_sup, new_prefix, result)

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
    frequent_itemsets, performance = ECLAT(sample, min_support_count)
    
    output_file = os.path.join(result_path, f'eclat_result.txt')
    
    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Sample size: {len(sample)} transactions\n')
        f.write(f'Minimum support count: {min_support_count}\n\n')
        f.write('Frequent itemsets found:\n')
        
        for itemset in frequent_itemsets:
            support_count = sum(1 for transaction in sample if all(item in transaction for item in itemset))
            support = support_count / len(sample)
            f.write(f'Itemset: {itemset}, Support: {support:.3f}, Count: {support_count}\n')
        
        f.write(f'\nTotal number of frequent itemsets: {len(frequent_itemsets)}\n\n')
        f.write(f'\nTotal number of frequent itemsets: {len(frequent_itemsets)}')
        f.write('\nPerformance Metrics:\n')
        f.write(f'Execution Time: {performance["execution_time"]:.8f} seconds\n')
        f.write(f'Peak Memory Usage: {performance["peak_memory"]:.8f} MB\n')
    
    print(f'Results have been saved to: {output_file}')