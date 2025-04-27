from typing import List, Dict, Set, Tuple
from collections import defaultdict
import os
from performance import PerformanceMonitor

root_path='/Users/Zhuanz/Documents/Junior/Data Mining'
result_path='/Users/Zhuanz/Documents/Junior/Data Mining/Result'

class FPNode:
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

def FPGrowth(dataset: List[List[str]], min_sup: int) -> Tuple[List[List[str]], Dict]:
    """
    FP-Growth算法实现
    :param dataset: 事务数据集
    :param min_sup: 最小支持度计数
    :return: 频繁项集列表和性能监控结果
    """
    monitor = PerformanceMonitor()
    monitor.start()
    
    # 第一次扫描，获取频繁1项集
    item_count = defaultdict(int)
    for transaction in dataset:
        for item in transaction:
            item_count[item] += 1
    
    # 筛选频繁1项集
    frequent_items = {k: v for k, v in item_count.items() if v >= min_sup}
    if not frequent_items:
        return [], {}
    
    # 构建FP树
    root = FPNode()
    header_table = {k: [v, None] for k, v in sorted(frequent_items.items(), 
                                                   key=lambda x: (-x[1], x[0]))}
    
    # 第二次扫描，构建FP树
    for transaction in dataset:
        # 筛选并排序频繁项
        filtered_tx = [item for item in transaction if item in frequent_items]
        filtered_tx.sort(key=lambda x: (-frequent_items[x], x))
        if filtered_tx:
            insert_tree(filtered_tx, root, header_table)
            monitor.record_memory()  # 记录树构建过程中的内存
    
    # 记录FP树节点数
    def count_nodes(node):
        count = 1
        for child in node.children.values():
            count += count_nodes(child)
        return count
    
    node_count = count_nodes(root)
    monitor.add_metric('node_count', node_count)
    monitor.add_metric('conditional_pattern_count', 0)
    
    # 挖掘频繁模式
    frequent_patterns = []
    
    def mine_fptree_with_monitor(header_table, min_sup, prefix, frequent_patterns):
        """挖掘FP树中的频繁模式（带性能监控）"""
        # 按支持度升序处理头表项
        sorted_items = sorted(header_table.items(), 
                            key=lambda x: (x[1][0], x[0]))
        
        for item, [count, node] in sorted_items:
            new_pattern = prefix + [item]
            frequent_patterns.append(new_pattern)
            
            # 收集条件模式基
            cond_pattern_bases = []
            while node:
                path = []
                parent = node.parent
                while parent and parent.item:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    cond_pattern_bases.append((path, node.count))
                node = node.next
            
            # 构建条件FP树
            if cond_pattern_bases:
                monitor.add_metric('conditional_pattern_count', len(cond_pattern_bases))
                monitor.record_memory()  # 记录条件模式基生成时的内存
                cond_tree_data = []
                for base_pattern, count in cond_pattern_bases:
                    cond_tree_data.extend([base_pattern] * count)
                FPGrowth(cond_tree_data, min_sup)
    
    mine_fptree_with_monitor(header_table, min_sup, [], frequent_patterns)
    monitor.end()
    return frequent_patterns, monitor.get_results()

def insert_tree(items: List[str], tree: FPNode, header_table: Dict):
    """插入事务到FP树"""
    if items:
        first_item = items[0]
        if first_item not in tree.children:
            tree.children[first_item] = FPNode(first_item, 1, tree)
            # 更新头表链表
            if header_table[first_item][1] is None:
                header_table[first_item][1] = tree.children[first_item]
            else:
                update_header_link(header_table[first_item][1], 
                                 tree.children[first_item])
        else:
            tree.children[first_item].count += 1
        # 递归插入剩余项
        insert_tree(items[1:], tree.children[first_item], header_table)

def update_header_link(start: FPNode, target: FPNode):
    """更新头表链表"""
    while start.next:
        start = start.next
    start.next = target

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
    frequent_itemsets, performance = FPGrowth(sample, min_support_count)
    
    output_file = os.path.join(result_path, f'fpgrowth_result.txt')
    
    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Sample size: {len(sample)} transactions\n')
        f.write(f'Minimum support count: {min_support_count}\n\n')
        f.write('Frequent itemsets found:\n')
        
        for itemset in frequent_itemsets:
            support_count = sum(1 for transaction in sample if all(item in transaction for item in itemset))
            support = support_count / len(sample)
            f.write(f'Itemset: {itemset}, Support: {support:.3f}, Count: {support_count}\n')
        
        f.write(f'\nTotal number of frequent itemsets: {len(frequent_itemsets)}\n')
        f.write('\nPerformance Metrics:\n')
        f.write('\nPerformance Metrics:\n')
        f.write(f'Execution Time: {performance["execution_time"]:.8f} seconds\n')
        f.write(f'Peak Memory Usage: {performance["peak_memory"]:.8f} MB\n')
    
    print(f'Results have been saved to: {output_file}')