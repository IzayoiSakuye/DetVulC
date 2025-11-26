# gnn_vuln_scanner/multilabel_graph_builder.py
import torch
import pandas as pd
import networkx as nx
import re
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import ast
import numpy as np


class MultiLabelGraphBuilder:
    """多标签图数据构建器"""

    def __init__(self):
        # 定义所有可能的漏洞类型
        self.vuln_types = [
            'buffer_overflow',
            'use_after_free',
            'double_free',
            'null_pointer',
            'integer_overflow',
            'format_string',
            'command_injection',
            'path_traversal',
            'race_condition',
            'memory_leak'
        ]

        # 创建漏洞类型到索引的映射
        self.vuln_type_to_idx = {vuln: idx for idx, vuln in enumerate(self.vuln_types)}
        self.num_classes = len(self.vuln_types)

        # LLVM IR模式
        self.instruction_patterns = {
            'call': r'call\s+(?:.*?)@(\w+)',
            'store': r'store\s+',
            'load': r'load\s+',
            'alloca': r'alloca\s+',
            'getelementptr': r'getelementptr\s+',
            'phi': r'phi\s+',
            'select': r'select\s+'
        }

        self.dangerous_functions = {
            'system', 'exec', 'execl', 'execv', 'popen', 'gets',
            'strcpy', 'strcat', 'sprintf', 'scanf', 'printf',
            'malloc', 'free', 'realloc'
        }

    def build_ast_graph(self, ir_code):
        """将LLVM IR代码构建为AST图"""
        # 创建图
        G = nx.DiGraph()
        node_id = 0
        lines = ir_code.strip().split('\n')

        # 第一遍：创建节点
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(';'):  # 跳过注释和空行
                continue

            # 提取节点特征
            features = self._extract_node_features(line)
            G.add_node(node_id,
                       line=line,
                       features=features,
                       line_num=i)
            node_id += 1

        # 第二遍：创建边（控制流和数据流）
        nodes = list(G.nodes())
        if len(nodes) > 1:
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1], edge_type='control_flow')
        elif len(nodes) == 1:
            # 自环边
            G.add_edge(nodes[0], nodes[0], edge_type='control_flow')

        return G

    def _extract_node_features(self, line):
        """提取节点特征"""
        features = {
            'instruction_type': 'unknown',
            'has_dangerous_call': False,
            'line_length': len(line),
            'num_operands': len(line.split(',')) if ',' in line else 1,
            'contains_ptr': '*' in line,
            'contains_array': '[' in line or ']' in line
        }

        # 识别指令类型
        for inst_type, pattern in self.instruction_patterns.items():
            if re.search(pattern, line):
                features['instruction_type'] = inst_type
                # 检查是否调用了危险函数
                if inst_type == 'call':
                    match = re.search(pattern, line)
                    if match and match.group(1) in self.dangerous_functions:
                        features['has_dangerous_call'] = True
                break

        return features

    def parse_detailed_vuln_types(self, detailed_vuln_str):
        """解析详细的漏洞类型字符串"""
        try:
            # 尝试解析JSON格式
            if isinstance(detailed_vuln_str, str):
                vuln_list = ast.literal_eval(detailed_vuln_str)
                if isinstance(vuln_list, list):
                    return vuln_list
        except:
            # 如果解析失败，尝试按逗号分割
            if isinstance(detailed_vuln_str, str):
                return [v.strip() for v in detailed_vuln_str.split(',') if v.strip()]

        return []

    def create_multilabel(self, detailed_vuln_types):
        """创建多标签向量"""
        # 初始化零向量
        label_vector = [0] * self.num_classes

        # 解析漏洞类型
        vuln_list = self.parse_detailed_vuln_types(detailed_vuln_types)

        # 设置对应位置为1
        for vuln_type in vuln_list:
            vuln_type = vuln_type.lower().strip()
            if vuln_type in self.vuln_type_to_idx:
                idx = self.vuln_type_to_idx[vuln_type]
                label_vector[idx] = 1

        return label_vector

    def graph_to_pyg_data(self, G, multilabel):
        """将NetworkX图转换为PyTorch Geometric Data对象"""
        if len(G.nodes()) == 0:
            # 创建一个默认节点以防图为空
            G.add_node(0, features=self._extract_node_features(""), line="")

        # 节点特征矩阵
        node_features = []
        for node_id in sorted(G.nodes()):
            features = G.nodes[node_id].get('features', self._extract_node_features(""))
            # 将特征转换为数值向量
            feature_vector = self._features_to_vector(features)
            node_features.append(feature_vector)

        x = torch.FloatTensor(node_features)

        # 边索引
        if len(G.edges()) > 0:
            edge_index = []
            for edge in G.edges():
                src, dst = edge
                edge_index.append([src, dst])
            edge_index = torch.LongTensor(edge_index).t().contiguous()
        else:
            # 如果没有边，创建自环边
            edge_index = torch.LongTensor([[0], [0]]) if len(G.nodes()) > 0 else torch.LongTensor([[], []])

        # 多标签
        y = torch.FloatTensor(multilabel)

        # 全局池化所需的批次信息（单图）
        batch = torch.zeros(x.size(0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, batch=batch)

    def _features_to_vector(self, features):
        """将特征字典转换为数值向量"""
        # 指令类型编码
        instruction_types = ['unknown', 'call', 'store', 'load', 'alloca',
                             'getelementptr', 'phi', 'select']
        inst_type_vec = [1 if features['instruction_type'] == t else 0
                         for t in instruction_types]

        # 其他数值特征
        other_features = [
            float(features['has_dangerous_call']),
            min(float(features['line_length']) / 1000.0, 1.0),  # 归一化并限制范围
            min(float(features['num_operands']) / 10.0, 1.0),  # 归一化并限制范围
            float(features['contains_ptr']),
            float(features['contains_array'])
        ]

        return inst_type_vec + other_features

    def build_from_csv(self, csv_file, output_file=None):
        """从CSV构建多标签图数据集"""
        print("🏗️  开始构建多标签图数据集...")

        # 读取数据
        df = pd.read_csv(csv_file)
        print(f"📊 原始数据: {len(df)} 条记录")

        # 清理数据
        df = df.dropna(subset=['ir_code'])
        df = df[df['ir_code'].str.len() > 10]
        print(f"🧹 清理后数据: {len(df)} 条记录")

        # 检查必需列
        required_columns = ['ir_code', 'detailed_vuln_types']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️  缺少必需列: {missing_columns}")
            # 创建默认的detailed_vuln_types列
            if 'detailed_vuln_types' not in df.columns:
                df['detailed_vuln_types'] = df.get('label', 0).apply(
                    lambda x: "['safe']" if x == 0 else "['buffer_overflow']"
                )

        graph_data_list = []
        failed_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="构建图"):
            try:
                # 构建AST图
                G = self.build_ast_graph(row['ir_code'])

                # 创建多标签
                multilabel = self.create_multilabel(row.get('detailed_vuln_types', "['safe']"))

                # 转换为PyG Data对象
                pyg_data = self.graph_to_pyg_data(G, multilabel)

                # 验证数据
                if pyg_data.x.size(0) == 0:
                    print(f"  ⚠️  跳过空图: 样本 {idx}")
                    failed_count += 1
                    continue

                # 添加元信息
                pyg_data.metadata = {
                    'index': idx,
                    'file_path': row.get('file_path', ''),
                    'primary_vuln_type': row.get('primary_vuln_type', 'safe'),
                    'detailed_vuln_types': row.get('detailed_vuln_types', "['safe']"),
                    'original_label': row.get('label', 0)  # 原始的二分类标签
                }

                graph_data_list.append(pyg_data)

            except Exception as e:
                print(f"  ⚠️  处理样本 {idx} 时出错: {e}")
                failed_count += 1
                continue

        print(f"✅ 多标签图数据集构建完成: {len(graph_data_list)} 个图 (失败: {failed_count})")

        # 统计标签分布
        if graph_data_list:
            self.analyze_label_distribution(graph_data_list)

        # 保存数据集
        if output_file and graph_data_list:
            with open(output_file, 'wb') as f:
                pickle.dump(graph_data_list, f)
            print(f"💾 图数据集已保存到: {output_file}")

        return graph_data_list

    def analyze_label_distribution(self, graph_data_list):
        """分析标签分布"""
        label_counts = [0] * self.num_classes
        sample_counts = []

        for data in graph_data_list:
            labels = data.y.tolist()
            sample_counts.append(sum(labels))  # 每个样本的漏洞数量
            for j, label in enumerate(labels):
                label_counts[j] += label

        print("📈 标签分布统计:")
        for i, (vuln_type, count) in enumerate(zip(self.vuln_types, label_counts)):
            print(f"   {vuln_type}: {int(count)} 个样本")

        if sample_counts:
            print(f"📊 平均每样本漏洞数: {sum(sample_counts) / len(sample_counts):.2f}")
            print(f"📊 样本漏洞数分布: {sorted(list(set(sample_counts)))}")


def main_build_multilabel_graphs():
    """构建多标签图数据集的主函数"""
    csv_file = "data/llvm_data.csv"
    output_file = "data/graph_dataset.pkl"

    try:
        builder = MultiLabelGraphBuilder()
        graph_data_list = builder.build_from_csv(csv_file, output_file)
        print(f"🎉 成功构建 {len(graph_data_list)} 个多标签图!")
        return graph_data_list
    except Exception as e:
        print(f"❌ 构建多标签图数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main_build_multilabel_graphs()
