# gnn_vuln_scanner/graph_builder.py
import torch
import pandas as pd
import networkx as nx
import re
from torch_geometric.data import Data
from tqdm import tqdm
import pickle


class LLVMASTGraphBuilder:
    """将LLVM IR代码转换为AST图"""

    def __init__(self):
        # LLVM IR的关键字模式
        self.instruction_patterns = {
            'call': r'call\s+(?:.*?)@(\w+)',
            'store': r'store\s+',
            'load': r'load\s+',
            'alloca': r'alloca\s+',
            'getelementptr': r'getelementptr\s+',
            'phi': r'phi\s+',
            'select': r'select\s+'
        }

        # 危险函数列表
        self.dangerous_functions = {
            'system', 'exec', 'execl', 'execv', 'popen', 'gets',
            'strcpy', 'strcat', 'sprintf', 'scanf', 'printf'
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
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i + 1], edge_type='control_flow')

        return G

    def _extract_node_features(self, line):
        """提取节点特征"""
        features = {
            'instruction_type': 'unknown',
            'has_dangerous_call': False,
            'line_length': len(line),
            'num_operands': len(line.split(',')),
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

    def graph_to_pyg_data(self, G, label):
        """将NetworkX图转换为PyTorch Geometric Data对象"""
        # 节点特征矩阵
        node_features = []
        for node_id in sorted(G.nodes()):
            features = G.nodes[node_id]['features']
            # 将特征转换为数值向量
            feature_vector = self._features_to_vector(features)
            node_features.append(feature_vector)

        x = torch.FloatTensor(node_features)

        # 边索引
        edge_index = []
        edge_attr = []
        for edge in G.edges(data=True):
            src, dst, attr = edge
            edge_index.append([src, dst])
            # 边属性（简化处理）
            edge_type_map = {'control_flow': 0, 'data_flow': 1, 'call': 2}
            edge_attr.append(edge_type_map.get(attr.get('edge_type', 'control_flow'), 0))

        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.LongTensor(edge_attr) if edge_attr else None

        # 标签
        y = torch.LongTensor([label])

        # 全局池化所需的批次信息（单图）
        batch = torch.zeros(x.size(0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)

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
            float(features['line_length']) / 1000.0,  # 归一化
            float(features['num_operands']) / 10.0,  # 归一化
            float(features['contains_ptr']),
            float(features['contains_array'])
        ]

        return inst_type_vec + other_features


class GraphDatasetBuilder:
    """图数据集构建器"""

    def __init__(self):
        self.ast_builder = LLVMASTGraphBuilder()

    def build_from_csv(self, csv_file, output_file=None):
        """从CSV文件构建图数据集"""

        # 读取CSV数据
        df = pd.read_csv(csv_file)
        print(f"原始数据: {len(df)} 条记录")

        # 清理数据
        df = df.dropna(subset=['ir_code', 'label'])
        df = df[df['ir_code'].str.len() > 10]
        print(f"清理后数据: {len(df)} 条记录")

        graph_data_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="构建图"):
            try:
                # 构建AST图
                G = self.ast_builder.build_ast_graph(row['ir_code'])

                # 转换为PyG Data对象
                pyg_data = self.ast_builder.graph_to_pyg_data(G, row['label'])

                # 添加元信息
                pyg_data.metadata = {
                    'file_path': row.get('file_path', ''),
                    'primary_vuln_type': row.get('primary_vuln_type', 'unknown'),
                    'detailed_vuln_types': row.get('detailed_vuln_types', 'unknown')
                }

                graph_data_list.append(pyg_data)

            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                continue

        print(f"图数据集构建完成: {len(graph_data_list)} 个图")

        # 保存图数据集
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(graph_data_list, f)
            print(f"图数据集已保存到: {output_file}")

        return graph_data_list


def main_build_graphs():
    """构建图数据集的主函数"""
    csv_file = "data/llvm_data.csv"
    output_file = "data/graph_dataset.pkl"

    try:
        builder = GraphDatasetBuilder()
        graph_data_list = builder.build_from_csv(csv_file, output_file)
        print(f"成功构建 {len(graph_data_list)} 个图!")
        return graph_data_list
    except Exception as e:
        print(f"构建图数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main_build_graphs()
