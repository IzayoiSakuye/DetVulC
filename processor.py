# data/advanced_vuln_type_extractor.py
import re
import pandas as pd
from pathlib import Path


class Extractor:
    def __init__(self):
        # 定义详细的漏洞类型模式
        self.vuln_patterns = {
            # 格式字符串漏洞
            'Uncontrolled_Format_String': [
                r'Uncontrolled_Format_String',
                r'Format_String_Vulnerability',
                r'Uncontrolled.*Format.*String'
            ],
            'Format_String_Error': [
                r'Format_String_Error',
                r'Format_String_Bug'
            ],

            # 命令注入
            'OS_Command_Injection': [
                r'OS_Command_Injection',
                r'Command_Injection',
                r'Shell_Injection'
            ],

            # 缓冲区溢出
            'Buffer_Overflow': [
                r'Buffer_Overflow',
                r'BufferOverrun',
                r'Buffer_Overrun'
            ],
            'Stack_Based_Buffer_Overflow': [
                r'Stack_Based_Buffer_Overflow',
                r'Stack_Buffer_Overflow'
            ],
            'Heap_Based_Buffer_Overflow': [
                r'Heap_Based_Buffer_Overflow',
                r'Heap_Buffer_Overflow'
            ],

            # 整数溢出
            'Integer_Overflow': [
                r'Integer_Overflow',
                r'IntegerOverFlow'
            ],
            'Integer_Underflow': [
                r'Integer_Underflow',
                r'IntegerUnderflow'
            ],

            # 内存相关
            'Use_After_Free': [
                r'Use_After_Free',
                r'UseAfterFree'
            ],
            'Double_Free': [
                r'Double_Free',
                r'DoubleFree'
            ],
            'Memory_Leak': [
                r'Memory_Leak',
                r'MemoryLeak'
            ],

            # 输入验证
            'Improper_Input_Validation': [
                r'Improper_Input_Validation',
                r'Input_Validation_Error'
            ],

            # 路径遍历
            'Path_Traversal': [
                r'Path_Traversal',
                r'Directory_Traversal'
            ],

            # 竞态条件
            'Race_Condition': [
                r'Race_Condition',
                r'Time_of_Check_Time_of_Use'
            ]
        }

    def extract_detailed_vuln_types(self, file_path):
        """从文件路径提取详细的漏洞类型"""
        if not file_path:
            return []

        found_types = []

        # 提取所有匹配的漏洞类型
        for vuln_type, patterns in self.vuln_patterns.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    found_types.append(vuln_type)
                    break  # 避免重复添加同一类型

        # 如果没找到具体类型，尝试提取CWE编号
        if not found_types:
            cwe_match = re.search(r'CWE[-_\s]*(\d+)', file_path, re.IGNORECASE)
            if cwe_match:
                found_types.append(f"CWE-{cwe_match.group(1)}")

        return found_types if found_types else ["unknown"]

    def extract_cwe_number(self, file_path):
        """提取CWE编号"""
        if not file_path:
            return "unknown"

        cwe_match = re.search(r'CWE[-_\s]*(\d+)', file_path, re.IGNORECASE)
        if cwe_match:
            return f"CWE-{cwe_match.group(1)}"
        return "unknown"

    def extract_all_vuln_info(self, file_path, ir_code):
        """提取所有漏洞信息"""
        # 提取详细漏洞类型
        detailed_types = self.extract_detailed_vuln_types(file_path)

        # 提取CWE编号
        cwe_number = self.extract_cwe_number(file_path)

        # 从IR代码推断额外信息
        inferred_info = self._infer_from_ir(ir_code)

        return {
            'detailed_vuln_types': detailed_types,
            'primary_vuln_type': detailed_types[0] if detailed_types else "unknown",
            'cwe_number': cwe_number,
            'inferred_info': inferred_info
        }

    def _infer_from_ir(self, ir_code):
        """从IR代码推断漏洞信息"""
        inferred = {
            'risk_functions': [],
            'dangerous_patterns': [],
            'memory_operations': []
        }

        # 识别风险函数
        risk_functions = [
            (r'call.*system', 'system_call'),
            (r'call.*exec', 'exec_call'),
            (r'call.*gets', 'unsafe_input'),
            (r'call.*strcpy', 'unsafe_copy'),
            (r'call.*sprintf', 'unsafe_format'),
            (r'call.*printf.*%', 'format_string'),
            (r'call.*free', 'memory_free'),
            (r'call.*malloc', 'memory_alloc')
        ]

        for pattern, func_type in risk_functions:
            if re.search(pattern, ir_code):
                inferred['risk_functions'].append(func_type)

        # 识别危险模式
        dangerous_patterns = [
            (r'add.*nsw', 'integer_overflow'),
            (r'mul.*nsw', 'integer_overflow'),
            (r'getelementptr.*\[.*\].*i64.*add', 'buffer_overflow'),
            (r'store.*i8\*.*load.*i8\*', 'memory_corruption')
        ]

        for pattern, pattern_type in dangerous_patterns:
            if re.search(pattern, ir_code):
                inferred['dangerous_patterns'].append(pattern_type)

        # 识别内存操作
        memory_ops = [
            (r'alloca', 'stack_allocation'),
            (r'call.*malloc', 'heap_allocation'),
            (r'call.*free', 'memory_deallocation'),
            (r'load', 'memory_read'),
            (r'store', 'memory_write')
        ]

        for pattern, op_type in memory_ops:
            if re.search(pattern, ir_code):
                inferred['memory_operations'].append(op_type)

        return inferred


# 改进的处理器
class Processor:
    def __init__(self, data_dir="data/iSeVCs_for_train_programs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/processed_output")
        self.output_dir.mkdir(exist_ok=True)
        self.vuln_extractor = Extractor()

    def parse_single_sample(self, sample_content, source_file=""):
        """解析单个样本并提取详细漏洞信息"""
        lines = sample_content.strip().split('\n')
        if not lines:
            return None

        # 查找标签行
        label_line = None
        ir_start_idx = 0

        for i, line in enumerate(lines):
            line = line.strip()
            # 匹配标签+文件路径格式
            if re.match(r'^\d+\s+\S', line):
                label_line = line
                ir_start_idx = i + 1
                break

        if label_line:
            # 解析标签和文件路径
            parts = label_line.split(' ', 1)
            label = int(parts[0])
            file_path = parts[1] if len(parts) > 1 else ""
        else:
            # 没有找到标签行
            label = 0  # 默认安全
            file_path = ""
            ir_start_idx = 0

        # 提取IR代码
        ir_lines = []
        for line in lines[ir_start_idx:]:
            line = line.strip()
            # 跳过空行和分隔符
            if line and not re.match(r'^\s*\[.*\]\s*$', line) and not line.startswith('----'):
                ir_lines.append(line)

        ir_code = '\n'.join(ir_lines)

        # 提取详细的漏洞信息
        vuln_info = self.vuln_extractor.extract_all_vuln_info(file_path, ir_code)

        return {
            'label': label,
            'file_path': file_path,
            'ir_code': ir_code,
            'source_file': source_file,
            # 详细漏洞信息
            'detailed_vuln_types': '|'.join(vuln_info['detailed_vuln_types']),
            'primary_vuln_type': vuln_info['primary_vuln_type'],
            'cwe_number': vuln_info['cwe_number'],
            'inferred_risk_functions': '|'.join(vuln_info['inferred_info']['risk_functions']),
            'inferred_dangerous_patterns': '|'.join(vuln_info['inferred_info']['dangerous_patterns']),
            'inferred_memory_operations': '|'.join(vuln_info['inferred_info']['memory_operations'])
        }

    def process_all_files(self):
        """处理并合并所有文件"""
        # 查找所有txt文件
        txt_files = list(self.data_dir.glob("*.txt"))
        print(f"找到 {len(txt_files)} 个txt文件")

        all_data = []

        for txt_file in txt_files:
            print(f"处理文件: {txt_file.name}")

            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 分割样本
                samples = self._split_samples(content)

                # 解析每个样本
                for sample in samples:
                    if sample.strip():
                        try:
                            parsed_data = self.parse_single_sample(sample, txt_file.name)
                            if parsed_data and parsed_data['ir_code'].strip():
                                all_data.append(parsed_data)
                        except Exception as e:
                            print(f"  解析样本失败: {e}")
                            continue

                print(f"  从 {txt_file.name} 解析了 {len([s for s in samples if s.strip()])} 个样本")

            except Exception as e:
                print(f"处理文件 {txt_file} 失败: {e}")
                continue

        print(f"\n总共解析了 {len(all_data)} 个样本")

        # 保存数据
        if all_data:
            self._save_and_analyze(all_data)

        return all_data

    def _split_samples(self, content):
        """分割样本"""
        # 查找分隔符
        separators = []

        # 查找 [数字] 格式的分隔符
        bracket_matches = list(re.finditer(r'^\s*\[(\d+)\]\s*$', content, re.MULTILINE))

        # 查找 ---- 格式的分隔符
        dash_matches = list(re.finditer(r'^\s*----+\s*$', content, re.MULTILINE))

        # 合并并排序
        all_separators = []
        for match in bracket_matches:
            all_separators.append(('bracket', match.start(), match.end()))
        for match in dash_matches:
            all_separators.append(('dash', match.start(), match.end()))

        all_separators.sort(key=lambda x: x[1])

        if not all_separators:
            return [content.strip()]

        # 分割内容
        samples = []
        start_pos = 0

        for sep_type, sep_start, sep_end in all_separators:
            sample_content = content[start_pos:sep_start].strip()
            if sample_content:
                samples.append(sample_content)
            start_pos = sep_end

        # 添加最后的内容
        last_content = content[start_pos:].strip()
        if last_content:
            samples.append(last_content)

        return samples

    def _save_and_analyze(self, all_data):
        """保存和分析数据"""
        # 转换为DataFrame
        df = pd.DataFrame(all_data)

        # 保存主文件
        output_file = self.output_dir / "llvm_data.csv"
        df.to_csv(output_file, index=False)
        print(f"数据已保存到: {output_file}")

# 运行完整处理
def run():
    """运行完整的数据处理"""
    processor = Processor()
    data = processor.process_all_files()
    return data


# 主函数
if __name__ == "__main__":
    print("\n" + "=" * 60 + "\n")
    # 完整处理
    print("开始处理所有数据文件...")
    data = run()
    if data:
        print(f"\n✅ 成功处理 {len(data)} 个样本!")

