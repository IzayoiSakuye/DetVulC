#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
processor_extractor_multi_fixed.py
支持你的精确多样本格式：

样本格式：
(1) 以如下行开头：
    1 000/119/416/arr/CWE78_OS_Command_Injection__char_file_execl_61a_0_dataBuffer[9999].final.ll
(2) 以如下行结尾：
    -------------------------

每个样本单独输出为 CSV 行
"""

import os
import re
import csv
import argparse

# ===============================================================
#  疏松但准确的漏洞类型提取器（支持多标签）
# ===============================================================

class Extractor:
    def __init__(self):
        self.vuln_patterns = {
            'OS_Command_Injection': [
                r'OS[_\s\-]*Command[_\s\-]*Injection',
                r'Command[_\s\-]*Injection',
            ],
            'Buffer_Overflow': [
                r'Buffer[_\s\-]*Overflow', r'strcpy', r'ncat', r'memcpy'
            ],
            'Use_After_Free': [
                r'Use[_\s\-]*After[_\s\-]*Free'
            ],
            'Integer_Overflow': [
                r'Integer[_\s\-]*Overflow'
            ]
        }

        self.cwe_map = {
            '78': 'OS_Command_Injection',
            '119': 'Buffer_Overflow',
            '416': 'Use_After_Free',
            '190': 'Integer_Overflow'
        }

    def extract_cwe(self, header_line):
        """从样本头中提取 CWE 编号"""
        m = re.search(r'CWE[_\-]?(\d+)', header_line)
        if m:
            return f"CWE-{m.group(1)}"
        return "unknown"

    def extract_path(self, header_line):
        """样本头本身包含路径"""
        parts = header_line.strip().split(None, 1)  # split only two parts
        if len(parts) == 2:
            return parts[1]
        return header_line.strip()

    def extract_vuln_types(self, header_line, ir_block):
        """从样本头 + IR 同时挖掘漏洞类型"""
        found = []

        # 1) 从文件名（CWE + 描述）匹配
        for vtype, patterns in self.vuln_patterns.items():
            for p in patterns:
                if re.search(p, header_line, re.IGNORECASE):
                    found.append(vtype)
                    break

        # 2) 从 CWE 映射
        m = re.search(r'CWE[_\-]?(\d+)', header_line)
        if m:
            cwe_id = m.group(1)
            if cwe_id in self.cwe_map:
                found.append(self.cwe_map[cwe_id])

        # 3) 从 IR 内容匹配
        for vtype, patterns in self.vuln_patterns.items():
            for p in patterns:
                if re.search(p, ir_block, re.IGNORECASE):
                    found.append(vtype)
                    break

        # 去重
        result = []
        for x in found:
            if x not in result:
                result.append(x)

        return result if result else ["unknown"]

class SampleSplitter:
    def __init__(self):
        # 样本头格式，例如：
        # 1 000/119/416/arr/CWE78_......
        self.header_re = re.compile(
            r'^\s*\d+\s+.+CWE[\-_]?\d+.*$', re.IGNORECASE
        )

        self.end_re = re.compile(r'^\s*-{5,}\s*$')

    def split(self, text):
        """
        返回列表：[ (header, llvm_text), ... ]
        """
        lines = text.splitlines()
        samples = []

        current_header = None
        buffer = []

        for ln in lines:
            if self.header_re.match(ln):
                # 若已有样本缓冲区，则先结束该样本
                if current_header is not None and buffer:
                    samples.append((current_header, "\n".join(buffer)))
                    buffer = []

                current_header = ln.strip()
                continue

            # 样本结束线
            if self.end_re.match(ln):
                if current_header is not None:
                    samples.append((current_header, "\n".join(buffer)))
                    current_header = None
                    buffer = []
                continue

            # 累积样本内容
            if current_header is not None:
                buffer.append(ln)

        # 尾部 flush
        if current_header is not None and buffer:
            samples.append((current_header, "\n".join(buffer)))

        return samples

class Processor:
    def __init__(self):
        self.extractor = Extractor()
        self.splitter = SampleSplitter()
        self.rows = []

    def process_file(self, file_path):
        try:
            text = open(file_path, "r", encoding="utf-8", errors="ignore").read()
        except:
            return

        samples = self.splitter.split(text)

        for idx, (header, ir) in enumerate(samples, 1):
            sample_path = self.extractor.extract_path(header)
            cwe = self.extractor.extract_cwe(header)
            vuln_types = self.extractor.extract_vuln_types(header, ir)

            self.rows.append({
                "sample_id": len(self.rows)+1,
                "source_file": file_path,
                "sample_path": sample_path,
                "cwe_number": cwe,
                "detailed_vuln_types": "|".join(vuln_types),
                "ir_snippet": ir[:500].replace("\n", "\\n")
            })

    def process_directory(self, root, ext=".txt"):
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(ext):
                    self.process_file(os.path.join(dirpath, f))

    def save_csv(self, out_csv):
        keys = [
            "sample_id", "source_file", "sample_path",
            "cwe_number", "detailed_vuln_types", "ir_snippet"
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    P = Processor()
    P.process_directory(args.input_dir)
    P.save_csv(args.output)

    print("处理完成，共生成样本：", len(P.rows))
