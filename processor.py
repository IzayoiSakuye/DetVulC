#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse


class Extractor:
    def __init__(self):
        # CWE 编号到漏洞类型的映射
        self.cwe_map = {
            '78': 'OS_Command_Injection',
            '89': 'SQL_Injection',
            '79': 'Cross_Site_Scripting',
            '90': 'LDAP_Injection',
            '91': 'XML_Injection',
            '94': 'Code_Injection',
            '134': 'Format_String_Vulnerability',
            '611': 'XML_External_Entity_Reference',
            '918': 'Server-Side_Request_Forgery',
            '119': 'Buffer_Overflow',
            '120': 'Buffer_Copy_without_Checking_Size',
            '121': 'Stack_based_Buffer_Overflow',
            '122': 'Heap_based_Buffer_Overflow',
            '124': 'Buffer_Underwrite',
            '125': 'Out-of-bounds_Read',
            '126': 'Buffer_Overread',
            '127': 'Buffer_Underread',
            '129': 'Improper_Validation_of_Array_Index',
            '787': 'Out-of-bounds_Write',
            '415': 'Double_Free',
            '416': 'Use_After_Free',
            '401': 'Missing_Release_of_Memory',
            '476': 'NULL_Pointer_Dereference',
            '824': 'Access_of_Uninitialized_Pointer',
            '825': 'Expired_Pointer_Dereference',
            '20': 'Improper_Input_Validation',
            '22': 'Path_Traversal',
            '23': 'Relative_Path_Traversal',
            '36': 'Absolute_Path_Traversal',
            '73': 'External_Control_of_File_Name_or_Path',
            '400': 'Uncontrolled_Resource_Consumption',
            '434': 'Unrestricted_Upload_of_File',
            '502': 'Deserialization_of_Untrusted_Data',
            '776': 'Improper_Restriction_of_Recursive_Entity_References',
            '835': 'Loop_with_Unreachable_Exit_Condition',
            '190': 'Integer_Overflow',
            '191': 'Integer_Underflow',
            '192': 'Integer_Coercion_Error',
            '197': 'Numeric_Truncation_Error',
            '681': 'Incorrect_Conversion_between_Numeric_Types',
            '682': 'Incorrect_Calculation',
            '259': 'Use_of_Hard-coded_Password',
            '276': 'Incorrect_Default_Permissions',
            '284': 'Improper_Access_Control',
            '287': 'Improper_Authentication',
            '306': 'Missing_Authentication_for_Critical_Function',
            '321': 'Use_of_Hard-coded_Cryptographic_Key',
            '327': 'Use_of_a_Broken_or_Risky_Cryptographic_Algorithm',
            '330': 'Use_of_Insufficiently_Random_Values',
            '862': 'Missing_Authorization',
            '863': 'Incorrect_Authorization',
            '200': 'Information_Exposure',
            '209': 'Generation_of_Error_Message_Containing_Sensitive_Information',
            '215': 'Insertion_of_Sensitive_Information_into_Debug-Logging',
            '532': 'Insertion_of_Sensitive_Information_into_Log_File',
            '250': 'Execution_with_Unnecessary_Privileges',
            '362': 'Race_Condition',
            '367': 'Time-of-check_Time-of-use',
            '426': 'Untrusted_Search_Path',
            '427': 'Uncontrolled_Search_Path_Element',
            '489': 'Active_Debug_Code',
            '676': 'Use_of_Potentially_Dangerous_Function',
            '759': 'Use_of_one-way_hash_without_a_salt',
        }

    def extract_path(self, header_line):
        """从样本头中提取文件路径"""
        parts = header_line.strip().split(None, 1)
        return parts[1] if len(parts) == 2 else header_line.strip()

    def extract_cwe_numbers(self, header_line):
        """从完整的样本头中提取所有CWE编号"""
        all_cwes = []
        path_part = self.extract_path(header_line)

        matches = re.findall(r'CWE[_\-]?(\d+)|[/\\](\d+)[/\\]', path_part)
        for m in matches:
            cwe_id = m[0] or m[1]  # m[0] for 'CWEddd', m[1] for '/ddd/'
            if cwe_id and cwe_id not in ['0', '00', '000']:
                all_cwes.append(cwe_id)

        unique_cwes = sorted(list(set(all_cwes)))
        return [f"CWE-{cwe}" for cwe in unique_cwes] if unique_cwes else ["unknown"]

    def extract_specific_vuln_type(self, header_line):
        """
        从文件名中提取一个或多个精确的漏洞类型描述。
        """
        path = self.extract_path(header_line)
        filename = os.path.basename(path)

        # 移除文件扩展名和可能的结尾部分
        core_filename = filename.split('.')[0]

        # 使用 '__' 作为分隔符来切分复合漏洞类型
        vuln_parts = core_filename.split('__')

        specific_types = []
        for part in vuln_parts:
            # 只处理以'CWE'开头的部分，这有助于过滤掉非漏洞描述的部分
            if part.startswith('CWE'):
                cleaned_part = re.sub(r'_\d+[a-z]?(_\d+)?.*$', '', part)
                specific_types.append(cleaned_part.strip('_'))

        if specific_types:
            return "|".join(specific_types)

        return "unknown"

    def get_general_vuln_types(self, cwe_numbers):
        """根据CWE编号列表，从cwe_map中查找通用的漏洞类型"""
        found = []
        cwe_ids = re.findall(r'\d+', "|".join(cwe_numbers))
        for cwe_id in cwe_ids:
            if cwe_id in self.cwe_map:
                found.append(self.cwe_map[cwe_id])

        result = sorted(list(set(found)))
        return result if result else ["unknown"]


class SampleSplitter:
    def __init__(self):
        self.header_re = re.compile(r'^\s*\d+\s+.*\.ll\s*$', re.IGNORECASE)
        self.end_re = re.compile(r'^\s*-{5,}\s*$')

    def split(self, text):
        lines = text.splitlines()
        samples, buffer = [], []
        current_header = None
        for ln in lines:
            if self.header_re.match(ln):
                if current_header: samples.append((current_header, "\n".join(buffer)))
                current_header, buffer = ln.strip(), []
            elif self.end_re.match(ln):
                if current_header:
                    samples.append((current_header, "\n".join(buffer)))
                    current_header, buffer = None, []
            elif current_header:
                buffer.append(ln)
        if current_header: samples.append((current_header, "\n".join(buffer)))
        return samples


class Processor:
    def __init__(self):
        self.extractor = Extractor()
        self.splitter = SampleSplitter()
        self.rows = []

    def process_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
            return
        samples = self.splitter.split(text)
        for header, ir in samples:
            cwe_numbers = self.extractor.extract_cwe_numbers(header)
            specific_type = self.extractor.extract_specific_vuln_type(header)
            general_types = self.extractor.get_general_vuln_types(cwe_numbers)
            self.rows.append({
                "sample_id": len(self.rows) + 1, "source_file": os.path.basename(file_path),
                "specific_vulnerability_type": specific_type, "cwe_number": "|".join(cwe_numbers),
                "general_vulnerability_type": "|".join(general_types),
                "sample_path": self.extractor.extract_path(header), "ir_code": ir.strip().replace("\n", "\\n")
            })

    def process_directory(self, root, ext=".txt"):
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if f.endswith(ext):
                    full_path = os.path.join(dirpath, f)
                    print(f"Processing {full_path}...")
                    self.process_file(full_path)

    def save_csv(self, out_csv):
        if not self.rows: print("Warning: No samples found."); return
        keys = ["sample_id", "source_file", "specific_vulnerability_type", "cwe_number",
                "general_vulnerability_type", "sample_path", "ir_code"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract comprehensive vulnerability data from LLVM IR samples.")
    parser.add_argument("--input-dir", required=True, type=str, help="Directory containing input .txt files.")
    parser.add_argument("--output", type=str, default="vulnerability_dataset_final.csv",
                        help="Path to the output CSV file.")
    args = parser.parse_args()
    processor = Processor()
    print(f"Starting enhanced processing of directory: {args.input_dir}")
    processor.process_directory(args.input_dir)
    processor.save_csv(args.output)
    print("\n" + "=" * 50 + f"\nProcessing complete.\nTotal samples generated: {len(processor.rows)}\n"
                            f"Results saved to: {args.output}\n" + "=" * 50)