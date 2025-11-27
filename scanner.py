import argparse
import os
import subprocess
import tempfile
import tensorflow as tf
import joblib
import numpy as np


#  配置
MODEL_PATH = './models/vulnerability_scanner_model.keras'
MLB_PATH = './models/mlb.pkl'
PREDICTION_THRESHOLD = 0.5


def check_clang():
    """检查系统中是否安装了 clang"""
    try:
        subprocess.run(['clang', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def c_to_llvm_ir(c_file_path: str) -> str | None:
    """
    使用 clang 将 C 语言源文件编译成 LLVM IR 字符串。
    """
    print(f"Compiling '{c_file_path}' to LLVM IR...")

    # 创建一个临时文件来存放输出的 .ll 代码
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.ll') as temp_ll_file:
        ll_path = temp_ll_file.name

    try:
        # 构建 clang 命令
        # -S: 生成汇编代码
        # -emit-llvm: 指定输出为 LLVM IR
        # -O0: 不进行优化，保留最原始的IR结构
        # -g: 生成调试信息，有时有助于分析，可选
        command = ['clang', '-S', '-emit-llvm', '-O0', '-g', c_file_path, '-o', ll_path]

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # 如果编译成功，读取临时文件的内容
        with open(ll_path, 'r') as f:
            ir_code = f.read()
        print("Compilation successful.")
        return ir_code

    except subprocess.CalledProcessError as e:
        print("\n--- Compilation Failed ---")
        print(f"Error compiling '{c_file_path}'. Clang returned a non-zero exit code.")
        print("Clang output (stderr):")
        print(e.stderr)
        return None
    finally:
        # 清理临时文件
        if os.path.exists(ll_path):
            os.remove(ll_path)


def predict_vulnerability(ir_code_snippet: str, model, mlb) -> list:
    """
    对单段LLVM IR代码进行漏洞预测。
    """
    if not ir_code_snippet.strip():
        print("Warning: Input code is empty or could not be generated.")
        return []

    input_tensor = tf.constant([ir_code_snippet], dtype=tf.string)
    predictions = model.predict(input_tensor, verbose=0)[0]

    vulnerabilities = []
    for i, prob in enumerate(predictions):
        if prob > PREDICTION_THRESHOLD:
            vulnerabilities.append({
                "type": mlb.classes_[i],
                "confidence": f"{prob:.2%}"
            })

    vulnerabilities.sort(key=lambda x: float(x['confidence'].strip('%')) / 100, reverse=True)
    return vulnerabilities


def main():
    parser = argparse.ArgumentParser(
        description="A deep learning-based vulnerability scanner for C and LLVM IR files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("filepath", help="Path to the source file (.c or .ll) to scan.")
    args = parser.parse_args()

    file_path = args.filepath

    # 检查输入文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    # 加载模型和标签编码器
    print("Loading vulnerability detection model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        mlb = joblib.load(MLB_PATH)
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading model files: {e}")
        print(f"Please make sure '{MODEL_PATH}' and '{MLB_PATH}' are in the same directory.")
        return
    print("Model loaded successfully.")

    # 获取LLVM IR代码
    ir_code = None
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.c':
        if not check_clang():
            print("Error: 'clang' command not found.")
            print("Please install clang and ensure it's in your system's PATH to scan .c files.")
            return
        ir_code = c_to_llvm_ir(file_path)
    elif file_extension == '.ll':
        print(f"Reading LLVM IR from '{file_path}'...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            ir_code = f.read()
    else:
        print(f"Error: Unsupported file type '{file_extension}'. Please provide a .c or .ll file.")
        return

    if not ir_code:
        print("Could not get LLVM IR code to analyze. Exiting.")
        return

    # 执行预测并显示结果
    print("\nScanning for vulnerabilities...")
    results = predict_vulnerability(ir_code, model, mlb)

    print("\n--- Scan Report ---")
    print(f"File: {os.path.basename(file_path)}")
    if results:
        print(f"Status: Vulnerabilities Detected!")
        for vuln in results:
            print(f"  - Type: {vuln['type']}, Confidence: {vuln['confidence']}")
    else:
        print("Status: No vulnerabilities detected.")
    print("-------------------\n")


if __name__ == "__main__":
    main()