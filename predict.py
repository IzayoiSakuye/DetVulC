import tensorflow as tf
import joblib
import numpy as np

# ===============================================================
#  配置和加载
# ===============================================================
MODEL_PATH = 'vulnerability_scanner_model.keras'
MLB_PATH = 'mlb.pkl'
PREDICTION_THRESHOLD = 0.5

print("Loading model and label binarizer...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
except (IOError, FileNotFoundError) as e:
    print(f"Error loading files: {e}")
    print(f"Please make sure '{MODEL_PATH}' and '{MLB_PATH}' are in the same directory.")
    exit()

print("Model and binarizer loaded successfully.")


# ===============================================================
#  预测函数
# ===============================================================
def predict_vulnerability(ir_code_snippet: str):
    """
    对单段LLVM IR代码进行漏洞预测。
    """
    if not ir_code_snippet.strip():
        print("Warning: Input code is empty.")
        return []

    # --- 关键修改在这里 ---
    # 将输入的Python字符串列表转换为TensorFlow的原生字符串张量 (tf.string)
    # 而不是NumPy数组。
    input_tensor = tf.constant([ir_code_snippet], dtype=tf.string)

    # 使用转换后的张量进行预测
    predictions = model.predict(input_tensor)[0]

    vulnerabilities = []
    for i, prob in enumerate(predictions):
        if prob > PREDICTION_THRESHOLD:
            vulnerabilities.append({
                "type": mlb.classes_[i],
                "confidence": f"{prob:.2%}"
            })

    # 按置信度排序
    vulnerabilities.sort(key=lambda x: float(x['confidence'].strip('%')) / 100, reverse=True)

    return vulnerabilities


# ===============================================================
#  示例用法
# ===============================================================
if __name__ == "__main__":
    sample_code_1 = """
    define dso_local i32 @main() #0 {
      %1 = alloca i32, align 4
      %2 = alloca [10 x i8], align 1
      store i32 0, i32* %1, align 4
      %3 = getelementptr inbounds [10 x i8], [10 x i8]* %2, i64 0, i64 0
      call i8* @strcpy(i8* %3, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i64 0, i64 0)) #3
      ret i32 0
    }
    declare i8* @strcpy(i8*, i8*) #1
    """

    sample_code_2 = """
    define dso_local i32 @safe_add(i32 %a, i32 %b) #0 {
      %1 = alloca i32, align 4
      %2 = alloca i32, align 4
      store i32 %a, i32* %1, align 4
      store i32 %b, i32* %2, align 4
      %3 = load i32, i32* %1, align 4
      %4 = load i32, i32* %2, align 4
      %5 = add nsw i32 %3, %4
      ret i32 %5
    }
    """

    print("\n--- Prediction for Sample 1 (strcpy risk) ---")
    results1 = predict_vulnerability(sample_code_1)
    if results1:
        for vuln in results1:
            print(f"  - Vulnerability: {vuln['type']}, Confidence: {vuln['confidence']}")
    else:
        print("  No vulnerabilities detected.")

    print("\n--- Prediction for Sample 2 (safe add) ---")
    results2 = predict_vulnerability(sample_code_2)
    if results2:
        for vuln in results2:
            print(f"  - Vulnerability: {vuln['type']}, Confidence: {vuln['confidence']}")
    else:
        print("  No vulnerabilities detected.")
