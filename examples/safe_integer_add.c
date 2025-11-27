#include <limits.h> // for INT_MAX
#include <stdlib.h>

void* safe_addition_and_alloc(int a, int b) {
    // 在相加前检查是否会溢出
    if (a > 0 && b > 0 && a > INT_MAX - b) {
        // 溢出，处理错误
        return NULL;
    }

    int size = a + b;
    void* buffer = malloc(size);
    return buffer;
}

int main() {
    void* p = safe_addition_and_alloc(100, 200);
    if (p) {
        free(p);
    }
    
    // 尝试一个可能溢出的情况
    void* p2 = safe_addition_and_alloc(INT_MAX - 10, 20);
    // safe_addition_and_alloc 会返回 NULL，避免了后续问题
    if (p2) {
        free(p2);
    }

    return 0;
}