#include <string.h>

void safe_copy() {
    char dest[10];
    char src[] = "This is a very long string";

    // 使用 strncpy 安全地复制
    // 它最多只会写 10 个字节
    strncpy(dest, src, sizeof(dest));

    // 关键：手动确保字符串以 null 结尾
    // 因为如果 src 的长度 >= dest 的大小, strncpy 不会添加 null 终止符
    dest[sizeof(dest) - 1] = '\0';
}

int main() {
    safe_copy();
    return 0;
}