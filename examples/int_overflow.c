#include <stdlib.h>
#include <string.h>

void integer_overflow_vulnerable(int len1, int len2, char* data1, char* data2) {
    // 如果 len1 和 len2 都接近 INT_MAX, 相加会导致溢出
    int total_len = len1 + len2;
    
    // 溢出后 total_len 可能是一个很小的数，导致 malloc 分配不足
    char* buffer = (char*)malloc(total_len);
    if (buffer == NULL) {
        return;
    }
    
    // 这里的 memcpy 会导致堆溢出
    memcpy(buffer, data1, len1);
    memcpy(buffer + len1, data2, len2);
    
    free(buffer);
}

int main() {
    char data1[] = "data_part_1";
    char data2[] = "data_part_2";
    // 示例调用，实际攻击会传入一个极大的 len
    integer_overflow_vulnerable(2147483640, 2147483640, data1, data2);
    return 0;
}