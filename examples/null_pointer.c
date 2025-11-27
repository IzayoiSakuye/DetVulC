#include <stdio.h>

int main() {
    FILE *fp;

    // 如果 "non_existent_file.txt" 不存在, fopen 返回 NULL
    fp = fopen("non_existent_file.txt", "r");

    // 没有检查 fp 是否为 NULL
    // 这会导致 NULL 指针解引用
    printf("The first character is: %c\n", fgetc(fp));
    
    fclose(fp);
    return 0;
}