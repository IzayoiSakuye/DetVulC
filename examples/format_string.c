#include <stdio.h>

void format_string_vulnerable(char* user_input) {
    printf("User provided message: ");
    // 危险：直接将用户输入作为格式化字符串
    printf(user_input);
    printf("\n");
}

int main() {
    // 攻击者可以传入 "%s%s%s%s%s%s" 来泄露栈上的信息
    char malicious_input[] = "This is a test %s%s%s%s";
    format_string_vulnerable(malicious_input);
    return 0;
}