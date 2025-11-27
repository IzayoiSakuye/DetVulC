#include <string.h>

void vulnerable_function() {
    char buffer[10];
    // This will cause a stack buffer overflow
    strcpy(buffer, "This is a very long string that will overflow the buffer");
}

int main() {
    vulnerable_function();
    return 0;
}