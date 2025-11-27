#include <stdlib.h>

typedef struct {
    int value;
    char name[10];
} Data;

int main() {
    Data* p = (Data*)malloc(sizeof(Data));
    if (p == NULL) {
        return 1;
    }
    
    p->value = 50;
    
    free(p);
    
    // 危险：在已释放的内存上进行写操作
    // This is a Use-After-Free vulnerability
    p->value = 100; 

    return 0;
}