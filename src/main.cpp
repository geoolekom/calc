#include <iostream>
#include <cstdlib>

#include "base/ExplicitScheme.h"
#include "schemes/SimpleScheme.h"
#include "schemes/LaxWendroffScheme.h"


int castToInt(char* str) {
    char* pointerEnd;
    const long int temp = strtol(str, &pointerEnd, 10);
    if (temp == 0) {
        std::cout << "Значение " << str << " не является ненулевым целым числом.\n";
    }
    return (int) temp;
}


int main(int argc, char* argv[]) {
    if (argc < 5) {
	    std::cout << "Недостаточно аргументов.\n";
		return 1;
	}
	int c = castToInt(argv[1]), n = castToInt(argv[2]), t = castToInt(argv[3]);
    const char* filename = argv[4];

    if (c != 0 && n != 0 && t != 0) {
        LaxWendroffScheme* s = new LaxWendroffScheme(n, c);
        s->solve(t);
        s->toFile(filename);
        delete s;
    }
	return 0;
}
