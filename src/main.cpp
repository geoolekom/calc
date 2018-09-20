#include <iostream>
#include <cstdlib>

#include "base/ExplicitScheme.h"
#include "schemes/LaxWendroffScheme.h"

int main(int argc, char* argv[]) {
	if (argc < 5) {
		std::cout << "Недостаточно аргументов.\n";
		return 1;
	}
	const int c = atoi(argv[1]), n = atoi(argv[2]), t = atoi(argv[3]);
    LaxWendroffScheme* s = new LaxWendroffScheme(n, c);
	s->solve(t);
	s->toFile(argv[4]);
	delete s;
	return 0;
}
