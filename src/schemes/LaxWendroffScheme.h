//
// Created by geoolekom on 21.09.18.
//

#ifndef PROJECT_LAXWENDROFFSCHEME_H
#define PROJECT_LAXWENDROFFSCHEME_H


#include "ExplicitScheme.h"

class LaxWendroffScheme : public ExplicitScheme {
public:
    LaxWendroffScheme(int size, int c): ExplicitScheme(size, c) {};
    virtual void step();
};


#endif //PROJECT_LAXWENDROFFSCHEME_H
