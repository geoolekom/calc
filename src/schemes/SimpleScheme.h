//
// Created by geoolekom on 21.09.18.
//

#include "ExplicitScheme.h"

#ifndef PROJECT_SIMPLESCHEME_H
#define PROJECT_SIMPLESCHEME_H

class SimpleScheme : public ExplicitScheme {
public:
    SimpleScheme(int size, int c): ExplicitScheme(size, c) {};
    virtual void step();
};


#endif //PROJECT_SIMPLESCHEME_H
