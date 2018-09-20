//
// Created by geoolekom on 21.09.18.
//

#include <cmath>
#include <iostream>
#include "SimpleScheme.h"

void SimpleScheme::step() {
    for (int i = 1; i < this->size - 1; i++) {
        this->u[i] = this->prev[i] - this->c * this->size / 1000.0 * (this->prev[i] - this->prev[i - 1]);
    }
}
