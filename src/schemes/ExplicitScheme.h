//
// Created by geoolekom on 21.09.18.
//

#ifndef PROJECT_SCHEME_H
#define PROJECT_SCHEME_H


class ExplicitScheme {
public:
    explicit ExplicitScheme(int size, int c, double timeStep = 0.001);
    ~ExplicitScheme();

    void solve(int);

    void setInitialValues();

    void toFile(const char*);

    virtual void step() = 0;

protected:
    double *u, *prev;
    int size, c;
    double timeStep;
};


#endif //PROJECT_SCHEME_H
