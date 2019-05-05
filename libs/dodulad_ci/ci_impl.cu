#include <assert.h>
#include <math.h>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <host_defines.h>
#include <cuda_runtime_api.h>
#include "ci.hpp"

#define EPS 1e-10

namespace ci {
    __device__ void elog(const char *name, double var, int index) {
        if (isnan(var)) {
            printf("%s: %f <- %d\n", name, var, index);
            assert(0);
        }
    }

    __device__ void cudaIter(node_calc* ncData, size_t ncSize, double* f1, double* f2) {
        double x[2], y[2], z[2], v[2];
        for (int i = 0; i < ncSize; i++) {
            const auto p = ncData + i;
            if (fabs(p->r - 1) > EPS) {
                x[0] = f1[p->i1l];
                x[1] = f1[p->i1m];
                elog("x0", x[0], p->i1l);
                elog("x1", x[1], p->i1m);

                z[0] = f2[p->i2l];
                z[1] = f2[p->i2m];
                elog("z0", z[0], p->i2l);
                elog("z1", z[1], p->i2m);

                y[0] = 1. - p->r;
                y[1] = p->r;
                elog("y0", y[0], 0);


                v[0] = pow(x[0], y[0]) * pow(z[0], y[0]);
                v[1] = pow(x[1], y[1]) * pow(z[1], y[1]);

                double rr5 = f1[p->i1];
                double rr6 = f2[p->i2];
                double d = ( - v[0] * v[1] + rr5 * rr6) * p->c;

                double dl = (1. - p->r) * d;
                double dm = p->r * d;

                f1[p->i1l] += dl;
                f2[p->i2l] += dl;
                f1[p->i1m] += dm;
                f2[p->i2m] += dm;
                f1[p->i1] -= d;
                f2[p->i2] -= d;

                if ((f1[p->i1l] < 0) ||
                    (f1[p->i1m] < 0) ||
                    (f2[p->i2l] < 0) ||
                    (f2[p->i2m] < 0) ||
                    (f1[p->i1 ] < 0) ||
                    (f2[p->i2 ] < 0)) {

                    f1[p->i1l] = x[0];
                    f1[p->i1m] = x[1];
                    f2[p->i2l] = z[0];
                    f2[p->i2m] = z[1];
                    f1[p->i1 ] = rr5;
                    f2[p->i2 ] = rr6;
                }
            } else {
                double g1 = f1[p->i1];
                double g2 = f2[p->i2];
                double g3 = f1[p->i1m];
                double g4 = f2[p->i2m];

                double d = (- g3*g4 + g1*g2) * p->c;

                f1[p->i1]  -= d;
                f2[p->i2]  -= d;
                f1[p->i1m] += d;
                f2[p->i2m] += d;

                if ((f1[p->i1m] < 0) ||
                    (f2[p->i2m] < 0) ||
                    (f1[p->i1] < 0) ||
                    (f2[p->i2] < 0))  {

                    f1[p->i1] = g1;
                    f2[p->i2] = g2;
                    f1[p->i1m] = g3;
                    f2[p->i2m] = g4;
                }
            }
        }
    }
}