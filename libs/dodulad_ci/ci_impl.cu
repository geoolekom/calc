#include <assert.h>
#include <math.h>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime_api.h>
#include "ci_cuda.h"

#define EPS 1e-10

namespace ci {
    __device__ void elog(const char *name, double var, int index) {
        if (isnan(var) || var < 0) {
            printf("%s: %.20f <- %d\n", name, var, index);
            assert(0);
        }
    }

    __device__ void cudaIter(node_calc* ncData, size_t ncSize, float* f1, float* f2) {
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
                elog("y1", y[1], 0);

                v[0] = pow(x[0], y[0]) * pow(z[0], y[0]);
                if (isnan(v[0])) {
                    printf("v0: x0: %.20f, z0: %.20f, y0: %.20f\n", x[0], z[0], y[0]);
                    assert(0);
                }

                v[1] = pow(x[1], y[1]) * pow(z[1], y[1]);
                if (isnan(v[1])) {
                    printf("v1: x1: %.20f, z1: %.20f, y1: %.20f\n", x[1], z[1], y[1]);
                    assert(0);
                }

                double rr5 = f1[p->i1];
                double rr6 = f2[p->i2];
                elog("rr5", rr5, p->i1);
                elog("rr6", rr6, p->i2);

                double d = ( - v[0] * v[1] + rr5 * rr6) * p->c;

                double dl = (1. - p->r) * d;
                double dm = p->r * d;

                f1[p->i1l] += dl;
                f2[p->i2l] += dl;
                f1[p->i1m] += dm;
                f2[p->i2m] += dm;
                f1[p->i1 ] -= d;
                f2[p->i2 ] -= d;

                if ((f1[p->i1l] < 0) ||
                    (f1[p->i1m] < 0) ||
                    (f2[p->i2l] < 0) ||
                    (f2[p->i2m] < 0) ||
                    (f1[p->i1 ] < 0) ||
                    (f2[p->i2 ] < 0)) {

                    f1[p->i1l] = (float) x[0];
                    f1[p->i1m] = (float) x[1];
                    f2[p->i2l] = (float) z[0];
                    f2[p->i2m] = (float) z[1];
                    f1[p->i1 ] = (float) rr5;
                    f2[p->i2 ] = (float) rr6;
                }
            } else {
                double g1 = f1[p->i1];
                double g2 = f2[p->i2];
                double g3 = f1[p->i1m];
                double g4 = f2[p->i2m];

                double d = (- g3*g4 + g1*g2) * p->c;

                f1[p->i1 ] -= d;
                f2[p->i2 ] -= d;
                f1[p->i1m] += d;
                f2[p->i2m] += d;

                if ((f1[p->i1m] < 0) ||
                    (f2[p->i2m] < 0) ||
                    (f1[p->i1 ] < 0) ||
                    (f2[p->i2 ] < 0))  {

                    f1[p->i1 ] = (float) g1;
                    f2[p->i2 ] = (float) g2;
                    f1[p->i1m] = (float) g3;
                    f2[p->i2m] = (float) g4;
                }
            }
        }
    }
}