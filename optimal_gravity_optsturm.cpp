#include <vector>
#include <iostream>
#include <numeric>
#include <Eigen/Dense>
#include "mex.h"
#include "matrix.h"
#include "charpoly.h"
#include "Polynomial.hpp"

using namespace Eigen;
using namespace std;

MatrixXd solver_opt(VectorXd &data)
{
    // Compute coefficients
    using polynomial::Polynomial;
    // Compute coefficients
    const double *p = data.data();
    
    Matrix<double, 9, 1> acoeffs;
    acoeffs << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]; //decreasing
    Polynomial<8> a(acoeffs);
    
    Matrix<double, 7, 1> bcoeffs;
    bcoeffs << p[9], p[10], p[11], p[12], p[13], p[14], p[15];
    Polynomial<6> b(bcoeffs);
    
    Matrix<double, 5, 1> ccoeffs;
    ccoeffs << p[16], p[17], p[18], p[19], p[20];
    Polynomial<4> c(ccoeffs);
    
    Matrix<double, 3, 1> dcoeffs;
    dcoeffs << 1, 0, 1;
    Polynomial<2> d(dcoeffs);
    
    Matrix<double, 9, 1> ecoeffs;
    ecoeffs << p[21], p[22], p[23], p[24], p[25], p[26], p[27], p[28], p[29];
    Polynomial<8> e(ecoeffs);
    
    Matrix<double, 7, 1> fcoeffs;
    fcoeffs << p[30], p[31], p[32], p[33], p[34], p[35], p[36];
    Polynomial<6> f(fcoeffs);
    
    Matrix<double, 5, 1> gcoeffs;
    gcoeffs << p[37], p[38], p[39], p[40], p[41];
    Polynomial<4> g(gcoeffs);
    
    Matrix<double, 29, 1> hcoeffs;
    hcoeffs = (a * (f * (f * (d * f - c * g) + g * (b * g - d * e)) - g * (g * (a * g - c * e) + d * e * f)) - e * (d * (e * (d * e - c * f) + b * (f * f - g * e)) + g * (b * (g * b - f * c) + e * (c * c - d * b) + a * (d * f - g * c)))).coefficients();
    Polynomial<28> h(hcoeffs);
    
    std::vector<double> roots;
    h.realRootsSturm(-1, 1, roots);
    
    // std::cout<<"roots.size() " << roots.size() << std::endl;
    
    MatrixXd sols(1, 28);
    sols.setZero();
    int k = 0;
    if (roots.size() > 0 && roots.size() < 10)
    {
        for (int i = 0; i < roots.size(); i++)
        {
            sols(0, i) = roots[i];
            k++;
        }
        sols.conservativeResize(1, k);
    }
    else
    {
        sols(0, 0) = 0;
        sols.conservativeResize(1, 1);
    }
    return sols;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    if (nrhs != 2)
    {
        mexPrintf("solveE: Wrong number of arguments.\n");
        mexPrintf("Usage: sols = solver(q1, q2)\n");
        return;
    }
    
    // Check the input
    const mxArray *q1 = prhs[0];
    const mxArray *q2 = prhs[1];
    
    // Check the dimensions of the arguments
    int ndimensions1 = mxGetNumberOfDimensions(q1);
    const mwSize *q1dim = mxGetDimensions(q1);
    
    int ndimensions2 = mxGetNumberOfDimensions(q2);
    const mwSize *q2dim = mxGetDimensions(q2);
    
    // Now check them
    if (ndimensions1 != 2 || q1dim[0] != 2 || ndimensions2 != 2 || q2dim[0] != 2)
    {
        mexPrintf("Bad input to mex function \n");
        mexPrintf(
                "Inputs q1 and q2 must have dimensions [2, n]\n");
        return;
    }
    
    if (q1dim[1] != q2dim[1])
    {
        mexPrintf("Bad input to mex function \n");
        mexPrintf("Inputs q1 and q2 must have same dimensions.\n");
        return;
    }
    
    // -------------------------------------------------------------------------
    // Read and reformat the input
    int npoints = q1dim[1];
    
    double qa[q1dim[1]][2], qb[q1dim[1]][2];
    
    double *p1 = (double *)mxGetData(q1);
    memcpy(&(qa[0][0]), p1, 2 * npoints * sizeof(double));
    
    double *p2 = (double *)mxGetData(q2);
    memcpy(&(qb[0][0]), p2, 2 * npoints * sizeof(double));
    
    double w[9] = {0};
    double a[5] = {0};
    double b[5] = {0};
    double c[5] = {0};
    double d[5] = {0};
    double e[5] = {0};
    double f[5] = {0};
    double k[5] = {0};
    double g[7] = {0};
    double p[7] = {0};
    double q[7] = {0};
    double n[5] = {0};
    double h[9] = {0};
    
    for (int i = 0; i < npoints; i++)
    {
        double u1 = qa[i][0] * qb[i][1];
        double u2 = qb[i][0] * qa[i][1];
        
        w[0] = -qb[i][1] - qa[i][1];
        w[1] = -2 * u1;
        w[2] = qb[i][1] - qa[i][1];
        w[3] = qb[i][0] - qa[i][0];
        w[4] = 2 + 2 * qa[i][0] * qb[i][0];
        w[5] = -w[3];
        w[6] = u1 + u2;
        w[7] = -2 * qb[i][1];
        w[8] = u2 - u1;
        
        a[4] += w[0] * w[0];
        a[3] += 2 * w[0] * w[1];
        a[2] += w[1] * w[1] + 2 * w[0] * w[2];
        a[1] += 2 * w[1] * w[2];
        a[0] += w[2] * w[2];
        b[4] += w[0] * w[3];
        b[3] += w[0] * w[4] + w[1] * w[3];
        b[2] += w[1] * w[4] - w[0] * w[3] + w[2] * w[3];
        b[1] += w[2] * w[4] - w[1] * w[3];
        b[0] -= w[2] * w[3];
        c[4] += w[0] * w[6];
        c[3] += w[0] * w[7] + w[1] * w[6];
        c[2] += w[0] * w[8] + w[1] * w[7] + w[2] * w[6];
        c[1] += w[1] * w[8] + w[2] * w[7];
        c[0] += w[2] * w[8];
        d[4] += w[3] * w[3];
        d[3] += 2 * w[3] * w[4];
        d[2] += w[4] * w[4] - 2 * w[3] * w[3];
        d[1] -= 2 * w[3] * w[4];
        d[0] += w[3] * w[3];
        e[4] += w[3] * w[6];
        e[3] += w[3] * w[7] + w[4] * w[6];
        e[2] += w[3] * w[8] - w[3] * w[6] + w[4] * w[7];
        e[1] += w[4] * w[8] - w[3] * w[7];
        e[0] -= w[3] * w[8];
        f[4] += w[6] * w[6];
        f[3] += 2 * w[6] * w[7];
        f[2] += w[7] * w[7] + 2 * w[6] * w[8];
        f[1] += 2 * w[7] * w[8];
        f[0] += w[8] * w[8];
    }
    
    k[4] = a[4] + d[4] + f[4];
    k[3] = a[3] + d[3] + f[3];
    k[2] = a[2] + d[2] + f[2];
    k[1] = a[1] + d[1] + f[1];
    k[0] = a[0] + d[0] + f[0];
    g[6] = -b[4] * b[4] - c[4] * c[4] - e[4] * e[4] + a[4] * d[4] + a[4] * f[4] + d[4] * f[4];
    g[5] = a[3] * d[4] - 2 * b[3] * b[4] + a[4] * d[3] - 2 * c[3] * c[4] + a[3] * f[4] + a[4] * f[3] + d[3] * f[4] + d[4] * f[3] - 2 * e[3] * e[4];
    g[4] = -b[3] * b[3] + b[4] * b[4] - 2 * b[2] * b[4] - c[3] * c[3] + c[4] * c[4] - 2 * c[2] * c[4] - e[3] * e[3] + e[4] * e[4] - 2 * e[2] * e[4] + a[2] * d[4] + a[3] * d[3] + a[4] * d[2] - a[4] * d[4] + a[2] * f[4] + a[3] * f[3] + a[4] * f[2] - a[4] * f[4] + d[2] * f[4] + d[3] * f[3] + d[4] * f[2] - d[4] * f[4];
    g[3] = 2 * b[3] * b[4] - 2 * b[2] * b[3] - 2 * b[1] * b[4] + a[1] * d[4] + a[2] * d[3] + a[3] * d[2] + a[4] * d[1] - a[3] * d[4] - a[4] * d[3] - 2 * c[1] * c[4] - 2 * c[2] * c[3] + 2 * c[3] * c[4] + a[1] * f[4] + a[2] * f[3] + a[3] * f[2] + a[4] * f[1] - a[3] * f[4] - a[4] * f[3] + d[1] * f[4] + d[2] * f[3] + d[3] * f[2] + d[4] * f[1] - 2 * e[1] * e[4] - 2 * e[2] * e[3] - d[3] * f[4] - d[4] * f[3] + 2 * e[3] * e[4];
    g[2] = -b[2] * b[2] + 2 * b[2] * b[4] + b[3] * b[3] - 2 * b[1] * b[3] - b[4] * b[4] - 2 * b[0] * b[4] - c[2] * c[2] + 2 * c[2] * c[4] + c[3] * c[3] - 2 * c[1] * c[3] - c[4] * c[4] - 2 * c[0] * c[4] - e[2] * e[2] + 2 * e[2] * e[4] + e[3] * e[3] - 2 * e[1] * e[3] - e[4] * e[4] - 2 * e[0] * e[4] + a[0] * d[4] + a[1] * d[3] + a[2] * d[2] + a[3] * d[1] + a[4] * d[0] - a[2] * d[4] - a[3] * d[3] - a[4] * d[2] + a[4] * d[4] + a[0] * f[4] + a[1] * f[3] + a[2] * f[2] + a[3] * f[1] + a[4] * f[0] - a[2] * f[4] - a[3] * f[3] - a[4] * f[2] + a[4] * f[4] + d[0] * f[4] + d[1] * f[3] + d[2] * f[2] + d[3] * f[1] + d[4] * f[0] - d[2] * f[4] - d[3] * f[3] - d[4] * f[2] + d[4] * f[4];
    g[1] = 2 * b[1] * b[4] - 2 * b[1] * b[2] - 2 * b[0] * b[3] + 2 * b[2] * b[3] - 2 * b[3] * b[4] + a[0] * d[3] + a[1] * d[2] + a[2] * d[1] + a[3] * d[0] - a[1] * d[4] - a[2] * d[3] - a[3] * d[2] - a[4] * d[1] + a[3] * d[4] + a[4] * d[3] - 2 * c[0] * c[3] - 2 * c[1] * c[2] + 2 * c[1] * c[4] + 2 * c[2] * c[3] - 2 * c[3] * c[4] + a[0] * f[3] + a[1] * f[2] + a[2] * f[1] + a[3] * f[0] - a[1] * f[4] - a[2] * f[3] - a[3] * f[2] - a[4] * f[1] + a[3] * f[4] + a[4] * f[3] + d[0] * f[3] + d[1] * f[2] + d[2] * f[1] + d[3] * f[0] - 2 * e[0] * e[3] - 2 * e[1] * e[2] - d[1] * f[4] - d[2] * f[3] - d[3] * f[2] - d[4] * f[1] + 2 * e[1] * e[4] + 2 * e[2] * e[3] + d[3] * f[4] + d[4] * f[3] - 2 * e[3] * e[4];
    g[0] = -b[1] * b[1] + 2 * b[1] * b[3] + b[2] * b[2] - 2 * b[2] * b[4] - 2 * b[0] * b[2] - b[3] * b[3] + b[4] * b[4] + 2 * b[0] * b[4] - c[1] * c[1] + 2 * c[1] * c[3] + c[2] * c[2] - 2 * c[2] * c[4] - 2 * c[0] * c[2] - c[3] * c[3] + c[4] * c[4] + 2 * c[0] * c[4] - e[1] * e[1] + 2 * e[1] * e[3] + e[2] * e[2] - 2 * e[2] * e[4] - 2 * e[0] * e[2] - e[3] * e[3] + e[4] * e[4] + 2 * e[0] * e[4] + a[0] * d[2] + a[1] * d[1] + a[2] * d[0] - a[0] * d[4] - a[1] * d[3] - a[2] * d[2] - a[3] * d[1] - a[4] * d[0] + a[2] * d[4] + a[3] * d[3] + a[4] * d[2] - a[4] * d[4] + a[0] * f[2] + a[1] * f[1] + a[2] * f[0] - a[0] * f[4] - a[1] * f[3] - a[2] * f[2] - a[3] * f[1] - a[4] * f[0] + a[2] * f[4] + a[3] * f[3] + a[4] * f[2] - a[4] * f[4] + d[0] * f[2] + d[1] * f[1] + d[2] * f[0] - d[0] * f[4] - d[1] * f[3] - d[2] * f[2] - d[3] * f[1] - d[4] * f[0] + d[2] * f[4] + d[3] * f[3] + d[4] * f[2] - d[4] * f[4];
    
    p[6] = b[4] * f[4] - c[4] * e[4];
    p[5] = b[3] * f[4] + b[4] * f[3] - c[3] * e[4] - c[4] * e[3];
    p[4] = b[2] * f[4] + b[3] * f[3] + b[4] * f[2] - c[2] * e[4] - c[3] * e[3] - c[4] * e[2] - b[4] * f[4] + c[4] * e[4];
    p[3] = b[1] * f[4] + b[2] * f[3] + b[3] * f[2] + b[4] * f[1] - c[1] * e[4] - c[2] * e[3] - c[3] * e[2] - c[4] * e[1] - b[3] * f[4] - b[4] * f[3] + c[3] * e[4] + c[4] * e[3];
    p[2] = b[0] * f[4] + b[1] * f[3] + b[2] * f[2] + b[3] * f[1] + b[4] * f[0] - c[0] * e[4] - c[1] * e[3] - c[2] * e[2] - c[3] * e[1] - c[4] * e[0] - b[2] * f[4] - b[3] * f[3] - b[4] * f[2] + c[2] * e[4] + c[3] * e[3] + c[4] * e[2] + b[4] * f[4] - c[4] * e[4];
    p[1] = b[0] * f[3] + b[1] * f[2] + b[2] * f[1] + b[3] * f[0] - c[0] * e[3] - c[1] * e[2] - c[2] * e[1] - c[3] * e[0] - b[1] * f[4] - b[2] * f[3] - b[3] * f[2] - b[4] * f[1] + c[1] * e[4] + c[2] * e[3] + c[3] * e[2] + c[4] * e[1] + b[3] * f[4] + b[4] * f[3] - c[3] * e[4] - c[4] * e[3];
    p[0] = b[0] * f[2] + b[1] * f[1] + b[2] * f[0] - c[0] * e[2] - c[1] * e[1] - c[2] * e[0] - b[0] * f[4] - b[1] * f[3] - b[2] * f[2] - b[3] * f[1] - b[4] * f[0] + c[0] * e[4] + c[1] * e[3] + c[2] * e[2] + c[3] * e[1] + c[4] * e[0] + b[2] * f[4] + b[3] * f[3] + b[4] * f[2] - c[2] * e[4] - c[3] * e[3] - c[4] * e[2] - b[4] * f[4] + c[4] * e[4];
    
    q[6] = a[4] * e[4] - b[4] * c[4];
    q[5] = a[3] * e[4] - b[4] * c[3] - b[3] * c[4] + a[4] * e[3];
    q[4] = b[4] * c[4] - b[3] * c[3] - b[4] * c[2] - b[2] * c[4] + a[2] * e[4] + a[3] * e[3] + a[4] * e[2] - a[4] * e[4];
    q[3] = b[3] * c[4] - b[2] * c[3] - b[3] * c[2] - b[4] * c[1] - b[1] * c[4] + b[4] * c[3] + a[1] * e[4] + a[2] * e[3] + a[3] * e[2] + a[4] * e[1] - a[3] * e[4] - a[4] * e[3];
    q[2] = b[2] * c[4] - b[1] * c[3] - b[2] * c[2] - b[3] * c[1] - b[4] * c[0] - b[0] * c[4] + b[3] * c[3] + b[4] * c[2] - b[4] * c[4] + a[0] * e[4] + a[1] * e[3] + a[2] * e[2] + a[3] * e[1] + a[4] * e[0] - a[2] * e[4] - a[3] * e[3] - a[4] * e[2] + a[4] * e[4];
    q[1] = b[1] * c[4] - b[1] * c[2] - b[2] * c[1] - b[3] * c[0] - b[0] * c[3] + b[2] * c[3] + b[3] * c[2] + b[4] * c[1] - b[3] * c[4] - b[4] * c[3] + a[0] * e[3] + a[1] * e[2] + a[2] * e[1] + a[3] * e[0] - a[1] * e[4] - a[2] * e[3] - a[3] * e[2] - a[4] * e[1] + a[3] * e[4] + a[4] * e[3];
    q[0] = b[0] * c[4] - b[1] * c[1] - b[2] * c[0] - b[0] * c[2] + b[1] * c[3] + b[2] * c[2] + b[3] * c[1] + b[4] * c[0] - b[2] * c[4] - b[3] * c[3] - b[4] * c[2] + b[4] * c[4] + a[0] * e[2] + a[1] * e[1] + a[2] * e[0] - a[0] * e[4] - a[1] * e[3] - a[2] * e[2] - a[3] * e[1] - a[4] * e[0] + a[2] * e[4] + a[3] * e[3] + a[4] * e[2] - a[4] * e[4];
    
    n[4] = -c[4] * c[4] + a[4] * f[4];
    n[3] = a[3] * f[4] - 2 * c[3] * c[4] + a[4] * f[3];
    n[2] = -c[3] * c[3] + 2 * c[4] * c[4] - 2 * c[2] * c[4] + a[2] * f[4] + a[3] * f[3] + a[4] * f[2] - 2 * a[4] * f[4];
    n[1] = 4 * c[3] * c[4] - 2 * c[2] * c[3] - 2 * c[1] * c[4] + a[1] * f[4] + a[2] * f[3] + a[3] * f[2] + a[4] * f[1] - 2 * a[3] * f[4] - 2 * a[4] * f[3];
    n[0] = -c[2] * c[2] + 4 * c[2] * c[4] + 2 * c[3] * c[3] - 2 * c[1] * c[3] - 3 * c[4] * c[4] - 2 * c[0] * c[4] + a[0] * f[4] + a[1] * f[3] + a[2] * f[2] + a[3] * f[1] + a[4] * f[0] - 2 * a[2] * f[4] - 2 * a[3] * f[3] - 2 * a[4] * f[2] + 3 * a[4] * f[4];
    
    h[8] = d[4] * n[4] - b[4] * p[6] - e[4] * q[6];
    h[7] = d[3] * n[4] + d[4] * n[3] - b[3] * p[6] - b[4] * p[5] - e[3] * q[6] - e[4] * q[5];
    h[6] = d[2] * n[4] + d[3] * n[3] + d[4] * n[2] - b[2] * p[6] - b[3] * p[5] - b[4] * p[4] + b[4] * p[6] - e[2] * q[6] - e[3] * q[5] - e[4] * q[4] + e[4] * q[6];
    h[5] = d[1] * n[4] + d[2] * n[3] + d[3] * n[2] + d[4] * n[1] - b[1] * p[6] - b[2] * p[5] - b[3] * p[4] - b[4] * p[3] + b[3] * p[6] + b[4] * p[5] - e[1] * q[6] - e[2] * q[5] - e[3] * q[4] - e[4] * q[3] + e[3] * q[6] + e[4] * q[5];
    h[4] = d[0] * n[4] + d[1] * n[3] + d[2] * n[2] + d[3] * n[1] + d[4] * n[0] - b[0] * p[6] - b[1] * p[5] - b[2] * p[4] - b[3] * p[3] - b[4] * p[2] + b[2] * p[6] + b[3] * p[5] + b[4] * p[4] - b[4] * p[6] - e[0] * q[6] - e[1] * q[5] - e[2] * q[4] - e[3] * q[3] - e[4] * q[2] + e[2] * q[6] + e[3] * q[5] + e[4] * q[4] - e[4] * q[6];
    h[3] = d[0] * n[3] + d[1] * n[2] + d[2] * n[1] + d[3] * n[0] - b[0] * p[5] - b[1] * p[4] - b[2] * p[3] - b[3] * p[2] - b[4] * p[1] + b[1] * p[6] + b[2] * p[5] + b[3] * p[4] + b[4] * p[3] - b[3] * p[6] - b[4] * p[5] - e[0] * q[5] - e[1] * q[4] - e[2] * q[3] - e[3] * q[2] - e[4] * q[1] + e[1] * q[6] + e[2] * q[5] + e[3] * q[4] + e[4] * q[3] - e[3] * q[6] - e[4] * q[5];
    h[2] = d[0] * n[2] + d[1] * n[1] + d[2] * n[0] - b[0] * p[4] - b[1] * p[3] - b[2] * p[2] - b[3] * p[1] - b[4] * p[0] + b[0] * p[6] + b[1] * p[5] + b[2] * p[4] + b[3] * p[3] + b[4] * p[2] - b[2] * p[6] - b[3] * p[5] - b[4] * p[4] + b[4] * p[6] - e[0] * q[4] - e[1] * q[3] - e[2] * q[2] - e[3] * q[1] - e[4] * q[0] + e[0] * q[6] + e[1] * q[5] + e[2] * q[4] + e[3] * q[3] + e[4] * q[2] - e[2] * q[6] - e[3] * q[5] - e[4] * q[4] + e[4] * q[6];
    h[1] = d[0] * n[1] + d[1] * n[0] - b[0] * p[3] - b[1] * p[2] - b[2] * p[1] - b[3] * p[0] + b[0] * p[5] + b[1] * p[4] + b[2] * p[3] + b[3] * p[2] + b[4] * p[1] - b[1] * p[6] - b[2] * p[5] - b[3] * p[4] - b[4] * p[3] + b[3] * p[6] + b[4] * p[5] - e[0] * q[3] - e[1] * q[2] - e[2] * q[1] - e[3] * q[0] + e[0] * q[5] + e[1] * q[4] + e[2] * q[3] + e[3] * q[2] + e[4] * q[1] - e[1] * q[6] - e[2] * q[5] - e[3] * q[4] - e[4] * q[3] + e[3] * q[6] + e[4] * q[5];
    h[0] = d[0] * n[0] - b[0] * p[2] - b[1] * p[1] - b[2] * p[0] + b[0] * p[4] + b[1] * p[3] + b[2] * p[2] + b[3] * p[1] + b[4] * p[0] - b[0] * p[6] - b[1] * p[5] - b[2] * p[4] - b[3] * p[3] - b[4] * p[2] + b[2] * p[6] + b[3] * p[5] + b[4] * p[4] - b[4] * p[6] - e[0] * q[2] - e[1] * q[1] - e[2] * q[0] + e[0] * q[4] + e[1] * q[3] + e[2] * q[2] + e[3] * q[1] + e[4] * q[0] - e[0] * q[6] - e[1] * q[5] - e[2] * q[4] - e[3] * q[3] - e[4] * q[2] + e[2] * q[6] + e[3] * q[5] + e[4] * q[4] - e[4] * q[6];
    
    VectorXd data(42);
    data << -h[8], -h[7], -h[6], -h[5], -h[4], -h[3], -h[2], -h[1], -h[0], g[6], g[5], g[4], g[3], g[2], g[1], g[0], -k[4], -k[3], -k[2], -k[1], -k[0], -h[7], 8*h[8] - 2*h[6], 7*h[7] - 3*h[5], 6*h[6] - 4*h[4], 5*h[5] - 5*h[3], 4*h[4] - 6*h[2], 3*h[3] - 7*h[1], 2*h[2] - 8*h[0], h[1], g[5], 2*g[4] - 6*g[6], 3*g[3] - 5*g[5], 4*g[2] - 4*g[4], 5*g[1] - 3*g[3], 6*g[0] - 2*g[2], -g[1], -k[3], 4*k[4] - 2*k[2], 3*k[3] - 3*k[1], 2*k[2] - 4*k[0], k[1];
    
    MatrixXd sols = solver_opt(data);
    MatrixXd sols_f = MatrixXd::Zero(4, 1);
    double thr = 10.;
    ArrayXd Vr = ArrayXd::Zero(3);
    for (int i = 0; i<sols.cols(); i++)
    {
        double c00 = (pow(sols(0,i),2)+1)*(pow(sols(0,i),2)+1);
        double c11 = (pow(sols(0,i),4)*a[4]+pow(sols(0,i),3)*a[3]+pow(sols(0,i),2)*a[2]+sols(0,i)*a[1]+a[0])/c00;
        double c12 = (pow(sols(0,i),4)*b[4]+pow(sols(0,i),3)*b[3]+pow(sols(0,i),2)*b[2]+sols(0,i)*b[1]+b[0])/c00;
        double c13 = (pow(sols(0,i),4)*c[4]+pow(sols(0,i),3)*c[3]+pow(sols(0,i),2)*c[2]+sols(0,i)*c[1]+c[0])/c00;
        double c22 = (pow(sols(0,i),4)*d[4]+pow(sols(0,i),3)*d[3]+pow(sols(0,i),2)*d[2]+sols(0,i)*d[1]+d[0])/c00;
        double c23 = (pow(sols(0,i),4)*e[4]+pow(sols(0,i),3)*e[3]+pow(sols(0,i),2)*e[2]+sols(0,i)*e[1]+e[0])/c00;
        double c33 = (pow(sols(0,i),4)*f[4]+pow(sols(0,i),3)*f[3]+pow(sols(0,i),2)*f[2]+sols(0,i)*f[1]+f[0])/c00;
        
        Matrix3d CC;
        CC << c11,c12,c13,c12,c22,c23,c13,c23,c33;
        SelfAdjointEigenSolver<Matrix3d> eigensolver(CC);
        const ArrayXcd &singularVals = eigensolver.eigenvalues();
        
        MatrixXd::Index minRow;
        const double &min = singularVals.real().minCoeff(&minRow); // smallest one of the three eigenvalues
        
        
        
//         MatrixXd ker = MatrixXd::Zero(3, 1);
//         JacobiSVD<MatrixXd> svd(CC, ComputeThinV);
//         Eigen::Vector3d D = svd.singularValues();
//         ker = svd.matrixV().rightCols(1);
//
//         ArrayXd Dr = D.real();
//         ArrayXd Vr = ker;
//
//         double min = Dr(2); // smallest one of the three eigenvalues
        
        if (fabs(min) < thr) // smallest eigenvalue corresponds to the correct solution
        {
            const VectorXd &ker = eigensolver.eigenvectors().col(minRow);
            sols_f(0, 0) = sols(0, i);	  // y, Ry = [(1-y^2) 0 2*y; 0 1+y^2 0; -2*y 0 (1-y^2)]/(1+y^2)
            sols_f(1, 0) = ker(0); // translation vector with sign ambiguous
            sols_f(2, 0) = ker(1);
            sols_f(3, 0) = ker(2);
            thr = fabs(min);
        }
        
    }
    
    plhs[0] = mxCreateDoubleMatrix(sols_f.rows(), sols_f.cols(), mxREAL);
    double *zr = mxGetPr(plhs[0]);
    for (Index i = 0; i < sols_f.size(); i++)
    {
        zr[i] = sols_f(i);
    }
}
