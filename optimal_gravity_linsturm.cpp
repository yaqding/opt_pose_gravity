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
    
    using namespace Eigen;
    using polynomial::Polynomial;
    // Compute coefficients
    const double *p = data.data();
    
    Matrix<double, 7, 1> acoeffs;
    acoeffs << p[0], p[1], p[2], p[3], p[4], p[5], p[6]; //decreasing
    Polynomial<6> a(acoeffs);
    
    Matrix<double, 5, 1> bcoeffs;
    bcoeffs << p[7], p[8], p[9], p[10], p[11];
    Polynomial<4> b(bcoeffs);
    
    Matrix<double, 3, 1> ccoeffs;
    ccoeffs << p[12], p[13], p[14];
    Polynomial<2> c(ccoeffs);
    
    Matrix<double, 6, 1> ecoeffs;
    ecoeffs << p[15], p[16], p[17], p[18], p[19], p[20];
    Polynomial<5> e(ecoeffs);
    
    Matrix<double, 4, 1> fcoeffs;
    fcoeffs << p[21], p[22], p[23], p[24];
    Polynomial<3> f(fcoeffs);
    
    Matrix<double, 2, 1> gcoeffs;
    gcoeffs << p[25], p[26];
    Polynomial<1> g(gcoeffs);
    
    Matrix<double, 16, 1> hcoeffs;
    Polynomial<9> h1 = a * f - b * e;
    Polynomial<7> h2 = a * g - c * e;
    
    // hcoeffs = (g*( g*(b*(a*f-b*e)-a*(a*g-c*e)) - c*(f*(a*f-b*e)-e*(a*g-c*e)) )).coefficients();
    hcoeffs = (g * (g * (b * h1 - a * h2) - c * (f * h1 - e * h2))).coefficients();
    Polynomial<15> h(hcoeffs);
    
    std::vector<double> roots;
    h.realRootsSturm(-1, 1, roots);
    
    // std::cout<<"roots.size() " << roots.size() << std::endl;
    
    MatrixXd sols(1, 15);
    sols.setZero();
    int k = 0;
    if (roots.size() > 0 && roots.size() < 6)
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
    
    double w[6] = {0};
    double a[3] = {0};
    double b[3] = {0};
    double c[3] = {0};
    double d[3] = {0};
    double e[3] = {0};
    double f[3] = {0};
    
    double k[3] = {0};
    double g[5] = {0};
    double p[7] = {0};
    
    double q[7] = {0};
    double n[5] = {0};
    double h[9] = {0};
    
    for (int i = 0; i < npoints; i++)
    {
        const double
                &x1o = qa[i][0],
                &y1o = qa[i][1],
                &x2o = qb[i][0],
                &y2o = qb[i][1];
        
        w[0] = y2o - y1o;
        w[1] = -x1o * y2o;
        w[2] = x1o - x2o;
        w[3] = 1 + x1o * x2o;
        w[4] = x2o * y1o - x1o * y2o;
        w[5] = -y2o;
        
        a[2] += w[1] * w[1];
        a[1] += 2 * w[1] * w[0];
        a[0] += w[0] * w[0];
        
        b[2] += w[1] * w[3];
        b[1] += w[0] * w[3] + w[1] * w[2];
        b[0] += w[0] * w[2];
        
        c[2] += w[1] * w[5];
        c[1] += w[0] * w[5] + w[1] * w[4];
        c[0] += w[0] * w[4];
        
        d[2] += w[3] * w[3];
        d[1] += 2 * w[2] * w[3];
        d[0] += w[2] * w[2];
        
        e[2] += w[3] * w[5];
        e[1] += w[2] * w[5] + w[3] * w[4];
        e[0] += w[2] * w[4];
        
        f[2] += w[5] * w[5];
        f[1] += 2 * w[4] * w[5];
        f[0] += w[4] * w[4];
    }
    
    k[2] = a[2] + d[2] + f[2];
    k[1] = a[1] + d[1] + f[1];
    k[0] = a[0] + d[0] + f[0];
    
    g[0] = -b[0] * b[0] - c[0] * c[0] - e[0] * e[0] + a[0] * d[0] + a[0] * f[0] + d[0] * f[0];
    g[1] = a[0] * d[1] - 2 * b[0] * b[1] + a[1] * d[0] - 2 * c[0] * c[1] + a[0] * f[1] + a[1] * f[0] + d[0] * f[1] + d[1] * f[0] - 2 * e[0] * e[1];
    g[2] = -b[1] * b[1] - c[1] * c[1] - e[1] * e[1] - 2 * b[0] * b[2] + a[0] * d[2] + a[1] * d[1] + a[2] * d[0] - 2 * c[0] * c[2] + a[0] * f[2] + a[1] * f[1] + a[2] * f[0] + d[0] * f[2] + d[1] * f[1] + d[2] * f[0] - 2 * e[0] * e[2];
    g[3] = a[1] * d[2] - 2 * b[1] * b[2] + a[2] * d[1] - 2 * c[1] * c[2] + a[1] * f[2] + a[2] * f[1] + d[1] * f[2] + d[2] * f[1] - 2 * e[1] * e[2];
    g[4] = -b[2] * b[2] - c[2] * c[2] - e[2] * e[2] + a[2] * d[2] + a[2] * f[2] + d[2] * f[2];
    
    p[6] = -f[2] * b[2] * b[2] + 2 * b[2] * c[2] * e[2] - d[2] * c[2] * c[2] - a[2] * e[2] * e[2] + a[2] * d[2] * f[2];
    p[5] = -f[1] * b[2] * b[2] + 2 * e[1] * b[2] * c[2] + 2 * c[1] * b[2] * e[2] - 2 * b[1] * f[2] * b[2] - d[1] * c[2] * c[2] + 2 * b[1] * c[2] * e[2] - 2 * c[1] * d[2] * c[2] - a[1] * e[2] * e[2] - 2 * a[2] * e[1] * e[2] + a[1] * d[2] * f[2] + a[2] * d[1] * f[2] + a[2] * d[2] * f[1];
    p[4] = -f[2] * b[1] * b[1] - 2 * f[1] * b[1] * b[2] + 2 * b[1] * c[1] * e[2] + 2 * b[1] * c[2] * e[1] - f[0] * b[2] * b[2] + 2 * b[2] * c[1] * e[1] + 2 * e[0] * b[2] * c[2] + 2 * c[0] * b[2] * e[2] - 2 * b[0] * f[2] * b[2] - d[2] * c[1] * c[1] - 2 * d[1] * c[1] * c[2] - d[0] * c[2] * c[2] + 2 * b[0] * c[2] * e[2] - 2 * c[0] * d[2] * c[2] - a[2] * e[1] * e[1] - 2 * a[1] * e[1] * e[2] - a[0] * e[2] * e[2] - 2 * a[2] * e[0] * e[2] + a[0] * d[2] * f[2] + a[1] * d[1] * f[2] + a[1] * d[2] * f[1] + a[2] * d[0] * f[2] + a[2] * d[1] * f[1] + a[2] * d[2] * f[0];
    p[3] = 2 * b[0] * c[1] * e[2] - c[1] * c[1] * d[1] - b[1] * b[1] * f[1] - 2 * b[0] * b[1] * f[2] - 2 * b[0] * b[2] * f[1] - a[1] * e[1] * e[1] + 2 * b[0] * c[2] * e[1] - 2 * b[1] * b[2] * f[0] + 2 * b[1] * c[0] * e[2] + 2 * b[1] * c[1] * e[1] + 2 * b[1] * c[2] * e[0] + 2 * b[2] * c[0] * e[1] + 2 * b[2] * c[1] * e[0] - 2 * c[0] * c[1] * d[2] - 2 * c[0] * c[2] * d[1] - 2 * c[1] * c[2] * d[0] + a[0] * d[1] * f[2] + a[0] * d[2] * f[1] - 2 * a[0] * e[1] * e[2] + a[1] * d[0] * f[2] + a[1] * d[1] * f[1] + a[1] * d[2] * f[0] - 2 * a[1] * e[0] * e[2] + a[2] * d[0] * f[1] + a[2] * d[1] * f[0] - 2 * a[2] * e[0] * e[1];
    p[2] = -f[2] * b[0] * b[0] - 2 * f[1] * b[0] * b[1] + 2 * e[2] * b[0] * c[0] + 2 * b[0] * c[1] * e[1] + 2 * c[2] * b[0] * e[0] - 2 * b[2] * f[0] * b[0] - f[0] * b[1] * b[1] + 2 * b[1] * c[0] * e[1] + 2 * b[1] * c[1] * e[0] - d[2] * c[0] * c[0] - 2 * d[1] * c[0] * c[1] + 2 * b[2] * c[0] * e[0] - 2 * c[2] * d[0] * c[0] - d[0] * c[1] * c[1] - a[2] * e[0] * e[0] - 2 * a[1] * e[0] * e[1] - 2 * a[0] * e[2] * e[0] - a[0] * e[1] * e[1] + a[0] * d[0] * f[2] + a[0] * d[1] * f[1] + a[0] * d[2] * f[0] + a[1] * d[0] * f[1] + a[1] * d[1] * f[0] + a[2] * d[0] * f[0];
    p[1] = -f[1] * b[0] * b[0] + 2 * e[1] * b[0] * c[0] + 2 * c[1] * b[0] * e[0] - 2 * b[1] * f[0] * b[0] - d[1] * c[0] * c[0] + 2 * b[1] * c[0] * e[0] - 2 * c[1] * d[0] * c[0] - a[1] * e[0] * e[0] - 2 * a[0] * e[1] * e[0] + a[0] * d[0] * f[1] + a[0] * d[1] * f[0] + a[1] * d[0] * f[0];
    p[0] = -f[0] * b[0] * b[0] + 2 * b[0] * c[0] * e[0] - d[0] * c[0] * c[0] - a[0] * e[0] * e[0] + a[0] * d[0] * f[0];
    
    VectorXd data(27);
    data << -p[6], -p[5], -p[4], -p[3], -p[2], -p[1], -p[0], g[4], g[3], g[2], g[1], g[0], -k[2], -k[1], -k[0], 6 * p[6], 5 * p[5], 4 * p[4], 3 * p[3], 2 * p[2], p[1], -4 * g[4], -3 * g[3], -2 * g[2], -g[1], 2 * k[2], k[1];
    
    MatrixXd sols = solver_opt(data);
    MatrixXd sols_f = MatrixXd::Zero(4, 1);
    double r = 0.0;
    Vector3d t;
    double thr = std::numeric_limits<double>::max();
    ArrayXd Vr = ArrayXd::Zero(3);
    for (int i = 0; i < sols.cols(); i++)
    {
        const double &sols0i = sols(0, i);
        
        double pow_2_sol_0_i = pow(sols0i, 2);
        // double pow_3_sol_0_i = pow(sols0i, 3);
        // double pow_4_sol_0_i = pow(sols0i, 4);
        
        // double c00 = (pow_2_sol_0_i + 1)*(pow_2_sol_0_i + 1);
        double c11 = (pow_2_sol_0_i * a[2] + sols0i * a[1] + a[0]);
        double c12 = (pow_2_sol_0_i * b[2] + sols0i * b[1] + b[0]);
        double c13 = (pow_2_sol_0_i * c[2] + sols0i * c[1] + c[0]);
        double c22 = (pow_2_sol_0_i * d[2] + sols0i * d[1] + d[0]);
        double c23 = (pow_2_sol_0_i * e[2] + sols0i * e[1] + e[0]);
        double c33 = (pow_2_sol_0_i * f[2] + sols0i * f[1] + f[0]);
        
        Matrix3d CC;
        CC << c11, c12, c13, c12, c22, c23, c13, c23, c33;
        SelfAdjointEigenSolver<Matrix3d> eigensolver(CC);
        const ArrayXcd &singularVals = eigensolver.eigenvalues();
        
        MatrixXd::Index minRow;
        const double &min =
                singularVals.real().minCoeff(&minRow); // smallest one of the three eigenvalues
        
        if (fabs(min) < thr) // smallest eigenvalue corresponds to the correct solution
        {
            const VectorXd &ker = eigensolver.eigenvectors().col(minRow);
            sols_f(0, 0) = sols(0, i);	  // y, Ry = [(1-y^2) 0 2*y; 0 1+y^2 0; -2*y 0 (1-y^2)]/(1+y^2)
            sols_f(1, 0) = ker(0, 0); // translation vector with sign ambiguous
            sols_f(2, 0) = ker(1, 0);
            sols_f(3, 0) = ker(2, 0);
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