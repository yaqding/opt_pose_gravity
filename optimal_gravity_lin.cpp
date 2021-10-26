#include <vector>
#include <iostream>
#include <numeric>
#include <Eigen/Dense>
#include "mex.h"
#include "matrix.h"

using namespace Eigen;
using namespace std;

MatrixXd solver_opt(VectorXd &data)
{
    // Compute coefficients
    const double *d = data.data();
    
    static const int coeffs0_ind[] = {6, 11, 14, 20, 24, 26, 6, 11, 14, 20, 24, 26, 20, 24, 26};
    static const int coeffs1_ind[] = {5, 10, 13, 19, 23, 25, 5, 10, 13, 19, 23, 25, 19, 23, 25};
    static const int coeffs2_ind[] = {4, 9, 12, 18, 22, 4, 9, 12, 18, 22, 18, 22};
    static const int coeffs3_ind[] = {3, 8, 17, 21, 3, 8, 17, 21, 17, 21};
    static const int coeffs4_ind[] = {2, 7, 16, 2, 7, 16, 16};
    static const int coeffs5_ind[] = {1, 15, 1, 15, 15};
    static const int coeffs6_ind[] = {0, 0};
    
    static const int C0_ind[] = {0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18, 22, 23, 24}; // C0 C1
    static const int C2_ind[] = {0, 1, 2, 5, 6, 11, 12, 13, 16, 17, 22, 23};			// C2
    static const int C3_ind[] = {0, 1, 5, 6, 11, 12, 16, 17, 22, 23};					// C3
    static const int C4_ind[] = {0, 1, 5, 11, 12, 16, 22};								// C4
    static const int C5_ind[] = {0, 5, 11, 16, 22};										// C5
    static const int C6_ind[] = {0, 11};												// C3 C5 C7
    
    Eigen::Matrix<double, 5, 5, RowMajor> C0;
    C0.setZero();
    Eigen::Matrix<double, 5, 5, RowMajor> C1;
    C1.setZero();
    Eigen::Matrix<double, 5, 5, RowMajor> C2;
    C2.setZero();
    Eigen::Matrix<double, 5, 5, RowMajor> C3;
    C3.setZero();
    Eigen::Matrix<double, 5, 5, RowMajor> C4;
    C4.setZero();
    Eigen::Matrix<double, 5, 5, RowMajor> C5;
    C5.setZero();
    Eigen::Matrix<double, 5, 5, RowMajor> C6;
    C6.setZero();
    
    for (int i = 0; i < 15; i++)
    {
        C0(C0_ind[i]) = d[coeffs0_ind[i]];
        C1(C0_ind[i]) = d[coeffs1_ind[i]];
    }
    for (int i = 0; i < 12; i++)
    {
        C2(C2_ind[i]) = d[coeffs2_ind[i]];
    }
    for (int i = 0; i < 10; i++)
    {
        C3(C3_ind[i]) = d[coeffs3_ind[i]];
    }
    for (int i = 0; i < 7; i++)
    {
        C4(C4_ind[i]) = d[coeffs4_ind[i]];
    }
    for (int i = 0; i < 5; i++)
    {
        C5(C5_ind[i]) = d[coeffs5_ind[i]];
    }
    for (int i = 0; i < 2; i++)
    {
        C6(C6_ind[i]) = d[coeffs6_ind[i]];
    }
    
    C0(3) = 1;
    C0(14) = 1;
    
    MatrixXd M(5, 21);
    M << C6.block<5, 2>(0, 0), C5.block<5, 3>(0, 0), C4.block<5, 3>(0, 0), C3.block<5, 4>(0, 0), C2.block<5, 4>(0, 0), C1;
    M = (-C0.fullPivLu().solve(M)).eval();
    
    Eigen::Matrix<double, 21, 21, RowMajor> K;
    K.setZero();
    static const int K_ind[] = {2, 24, 47, 69, 91, 113, 135, 157, 180, 202, 224, 246, 268, 290, 312, 334};
    for (int i = 0; i < 16; i++)
    {
        K(K_ind[i]) = 1;
    }
    
    K.block<5, 21>(16, 0) = M;
    
    RealSchur<MatrixXd> schur(21);
    schur.compute(K, false);
    MatrixXd sols = MatrixXd::Zero(1, 21);
    MatrixXd S = schur.matrixT();
    int k = 0;
    for (int i = 0; i < 20; i++) // find real eigenvalues
    {
        if (((S(i, i) > 1) || (S(i, i) < -1)) && (S(i + 1, i) == 0))
        {
            sols(0, k) = 1 / S(i, i);
            k++;
        }
    }
    if (((S(20, 20) > 1) || (S(20, 20) < -1)) && (S(19, 20) == 0))
    {
        sols(0, k) = 1 / S(20, 20);
        k++;
    }
    
    
    sols.conservativeResize(1, k);
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