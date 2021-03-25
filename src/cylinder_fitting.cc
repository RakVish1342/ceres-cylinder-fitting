#include "ceres/ceres.h"
#include "glog/logging.h"

#include <math.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


const int kNumObservations = 11;
const double data[] = {
    +5.0,                       0.0,                        0.0, 
    +5.0*std::cos(M_PI/6.0),    5.0*std::sin(M_PI/6.0),     0.0, 
    +5.0*std::cos(M_PI/3.0),    5.0*std::sin(M_PI/3.0),     0.0, 
    -5.0*std::cos(M_PI/3.0),    5.0*std::sin(M_PI/3.0),     0.0, 
    -5.0*std::cos(M_PI/6.0),    5.0*std::sin(M_PI/6.0),     0.0, 
    -5.0,                       0.0,                        0.0, 
    +5.0,                       0.0,                        1.0, 
    +5.0*std::cos(M_PI/4.0),    5.0*std::sin(M_PI/4.0),     1.0, 
    +5.0*std::cos(M_PI/2.0),    5.0*std::sin(M_PI/2.0),     1.0, 
    -5.0*std::cos(M_PI/4.0),    5.0*std::sin(M_PI/4.0),     1.0, 
    -5.0,                       0.0,                        1.0,    
};


struct CostFunctor
{
    CostFunctor(double x, double y, double z): 
        p(Eigen::Vector3d(x, y, z)) 
    {}

    // Distance from cylinder
    template <typename T>
    bool operator()(const T *const rho, 
                    const T *const kappa, 
                    const T *const phi, 
                    const T *const theta, 
                    const T *const alpha, 
                    T *residual) const
    {
        // Remember that all variables accepted as args are assumed to be "const double *const &" (reference to a const pointer to a const double variable)
        // and treated as vectors/arrays of const ceres::Jet& type. So, need to reference [0]th element usually even if it is a scalar value.
        Eigen::Vector3d n_recon = Eigen::Vector3d(  std::cos(phi[0])*std::sin(theta[0]), 
                                            std::sin(phi[0])*std::sin(theta[0]),
                                            std::cos(theta[0])
                                        );
        // Partial deriv of n wrt theta
        Eigen::Vector3d n_theta = Eigen::Vector3d(  + std::cos(phi[0])*std::cos(theta[0]), 
                                    + std::sin(phi[0])*std::cos(theta[0]),
                                    - std::sin(theta[0])
                                );
        // n_phi / sin(theta)
        Eigen::Vector3d n_phi_bar = Eigen::Vector3d(    - std::sin(phi[0]), 
                                        + std::cos(phi[0]),
                                        + 0
                                    );
        Eigen::Vector3d a_recon = n_theta*std::cos(alpha[0]) + n_phi_bar*std::sin(alpha[0]);
        residual[0] = (kappa[0]/2.0)*( p.norm()*p.norm() - 2*rho[0]*p.dot(n_recon) - p.dot(a_recon)*p.dot(a_recon) + rho[0]*rho[0] )
                        + rho[0] - p.dot(n_recon);

        // OR
        // Eigen::Vector3d p_hat = p - rho*n_recon;
        // residual[0] = (kappa/2.0)*( p_hat.cross(a_recon)*p_hat.cross(a_recon) ) - p_hat.dot(n_recon);

        return true;
    }

private:
    Eigen::Vector3d p;

    // The () operator overload of CostFunctor needs to be of const type when provided to AddResidualBlock, else is not recognized
    // Making the member function const prevents altering all (even non-const) member variables within that function's scope
    // So, make all temporary/intermediate variables into local variables and not member variabels. Anyway they aren't actually 
    // used elsewhere within the class.
    // Eigen::Vector3d n_recon;
    // Eigen::Vector3d n_theta;
    // Eigen::Vector3d n_phi;
    // Eigen::Vector3d n_phi_bar;
    // Eigen::Vector3d a_recon;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);

    // ceres::problem::AddResidualBlock mandates double precision type.
    // float not accepted.
    double rho = 0.0;
    double kappa = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    double alpha = 0.0;

    Problem problem;
    for (int i = 0; i < kNumObservations; ++i)
    {
        
        problem.AddResidualBlock(
            new AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1, 1>(    // Dimensions of: residual, rho, kappa, theta, phi, alpha
                new CostFunctor(data[3*i], data[3*i + 1], data[3*i + 2]) ),
            NULL,
            &rho,
            &kappa,
            &theta,
            &phi,
            &alpha);
    }

    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "--- INIT ---" << std::endl;
    std::cout << rho << ", " << kappa << ", " << theta << ", " << phi << ", " << alpha << std::endl;
    std::cout << "--- FINAL ---" << std::endl;
    std::cout << rho << ", " << kappa << ", " << theta << ", " << phi << ", " << alpha << std::endl;

    return 0;
}