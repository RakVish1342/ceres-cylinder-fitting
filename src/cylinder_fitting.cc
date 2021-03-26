#include "ceres/ceres.h"
#include "ceres/jet.h"

#include <math.h>
#include <random>

#include "glog/logging.h"

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
        x(x), 
        y(y), 
        z(z), 
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
    
        /*

        Eigen::Matrix<T, 3, 1> n_recon = Eigen::Matrix<T, 3, 1>(ceres::cos(phi[0])*ceres::sin(theta[0]), 
                                                                ceres::sin(phi[0])*ceres::sin(theta[0]),
                                                                ceres::cos(theta[0])
                                                                );

        // Partial deriv of n wrt theta
        Eigen::Matrix<T, 3, 1> n_theta = Eigen::Matrix<T, 3, 1>(+ ceres::cos(phi[0])*ceres::cos(theta[0]), 
                                                                + ceres::sin(phi[0])*ceres::cos(theta[0]),
                                                                - ceres::sin(theta[0])
                                                                );

        // n_phi / sin(theta)
        // HOW DO YOU ADD A CONSTANT VALUE AS THE THIRD ELEMENT?? NEED TO USE A JET STRUCT OBJ WITH a=0.0?? 
        // Even if I figure out how to do that, I think it will cause an issue in the evaluation step when a double value number will be reqd.
        // It will only work in the jacobian/derivative phase. 
        // const T CNST 
        // Jet<T, N>(cos(f.a), -sin(f.a) * f.v);
        Eigen::Matrix<T, 3, 1> n_phi_bar = Eigen::Matrix<T, 3, 1>(  - ceres::sin(phi[0]), 
                                                                    + ceres::cos(phi[0]),
                                                                    + 0.0
                                                                    // ceres::Jet<T, 1> ()
                                                                    // ceres::sin(0)
                                                                    // const ceres::Jet<double, 5>& ()
                                                                    // CNST
                                                                );

        Eigen::Matrix<T, 3, 1> a_recon = n_theta*ceres::cos(alpha[0]) + n_phi_bar*ceres::sin(alpha[0]);

        residual[0] = (kappa[0]/2.0)*( p.norm()*p.norm() - 2*rho[0]*p.dot(n_recon) - p.dot(a_recon)*p.dot(a_recon) + rho[0]*rho[0] )
                        + rho[0] - p.dot(n_recon);

        */

       residual[0] = (kappa[0]/2.0) *(  ceres::sqrt(x*x + y*y + z*z)*ceres::sqrt(x*x + y*y + z*z)  

                                        - 2.0*rho[0]*( x*ceres::cos(phi[0])*ceres::sin(theta[0]) + y*ceres::sin(phi[0])*ceres::sin(theta[0]) + z*ceres::cos(theta[0]) )
       
                                        - ( + x*(ceres::cos(phi[0])*ceres::cos(theta[0])*ceres::cos(alpha[0]) - ceres::sin(phi[0])*ceres::sin(alpha[0]) )
                                            + y*(ceres::sin(phi[0])*ceres::cos(theta[0])*ceres::cos(alpha[0]) + ceres::cos(phi[0])*ceres::sin(alpha[0]) )
                                            + z*(- ceres::sin(theta[0])*ceres::cos(alpha[0]) ) 
                                        ) 
                                        * ( + x*(ceres::cos(phi[0])*ceres::cos(theta[0])*ceres::cos(alpha[0]) - ceres::sin(phi[0])*ceres::sin(alpha[0]) )
                                            + y*(ceres::sin(phi[0])*ceres::cos(theta[0])*ceres::cos(alpha[0]) + ceres::cos(phi[0])*ceres::sin(alpha[0]) )
                                            + z*(- ceres::sin(theta[0])*ceres::cos(alpha[0]) )
                                        )
       
                                        + rho[0]*rho[0] 
       
                                    )
       
                                    + rho[0]
       
                                    - ( x*ceres::cos(phi[0])*ceres::sin(theta[0]) + y*ceres::sin(phi[0])*ceres::sin(theta[0]) + z*ceres::cos(theta[0]) );
      

        // OR
        // Eigen::Vector3d p_hat = p - rho*n_recon;
        // residual[0] = (kappa/2.0)*( p_hat.cross(a_recon)*p_hat.cross(a_recon) ) - p_hat.dot(n_recon);

        // Testing inbuilt cos function
        /*
        // residual[0] = std::cos(phi[0]); // from cmath.h
        // residual[0] = cos(phi[0]); // Same as ceres::cos()
        residual[0] = ceres::cos(phi[0]);
        */

        return true;
    }

private:
    double x;
    double y; 
    double z;
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
    double kappa = 4.0;
    double theta = 3.1415/2;
    double phi = 0.0;
    double alpha = 0.0;

    Problem problem;
    // for (int i = 0; i < kNumObservations; ++i)
    // {
        
    //     problem.AddResidualBlock(
    //         new AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1, 1>(    // Dimensions of: residual, rho, kappa, theta, phi, alpha
    //             new CostFunctor(data[3*i], data[3*i + 1], data[3*i + 2]) ),
    //         NULL,
    //         &rho,
    //         &kappa,
    //         &theta,
    //         &phi,
    //         &alpha);
    // }

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    // std::default_random_engine gen; // uses a fixed seed
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    double epsilon = 0.01;
    std::uniform_real_distribution<double> dis(-epsilon, +epsilon);

    for (int i = 0; i < kNumObservations; ++i)
    {
        problem.AddResidualBlock(
            new AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1, 1>(    // Dimensions of: residual, rho, kappa, theta, phi, alpha
                new CostFunctor(data[3*i]+dis(gen), data[3*i + 1]+dis(gen), data[3*i + 2]+dis(gen)) ),
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




    std::cout << "--- FINAL CYLINDER VECTORS ---" << std::endl;

    Eigen::Vector3d n_recon (   std::cos(phi)*std::sin(theta), 
                                std::sin(phi)*std::sin(theta),
                                std::cos(theta)
                            );

    // Partial deriv of n wrt theta
    Eigen::Vector3d n_theta (   + std::cos(phi)*std::cos(theta), 
                                + std::sin(phi)*std::cos(theta),
                                - std::sin(theta)
                            );

    // n_phi / sin(theta)
    Eigen::Vector3d n_phi_bar ( - std::sin(phi), 
                                + std::cos(phi),
                                + 0.0
                            );

    Eigen::Vector3d a_recon = n_theta*std::cos(alpha) + n_phi_bar*ceres::sin(alpha);

    std::cout << "--- n ---" << std::endl;
    std::cout << n_recon << std::endl;
    std::cout << "--- a ---" << std::endl;
    std::cout << a_recon << std::endl;

    return 0;
}


// TODO:
/*

*   /usr/local/include/ceres/autodiff_cost_function.h:200:8:   required from here
    /home/rxth/rakshith/tmp/tmp_ceres/curve_fitting/src/cylinder_fitting.cc:105:61: error: no matching function for call to ‘cos(const ceres::Jet<double, 5>&)’
    105 |         Eigen::Vector3d n_recon = Eigen::Vector3d(  std::cos(phi[0])*std::sin(theta[0]),

*Possible options: Numerical diff OR analytical diff
*Numerical slow and error prone
*Analytical is complex to code up
**Recommended to use autodiff as much as possible

StackOverflow answers about Eigen and ceres::Jet type: 
**https://groups.google.com/g/ceres-solver/c/L6nBBI07dE8 (NumericDiffCostFunction and cost_function_to_functor.h)
    http://ceres-solver.org/interfacing_with_autodiff.html

**https://groups.google.com/g/ceres-solver/c/O-jk48z6qvU (Use templated T Matrices while coding up Eigen vectors instead of float/double type vectors...but latter few answers in above link say that won't work )

What I need to do:
1. I can avoid use of eigen, by performing norm and dot pdt calculation manually
2. BUUUT, I need to avoid cos/sin?? How can I do this? Do I need to provide the jacobian manually using analytical differentiation? If so, look at more complex examples that tutorial example (to see how to take diff of function which depends on multiple variables.)

*** Jet.h seems to have implementation of differentials for standard non-linear functions (eg. trigonometric functions)

*/  








/*

rxth@alienware:~/rakshith/installs/ceres/ceres-solver-2.0.0/examples$ grep -Rin eigen
slam/pose_graph_2d/pose_graph_2d_error_term.h:36:#include "Eigen/Core"
slam/pose_graph_2d/pose_graph_2d_error_term.h:42:Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw_radians) {
slam/pose_graph_2d/pose_graph_2d_error_term.h:46:  Eigen::Matrix<T, 2, 2> rotation;
slam/pose_graph_2d/pose_graph_2d_error_term.h:65:                       const Eigen::Matrix3d& sqrt_information)
slam/pose_graph_2d/pose_graph_2d_error_term.h:78:    const Eigen::Matrix<T, 2, 1> p_a(*x_a, *y_a);
slam/pose_graph_2d/pose_graph_2d_error_term.h:79:    const Eigen::Matrix<T, 2, 1> p_b(*x_b, *y_b);
slam/pose_graph_2d/pose_graph_2d_error_term.h:81:    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals_ptr);
slam/pose_graph_2d/pose_graph_2d_error_term.h:98:                                     const Eigen::Matrix3d& sqrt_information) {
slam/pose_graph_2d/pose_graph_2d_error_term.h:105:  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
slam/pose_graph_2d/pose_graph_2d_error_term.h:109:  const Eigen::Vector2d p_ab_;
slam/pose_graph_2d/pose_graph_2d_error_term.h:113:  const Eigen::Matrix3d sqrt_information_;
slam/pose_graph_2d/types.h:40:#include "Eigen/Core"
slam/pose_graph_2d/types.h:75:  Eigen::Matrix3d information;
slam/pose_graph_2d/pose_graph_2d.cc:87:    const Eigen::Matrix3d sqrt_information =
slam/pose_graph_3d/pose_graph_3d.cc:62:      new EigenQuaternionParameterization;
slam/pose_graph_3d/pose_graph_3d.cc:77:    const Eigen::Matrix<double, 6, 6> sqrt_information =
slam/pose_graph_3d/pose_graph_3d.cc:136:                Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
slam/pose_graph_3d/pose_graph_3d.cc:143:                   Eigen::aligned_allocator<std::pair<const int, Pose3d>>>::
slam/pose_graph_3d/types.h:39:#include "Eigen/Core"
slam/pose_graph_3d/types.h:40:#include "Eigen/Geometry"
slam/pose_graph_3d/types.h:46:  Eigen::Vector3d p;
slam/pose_graph_3d/types.h:47:  Eigen::Quaterniond q;
slam/pose_graph_3d/types.h:52:  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
slam/pose_graph_3d/types.h:67:                 Eigen::aligned_allocator<std::pair<const int, Pose3d>>>
slam/pose_graph_3d/types.h:83:  Eigen::Matrix<double, 6, 6> information;
slam/pose_graph_3d/types.h:88:  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
slam/pose_graph_3d/types.h:106:typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d>>
slam/pose_graph_3d/pose_graph_3d_error_term.h:34:#include "Eigen/Core"
slam/pose_graph_3d/pose_graph_3d_error_term.h:73:                       const Eigen::Matrix<double, 6, 6>& sqrt_information)
slam/pose_graph_3d/pose_graph_3d_error_term.h:82:    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
slam/pose_graph_3d/pose_graph_3d_error_term.h:83:    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);
slam/pose_graph_3d/pose_graph_3d_error_term.h:85:    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
slam/pose_graph_3d/pose_graph_3d_error_term.h:86:    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);
slam/pose_graph_3d/pose_graph_3d_error_term.h:89:    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
slam/pose_graph_3d/pose_graph_3d_error_term.h:90:    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;
slam/pose_graph_3d/pose_graph_3d_error_term.h:93:    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);
slam/pose_graph_3d/pose_graph_3d_error_term.h:96:    Eigen::Quaternion<T> delta_q =
slam/pose_graph_3d/pose_graph_3d_error_term.h:102:    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
slam/pose_graph_3d/pose_graph_3d_error_term.h:115:      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
slam/pose_graph_3d/pose_graph_3d_error_term.h:120:  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
slam/pose_graph_3d/pose_graph_3d_error_term.h:126:  const Eigen::Matrix<double, 6, 6> sqrt_information_;
slam/pose_graph_3d/README.md:11:The example also illustrates how to use Eigen's geometry module with Ceres'
slam/pose_graph_3d/README.md:13:use Eigen's quaternion which uses the Hamiltonian convention but has different
slam/pose_graph_3d/README.md:16:order for Ceres's quaternion is [q_w, q_x, q_y, q_z] where as Eigen's quaternion
bal_problem.cc:39:#include "Eigen/Core"
bal_problem.cc:47:typedef Eigen::Map<Eigen::VectorXd> VectorRef;
bal_problem.cc:48:typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
bal_problem.cc:228:  Eigen::VectorXd inverse_rotation = -angle_axis_ref;
bal_problem.cc:252:  Eigen::Vector3d median;
nist.cc:78:#include "Eigen/Core"
nist.cc:122:DEFINE_bool(approximate_eigenvalue_bfgs_scaling,
nist.cc:124:            "Use approximate eigenvalue scaling in (L)BFGS line search.");
nist.cc:155:using Eigen::Dynamic;
nist.cc:156:using Eigen::RowMajor;
nist.cc:157:typedef Eigen::Matrix<double, Dynamic, 1> Vector;
nist.cc:158:typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> Matrix;
nist.cc:491:  options->use_approximate_eigenvalue_bfgs_scaling =
nist.cc:492:      FLAGS_approximate_eigenvalue_bfgs_scaling;
nist.cc:606:      ceres::TinySolverCostFunctionAdapter<Eigen::Dynamic, num_parameters> cfa(
nist.cc:609:          ceres::TinySolverCostFunctionAdapter<Eigen::Dynamic, num_parameters>>
nist.cc:618:      Eigen::Matrix<double, num_parameters, 1> x;
denoising.cc:76:              "Options are: suite_sparse, cx_sparse and eigen_sparse");
Makefile.example:42:# The place you unpacked or cloned Eigen. If Eigen was installed from packages,
Makefile.example:44:EIGEN_SRC_DIR := /home/keir/src/eigen-3.0.5
Makefile.example:47:            -I$(EIGEN_SRC_DIR)
Makefile.example:75:# Disabling debug asserts via -DNDEBUG helps make Eigen faster, at the cost of
libmv_homography.cc:66:typedef Eigen::NumTraits<double> EigenDouble;
libmv_homography.cc:68:typedef Eigen::MatrixXd Mat;
libmv_homography.cc:69:typedef Eigen::VectorXd Vec;
libmv_homography.cc:70:typedef Eigen::Matrix<double, 3, 3> Mat3;
libmv_homography.cc:71:typedef Eigen::Matrix<double, 2, 1> Vec2;
libmv_homography.cc:72:typedef Eigen::Matrix<double, Eigen::Dynamic, 8> MatX8;
libmv_homography.cc:73:typedef Eigen::Vector3d Vec3;
libmv_homography.cc:109:void SymmetricGeometricDistanceTerms(const Eigen::Matrix<T, 3, 3>& H,
libmv_homography.cc:110:                                     const Eigen::Matrix<T, 2, 1>& x1,
libmv_homography.cc:111:                                     const Eigen::Matrix<T, 2, 1>& x2,
libmv_homography.cc:114:  typedef Eigen::Matrix<T, 3, 1> Vec3;
libmv_homography.cc:155:  typedef Eigen::Matrix<T, 8, 1> Parameters;     // a, b, ... g, h
libmv_homography.cc:156:  typedef Eigen::Matrix<T, 3, 3> Parameterized;  // H
libmv_homography.cc:250:    typedef Eigen::Matrix<T, 3, 3> Mat3;
libmv_homography.cc:251:    typedef Eigen::Matrix<T, 2, 1> Vec2;
libmv_homography.cc:322:      x1, x2, H, EigenDouble::dummy_precision());
bundle_adjuster.cc:95:DEFINE_string(dense_linear_algebra_library, "eigen",
bundle_adjuster.cc:96:              "Options are: eigen and lapack.");
ellipse_approximation.cc:278:                                        const Eigen::Vector2d& y)
ellipse_approximation.cc:329:                                     const Eigen::Vector2d& y) {
ellipse_approximation.cc:339:  const Eigen::Vector2d y_;
ellipse_approximation.cc:386:  // Eigen::MatrixXd is column major so we define our own MatrixXd which is
ellipse_approximation.cc:387:  // row major. Eigen::VectorXd can be used directly.
ellipse_approximation.cc:388:  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ellipse_approximation.cc:390:  using Eigen::VectorXd;
BUILD:38:    "@com_gitlab_libeigen_eigen//:eigen",
libmv_bundle_adjuster.cc:117:typedef Eigen::Matrix<double, 3, 3> Mat3;
libmv_bundle_adjuster.cc:118:typedef Eigen::Matrix<double, 6, 1> Vec6;
libmv_bundle_adjuster.cc:119:typedef Eigen::Vector3d Vec3;
libmv_bundle_adjuster.cc:120:typedef Eigen::Vector4d Vec4;
rxth@alienware:~/rakshith/installs/ceres/ceres-solver-2.0.0/examples$ 
rxth@alienware:~/rakshith/installs/ceres/ceres-solver-2.0.0/examples$ 
rxth@alienware:~/rakshith/installs/ceres/ceres-solver-2.0.0/examples$ 

*/