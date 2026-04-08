/* ---------------------------------------------------------------------
 * Declaration of MixedStokesProblem class template
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Ghumman, University of Pittsburgh, 2024
 * and Manu Jayadharan, Northwestern University, 2024
 * based on the Eldar Khattatov's Elasticity DD implementation from 2017
 */

#ifndef STOKES_MFE_STOKES_MFE_H
#define STOKES_MFE_STOKES_MFE_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
// #include "utilities.h"
#include "projector.h"

namespace mfe_stokes
{
  using namespace dealii;

  // Mixed Stokes Domain Decomposition class template
  template <int dim>
  class MixedStokesProblem
  {
  using FEFunction = Functions::FEFieldFunction<dim, BlockVector<double>>;
  public:
  MixedStokesProblem (ParameterHandler &prm,
                      unsigned int &degree,
                      unsigned int &degree_biot,
                      std::vector<unsigned int> &n_domains,
                      const unsigned int &subdomain_type,
                      std::vector<int> &neighbors,
                      std::vector<unsigned int> &neighbor_type,
                      std::vector<unsigned int> &initial_refinement,
                      std::vector<unsigned int> &initial_refinement_mortar,
                      MPI_Comm mpi_communicator_stokes,
                      const unsigned int preconditioner_type,
                      const unsigned int mortar_degree = 2,
                      const unsigned int mortar_type = 0,
                      const unsigned int mortar_degree_biot = 1,
                      const bool cont_mortar_flag = 1);

    void
    run(unsigned int cycle);

    void
    assemble_rhs_star(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                      FEFaceValues<dim> &fe_face_values,
                      FEFaceValues<dim> &fe_face_values_mortar_biot);

    void 
    preconditioner(unsigned int cycle,
                   unsigned int preconditioner_type);

    void
    solve_bar();

    void
    solve_star();

    void 
    write_dof_locations(const std::string &  filename) const;

    void
    output_dof_results(const unsigned int cycle) const;

    void
    output_results(const unsigned int cycle);

    void 
    compute_errors(const unsigned int &cycle);

    void
    print_individual_convg_table_l2();

    void
    print_individual_convg_table_h1();

    void
    print_total_convg_table_l2();

    void
    print_total_convg_table_h1();

    void
    local_cg();

    void
    local_gmres(unsigned int &cycle);


  private:
    MPI_Comm   mpi_communicator;
    MPI_Comm   mpi_communicator_stokes;
    MPI_Status mpi_status;
    const unsigned int this_mpi;
    const unsigned int n_processes;
    const unsigned int this_mpi_stokes;
    const unsigned int n_processes_stokes;
    ParameterHandler &prm;
    ParameterHandler prm_fxn;
    unsigned int degree;
    unsigned int qdegree;
    double gamma;
    const unsigned int subdomain_type;
    bool stokes_neighbor;
    bool biot_neighbor;
    std::vector<unsigned int> external_bdry_ids;
    bool cont_mortar_flag;
    const unsigned int mortar_degree;
    const unsigned int mortar_type;
    const unsigned int mortar_degree_biot;
    double       tol;
    double       initial_val;
    double       avg_eigenvalue;
    std::vector<double> nbr_avg_eigenvalue;
    std::vector<std::vector<double>> jacobi_scaling;
    bool multiscale_basis_flag;
    const unsigned int preconditioner_type;
    std::vector<unsigned int> initial_refinement; // initial refinement for this subdomain mesh
    std::vector<unsigned int> initial_refinement_mortar; // initial refinement for the mortar mesh
    // std::vector<int> neighbors;
    // std::vector<unsigned int> neighbor_type;

    void
    interpret_parameters();

    void
    setup_system();

    void
    setup_system_biot();

    void
    assemble_system();

    void
    get_interface_dofs();

    double
    vect_norm(std::vector<double> v);

    void
    givens_rotation(double v1, double v2, double &cs, double &sn);

    void
	  apply_givens_rotation(std::vector<double> &h, 
                          std::vector<double> &cs, 
                          std::vector<double> &sn,
    							        unsigned int k_iteration);

    void
	  back_solve(const std::vector<std::vector<double>> &H, 
               const std::vector<double>              &beta, 
               std::vector<double>                    &y,
               const unsigned int                     &k);

    void
    interface_inner_product(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                            std::unique_ptr<FEFunction>      &solution_bar_function,
                            std::unique_ptr<FEFunction>      &solution_star_function,
                            FEFaceValues<dim>                &fe_face_values_mortar,
                            FEFaceValues<dim>                &fe_face_values_mortar_biot,
                            std::vector<Projector::Projector<dim>> &P_mortar);

    double
    inner_product_mortar(std::vector<std::vector<double>> &v_1,
                         std::vector<std::vector<double>> &v_2,
                         FEFaceValues<dim> &fe_face_values_mortar,
                         FEFaceValues<dim> &fe_face_values_mortar_biot);

    void
    compute_residual(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                    std::unique_ptr<FEFunction>      &solution_bar_function,
                    std::unique_ptr<FEFunction>      &solution_star_function,
                    FEFaceValues<dim>               &fe_face_values,
                    FEFaceValues<dim>               &fe_face_values_mortar,
                    FEFaceValues<dim>                &fe_face_values_mortar_biot,
                    std::vector<std::vector<double>> &interface_data,
                    std::vector<Projector::Projector<dim>> &P_mortar,
                    std::vector<std::vector<double>> &r,
                    double                           &r_norm,
                    std::vector<std::vector<double>> &r_b,
                    double                           &b_norm);

    void 
    interface_operator(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                       std::unique_ptr<FEFunction>       &solution_bar_function,
                       std::unique_ptr<FEFunction>       &solution_star_function,
                       std::vector<std::vector<double>> &interface_data,
                       std::vector<std::vector<double>> &interface_data_send,
                       std::vector<std::vector<double>> &interface_data_receive,
                       FEFaceValues<dim>                &fe_face_values,
                       FEFaceValues<dim>                &fe_face_values_mortar,
                       FEFaceValues<dim>                &fe_face_values_mortar_biot,
                       std::vector<Projector::Projector<dim>> &P_mortar,
                       std::vector<std::vector<double>> &Ap);
              
    void
    arnoldi(std::vector<std::unique_ptr<FEFunction>> &interface_function,
            std::unique_ptr<FEFunction>       &solution_bar_function,
            std::unique_ptr<FEFunction>       &solution_star_function,
            unsigned int                      &k_counter, 
            std::vector<std::vector<double>> &interface_data,
            std::vector<std::vector<double>> &interface_data_send,
            std::vector<std::vector<double>> &interface_data_receive,
            FEFaceValues<dim>                &fe_face_values,
            FEFaceValues<dim>                &fe_face_values_mortar,
            FEFaceValues<dim>                &fe_face_values_mortar_biot,
            std::vector<Projector::Projector<dim>> &P_mortar,
            std::vector<std::vector<double>> &Ap,
            std::vector<std::vector<double>> &q,
            std::vector<std::vector<std::vector<double>>>	&Q_side,
            std::vector<std::vector<double>>	            &H);

    void
    compute_multiscale_basis(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                            FEFaceValues<dim>                &fe_face_values,
                            FEFaceValues<dim>                &fe_face_values_mortar_biot,
                            std::vector<Projector::Projector<dim>> &P_mortar);

    unsigned int       gmres_iteration;
    unsigned int       maxiter;
    double             residual;
    // Number of subdomains in the computational domain
    std::vector<unsigned int> n_domains;

    // FE degree and DD parameters
    // const unsigned int degree;
    // const bool ess_dir_flag;
    // const bool mortar_flag;
    // const bool cont_mortar_flag; // Flag for continuous mortar
    // const bool print_interface_matrix_flag; // Flag to print interface matrix
    // const unsigned int mortar_degree;
    // const unsigned int iter_meth_flag;
    // unsigned int       cg_iteration;
    // const double       tolerance;
    // unsigned int       qdegree;
    double h;
    double h_old;
    double u_l2_error;
    double p_l2_error;
    double u_l2_error_old;
    double p_l2_error_old;
    double u_l2_error_total;
    double p_l2_error_total;
    double u_l2_error_old_total;
    double p_l2_error_old_total;
    

    // Neighbors and interface information
    std::vector<int>                       neighbors;
    std::vector<unsigned int>              neighbor_type;
    std::vector<unsigned int>              faces_on_interface;
    std::vector<unsigned int>              faces_on_interface_mortar;
    std::vector<std::vector<unsigned int>> interface_dofs;
    std::vector<std::vector<unsigned int>> interface_dofs_fe;
    std::vector<unsigned int>              interface_dofs_total;
    std::vector<std::vector<unsigned int>> interface_dofs_find_neumann;
    std::vector<unsigned int>              repeated_dofs;
    std::vector<unsigned int>              repeated_dofs_neumann;
    std::vector<unsigned int>              repeated_dofs_neumann_corner;

    unsigned long n_velocity_interface;
    unsigned long n_velocity_interface_fe;
    unsigned long n_stress_mortar;
    unsigned long n_disp_mortar;
    unsigned long n_rot_mortar;
    unsigned long n_flux_mortar;
    unsigned long n_pressure_mortar;
    int           interface_dofs_size;

    // Subdomain coordinates (assuming logically rectangular blocks)
    Point<dim> p1;
    Point<dim> p2;

    // Fine triangulation
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    // Mortar triangulation
    Triangulation<dim> triangulation_mortar;
    FESystem<dim>      fe_mortar;
    DoFHandler<dim>    dof_handler_mortar;

    // The fe space of Biot stress and velocity 
    // in the meaning of displacement and pressure, respectively
    // FESystem<dim>      fe_biot_stress;
    // DoFHandler<dim>    dof_handler_biot_stress;
    FESystem<dim>      fe_biot_mortar;
    DoFHandler<dim>    dof_handler_biot_mortar;   

    // // The fe space of Biot velocity in the meaning of pressure
    // FESystem<dim>      fe_darcy_vel;
    // DoFHandler<dim>    dof_handler_darcy_vel;

    // Star and bar problem data structures
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> system_matrix_star;
    SparseDirectUMFPACK       A_direct;
    SparseDirectUMFPACK       B_direct;
    std::shared_ptr<SparseDirectUMFPACK> A_preconditioner;
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_mortar;
    AffineConstraints<double> constraints_star;
    BlockSparsityPattern      preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;
    FullMatrix<double>        interface_matrix;


    BlockVector<double> solution_bar;
    BlockVector<double> solution_star;
    BlockVector<double> solution;
    BlockVector<double> exact_solution_at_nodes;
    BlockVector<double> biot_stress_vec;
    BlockVector<double> darcy_velocity_vec;
    std::vector<BlockVector<double>> exact_normal_stress_at_nodes_fe;
    std::vector<BlockVector<double>> exact_normal_stress_at_nodes_mortar;
    BlockVector<double> system_rhs_bar;
    BlockVector<double> system_rhs_star;
    std::vector<BlockVector<double>> interface_fe_function;
    std::vector<BlockVector<double>> interface_fe_function_fe;

    

    // Mortar data structures
    std::vector<BlockVector<double>>   interface_fe_function_mortar;
    std::vector<BlockVector<double>>   interface_fe_function_mortar_fe;
    std::vector<BlockVector<double>>   solution_bar_mortar;
    std::vector<BlockVector<double>>   solution_star_mortar;
    std::vector<BlockVector<double>>   solution_bar_mortar_temp;
    std::vector<BlockVector<double>>   solution_star_mortar_temp;
    BlockVector<double>                mortar_vec;
    BlockVector<double>                mortar_vec_biot;
    std::vector<BlockVector<double>>   multiscale_basis;


    // Output extra
    ConditionalOStream pcout;
    ConvergenceTable   convergence_table_l2;
    ConvergenceTable   convergence_table_h1;
    ConvergenceTable   convergence_table_l2_total;
    ConvergenceTable   convergence_table_h1_total;
    // TimerOutput        computing_timer;
  };
} // namespace mfe_stokes

#endif // STOKES_MFE_STOKES_MFE_H
