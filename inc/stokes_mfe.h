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
  public:
  MixedStokesProblem (ParameterHandler &prm,
                      unsigned int &degree,
                      unsigned int &degree_biot,
                      std::vector<unsigned int> n_domains,
                      const unsigned int &subdomain_type,
                      std::vector<int> &neighbors,
                      std::vector<unsigned int> &neighbor_type,
                      const bool mortar_flag = 0,
                      const unsigned int mortar_degree = 0,
                      const bool cont_mortar_flag = 0);

    void
    run(unsigned int cycle);

    void
    assemble_rhs_star(FEFaceValues<dim> &fe_face_values);

    void
    solve_bar();

    void
    solve_star();

    void
    output_results(const unsigned int cycle) const;

    void 
    compute_errors(const unsigned int &cycle);


  private:
    MPI_Comm   mpi_communicator;
    MPI_Status mpi_status;
    const unsigned int this_mpi;
    const unsigned int n_processes;
    ParameterHandler &prm;
    ParameterHandler prm_fxn;
    unsigned int degree;
    unsigned int qdegree;
    const unsigned int subdomain_type;
    bool stokes_neighbor;
    bool biot_neighbor;
    std::vector<unsigned int> external_bdry_ids;
    bool mortar_flag, cont_mortar_flag;
    unsigned int mortar_degree;
    double       tol;
    std::vector<std::vector<unsigned int>> reps; // initial refinement for this subdomain
    // std::vector<int> neighbors;
    // std::vector<unsigned int> neighbor_type;

    Projector::Projector<dim> P_coarse2fine;
    Projector::Projector<dim> P_fine2coarse;

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

    // void
    // get_interface_dofs_biot();


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
    std::vector<double>                    interface_dofs_total;
    std::vector<std::vector<unsigned int>> interface_dofs_find_neumann;
    std::vector<unsigned int>              repeated_dofs;
    std::vector<unsigned int>              repeated_dofs_neumann;
    std::vector<unsigned int>              repeated_dofs_neumann_corner;

    unsigned long n_velocity_interface;
    unsigned long n_velocity_interface_fe;
    unsigned long n_biot_stress;
    unsigned long n_darcy_vel;
    int           interface_dofs_size;

    // Subdomain coordinates (assuming logically rectangular blocks)
    Point<dim> p1;
    Point<dim> p2;

    // Fine triangulation
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    // The fe space of Biot stress in the meaning of displacement
    FESystem<dim>      fe_biot_stress;
    DoFHandler<dim>    dof_handler_biot_stress;

    // The fe space of Biot velocity in the meaning of pressure
    FESystem<dim>      fe_darcy_vel;
    DoFHandler<dim>    dof_handler_darcy_vel;

    // Mortar triangulation
    // Triangulation<dim> triangulation_mortar;
    // FESystem<dim>      fe_mortar;
    // DoFHandler<dim>    dof_handler_mortar;

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


    BlockVector<double> solution_bar_stokes;
    BlockVector<double> solution_star_stokes;
    BlockVector<double> solution;
    BlockVector<double> exact_solution_at_nodes;
    BlockVector<double> biot_stress_vec;
    BlockVector<double> darcy_velocity_vec;
    std::vector<BlockVector<double>> exact_normal_stress_at_nodes_fe;
    std::vector<BlockVector<double>> exact_normal_stress_at_nodes_mortar;
    BlockVector<double> system_rhs_bar_stokes;
    BlockVector<double> system_rhs_star_stokes;
    std::vector<BlockVector<double>> interface_fe_function;
    std::vector<BlockVector<double>> interface_fe_function_fe;

    

    // Mortar data structures
    std::vector<BlockVector<double>>   interface_fe_function_mortar;
    std::vector<BlockVector<double>>   interface_fe_function_mortar_fe;
    BlockVector<double>                solution_bar_mortar;
    BlockVector<double>                solution_star_mortar;
    std::vector<BlockVector<double>>   multiscale_basis;


    // Output extra
    ConditionalOStream pcout;
    ConvergenceTable   convergence_table;
    ConvergenceTable   convergence_table_total;
    // TimerOutput        computing_timer;
    


  };
} // namespace mfe_stokes

#endif // STOKES_MFE_STOKES_MFE_H
