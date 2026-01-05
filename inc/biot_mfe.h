/* ---------------------------------------------------------------------
 * Declaration of MixedBiotProblem class template: see source files for more details
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Eldar Khattatov, University of Pittsburgh, 2018-2019
 */

#ifndef BIOT_MFE_BIOT_MFE_H
#define BIOT_MFE_BIOT_MFE_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iostream>
#include <cstdlib>

#include "projector.h"



namespace mfe_biot
{
  using namespace dealii;

  // Mixed Biot Domain Decomposition class template
  template<int dim>
  class MixedBiotProblem
  {
    public:
    MixedBiotProblem(ParameterHandler &prm,
                      unsigned int &degree,
                      std::vector<unsigned int> n_domains,
                      const unsigned int &subdomain_type,
                      std::vector<int> &neighbors,
                      std::vector<unsigned int> &neighbor_type,
                      MPI_Comm mpi_communicator_biot,
                      const bool mortar_flag,
                      const unsigned int mortar_degree);

    void run(unsigned int cycle);

    void
    output_dof_results(const unsigned int cycle) const;

    void
    output_results(const unsigned int cycle) const;

    void 
    compute_errors(const unsigned int &cycle) 

    private:
    MPI_Comm mpi_communicator;
    MPI_Comm mpi_communicator_biot;
    MPI_Status mpi_status;
    const unsigned int this_mpi;
    const unsigned int n_processes;

    Projector::Projector <dim> P_coarse2fine;
    Projector::Projector <dim> P_fine2coarse;

    void interpret_parameters();
    void setup_system();
    void assemble_system();
    // void assemble_system_non_diff(); //assembing system matrix for non-differentiated elasticity
    // void assemble_system_elast();
    // void assemble_system_darcy();
    void get_interface_dofs();
    // void get_interface_dofs_elast();
    // void get_interface_dofs_darcy();
    void assemble_rhs_bar();
    // void assemble_rhs_bar_elast();
    // void assemble_rhs_bar_darcy();
    void assemble_rhs_star(FEFaceValues<dim> &fe_face_values);
    // void assemble_rhs_star_elast(FEFaceValues<dim> &fe_face_values);
    // void assemble_rhs_star_darcy(FEFaceValues<dim> &fe_face_values);

    void solve_bar();
    // void solve_bar_elast();
    // void solve_bar_darcy();

    void solve_star();
    // void solve_star_elast();
    // void solve_star_darcy();

    // void solve_timestep(unsigned int maxiter);
    // void Find_IC(unsigned int maxiter);

    void compute_multiscale_basis();

    // std::vector<double> compute_interface_error(); //return_vector[0] gives interface_error for elast part and return_vector[1] gives that of flow part.
    // std::vector<double> compute_interface_error_l2();
    void compute_errors(const unsigned int cycle);
    void output_results(const unsigned int cycle, const unsigned int refine);

    // void set_current_errors_to_zero();
    void reset_mortars();
    // //For implementing GMRES
    // void
    // givens_rotation(double v1, double v2, double &cs, double &sn);

    // void
    // apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
    // 							unsigned int k_iteration);

    // void
    // back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y, unsigned int k_iteration);

    // void
    // local_gmres(const unsigned int maxiter, const unsigned int solver_flag=0); //solver_order = 0(monolithic), 1(Elasticity), 2(Darcy)
    // double vect_norm(std::vector<double> v);
    // //just to test the local_gmres algorithm
    //   void
    // testing_gmres(const unsigned int &maxiter);


    unsigned int       gmres_iteration;
    unsigned int       maxiter;
    double             residual;

    // Number of subdomains in the computational domain



    // Physical parameters
    ParameterHandler &prm;
    ParameterHandler prm_fxn;


    double grid_diameter;

    // Number of subdomains in the computational domain
    std::vector<unsigned int> n_domains;

    // FE degree and DD parameters
    const unsigned int degree;
    const unsigned int mortar_degree;
    const unsigned int mortar_flag;
    double tol;
    unsigned int qdegree;


    // Neighbors and interface information
    std::vector<int> neighbors;
    std::vector<unsigned int> neighbor_type;
    const unsigned int subdomain_type;
    bool stokes_neighbor;
    bool biot_neighbor;
    std::vector<unsigned int> external_bdry_ids;
    std::vector<std::vector<unsigned int>> reps; // initial refinement for this subdomain
    std::vector<unsigned int> faces_on_interface;
    std::vector<unsigned int> faces_on_interface_mortar;
    std::vector <std::vector<unsigned int>> interface_dofs;
    std::vector <unsigned int> interface_dofs_total;


    unsigned long n_stress;
    unsigned long n_disp;
    unsigned long n_rot;
    unsigned long n_flux;
    unsigned long n_pressure;
    unsigned long n_Elast;

    unsigned long n_stress_mortar;
    unsigned long n_disp_mortar;
    unsigned long n_rot_mortar;
    unsigned long n_flux_mortar;
    unsigned long n_pressure_mortar;
    unsigned long n_Elast_mortar;
    unsigned int n_mortar_dofs;

    // Subdomain coordinates (assuming logically rectangular blocks)
    Point <dim> p1;
    Point <dim> p2;

    // Fine triangulation
    Triangulation <dim> triangulation;
    FESystem <dim> fe;
    DoFHandler <dim> dof_handler;

    // Mortar triangulation
    // Triangulation <dim> triangulation_mortar;
    // FESystem <dim> fe_mortar;
    // DoFHandler <dim> dof_handler_mortar;

    // Star and bar problem data structures
    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    SparseDirectUMFPACK A_direct;
    SparseDirectUMFPACK A_direct_non_diff; //for solving non-differentiated elasticity form for monolithic BIot

    BlockVector<double> solution_bar;

    BlockVector<double> solution_star;

    BlockVector<double> solution;

    // BlockVector<double> old_solution;
    // BlockVector<double> older_solution; //solution in previous to previous timestep(used in fixed stress split)
    // BlockVector<double> intermediate_solution;
    // BlockVector<double> intermediate_solution_old; //solution in previous time step of intermediate solution(used only in fixed stress)

    BlockVector<double> system_rhs_bar;

    BlockVector<double> system_rhs_star;

    BlockVector<double> interface_fe_function;

    std::vector<std::vector<double>> lambda_guess;
    std::vector<std::vector<double>> Alambda_guess;

    // Mortar data structures
    BlockVector<double> interface_fe_function_mortar;
    BlockVector<double> solution_bar_mortar;
    BlockVector<double> solution_star_mortar;
    std::vector <BlockVector<double>> multiscale_basis;

    // Output extra
    ConditionalOStream pcout;
    ConvergenceTable convergence_table;
    // TimerOutput computing_timer;
  };
}// namespace mfe_biot

#endif //BIOT_MFE_BIOT_MFE_H
