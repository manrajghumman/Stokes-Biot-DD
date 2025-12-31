/* ---------------------------------------------------------------------
 * Declaration of MixedStokesProblem class template
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Ghumman, University of Pittsburgh, 2024
 * and Manu Jayadharan, Northwestern University, 2024
 * based on the Eldar Khattatov's Elasticity DD implementation from 2017
 */

#ifndef STOKES_BIOT_DD_H
#define STOKES_BIOT_DD_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
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
#include "stokes_mfe.h"
#include "biot_mfe.h"

namespace dd_stokes_biot
{
  using namespace dealii;
  using namespace mfe_stokes;
  using namespace mfe_biot;

  // Mixed Stokes Domain Decomposition class template
  template <int dim>
  class MixedStokesBiotProblem
  {
  public:
  MixedStokesBiotProblem (ParameterHandler &prm);
  // MixedStokesBiotProblem (const unsigned int degree,
  //                       const bool ess_dir_flag                = 0,
  //                       const bool mortar_flag                 = 0,
  //                       const unsigned int mortar_degree       = 0,
  //                       const unsigned int iter_meth_flag      = 0,
  //                       const bool cont_mortar_flag            = 0,
  //                       const bool print_interface_matrix_flag = 0);

    void
    run();


  private:
    MPI_Comm   mpi_communicator;
    MPI_Status mpi_status;
    const unsigned int this_mpi;
    const unsigned int n_processes;

    MixedStokesProblem<dim> *stokes;
    MixedBiotProblem<dim> *biot;

    // Projector::Projector<dim> P_coarse2fine;
    // Projector::Projector<dim> P_fine2coarse;

    void interpret_parameters(dealii::ParameterHandler &prm,
                              unsigned int &refine,
                              std::vector<std::vector<unsigned int>> &dd_info_matrix,
                              const unsigned int &n_processes);

    void
    make_grid(const std::vector<unsigned int> &boundary_def);

    void
    local_gmres(const unsigned int &maxiter, unsigned int &cycle);

    //For implementing GMRES
    void
	  givens_rotation(double v1, double v2, double &cs, double &sn);

    void
	  apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
    							unsigned int k_iteration);

    void
	  back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y);

    double vect_norm(std::vector<double> v);

    ParameterHandler &prm;
    std::vector<std::vector<unsigned int>> dd_info_matrix;
    unsigned int subdomain_type;
    std::vector<int> neighbors;
    std::vector<unsigned int> neighbor_type;
    std::vector<unsigned int> external_bdry_ids;
    bool stokes_mortar_flag, stokes_cont_mortar_flag;
    unsigned int stokes_mortar_degree;
    unsigned int                           gmres_iteration;
    double                                 residual;

    // Subdomain coordinates (assuming logically rectangular blocks)
    Point<dim> p1;
    Point<dim> p2;
    // Number of subdomains in the computational domain
    std::vector<unsigned int> n_domains;

    // Output extra
    ConditionalOStream pcout;

  };
}// namespace dd_stokes_biot

#endif // STOKES_BIOT_DD_H