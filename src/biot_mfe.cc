/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is part of a program that  implements DD for 3 Different Schemes for Biot: Monolithic, Dranined SPlit and Fixed Stress. This file is specific to Example 1 in paper on DD for BIot schemes.
 *update: The code is modified to include nonmatching subdomain grid using mortar spaces and multiscale basis.
 * ---------------------------------------------------------------------
 *
 * Authors: Manu Jayadharan, Eldar Khattatov, University of Pittsburgh:2018- 2019
 */

// Internals,.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
// Extra for MPI and mortars
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/timer.h>
// C++
#include <fstream>
#include <iostream>
#include <random>
// Utilities, data, etc.
#include "../inc/biot_mfe.h"
#include "../inc/utilities.h"
#include "../inc/data_biot.h"


namespace mfe_biot
{
  using namespace dealii;
  using namespace dd_stokes_biot;

  // MixedBiotProblem class constructor
  template <int dim>
  MixedBiotProblem<dim>::MixedBiotProblem (const unsigned int degree,
                                           const BiotParameters &bprm,
                                           const unsigned int mortar_flag,
                                           const unsigned int mortar_degree,
                                           unsigned int split_flag,
                                           unsigned int ic_constr_flag)
    : mpi_communicator (MPI_COMM_WORLD),
      P_coarse2fine (false),
      P_fine2coarse (false),
      n_domains(dim,0),
      prm(bprm),
      degree (degree),
      mortar_degree(mortar_degree),
      mortar_flag (mortar_flag),
      cg_iteration(0),
      gmres_iteration(0),
      max_cg_iteration(0),
      max_cg_iteration_darcy(0),
      qdegree(11),
      split_flag(split_flag),
      ic_constr_flag(ic_constr_flag),
      fe (FE_BDM<dim>(degree), dim,
          FE_DGQ<dim>(degree-1), dim,
          FE_DGQ<dim>(degree-1), 0.5*dim*(dim-1),
          FE_BDM<dim>(degree), 1,
          FE_DGQ<dim>(degree-1), 1),
      pcout (std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer (mpi_communicator,
                        pcout,
                        TimerOutput::summary,
                        TimerOutput::wall_times),
      grid_diameter(0)
  {}

  // MixedBiotProblem::setup_system
  template <int dim>
  void MixedBiotProblem<dim>::setup_system ()
  {	    	
    TimerOutput::Scope t(computing_timer, "setup system");
  }


  // MixedBiotProblem - assemble_system
  template <int dim>
  void MixedBiotProblem<dim>::assemble_system ()
  {
    TimerOutput::Scope t(computing_timer, "Assemble system");          
  }

  // MixedBiotProblemDD - initialize the interface data structure for coupled monolithic sheme
  template <int dim>
  void MixedBiotProblem<dim>::get_interface_dofs ()
  {

  }


  // MixedBiotProblem - assemble RHS of star problems
  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs_bar ()
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS bar");  
  }


  // MixedBiotProblem - assemble RHS of star problems
  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs_star (FEFaceValues<dim> &fe_face_values)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS star");
  }


    
  // MixedBiotProblem::solvers
  template <int dim>
  void MixedBiotProblem<dim>::solve_bar ()
  {
    TimerOutput::Scope t(computing_timer, "Solve bar");
  }

  template <int dim>
  void MixedBiotProblem<dim>::solve_star ()
  {
      TimerOutput::Scope t(computing_timer, "Solve star");
  }

  template<int dim>
  void MixedBiotProblem<dim>::Find_IC(unsigned int maxiter)
  {

  }

  template<int dim>
  void MixedBiotProblem<dim>::solve_timestep(unsigned int maxiter)
  {
  
  }
     
  template <int dim>
  void MixedBiotProblem<dim>::reset_mortars()
  {
  
  }

  // MixedBiotProblem::run
  template <int dim>
  void MixedBiotProblem<dim>::run (const unsigned int refine,
                                            const std::vector<std::vector<unsigned int>> &reps,
                                            double tol,
                                            unsigned int maxiter,
                                            unsigned int quad_degree)
  {
      
  }
    template class MixedBiotProblem<2>;
    template class MixedBiotProblem<3>;
}
