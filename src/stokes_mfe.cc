/* ---------------------------------------------------------------------
 * Implementation of the MixedStokesProblem class
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Ghumman, University of Pittsburgh, 2025
 *         Manu Jayadharan, Northwestern University, 2024.
 * based on the Eldar Khattatov's Elasticity DD implementation from 2017.
 */

// Internals
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/sparse_direct.h>
#include <Eigen/Dense>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
// Extra for MPI and mortars
#include <deal.II/base/timer.h>
#include <deal.II/numerics/fe_field_function.h>
// C++
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
// Utilities, data, etc.
#include "../inc/data_stokes.h"
#include "../inc/stokes_mfe.h"
#include "../inc/utilities.h"


namespace mfe_stokes
{
  using namespace dealii;

  // MixedStokesBiotDD class constructor
  template <int dim>
  MixedStokesProblem<dim>::MixedStokesProblem(
    const unsigned int degree,
    const bool ess_dir_flag,
    const bool mortar_flag,
    const unsigned int mortar_degree,
    const unsigned int iter_meth_flag,
    const bool cont_mortar_flag,
    const bool print_interface_matrix_flag)
    : degree(degree)
    , ess_dir_flag(ess_dir_flag)
    , mortar_flag(mortar_flag)
    , mortar_degree(mortar_degree)
    , iter_meth_flag(iter_meth_flag)
    , cont_mortar_flag((mortar_flag == 1 && mortar_degree == 0) ? false: cont_mortar_flag) // if mortar_degree is 0, use RT<dim> (0)
    , print_interface_matrix_flag(print_interface_matrix_flag)
    , mpi_communicator(MPI_COMM_WORLD)
    , P_coarse2fine(false)
    , P_fine2coarse(false)
    , n_domains(dim, 0)//vector of type unsigned int initialized to size dim with entries = 0
    , gmres_iteration(0)
	  , cg_iteration(0)
    , fe(FE_Q<dim>(degree+1),//fe for velocity
         dim,
         FE_Q<dim>(degree),//fe for pressure
         1)
    , dof_handler(triangulation)
    , dof_handler_mortar(triangulation_mortar)
    ,fe_mortar([mortar_degree, cont_mortar_flag]() -> FESystem<dim> {
      if (mortar_degree > 0 && cont_mortar_flag)
        return FESystem<dim>(FE_Q<dim>(mortar_degree), dim, FE_Nothing<dim>(), 1);
      else if (mortar_degree == 0 || !cont_mortar_flag)
        return FESystem<dim>(FE_RaviartThomas<dim>(mortar_degree), 1, FE_Nothing<dim>(), 1);
      else
        return FESystem<dim>(FE_Nothing<dim>(), dim, FE_Nothing<dim>(), 1);
    }()) // Conditional initialization using a lambda function
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  // MixedStokesProblem::make_grid_and_dofs
  template <int dim>
  void
  MixedStokesProblem<dim>::setup_system(const std::vector<unsigned int> &boundary_def)
  {
    TimerOutput::Scope t(computing_timer, "make_grid_and_dofs");
  }


  // MixedStokesProblemDD - assemble_system
  template <int dim>
  void
  MixedStokesProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "Assemble system");
    
  }

  
  
  
  // MixedStokesProblemDD - initialize the interface data structure
  template <int dim>
  void
  MixedStokesProblem<dim>::get_interface_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Get interface DoFs");
    
  }


  // MixedStokesProblem - assemble RHS of star problems
  template <int dim>
  void
  MixedStokesProblem<dim>::assemble_rhs_star(
    FEFaceValues<dim> &fe_face_values)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS star stokes");
  }


  // MixedStokesProblemDD::solvers
  template <int dim>
  void MixedStokesProblem<dim>::solve_bar()
  {
      TimerOutput::Scope t(computing_timer, "Solve bar stokes");
    
  }


  template <int dim>
  void MixedStokesProblem<dim>::solve_star()
  {
    TimerOutput::Scope t(computing_timer, "Solve star");
    
  }

  template <int dim>
  void
  MixedStokesProblem<dim>::output_results(const unsigned int cycle) const
  {
    // TimerOutput::Scope t(computing_timer, "Output results");
  }

  // MixedStokesProblemDD::run
  template <int dim>
  void
  MixedStokesProblem<dim>::run(

    const unsigned int                            refine,
    const std::vector<unsigned int>              &boundary_def,
    std::vector<std::vector<unsigned int>> &reps,
    double                                        tol,
    std::string                                   name,
    unsigned int                                  maxiter,
    unsigned int                                  quad_degree)
  {




    tolerance = tol;
    qdegree   = quad_degree;

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    Assert(reps[0].size() == dim, ExcDimensionMismatch(reps[0].size(), dim));

    if (mortar_flag)
      {
        Assert(n_processes > 1,
               ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
        Assert(reps.size() >= n_processes + 1,
               ExcMessage("Some of the mesh parameters were not provided"));
      }

    for (unsigned int cycle = 0; cycle < refine; ++cycle)
      {
        gmres_iteration = 0;
        interface_dofs.clear();
        interface_dofs_fe.clear();
        interface_dofs_find_neumann.clear();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
          {
            pcout << "Making grid and DOFs..."
              << "\n";
            
            setup_system(boundary_def);

            pcout << "Assembling system..."
                  << "\n";
        
            assemble_system();
            pcout<<"assembled system!!"<<std::endl;
            solve_bar();
            solution = solution_bar_stokes;

            // compute_errors(cycle, reps);

            // computing_timer.print_summary();
            computing_timer.reset();

            // output_results(cycle, refine, name);
            // output_dof_results(cycle);
            pcout << "dof output written" << std::endl;
            output_results(cycle);
            pcout << "solution data output written" << std::endl;
            // output_interface_results(cycle);

            triangulation.clear();
            dof_handler.clear();
            // convergence_table.clear();
            faces_on_interface.clear();
            faces_on_interface_mortar.clear();
            interface_dofs.clear();
            interface_fe_function.clear();
            interface_fe_function_mortar.clear();
            if (mortar_flag)
              {
                triangulation_mortar.clear();
                P_fine2coarse.reset();
                P_coarse2fine.reset();
              }
            dof_handler_mortar.clear();
          }
        else
          {
            pcout << "Making grid and DOFs..."
              << "\n";
            setup_system(boundary_def);

            pcout << "Getting interface dofs" << std::endl;
            get_interface_dofs();
            pcout << "Assembling system..."
                  << "\n";
        
            assemble_system();
            pcout<<"Assembled system!!"<<std::endl;
            // output_dof_results(cycle);
            pcout << "dof output written" << std::endl;
            if (mortar_flag)
            {
              // output_dof_results_mortar(cycle);
                pcout << "dof mortar output written" << std::endl;
            }

            if (iter_meth_flag == 0)
            {
              pcout << "Starting CG iterations..."
                  << "\n";
              // local_cg(maxiter, cycle);
            }
            else if (iter_meth_flag == 1)
            {
              pcout << "Starting GMRES iterations..."
                  << "\n";
              // local_gmres(maxiter, cycle);
            }
            else
            {
              AssertThrow(false, ExcMessage("Unknown iterative method!"));
            }
            // compute_errors(cycle, reps);

            MPI_Barrier(mpi_communicator);

            output_results(cycle);
            // computing_timer.print_summary();
            computing_timer.reset();
            pcout << "----------------------------------------------" << std::endl;
            
            triangulation.clear();
            dof_handler.clear();
            faces_on_interface.clear();
            faces_on_interface_mortar.clear();
            interface_dofs.clear();
            interface_dofs_fe.clear();
            interface_fe_function.clear();
            interface_fe_function_mortar.clear();
            if (mortar_flag)
              {
                triangulation_mortar.clear();
                P_fine2coarse.reset();
                P_coarse2fine.reset();
              }
            dof_handler_mortar.clear();
          }
      }

    // print_individual_table_l2<dim>(convergence_table,
    //                                iter_meth_flag,
    //                                this_mpi,
    //                                n_processes,
    //                                mpi_communicator);
    // print_total_table_l2<dim>(convergence_table_total,
    //                           iter_meth_flag,
    //                           print_interface_matrix_flag,
    //                           this_mpi);
                              
    convergence_table.clear();
    convergence_table_total.clear();
    faces_on_interface.clear();
    faces_on_interface_mortar.clear();
  }

  template class MixedStokesProblem<2>;
  // template class MixedStokesProblem<3>;
} // namespace mfe_stokes
