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
#include <deal.II/base/parameter_handler.h>

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

// C++
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
// Utilities, data, etc.
#include "../inc/sb_dd.h"
#include "../inc/data_stokes.h"
#include "../inc/data_biot.h"
#include "../inc/stokes_mfe.h"
#include "../inc/biot_mfe.h"
#include "../inc/utilities.h"

namespace dd_stokes_biot
{
  using namespace dealii;

  // MixedStokesBiotProblem class constructor
  template <int dim>
  MixedStokesBiotProblem<dim>::MixedStokesBiotProblem (ParameterHandler &prm)
    : prm(prm),
      mpi_communicator (MPI_COMM_WORLD),
      this_mpi (Utilities::MPI::this_mpi_process(mpi_communicator)),
      n_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
      pcout(std::cout,
        (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}

    // MixedStokesBiotProblem - run function
    template <int dim>
    void MixedStokesBiotProblem<dim>::run ()
    {
      // TO DO: implement the run function for Stokes-Biot DD
      pcout << "Running Mixed Stokes-Biot Domain Decomposition Problem..." << std::endl;
    
      unsigned int refine;
      std::vector<std::vector<unsigned int>> reps;
      interpret_parameters(prm, refine, reps, dd_info_matrix, this_mpi, n_processes);

      for (auto &r : dd_info_matrix) {
        for (auto & c : r) {
            if (c == 1)
                pcout << "S ";
            else if (c == 2)
                pcout << "B ";
        }
        pcout << std::endl;
      }

      for (unsigned int cycle = 0; cycle < refine; ++cycle)
      {
        gmres_iteration = 0;

        if (cycle == 0)
          {
            // Partitioning into subdomains (simple bricks)
            find_divisors<dim>(n_processes, n_domains);

            switch (dim)
            {
              case 2:
                Assert(n_domains[0] == dd_info_matrix[0].size(),
                       ExcMessage("n_domains in x (" + std::to_string(n_domains[0]) +
                                  ") must be equal to subdomains in x (" +
                                  std::to_string(dd_info_matrix[0].size()) + ")"));
                Assert(n_domains[1] == dd_info_matrix.size(),
                       ExcMessage("n_domains in y (" + std::to_string(n_domains[1]) +
                                  ") must be equal to subdomains in y (" +
                                  std::to_string(dd_info_matrix.size()) + ")"));
                break;
              case 3:
                Assert(false, ExcMessage("Dimension = 3 not  yet supported"));
                // Assert(n_domains[0] == dd_info_matrix[0][0].size(),
                //        ExcMessage("n_domains in x (" + std::to_string(n_domains[0][0]) +
                //                   ") must be equal to subdomains in x (" +
                //                   std::to_string(dd_info_matrix[0][0].size()) + ")"));
                // Assert(n_domains[1] == dd_info_matrix[0].size(),
                //        ExcMessage("n_domains in y (" + std::to_string(n_domains[1]) +
                //                   ") must be equal to subdomains in y (" +
                //                   std::to_string(dd_info_matrix[0].size()) + ")"));
                // Assert(n_domains[2] == dd_info_matrix.size(),
                //        ExcMessage("n_domains in y (" + std::to_string(n_domains[2]) +
                //                   ") must be equal to subdomains in y (" +
                //                   std::to_string(dd_info_matrix.size()) + ")"));                  
                break;
              default:
                Assert(false, ExcMessage("Dimension = " + std::to_string(dim) 
                                                + " not supported"));
            }

            // // Dimensions of the domain (unit hypercube)
            // std::vector<double> subdomain_dimensions(dim);
            // for (unsigned int d = 0; d < dim; ++d)
            //   subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

            // get_subdomain_coordinates<dim>(
            //   this_mpi, n_domains, subdomain_dimensions, p1, p2);
          }

          
          /* 
                if you want to refine in a different way than uniform refinement,
                uncomment and implement the following code
          */
        // else
        //   {
        //     // Partitioning into subdomains (simple bricks)
        //     find_divisors<dim>(n_processes, n_domains);

        //     // Dimensions of the domain (unit hypercube)
        //     std::vector<double> subdomain_dimensions(dim);
        //     for (unsigned int d = 0; d < dim; ++d)
        //       subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

        //     get_subdomain_coordinates<dim>(
        //       this_mpi, n_domains, subdomain_dimensions, p1, p2);
            
        //     for (unsigned long i = 0; i < reps[this_mpi].size(); ++i)
        //     {
        //       reps[this_mpi][i] = reps[this_mpi][i] * 2;
        //       reps[n_processes][i] = reps[n_processes][i] * 2;
        //     }

        //   }

        
        // construct a stokes and biot triangulation for each subdomain
        // using the coordinates p1 and p2
        // then store the material_id for each subdomain triangulation
        // after looking up in dd_info_matrix


        if (n_processes == 1)
        {
            // pcout << "Making grid and DOFs..."
            //   << "\n";
            
            // make_grid_and_dofs(boundary_def);

        }
        else
        {
            // pcout << "Making grid and DOFs..."
            //   << "\n";
            // make_grid_and_dofs(boundary_def);

        }
            // computing_timer.reset();
            pcout << "----------------------------------------------" << std::endl;
      }
            
    }

  // Explicit template instantiation
  template class MixedStokesBiotProblem<2>;
//   template class MixedStokesBiotProblem<3>;

} //namespace dd_stokes_biot