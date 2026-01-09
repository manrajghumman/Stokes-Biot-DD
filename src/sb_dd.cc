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
#include "../inc/files.h"

namespace dd_stokes_biot
{
  using namespace dealii;
  using namespace dd_utilities;
  using namespace mfe_stokes;
  using namespace mfe_biot;

  // MixedStokesBiotProblem class constructor
  template <int dim>
  MixedStokesBiotProblem<dim>::MixedStokesBiotProblem (ParameterHandler &prm)
    : prm(prm)
    , n_domains(dim, 0)
    , mpi_communicator (MPI_COMM_WORLD)
    , this_mpi (Utilities::MPI::this_mpi_process(mpi_communicator))
    , n_processes (Utilities::MPI::n_mpi_processes(mpi_communicator))
    , pcout(std::cout,
        (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}



  // MixedStokesBiotProblem - run function
  template <int dim>
  void MixedStokesBiotProblem<dim>::run ()
  {
    // TO DO: implement the run function for Stokes-Biot DD
    pcout << "Running Mixed Stokes-Biot Domain Decomposition Problem..." << std::endl;
    clean_files<dim>();
    unsigned int refine;
    // interpret information for refinement, dd structure and other parameters
    interpret_parameters(prm, refine, dd_info_matrix, n_processes);

    // Print dd_info_matrix
    pcout << "Subdomain type matrix (S: Stokes, B: Biot):" << std::endl;
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

        // construct a stokes and biot triangulation for each subdomain
        // using the coordinates p1 and p2
        // then store the material_id for each subdomain triangulation
        // after looking up in dd_info_matrix

        // Find subdomain position
        switch (dim) 
        {
          case 2:
          {
            int x, y;
            //determine the x position
            if (this_mpi == 0) {x = 0;}
            else if ((this_mpi + 1) % n_domains[0] != 0) {
              x = this_mpi % n_domains[0];
            } else {
              x = n_domains[0] - 1;
            }
            //determine the y position
            if (this_mpi == 0) {y = n_domains[1] - 1;}
            else {y = n_domains[1] - this_mpi / n_domains[0] - 1;}

            subdomain_type = dd_info_matrix[y][x]; // for 2D
            break;
          }     
          case 3:
            Assert(false, ExcMessage("Dimension = 3 not  yet supported"));
            // update how you input the dd_info_matrix for 3D first
            break;
          default: 
            Assert(false, ExcMessage("Dimension = " + std::to_string(dim) 
                                              + " not supported"));
        }
        pcout << "--------------------------------------------------------" << std::endl;
        MPI_Barrier(mpi_communicator);

        
        // Find neighbors
        neighbors.resize(GeometryInfo<dim>::faces_per_cell, 0);
        neighbor_type.resize(GeometryInfo<dim>::faces_per_cell, 0);
        find_neighbors<dim>(this_mpi, 
                            n_domains,
                            dd_info_matrix,
                            neighbors, 
                            neighbor_type);

        // MPI_Barrier(mpi_communicator);
        for (unsigned int proc = 0; proc < n_processes; ++proc)
        {
          if (this_mpi == proc) 
          {
            std::cout << "Process " << this_mpi 
                    << " is subdomain type " 
                    << (subdomain_type == 1 ? "Stokes" : "Biot") 
                    << std::endl;
            std::cout << " with neighbors: ";
            for (unsigned int f = 0; f < neighbors.size(); ++f) 
            {
              std::cout << "side : " << f << " (" << (neighbor_type[f] == 1 ? "S" : 
                          (neighbor_type[f] == 2 ? "B" : 
                          (neighbor_type[f] == 0 ? "None" : "Neighbor type error. "
                              "Neighbor type is " + std::to_string(neighbor_type[f])))) 
                      << ") " ;
            }
            std::cout << std::endl;
          }
          MPI_Barrier(mpi_communicator);
        }
        MPI_Barrier(mpi_communicator);

        // initialize the respective stokes and biot problem objects
        // based on subdomain_type
        if (subdomain_type == 1) // Stokes
        {
          // pcout << "Initializing Stokes subdomain on process " << this_mpi << "\n";
          MPI_Comm_split(mpi_communicator, 1, this_mpi, &mpi_communicator_stokes);
          unsigned int degree, mortar_degree;
          bool mortar_flag, cont_mortar_flag;
          prm.enter_subsection("Stokes Parameters");
          {
              degree   = prm.get_integer("degree");
              mortar_flag = prm.get("mortar") == "yes" ? 1 : 0;
              if (mortar_flag)
              {
                  cont_mortar_flag = prm.get("continuous_mortar") == "yes" ? 1 : 0;
                  mortar_degree = prm.get_integer("mortar_degree");
              }
              else
              {
                  cont_mortar_flag = 1; // if no mortar, continuous mortar is default
                  mortar_degree = 0; // if no mortar, mortar degree is 0
              }
          }
          prm.leave_subsection();
          unsigned int degree_biot;
          prm.enter_subsection("Biot Parameters");
          {
              degree_biot = prm.get_integer("degree");
          }
          prm.leave_subsection();
          stokes = new MixedStokesProblem<dim>(prm,
                                              degree,
                                              degree_biot,
                                              n_domains,
                                              subdomain_type,
                                              neighbors,
                                              neighbor_type,
                                              mpi_communicator_stokes,
                                              mortar_flag,
                                              cont_mortar_flag,
                                              mortar_degree);
          stokes->run(cycle);
        
        }
        else if (subdomain_type == 2) // Biot
        {
          // pcout << "Initializing Stokes subdomain on process " << this_mpi << "\n";
          MPI_Comm_split(mpi_communicator, 2, this_mpi, &mpi_communicator_biot);
          unsigned int degree, mortar_degree;
          bool mortar_flag;
          prm.enter_subsection("Biot Parameters");
          {
              degree   = prm.get_integer("degree");
              mortar_flag = prm.get("mortar") == "yes" ? 1 : 0;
              if (mortar_flag)
                  mortar_degree = prm.get_integer("mortar_degree");
              else
                  mortar_degree = 0; // if no mortar, mortar degree is 0
          }
          prm.leave_subsection();

          biot = new MixedBiotProblem<dim>(prm,
                                          degree,
                                          n_domains,
                                          subdomain_type,
                                          neighbors,
                                          neighbor_type,
                                          mpi_communicator_biot,
                                          mortar_flag,
                                          mortar_degree);
          biot->run(cycle);
        }
      }
      else
      {
        // refine each subdomain grid uniformly and run the respective problems
        if(subdomain_type == 1) // Stokes
        {
          // stokes->reset_system();
          stokes->run(cycle);
        }
        else if (subdomain_type == 2) // Biot
        {
          // biot->reset_system();
          biot->run(cycle);
        }
      }

      // Now we must call the solve, output, 
      // compute error and convegence table functions
      if (n_processes == 1)
      {
        if (subdomain_type == 1)
        {
          stokes->output_results(cycle);
          stokes->output_dof_results(cycle);
          stokes->compute_errors(cycle);
          stokes->print_individual_convg_table_l2();
          stokes->print_individual_convg_table_h1();
        }
        else if (subdomain_type == 2)
        {
          biot->output_results(cycle);
          biot->output_dof_results(cycle);
          biot->compute_errors(cycle);
          biot->print_individual_convg_table_l2();
          biot->print_individual_convg_table_hdiv();
        }
      }


      
      MPI_Barrier(mpi_communicator);
          // computing_timer.reset();
      pcout << "----------------------------------------------" << std::endl;
    }
    // clean up
    if (subdomain_type == 1)
    {
      delete stokes;
      stokes = nullptr;
    }
    else if (subdomain_type == 2)
    {
      delete biot;
      biot = nullptr;
    }
          
  }

  // Explicit template instantiation
  template class MixedStokesBiotProblem<2>;
//   template class MixedStokesBiotProblem<3>;

} //namespace dd_stokes_biot