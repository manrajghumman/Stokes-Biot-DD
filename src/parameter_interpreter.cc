/* ---------------------------------------------------------------------
 * Utilities:
 *  - Stokes Biot - related to interpreting parameters from parameter file
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Singh Ghumman, University of Pittsburgh, 2025 - 2026
 */

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/exceptions.h>
#include <vector>
#include <string>
#include "../inc/sb_dd.h"
#include "../inc/stokes_mfe.h"
#include "../inc/biot_mfe.h"

namespace dd_stokes_biot
{
    using namespace dealii;
    
    /*
     * This function interprets parameters read from a parameter file 
     * for a Stokes Biot problem in either 2D or 3D, including type of domain,
     * initial refinement of each subdomain and external boundary conditions.
     * Read from the parameter handler object prm given in input.
     */
    template <int dim>
    void MixedStokesBiotProblem<dim>::interpret_parameters(dealii::ParameterHandler &prm,
                                                        unsigned int &refine,
                                                        std::vector<std::vector<unsigned int>> &dd_info_matrix,
                                                        const unsigned int &n_processes)
        {
            // make initial grid i.e. fill n_domains vector for each subdomain
            switch(dim)
            {
                case 2:
                {
                    // Get the string and split into a vector of strings
                    std::vector<int> bvec = Utilities::string_to_int(
                                    Utilities::split_string_list(prm.get("initial_grid_2D"))); 
                    AssertThrow(bvec.size() == dim, 
                                ExcMessage("Initial grid size must match the dimension."));
                    // Initialize n_domains with 2D grid sizes
                    n_domains[0] = bvec[0];
                    n_domains[1] = bvec[1];
                    break;
                }
                case 3:
                {
                    // Get the string and split into a vector of strings
                    std::vector<int> bvec = Utilities::string_to_int(
                                    Utilities::split_string_list(prm.get("initial_grid_3D"))); 
                    AssertThrow(bvec.size() == dim, 
                                ExcMessage("Initial grid size must match the dimension."));
                    // Initialize n_domains with 3D grid sizes
                    n_domains[0] = bvec[0];
                    n_domains[1] = bvec[1];
                    n_domains[2] = bvec[2];
                    break;
                }
                default:
                    AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
            }

            // enter the number of refinements
            AssertThrow(dim == 2 || dim == 3, ExcMessage("Dimension must be either 2 or 3."));
            refine = prm.get_integer("refinements");

            // make a grid matrix (dd_info_matrix) to differentiate Stokes and Biot subdomains
            std::string dd_info = prm.get("grid");
            auto rows = Utilities::split_string_list(dd_info, ';');

            for (auto &r : rows) {
                auto int_vec = Utilities::string_to_int(Utilities::split_string_list(r, ','));
                for (unsigned int i = 0; i < int_vec.size(); ++i)
                    Assert(int_vec[i] >= 0,
                            ExcMessage("Negative value at index " + std::to_string(i) +
                                    " in row " + std::to_string(dd_info.size()) +
                                    ".\n Each entry in the grid must be Stokes (1) or Biot (2)."));
                std::vector<unsigned int> uint_vec(int_vec.begin(), int_vec.end());
                dd_info_matrix.push_back(uint_vec);
            }
            unsigned int total_domains = 0;
            for (auto &r : dd_info_matrix)
                total_domains += r.size();
            Assert(total_domains == n_processes,
                ExcMessage("Total number of subdomains in the grid (" +
                        std::to_string(total_domains) +
                        ") does not match the number of MPI processes (" +
                        std::to_string(n_processes) + ")."));
        }

  // Explicit template instantiation
  template class MixedStokesBiotProblem<2>;
  template class MixedStokesBiotProblem<3>;

} // namespace dd_stokes_biot

namespace mfe_stokes
{
    using namespace dealii;
    
    /*
     * This function interprets parameters read from a parameter file 
     * for the Stokes part of Stokes Biot problem in either 2D or 3D, including type of domain,
     * initial refinement of each subdomain and external boundary conditions.
     * Read from the parameter handler object prm given in input.
     */
    template <int dim>
    void MixedStokesProblem<dim>::interpret_parameters()
    {
        prm.enter_subsection("Stokes Parameters");
        {
            qdegree  = prm.get_integer("quad_degree");  
            gamma    = prm.get_double("gamma");
        }
        prm.leave_subsection();
        // make initial refinement matrix (reps) for each subdomain
        switch (dim)
        {   
            case 2:
            {
                // Get the string and split into a vector of strings
                std::vector<int> bvec = Utilities::string_to_int(
                                Utilities::split_string_list(prm.get("initial_refinement_2D"))); 
                AssertThrow(bvec.size() == dim, 
                            ExcMessage("Initial grid size must match the dimension."));
                // Initialize mesh_m2d with 2D grid sizes
                // Uniform initial mesh for all processes
                for (unsigned int i = 0; i < n_processes + 1; ++i)
                    reps.push_back({static_cast<unsigned int>(bvec[0]), static_cast<unsigned int>(bvec[1])}); 
                break;
            }
            case 3:
            {
                // Get the string and split into a vector of strings
                std::vector<int> bvec = Utilities::string_to_int(
                                Utilities::split_string_list(prm.get("initial_refinement_3D"))); 
                AssertThrow(bvec.size() == dim, 
                            ExcMessage("Initial refinement must match the dimension."));
                // Initialize mesh_m2d with 3D grid sizes
                // Uniform initial mesh for all processes
                for (unsigned int i = 0; i < n_processes + 1; ++i)
                    reps.push_back({static_cast<unsigned int>(bvec[0]), 
                        static_cast<unsigned int>(bvec[1]), static_cast<unsigned int>(bvec[2])}); 
                break;
            }
            default:
                AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
        }
        
        // Interpret external boundary conditions
        prm.enter_subsection("External Boundary Conditions");
        {
            // ess_dir_flag = prm.get("essential_dirichlet") == "yes" ? 1 : 0;
            switch (dim)
            {
                case 2:
                {
                    // Get the string and split into a vector of strings
                    std::vector<std::string> bvec 
                                = Utilities::split_string_list(prm.get("boundary_2D")); 
                    // Convert each string to a char
                    for (const std::string &s : bvec)
                        if (s[0] == 'N')
                            external_bdry_ids.push_back(7); // Neumann boundary
                        else if (s[0] == 'D')
                            external_bdry_ids.push_back(0); // Dirichlet boundary
                        else
                            AssertThrow(false, ExcMessage("Invalid boundary condition type." 
                                " Use 'N' for Neumann or 'D' for Dirichlet."));
                    // Ensure the boundary vector has exactly 4 elements
                    AssertThrow(external_bdry_ids.size() == 4, 
                            ExcMessage("Boundary vector must have exactly 4 elements/edges for 2D."));
                    break;
                }
                case 3:
                {
                    // Get the string and split into a vector of strings
                    std::vector<std::string> bvec 
                                = Utilities::split_string_list(prm.get("boundary_3D")); 
                    // Convert each string to a char
                    for (const std::string &s : bvec)
                        if (s[0] == 'N')
                            external_bdry_ids.push_back(1); // Neumann boundary
                        else if (s[0] == 'D')
                            external_bdry_ids.push_back(0); // Dirichlet boundary
                        else
                            AssertThrow(false, 
                                ExcMessage("Invalid boundary condition type."
                                            " Use 'N' for Neumann or 'D' for Dirichlet."));
                    // Ensure the boundary vector has exactly 4 elements
                    AssertThrow(external_bdry_ids.size() == 6, 
                                ExcMessage("Boundary vector must have exactly 6 elements/faces for 3D."));
                    break;
                }
                default:
                    AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
            }
        }
        prm.leave_subsection();

        prm.enter_subsection("Solver");
        {
            maxiter = prm.get_integer("max_iter");
            tol = prm.get_double("tol");
        }
    }
    // Explicit template instantiation
  template class MixedStokesProblem<2>;
  template class MixedStokesProblem<3>;
} // namespace mfe_stokes

namespace mfe_biot
{
    using namespace dealii;
    
    /*
     * This function interprets parameters read from a parameter file 
     * for the Biot part of Stokes Biot problem in either 2D or 3D, including 
     * type of domain, initial refinement of each subdomain and external 
     * boundary conditions. Read from the parameter handler object prm given 
     * in input.
     */
    template <int dim>
    void MixedBiotProblem<dim>::interpret_parameters()
    {
        prm.enter_subsection("Biot Parameters");
        {
            qdegree  = prm.get_integer("quad_degree");  
        }
        prm.leave_subsection();
        // make initial refinement matrix (reps) for each subdomain
        switch (dim)
        {   
            case 2:
            {
                // Get the string and split into a vector of strings
                std::vector<int> bvec = Utilities::string_to_int(
                                Utilities::split_string_list(prm.get("initial_refinement_2D"))); 
                AssertThrow(bvec.size() == dim, 
                            ExcMessage("Initial grid size must match the dimension."));
                // Initialize mesh_m2d with 2D grid sizes
                // Uniform initial mesh for all processes
                for (unsigned int i = 0; i < n_processes + 1; ++i)
                    reps.push_back({static_cast<unsigned int>(bvec[0]), static_cast<unsigned int>(bvec[1])}); 
                break;
            }
            case 3:
            {
                // Get the string and split into a vector of strings
                std::vector<int> bvec = Utilities::string_to_int(
                                Utilities::split_string_list(prm.get("initial_refinement_3D"))); 
                AssertThrow(bvec.size() == dim, 
                            ExcMessage("Initial refinement must match the dimension."));
                // Initialize mesh_m2d with 3D grid sizes
                // Uniform initial mesh for all processes
                for (unsigned int i = 0; i < n_processes + 1; ++i)
                    reps.push_back({static_cast<unsigned int>(bvec[0]), 
                        static_cast<unsigned int>(bvec[1]), static_cast<unsigned int>(bvec[2])}); 
                break;
            }
            default:
                AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
        }
        
        // Interpret external boundary conditions
        prm.enter_subsection("External Boundary Conditions");
        {
            // ess_dir_flag = prm.get("essential_dirichlet") == "yes" ? 1 : 0;
            switch (dim)
            {
                case 2:
                {
                    // Get the string and split into a vector of strings
                    std::vector<std::string> bvec 
                                = Utilities::split_string_list(prm.get("boundary_2D")); 
                    // Convert each string to a char
                    for (const std::string &s : bvec)
                        if (s[0] == 'N')
                            external_bdry_ids.push_back(7); // Neumann boundary
                        else if (s[0] == 'D')
                            external_bdry_ids.push_back(0); // Dirichlet boundary
                        else
                            AssertThrow(false, ExcMessage("Invalid boundary condition type." 
                                " Use 'N' for Neumann or 'D' for Dirichlet."));
                    // Ensure the boundary vector has exactly 4 elements
                    AssertThrow(external_bdry_ids.size() == 4, 
                            ExcMessage("Boundary vector must have exactly 4 elements/edges for 2D."));
                    break;
                }
                case 3:
                {
                    // Get the string and split into a vector of strings
                    std::vector<std::string> bvec 
                                = Utilities::split_string_list(prm.get("boundary_3D")); 
                    // Convert each string to a char
                    for (const std::string &s : bvec)
                        if (s[0] == 'N')
                            external_bdry_ids.push_back(1); // Neumann boundary
                        else if (s[0] == 'D')
                            external_bdry_ids.push_back(0); // Dirichlet boundary
                        else
                            AssertThrow(false, 
                                ExcMessage("Invalid boundary condition type."
                                            " Use 'N' for Neumann or 'D' for Dirichlet."));
                    // Ensure the boundary vector has exactly 4 elements
                    AssertThrow(external_bdry_ids.size() == 6, 
                                ExcMessage("Boundary vector must have exactly 6 elements/faces for 3D."));
                    break;
                }
                default:
                    AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
            }
        }
        prm.leave_subsection();

        prm.enter_subsection("Solver");
        {
            maxiter = prm.get_integer("max_iter");
            tol = prm.get_double("tol");
        }
    }
    // Explicit template instantiation
  template class MixedBiotProblem<2>;
  template class MixedBiotProblem<3>;
} // namespace mfe_biot