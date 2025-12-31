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
#include <deal.II/fe/fe_system.h>
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
  using namespace dd_utilities;

  // MixedBiotProblem class constructor
  template <int dim>
  MixedBiotProblem<dim>::MixedBiotProblem (ParameterHandler &prm,
                                            unsigned int &degree,
                                            std::vector<unsigned int> n_domains,
                                            const unsigned int &subdomain_type,
                                            std::vector<int> &neighbors,
                                            std::vector<unsigned int> &neighbor_type,
                                            const bool mortar_flag,
                                            const unsigned int mortar_degree)
    : prm(prm),
      degree(degree),
      n_domains(n_domains),
      subdomain_type(subdomain_type),
      neighbors(neighbors),
      neighbor_type(neighbor_type),
      stokes_neighbor(false),
      biot_neighbor(false),
      mortar_flag(mortar_flag),
      mortar_degree(mortar_degree),
      mpi_communicator (MPI_COMM_WORLD),
      P_coarse2fine (false),
      P_fine2coarse (false),
      gmres_iteration(0),
      grid_diameter(0),
      fe (FE_BDM<dim>(degree), dim,
          FE_DGQ<dim>(degree-1), dim,
          FE_DGQ<dim>(degree-1), 0.5*dim*(dim-1),
          FE_BDM<dim>(degree), 1,
          FE_DGQ<dim>(degree-1), 1),
      dof_handler (triangulation),
      this_mpi (Utilities::MPI::this_mpi_process(mpi_communicator)),
      n_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
      pcout (std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
      // computing_timer (mpi_communicator,
      //                   pcout,
      //                   TimerOutput::summary,
      //                   TimerOutput::wall_times)
  {
    interpret_parameters();
  }

  // MixedBiotProblem::setup_system
  template <int dim>
  void MixedBiotProblem<dim>::setup_system ()
  {	    	
    // TimerOutput::Scope t(computing_timer, "setup system");
    system_matrix.clear();

    // Make interface data structures
    faces_on_interface.resize(GeometryInfo<dim>::faces_per_cell, 0);
    // faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell, 0);

    // Label interface faces and count how many of them there are per interface
    mark_interface_faces<dim>(triangulation, neighbors, p1, p2, faces_on_interface);
    // if (mortar_flag)
    //   mark_interface_faces(
    //     triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);
    // //this is a function in utilities.h

    std::cout << "Biot Part:\n boundary condition:";
    for (unsigned int i = 0; i < external_bdry_ids.size(); ++i)
    {
      if (external_bdry_ids[i] == 0)
        external_bdry_ids[i] = 0;
      else 
        external_bdry_ids[i] = 7;
      if (external_bdry_ids[i] == 7)
        std::cout << " N";
      else if (external_bdry_ids[i] == 0)
        std::cout << " D";
      else
        AssertThrow(false, ExcMessage("Invalid boundary condition type. Use 0 for Dirichlet or 7 for Neumann."));
    }
    std::cout << "" << std::endl;
    for (const auto &cell : triangulation.active_cell_iterators()){
      for (const auto &face : cell->face_iterators()){
        if (face->center()[dim - 1] == 0)
          if (external_bdry_ids[0] != 0)
            face->set_all_boundary_ids(external_bdry_ids[0]);
        if (face->center()[dim - 2] == 2)
          if (external_bdry_ids[1] != 0)
            face->set_all_boundary_ids(external_bdry_ids[1]);
        if (face->center()[dim - 1] == 2)
          if (external_bdry_ids[2] != 0)
            face->set_all_boundary_ids(external_bdry_ids[2]);
        if (face->center()[dim - 2] == 0)
          if (external_bdry_ids[3] != 0)
            face->set_all_boundary_ids(external_bdry_ids[3]);
      }
    }

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_component =
            DoFTools::count_dofs_per_fe_component (dof_handler);

    unsigned int n_s=0, n_u=0, n_g=0, n_z = 0, n_p = 0;
    for (unsigned int i=0; i<dim; ++i)
    {
        n_s += dofs_per_component[i*dim];
        n_u += dofs_per_component[dim*dim + i];

        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
            n_g = dofs_per_component[dim*dim + dim];
        else if (dim == 3)
            n_g += dofs_per_component[dim*dim + dim + i];
    }

    n_z = dofs_per_component[dim*dim + dim + 0.5*dim*(dim-1)];
    n_p = dofs_per_component[dim*dim + dim + 0.5*dim*(dim-1) + dim];

    n_stress = n_s;
    n_disp= n_u;
    n_rot=n_g;
    n_flux = n_z;
    n_pressure= n_p;
    // n_Elast= n_s+n_u+n_g;

    std::cout << "Biot part:\n N stress dofs: " << n_s 
              << "\n N displacement dofs: " << n_u
              << "\n N rotation dofs: " << n_g
              << "\n N flux dofs: " << n_z
              << "\n N pressure dofs: " << n_p << std::endl;

    BlockDynamicSparsityPattern dsp(5, 5);
			dsp.block(0, 0).reinit (n_s, n_s);
			dsp.block(0, 1).reinit (n_s, n_u);
			dsp.block(0, 2).reinit (n_s, n_g);
			dsp.block(0, 3).reinit (n_s, n_z);
			dsp.block(0, 4).reinit (n_s, n_p);

			dsp.block(1, 0).reinit (n_u, n_s);
			dsp.block(1, 1).reinit (n_u, n_u);
			dsp.block(1, 2).reinit (n_u, n_g);
			dsp.block(1, 3).reinit (n_u, n_z);
			dsp.block(1, 4).reinit (n_u, n_p);

			dsp.block(2, 0).reinit (n_g, n_s);
			dsp.block(2, 1).reinit (n_g, n_u);
			dsp.block(2, 2).reinit (n_g, n_g);
			dsp.block(2, 3).reinit (n_g, n_z);
			dsp.block(2, 4).reinit (n_g, n_p);

			dsp.block(3, 0).reinit (n_z, n_s);
			dsp.block(3, 1).reinit (n_z, n_u);
			dsp.block(3, 2).reinit (n_z, n_g);
			dsp.block(3, 3).reinit (n_z, n_z);
			dsp.block(3, 4).reinit (n_z, n_p);

			dsp.block(4, 0).reinit (n_p, n_s);
			dsp.block(4, 1).reinit (n_p, n_u);
			dsp.block(4, 2).reinit (n_p, n_g);
			dsp.block(4, 3).reinit (n_p, n_z);
			dsp.block(4, 4).reinit (n_p, n_p);

			dsp.collect_sizes ();
			DoFTools::make_sparsity_pattern (dof_handler, dsp);

			// Initialize system matrix
			sparsity_pattern.copy_from(dsp);
			system_matrix.reinit (sparsity_pattern);

			// Reinit solution and RHS vectors
			solution_bar.reinit (5);
			solution_bar.block(0).reinit (n_s);
			solution_bar.block(1).reinit (n_u);
			solution_bar.block(2).reinit (n_g);
			solution_bar.block(3).reinit (n_z);
			solution_bar.block(4).reinit (n_p);
			solution_bar.collect_sizes ();
			solution_bar = 0;

			// Reinit solution and RHS vectors
			solution_star.reinit (5);
			solution_star.block(0).reinit (n_s);
			solution_star.block(1).reinit (n_u);
			solution_star.block(2).reinit (n_g);
			solution_star.block(3).reinit (n_z);
			solution_star.block(4).reinit (n_p);
			solution_star.collect_sizes ();
			solution_star = 0;

			system_rhs_bar.reinit (5);
			system_rhs_bar.block(0).reinit (n_s);
			system_rhs_bar.block(1).reinit (n_u);
			system_rhs_bar.block(2).reinit (n_g);
			system_rhs_bar.block(3).reinit (n_z);
			system_rhs_bar.block(4).reinit (n_p);
			system_rhs_bar.collect_sizes ();
			system_rhs_bar = 0;

			system_rhs_star.reinit (5);
			system_rhs_star.block(0).reinit (n_s);
			system_rhs_star.block(1).reinit (n_u);
			system_rhs_star.block(2).reinit (n_g);
			system_rhs_star.block(3).reinit (n_z);
			system_rhs_star.block(4).reinit (n_p);
			system_rhs_star.collect_sizes ();
			system_rhs_star = 0;
  }

  // MixedBiotProblemDD - initialize the interface data structure for coupled monolithic sheme
  template <int dim>
  void MixedBiotProblem<dim>::get_interface_dofs ()
  {
    interface_dofs.resize(GeometryInfo<dim>::faces_per_cell, std::vector<unsigned int> ());
    std::vector<types::global_dof_index> local_face_dof_indices;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    typename DoFHandler<dim>::active_cell_iterator cell, endc;
    unsigned int side = 0;

    cell = dof_handler.begin_active(), endc = dof_handler.end();
    local_face_dof_indices.resize(fe.dofs_per_face);

    for (; cell != endc; ++cell)
      for (unsigned int face_n = 0; face_n < n_faces_per_cell; ++face_n)
        if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0 && 
                cell->face(face_n)->boundary_id() !=7))
        {
          cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
          side = cell->face(face_n)->boundary_id() - 1;
          // if (neighbor_type[side] == subdomain_type) // Stokes-Stokes interface -> don't have to do this for Biot subdomains
          for (auto el : local_face_dof_indices)
          {
            if (el < n_stress){
              interface_dofs[side].push_back(el);
              interface_dofs_total.push_back(el);                                                    
            }
            else if(el >= n_stress+n_disp+n_rot && el < n_stress+n_disp+n_rot+n_flux)
            {
              interface_dofs[side].push_back(el);
              interface_dofs_total.push_back(el);        
            }
          }
        }

    for (unsigned int i = 0; i < n_faces_per_cell; ++i)
    {
      if (neighbors[i] >= 0)
      {
        std::cout << "Biot Part:\n Number of interface dofs on face " << i
              << " : " << interface_dofs[i].size() << std::endl;
      }
    }
  }

  // MixedBiotProblem - assemble_system
  template <int dim>
  void MixedBiotProblem<dim>::assemble_system ()
  {
    // TimerOutput::Scope t(computing_timer, "Assemble system");          
    system_matrix = 0;
    //system_rhs_bar = 0;

    QGauss<dim>   quadrature_formula(degree+3);
    //QGauss<dim-1> face_quadrature_formula(qdegree);

    FEValues<dim> fe_values (fe, quadrature_formula,
                              update_values    | update_gradients |
                              update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    //const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const LameParameters<dim> lame_function;
    const KInverse<dim> k_inverse;

    std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));
    std::vector<Tensor<2,dim>>  k_inverse_values(n_q_points);

    // Rotation variable is either a scalar(2d) or a vector(3d)
    const unsigned int rotation_dim = 0.5*dim*(dim-1);
    // Stress DoFs vectors
    std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
    std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());
    // Displacement DoFs
    const FEValuesExtractors::Vector displacement (dim*dim);
    // Velocity and Pressure DoFs
    const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
    const FEValuesExtractors::Scalar pressure (dim*dim + dim + 0.5*dim*(dim-1) + dim);

    for (unsigned int i=0; i<dim; ++i)
    {
        const FEValuesExtractors::Vector tmp_stress(i*dim);
        stresses[i].first_vector_component = tmp_stress.first_vector_component;
        if (dim == 2 && i == 0)
        {
            const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
            rotations[i].component = tmp_rotation.component;
        } else if (dim == 3) {
            const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + i);
            rotations[i].component = tmp_rotation.component;
        }
    }

  }


  // MixedBiotProblem - assemble RHS of star problems
  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs_bar ()
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS bar");  
  }


  // MixedBiotProblem - assemble RHS of star problems
  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs_star (FEFaceValues<dim> &fe_face_values)
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS star");
  }


    
  // MixedBiotProblem::solvers
  template <int dim>
  void MixedBiotProblem<dim>::solve_bar ()
  {
    // TimerOutput::Scope t(computing_timer, "Solve bar");
  }

  template <int dim>
  void MixedBiotProblem<dim>::solve_star ()
  {
      // TimerOutput::Scope t(computing_timer, "Solve star");
  }

  // MixedBiotProblem::run
  template <int dim>
  void MixedBiotProblem<dim>::run (unsigned int cycle)
  {
    Assert(reps[0].size() == dim, ExcDimensionMismatch(reps[0].size(), dim));

    if (mortar_flag)
      {
        Assert(n_processes > 1,
               ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
        Assert(reps.size() >= n_processes + 1,
               ExcMessage("Some of the mesh parameters were not provided"));
      }
    
    if (cycle == 0)
    {
      // Dimensions of the domain (unit hypercube)
      std::vector<double> subdomain_dimensions(dim);
      for (unsigned int d = 1; d < dim; ++d)
        subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

      subdomain_dimensions[0] = 2.0 / double(n_domains[0]);
      
      get_subdomain_coordinates<dim>(
        this_mpi, n_domains, subdomain_dimensions, p1, p2);

      if (mortar_flag)
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  reps[this_mpi],
                                                  p1,
                                                  p2);
      else
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  reps[0],
                                                  p1,
                                                  p2);

      // if (mortar_flag)
      //   GridGenerator::subdivided_hyper_rectangle(triangulation_mortar,
      //                                             reps[n_processes],
      //                                             p1,
      //                                             p2);
      for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell; ++side)
      {
        if (neighbor_type[side] == 1)
          stokes_neighbor = true; // Stokes-Stokes interface
        else if (neighbor_type[side] == 2) 
          biot_neighbor = true; // Stokes-Biot interface
        else if (neighbor_type[side] != 0)
          Assert(false, ExcMessage("Unknown neighbor type! The type must "
            "be 1 for Stokes or 2 for Biot. But got " + std::to_string(neighbor_type[side])
            + " for the subdomain " + std::to_string(this_mpi) + " of type " 
            + std::to_string(subdomain_type) + " on side " 
            + std::to_string(side) + "."));
      }
    }
    else {
      triangulation.refine_global();
    }

    // for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell; ++side)
    //   {
    //     std::cout << "Neighbor type on side " << side
    //           << " : " << neighbor_type[side] << std::endl;
    //   }
    
    setup_system();
      
    get_interface_dofs();

    assemble_system();
  }
    template class MixedBiotProblem<2>;
    // template class MixedBiotProblem<3>;
}
