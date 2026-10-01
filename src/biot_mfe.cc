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
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_rt_bubbles.h>
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
#include "../inc/biot_parameter_reader.h"


namespace mfe_biot
{
  using namespace dealii;
  using namespace dd_utilities;

  // MixedBiotProblem class constructor
  template <int dim>
  MixedBiotProblem<dim>::MixedBiotProblem(
    ParameterHandler &prm,
    unsigned int &degree,
    std::vector<unsigned int> n_domains,
    const unsigned int &subdomain_type,
    std::vector<int> &neighbors,
    std::vector<unsigned int> &neighbor_type,
    std::vector<unsigned int> &initial_refinement,
    std::vector<unsigned int> &initial_refinement_mortar,
    MPI_Comm mpi_communicator_biot,
    const unsigned int preconditioner_type,
    const unsigned int mortar_type,
    const unsigned int mortar_degree)
    : mpi_communicator (MPI_COMM_WORLD),
      mpi_communicator_biot(mpi_communicator_biot),
      this_mpi (Utilities::MPI::this_mpi_process(mpi_communicator)),
      n_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_biot (Utilities::MPI::this_mpi_process(mpi_communicator_biot)),
      n_processes_biot (Utilities::MPI::n_mpi_processes(mpi_communicator_biot)),
      prm(prm),
      degree(degree),
      n_domains(n_domains),
      subdomain_type(subdomain_type),
      neighbors(neighbors),
      neighbor_type(neighbor_type),
      initial_refinement(initial_refinement),
      initial_refinement_mortar(initial_refinement_mortar),
      stokes_neighbor(false),
      biot_neighbor(false),
      mortar_type(mortar_type),
      mortar_degree(mortar_degree),
      // P_coarse2fine (false),
      // P_fine2coarse (false),
      gmres_iteration(0),
      preconditioner_type(preconditioner_type),
      grid_diameter(0),
      fe (FE_BDM<dim>(degree), dim,                // stress
          FE_DGQ<dim>(degree-1), dim,              // displacement
          FE_DGQ<dim>(degree-1), 0.5*dim*(dim-1),  // rotation
          FE_BDM<dim>(degree), 1,                  // flux
          FE_DGQ<dim>(degree-1), 1),               // pressure
      dof_handler (triangulation),
      fe_mortar([mortar_degree, mortar_type]() -> FESystem<dim> {
        if (mortar_type == 0) // BDM(k), k>0; k = 1 is no mortar case
          return FESystem<dim>(FE_BDM<dim>(mortar_degree), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_BDM<dim>(mortar_degree), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else if (mortar_type == 1) // RT(k), k>=0
          return FESystem<dim>(FE_BDM<dim>(mortar_degree), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_RaviartThomas<dim>(mortar_degree), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else if (mortar_type == 2) // Q(k), k>0
          return FESystem<dim>(FE_Q<dim>(mortar_degree), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_Q<dim>(mortar_degree), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else if (mortar_type == 3) // DQ(k), k>=0
          return FESystem<dim>(FE_DGQ<dim>(mortar_degree), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_DGQ<dim>(mortar_degree), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else
          AssertThrow(false, ExcMessage("Invalid mortar type. Use 0 for no mortar, 1 for BDM, or 2 for RT."));
      }()),
      dof_handler_mortar (triangulation_mortar),
      pcout (std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator_biot) == 0))
      // computing_timer (mpi_communicator,
      //                   pcout,
      //                   TimerOutput::summary,
      //                   TimerOutput::wall_times)
  {
    interpret_parameters();
    BiotParameterReader param(prm_fxn);
    param.read_parameters("../test function/dd test/biot.prm");
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
    // mark_interface_faces(
    //   triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);
    // //this is a function in utilities.h

    pcout << "Biot Part:\n boundary condition:";
    for (unsigned int i = 0; i < external_bdry_ids.size(); ++i)
    {
      // if (this_mpi == 1)
      //   external_bdry_ids[0] = 9; // Velocity Neumann on bottom boundary for Biot domain 1
      if (external_bdry_ids[i] == 7)
        pcout << " N";
      else if (external_bdry_ids[i] == 8)
        pcout << " S";
      else if (external_bdry_ids[i] == 9)
        pcout << " V";
      else if (external_bdry_ids[i] == 0)
        pcout << " D";
      else
        AssertThrow(false, ExcMessage("Invalid boundary condition type. Use 0 for Dirichlet or 7 for Neumann."));
    }
    pcout << "" << std::endl;
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

    pcout << "Biot part:\n N stress dofs: " << n_s 
          << "\n N displacement dofs: " << n_u
          << "\n N rotation dofs: " << n_g
          << "\n N flux dofs: " << n_z
          << "\n N pressure dofs: " << n_p << std::endl;

    make_neumann_stress_data();
    make_neumann_velocity_data();                                

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

    solution_star.reinit(solution_bar);
    system_rhs_bar.reinit(solution_bar);
    system_rhs_star.reinit(solution_bar);

    solution_bar = 0;
    solution_star = 0;
    system_rhs_bar = 0;
    system_rhs_star = 0;

    // Mortar spaces setup
    faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell, 0);
    mark_interface_faces(
      triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);

    dof_handler_mortar.distribute_dofs(fe_mortar);
    DoFRenumbering::component_wise(dof_handler_mortar);

    std::vector<types::global_dof_index> dofs_per_component_mortar =
            DoFTools::count_dofs_per_fe_component (dof_handler_mortar);
    unsigned int n_s_mortar=0, n_u_mortar=0, n_g_mortar=0, n_z_mortar = 0, n_p_mortar = 0;

    // when using displacement mortar space 
    // as normal trace of a tensor space
    if (mortar_type < 2)
    {
      for (unsigned int i=0; i<dim; ++i)
      {
        n_s_mortar += dofs_per_component_mortar[i*dim];
        n_u_mortar += dofs_per_component_mortar[dim*dim + i];

        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
            n_g_mortar = dofs_per_component_mortar[dim*dim + dim];
        else if (dim == 3)
            n_g_mortar += dofs_per_component_mortar[dim*dim + dim + i];
      }
      n_z_mortar = dofs_per_component_mortar[dim*dim + dim + 0.5*dim*(dim-1)];
      n_p_mortar = dofs_per_component_mortar[dim*dim + dim +0.5*dim*(dim-1) + dim];
    }
    else
    {
      for (unsigned int i=0; i<dim; ++i)
      {
        n_s_mortar += dofs_per_component_mortar[i];
        n_u_mortar += dofs_per_component_mortar[dim + i];

        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
            n_g_mortar = dofs_per_component_mortar[2*dim];
        else if (dim == 3)
            n_g_mortar += dofs_per_component_mortar[2*dim + i];
      }
      // n_s_mortar += dofs_per_component_mortar[0];
      n_z_mortar = dofs_per_component_mortar[2*dim + 0.5*dim*(dim-1)];
      n_p_mortar = dofs_per_component_mortar[2*dim +0.5*dim*(dim-1) + 1];
    }

    n_stress_mortar = n_s_mortar;
    n_disp_mortar = n_u_mortar;
    n_rot_mortar = n_g_mortar;
    n_flux_mortar = n_z_mortar;
    n_pressure_mortar = n_p_mortar;
    // n_Elast= n_s+n_u+n_g;

    pcout << "Biot mortar part:\n N stress mortar dofs: " << n_s_mortar
          << "\n N displacement mortar dofs: " << n_u_mortar
          << "\n N rotation mortar dofs: " << n_g_mortar
          << "\n N flux mortar dofs: " << n_z_mortar
          << "\n N pressure mortar dofs: " << n_p_mortar << std::endl;

    mortar_vec.reinit(5);
    mortar_vec.block(0).reinit (n_s_mortar);
    mortar_vec.block(1).reinit (n_u_mortar);
    mortar_vec.block(2).reinit (n_g_mortar);
    mortar_vec.block(3).reinit (n_z_mortar);
    mortar_vec.block(4).reinit (n_p_mortar);
    mortar_vec.collect_sizes ();
  }

  // MixedBiotProblem::make_neumann_stress_data
  template <int dim>
  void MixedBiotProblem<dim>::make_neumann_stress_data ()
  {
    // clear the constraints objects in case you are
    // refining this, they will be closed after filling
    // the velocity data in the following method, since
    // we could have Neumann BCs on both stress and flux 
    // for Biot
    constraints_external_bar.clear();
    constraints_external_star.clear();

    // neumann stress boundary conditions 
    Functions::ParsedFunction<dim> neumann_stress(dim*dim);
    prm_fxn.enter_subsection("BC Neumann Stress");
    neumann_stress.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    const QGauss<dim - 1> face_quadrature(degree + 2);

    // Neumann boundary ids for Biot 
    // 7 for Neumann stress + Neumann Darcy velocity, 
    // 8 for Neumann stress + pressure dirichlet, 
    // 9 for Neumann Darcy velocity + displacement
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[7] = &neumann_stress;
    boundary_functions[8] = &neumann_stress;
    // none for velocity, those are in the next method

    std::map<types::global_dof_index, double> projected_boundary_values;
    std::vector<unsigned int> component_mapping(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1);

    for (unsigned int d = 0; d < dim*dim; ++d)
      component_mapping[d] = d;

    // Dummy mapping. We filter out these DoFs below.
    // The project boundary values method needs a mapping for all 
    // components. We just need to put something there to satisfy 
    // the api of the function project_boundary_values().
    for (unsigned int d = dim*dim; d < dim*dim + dim + 0.5*dim*(dim-1) + dim + 1; ++d)
      component_mapping[d] = 0;

    VectorTools::project_boundary_values(dof_handler,
                                        boundary_functions,
                                        face_quadrature,
                                        projected_boundary_values,
                                        component_mapping);

    // we do this to make sure that the pressure dofs are not 
    // constrained by the projected boundary values, since we 
    // only have Neumann BCs on flux in this example.
    // also, we set the inhomogeneity for the star constraints to 0
    for (const auto &[dof_index, value] : projected_boundary_values)
    {
      if (dof_index < n_stress)
      {
        constraints_external_bar.add_line(dof_index);
        constraints_external_bar.set_inhomogeneity(dof_index, value);

        constraints_external_star.add_line(dof_index);
        constraints_external_star.set_inhomogeneity(dof_index, 0.0);
      }
    }
  }

  // MixedBiotProblem::make_neumann_velocity_data
  template <int dim>
  void MixedBiotProblem<dim>::make_neumann_velocity_data ()
  {
    // neumann darcy boundary conditions 
    Functions::ParsedFunction<dim> neumann_velocity(dim);
    prm_fxn.enter_subsection("BC Neumann Velocity");
    neumann_velocity.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();  
    
    const QGauss<dim - 1> face_quadrature(degree + 2);
    
    // Neumann boundary ids for Biot 
    // 7 for Neumann stress + Neumann Darcy velocity, 
    // 8 for Neumann stress + pressure dirichlet, 
    // 9 for Neumann Darcy velocity + displacement
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[7] = &neumann_velocity;
    boundary_functions[9] = &neumann_velocity;
    // neumann_ids = 8 and other stress related data was 
    // filled earlier with the neumann_stress function

    std::map<types::global_dof_index, double> projected_boundary_values;
    std::vector<unsigned int> component_mapping(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1);

    for (unsigned int d = dim*dim + dim + 0.5*dim*(dim-1); d < dim*dim + dim + 0.5*dim*(dim-1)+dim; ++d)
      component_mapping[d] = d-(dim*dim + dim + 0.5*dim*(dim-1));

    // Dummy pressure mapping. We filter pressure DoFs out below.
    // The project boundary values method needs a mapping for all 
    // components. We just need to put something there to satisfy 
    // the api of the function project_boundary_values().
    for (unsigned int d = 0; d < dim*dim + dim + 0.5*dim*(dim-1); ++d)
      component_mapping[d] = 0; // elasticity components
    component_mapping[dim*dim + dim + 0.5*dim*(dim-1) + dim] = 0; // pressure component

    VectorTools::project_boundary_values(dof_handler,
                                        boundary_functions,
                                        face_quadrature,
                                        projected_boundary_values,
                                        component_mapping);

    // we do this to make sure that the pressure dofs are not 
    // constrained by the projected boundary values, since we 
    // only have Neumann BCs on flux in this example.
    // also, we set the inhomogeneity for the star constraints to 0
    for (const auto &[dof_index, value] : projected_boundary_values)
    {
      if (n_stress+n_disp+n_rot <= dof_index && dof_index < n_stress+n_disp+n_rot+n_flux)
      {
        constraints_external_bar.add_line(dof_index);
        constraints_external_bar.set_inhomogeneity(dof_index, value);

        constraints_external_star.add_line(dof_index);
        constraints_external_star.set_inhomogeneity(dof_index, 0.0);
      }
    }
    
    constraints_external_bar.close();
    constraints_external_star.close();
  }

  // MixedBiotProblemDD - initialize the interface data structure for coupled monolithic sheme
  template <int dim>
  void MixedBiotProblem<dim>::get_interface_dofs ()
  {
    interface_dofs.clear();
    interface_dofs_total.clear();
    interface_dofs_size = 0;
    interface_dofs.resize(GeometryInfo<dim>::faces_per_cell, std::vector<unsigned int> ());
    std::vector<types::global_dof_index> local_face_dof_indices;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    unsigned int side = 0;

    if (mortar_type >= 2) // using Qk or DQk mortar space
    {
      local_face_dof_indices.resize(fe_mortar.dofs_per_cell);
      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
        for (unsigned int face_n = 0; face_n < n_faces_per_cell; ++face_n)
          if (cell->at_boundary(face_n) &&
                (cell->face(face_n)->boundary_id() > 0 && 
                  cell->face(face_n)->boundary_id() < 7))
          {
            // cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
            cell->get_dof_indices(local_face_dof_indices);
            side = cell->face(face_n)->boundary_id() - 1;
            // if (neighbor_type[side] == subdomain_type) // Stokes-Stokes interface -> don't have to do this for Biot subdomains
            for (unsigned int i = 0; i < local_face_dof_indices.size(); ++i)
            {
              auto el = local_face_dof_indices[i];
              if (fe_mortar.has_support_on_face(i, face_n))
              {
                if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                                              == interface_dofs[side].end())
                {
                  if (el < n_stress_mortar)
                  {
                    interface_dofs[side].push_back(el);
                    interface_dofs_total.push_back(el);
                    // pcout << "hi disp mortar dof: " << el << std::endl;                                                
                  }
                  else if(el >= n_stress_mortar+n_disp_mortar+n_rot_mortar && 
                          el < n_stress_mortar+n_disp_mortar+n_rot_mortar+n_flux_mortar)
                  {
                    interface_dofs[side].push_back(el);
                    interface_dofs_total.push_back(el);    
                    // pcout << "hi pressure mortar dof: " << el << std::endl;      
                  }
                }
              }
            }
          }
    }
    else if (mortar_type < 2)
    {
      local_face_dof_indices.resize(fe_mortar.dofs_per_face);
      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
        for (unsigned int face_n = 0; face_n < n_faces_per_cell; ++face_n)
          if (cell->at_boundary(face_n) &&
                (cell->face(face_n)->boundary_id() > 0 && 
                  cell->face(face_n)->boundary_id() < 7))
          {
            cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
            side = cell->face(face_n)->boundary_id() - 1;
            // if (neighbor_type[side] == subdomain_type) // Stokes-Stokes interface -> don't have to do this for Biot subdomains
            for (auto el : local_face_dof_indices)
            {
              if (el < n_stress_mortar){
                interface_dofs[side].push_back(el);
                interface_dofs_total.push_back(el);                                                    
              }
              else if(el >= n_stress_mortar+n_disp_mortar+n_rot_mortar && el < n_stress_mortar+n_disp_mortar+n_rot_mortar+n_flux_mortar)
              {
                interface_dofs[side].push_back(el);
                interface_dofs_total.push_back(el);        
              }
            }
          }
    }
    else
    {
      AssertThrow(false, ExcMessage("Invalid mortar type."
                  " Use 0 for no mortar, 1 for BDM, or 2 for RT."));
    }

    if (this_mpi == 2)
    {
      for (unsigned int i = 0; i < n_faces_per_cell; ++i)
      {
        if (neighbors[i] >= 0)
        {
          std::cout << "Biot Part:\n Number of interface dofs on face " << i
                << " : " << interface_dofs[i].size() << std::endl;
        }
      }
    }
    compute_interface_dofs_size<dim>(interface_dofs_total, 
                                     mpi_communicator, 
                                     this_mpi, 
                                     n_processes,
                                     n_domains,
                                     interface_dofs_size);
    // if (this_mpi == 0)
    //   for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    //     if (neighbors[side] >= 0)
    //     {
    //       pcout << "\ninterface_dofs[" << side << "].size = " << interface_dofs[side].size() << " : ";
    //       for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
    //         pcout << interface_dofs[side][i] << " ";
    //     }
    //   pcout << std::endl;
  }

  // MixedBiotProblem - assemble_system
  template <int dim>
  void MixedBiotProblem<dim>::assemble_system ()
  {
    pcout << "Assembling system..." << std::endl;
    // TimerOutput::Scope t(computing_timer, "Assemble system"); 
    // double s0 = 1.0;
    // double alpha = 1.0;         
    system_matrix = 0;
    system_rhs_bar = 0;

    QGauss<dim>   quadrature_formula(degree+3);
    QGauss<dim-1> face_quadrature_formula(qdegree);

    FEValues<dim> fe_values (fe, quadrature_formula,
                              update_values    | update_gradients |
                              update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                            update_values    | update_normal_vectors |
                                            update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    //for Dirichlet BC from displacement eta = eta_0
    Functions::ParsedFunction<dim> boundary_values_dir_disp(dim);
    prm_fxn.enter_subsection("BC Dirichlet Displacement");
    boundary_values_dir_disp.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    //for Dirichlet BC from pressure p = p_0
    Functions::ParsedFunction<dim> boundary_values_dir_press(1);
    prm_fxn.enter_subsection("BC Dirichlet Pressure");
    boundary_values_dir_press.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    //for f = -div sigma
    Functions::ParsedFunction<dim> rhs_vec(dim);
    prm_fxn.enter_subsection("RHS vector");
    rhs_vec.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    Tensor<1, dim> rhs_vec_val;

    //for q = s0 * p + alpha * (div eta) + div u
    Functions::ParsedFunction<dim> rhs_scalar(1);
    prm_fxn.enter_subsection("RHS scalar");
    rhs_scalar.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    const LameParameters<dim> lame_function(lame_lambda, lame_mu);
    const KInverse<dim> k_inverse(K_permeability);

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
    
    for(const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit (cell);
      local_matrix = 0;
      local_rhs = 0;

      lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);
      k_inverse.value_list (fe_values.get_quadrature_points(), k_inverse_values);

      // Velocity and pressure
      std::vector<Tensor<1,dim>>                phi_u(dofs_per_cell);
      std::vector <double>                      div_phi_u(dofs_per_cell);
      std::vector <double>                      phi_p(dofs_per_cell);

      // Stress, displacement and rotation
      std::vector<std::vector<Tensor<1,dim>>> phi_s(dofs_per_cell, std::vector<Tensor<1,dim>> (dim));
      std::vector<Tensor<1,dim>> div_phi_s(dofs_per_cell);
      std::vector<Tensor<1,dim>> phi_d(dofs_per_cell);
      std::vector<Tensor<1,rotation_dim>> phi_r(dofs_per_cell);
      std::vector<double> div_phi_d(dofs_per_cell);

      Tensor<2,dim> sigma_i, asigma_j, apId_j;
      Tensor<1,rotation_dim> asym_i, asym_j;
      // to get boundary values for displacement and tau.n_p
      Tensor<1,dim> gd_disp; 
      std::vector<Tensor<1,dim>> tau_n_i(dofs_per_cell);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dim; ++i)
            rhs_vec_val[i] = rhs_vec.value(fe_values.quadrature_point(q), i);

        // Evaluate test functions 
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          phi_u[k] = fe_values[velocity].value (k, q);
          phi_p[k] = fe_values[pressure].value (k, q);

          div_phi_u[k] = fe_values[velocity].divergence (k, q);

          for (unsigned int s_i=0; s_i<dim; ++s_i)
          {
            phi_s[k][s_i] = fe_values[stresses[s_i]].value (k, q);
            div_phi_s[k][s_i] = fe_values[stresses[s_i]].divergence (k, q);
          }
          phi_d[k] = fe_values[displacement].value (k, q);
          div_phi_d[k] = fe_values[displacement].divergence (k, q);

          for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
            phi_r[k][r_i] = fe_values[rotations[r_i]].value (k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const double mu = lame_parameters_values[q][1];
          const double lambda = lame_parameters_values[q][0];

          make_asymmetry_tensor(phi_s[i], asym_i);
          make_tensor(phi_s[i], sigma_i);

          for (unsigned int j=0; j<dofs_per_cell; ++j)// i is the test function, multiplied from left
          {
            make_asymmetry_tensor(phi_s[j], asym_j);
            compliance_tensor(phi_s[j], mu, lambda, asigma_j);
            compliance_tensor_pressure(phi_p[j], mu, lambda, apId_j);

            local_matrix(i, j) += // Darcy law
                                  (viscosity * k_inverse_values[q] * phi_u[j] * phi_u[i] - phi_p[j] * div_phi_u[i]   // (Kinv u_p, v_p) - (p, div v_p)
                                  // Momentum           
                                    + div_phi_u[j] * phi_p[i] + s0*phi_p[i]*phi_p[j] + alpha*trace(asigma_j)*phi_p[i]  // (div u_p, w_p) + s0(p, w_p) + alpha(Asigma, w_p I)
                                    + alpha * alpha * trace(apId_j)*phi_p[i] // alpha^2 (Ap I, w_p I)
                                  // Mixed elasticity eq-ns
                                    + scalar_product(asigma_j, sigma_i) + alpha*scalar_product(apId_j, sigma_i)  // (Asigma, tau) + alpha(ApI, tau)      
                                    + scalar_product(phi_d[j], div_phi_s[i])              // (eta_p, div tau)
                                    + scalar_product<1, rotation_dim>(phi_r[j], asym_i)   // (gamma, tau)
                                    + scalar_product(phi_d[i], div_phi_s[j])              // (v_s, div sigma)
                                    + scalar_product<1, rotation_dim>(phi_r[i], asym_j) ) // (chi, sigma)
                                  * fe_values.JxW(q);
          }
          local_rhs(i) +=  (- phi_d[i] * rhs_vec_val // - (f_p, v_s)
                          + phi_p[i] * rhs_scalar.value(fe_values.quadrature_point(q))  ) // (q_p, w_p)
                          * fe_values.JxW(q);
        }
      }

      //Entering boundary integrals
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary())
        { 
          fe_face_values.reinit(cell, face);
          if (face->boundary_id() == 0 || face->boundary_id() == 8)//Entering pressure related Dirichlet Condition
          {
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            { 
              // Evaluate test functions  
              for (unsigned int k=0; k<dofs_per_cell; ++k)
                phi_u[k] = fe_face_values[velocity].value (k, q);
          
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                local_rhs(i) += -(phi_u[i] * fe_face_values.normal_vector(q) // - <v.n, g_p^D_p>
                                  * boundary_values_dir_press.value(fe_face_values.quadrature_point(q))
                                  * fe_face_values.JxW(q));                               
              }
            }
          }
          if (face->boundary_id() == 0 || face->boundary_id() == 9) //Entering displacement related Dirichlet Condition
          {
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              // Get boundary values displacement
              for (unsigned int i = 0; i < dim; ++i)
                gd_disp[i] = boundary_values_dir_disp.value(fe_face_values.quadrature_point(q), i);
              
              // Evaluate test functions  
              for (unsigned int k=0; k<dofs_per_cell; ++k)
                for (unsigned int s_i=0; s_i<dim; ++s_i)
                {
                  tau_n_i[k][s_i] = fe_face_values[stresses[s_i]].value (k, q)
                                    * fe_face_values.normal_vector(q);
                }
          
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                local_rhs(i) += ( tau_n_i[i] * gd_disp     // <tau.n, g_p^D_d>
                                  * fe_face_values.JxW(q));                               
              }
            }
          }
        }
      }

      cell->get_dof_indices (local_dof_indices);
      // Add local contributions to global system
      constraints_external_bar.distribute_local_to_global(local_matrix, 
                                                        local_rhs, 
                                                        local_dof_indices, 
                                                        system_matrix, 
                                                        system_rhs_bar);
      // for (unsigned int i=0; i<dofs_per_cell; ++i)
      // {
      //   system_rhs_bar(local_dof_indices[i]) += local_rhs(i);

      //   for (unsigned int j=0; j<dofs_per_cell; ++j)
      //     system_matrix.add (local_dof_indices[i],
      //                       local_dof_indices[j],
      //                       local_matrix(i,j));
      // }
    }
    pcout << "System assembly complete." << std::endl;
  }


  // MixedBiotProblem - assemble RHS of star problems
  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs_star (std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                                 FEFaceValues<dim> &fe_face_values)
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS star");
    system_rhs_star = 0;

    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Vector<double>       local_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double> phi_v_dot_n(dofs_per_cell);
    double lambda_d;

    int side;

    std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

    for (unsigned int d=0; d<dim; ++d)
    {
      const FEValuesExtractors::Vector tmp_stress(d*dim);
      stresses[d].first_vector_component = tmp_stress.first_vector_component;
    }

    const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));

    if (mortar_type < 2)
    {
      std::vector<std::vector<double>> interface_values_flux(dim, std::vector<double> (n_face_q_points));
      std::vector<std::vector<std::vector<double>>> interface_values_stress(dim, 
                                                      std::vector<std::vector<double>>(
                                                      dim, std::vector<double>(n_face_q_points)));
      
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        local_rhs = 0;
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) && 
            (cell->face(face_n)->boundary_id() > 0) && (cell->face(face_n)->boundary_id() < 7))
            {
              fe_face_values.reinit (cell, face_n);
              side = cell->face(face_n)->boundary_id() - 1;

              for (unsigned int s = 0; s < dim; ++s)
                interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                                    interface_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);

              for (unsigned int s = 0 ; s < dim*dim; ++s)
                interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                              interface_values_stress[s/dim][s%dim], s);

              for (unsigned int q=0; q<n_face_q_points; ++q)
              {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  for (unsigned int s=0; s<dim; ++s)
                  {
                    lambda_d = 0.0;
                    for (unsigned int l=0; l<dim; ++l)
                      lambda_d += interface_values_stress[s][l][q] * fe_face_values.normal_vector(q)[l];

                    local_rhs(i) += fe_face_values[stresses[s]].value (i, q)
                                    * fe_face_values.normal_vector(q)  // <tau.n, lambda_d>
                                    * get_normal_direction<dim>(side)
                                    * lambda_d
                                    * fe_face_values.JxW(q);
                    // lambda_p = interface_values_flux[s][q] * fe_face_values.normal_vector(q)[s];
                    local_rhs(i) += - fe_face_values[velocity].value (i, q) 
                                  * fe_face_values.normal_vector(q)
                                  * get_normal_direction<dim>(side)
                                  * interface_values_flux[s][q] * fe_face_values.normal_vector(q)[s] 
                                  * fe_face_values.JxW(q); // <v.n, lambda_p>                                 
                  }
                }
              }
            }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          system_rhs_star(local_dof_indices[i]) += local_rhs(i);
      }
    }
    else
    {
      std::vector<double> interface_values_press(n_face_q_points);
      std::vector<std::vector<double>> interface_values_disp(dim, std::vector<double>(n_face_q_points));

      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        local_rhs = 0;
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) && 
            (cell->face(face_n)->boundary_id() > 0) && (cell->face(face_n)->boundary_id() < 7))
          {
            fe_face_values.reinit (cell, face_n);
            side = cell->face(face_n)->boundary_id() - 1;

            for (unsigned int s = 0; s < dim; ++s)
            {
              interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                            interface_values_disp[s], s);
            }
            interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                                  interface_values_press, 2*dim + 0.5*dim*(dim-1));

            for (unsigned int q=0; q<n_face_q_points; ++q)
            {
              for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int s=0; s<dim; ++s)
                {
                  // lambda_d = 0.0;
                  // for (unsigned int l=0; l<dim; ++l)
                  //   lambda_d += interface_values_stress[s][l][q] * fe_face_values.normal_vector(q)[l];

                  local_rhs(i) += fe_face_values[stresses[s]].value (i, q)
                                  * fe_face_values.normal_vector(q)  // <tau.n, lambda_d>
                                  * interface_values_disp[s][q]
                                  * fe_face_values.JxW(q);                                                 
                }
                local_rhs(i) += - fe_face_values[velocity].value (i, q) 
                                * fe_face_values.normal_vector(q)
                                * interface_values_press[q]
                                * fe_face_values.JxW(q); // <v.n, lambda_p> 
              }
            }
          }
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          system_rhs_star(local_dof_indices[i]) += local_rhs(i);
      }
    }
    
  }

  // MixedStokesProblemDD - simple jacobi type preconditioner for the 
  // interface problem penalizing the difference in central eigenvalues of 
  // two neighbouring subdomain problems on their shared interface
  template <int dim>
  void MixedBiotProblem<dim>::preconditioner(std::unique_ptr<FEFunction> &solution_bar_function,
                                            std::unique_ptr<FEFunction> &solution_star_function,
                                            std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                            std::vector<std::vector<double>> &interface_data,
                                            std::vector<std::vector<double>> &interface_data_send,
                                            std::vector<std::vector<double>> &interface_data_receive,
                                            FEFaceValues<dim>                &fe_face_values,
                                            FEFaceValues<dim>                &fe_face_values_mortar,
                                            std::vector<Projector::Projector<dim>> &P_mortar,
                                            std::vector<std::vector<double>> &Ap)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    if (preconditioner_type == 1)
    {
      double trace = 0.0;
      for (unsigned int b = 0; b < system_matrix.n_block_rows(); ++b)
      {
        const SparseMatrix<double> &block = system_matrix.block(b,b);

        for (unsigned int i = 0; i < block.m(); ++i)
          trace += block.diag_element(i);
      }
      avg_eigenvalue = trace / system_matrix.m();

      nbr_avg_eigenvalue.resize(n_faces_per_cell);
      jacobi_scaling.resize(n_faces_per_cell);

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        {
          jacobi_scaling[side].resize(interface_dofs[side].size());
          MPI_Sendrecv(&avg_eigenvalue,
                        1,
                        MPI_DOUBLE,
                        neighbors[side],
                        this_mpi,  // send tag

                        &nbr_avg_eigenvalue[side],
                        1,
                        MPI_DOUBLE,
                        neighbors[side],
                        neighbors[side],  // recv tag
                        mpi_communicator,
                        &mpi_status);

          // jacobi_scaling[side] = 1.0;
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            if (abs(nbr_avg_eigenvalue[side])>abs(avg_eigenvalue))
              jacobi_scaling[side][i] = abs(nbr_avg_eigenvalue[side]/avg_eigenvalue);
            else
              jacobi_scaling[side][i] = abs(avg_eigenvalue/nbr_avg_eigenvalue[side]);
          }
          // if (neighbor_type[side] == 1)
          //   jacobi_scaling[side] /= std::pow(2, cycle);
        }
      double max_scaling = 1.0;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
       if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          max_scaling = std::max(jacobi_scaling[side][i], max_scaling);
        }
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
       if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          jacobi_scaling[side][i] = sqrt(max_scaling * std::pow(2, cycle_number));
        }
    }
    else if (preconditioner_type == 2) // implement jacobi preconditioner
    {
      //compute <Se_i,e_i> on the interface where e_i is the
      // i-th interface basis function, and S is the interface
      // operator, to use as a scaling in a jacobi preconditioner 
      // for the interface problem. multiscale basis functions
      // are used to compute the action of the interface operator on e_i
      unsigned int ind = 0;
      jacobi_scaling.resize(n_faces_per_cell);
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        jacobi_scaling[side].resize(interface_dofs[side].size());
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          if (neighbors[side] >= 0)
          {
            // interface_fe_function[side] = 0;
            // multiscale_basis[side][i].reinit(solution_star);
            // multiscale_basis[side][i] = 0;

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
              {
                Ap[face][j] = 0;
                interface_data[face][j] = 0;
              }

            // // interface_fe_function[side]                          = 0;
            // // interface_fe_function[side][interface_dofs[side][i]] = 1.0;
            // interface_data[side][i] = 1.0;

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              interface_fe_function[face] = 0;

            // interface_fe_function[side]                          = 0;
            interface_fe_function[side][interface_dofs[side][i]] = 1.0;
            
            // interface_operator(interface_function,
            //                   solution_bar_function,
            //                   solution_star_function,
            //                   interface_data,
            //                   interface_data_send,
            //                   interface_data_receive,
            //                   fe_face_values,
            //                   fe_face_values_mortar,
            //                   fe_face_values_mortar_biot,
            //                   P_mortar,
            //                   Ap);

            // interface operator has to be done explitly ow wrong info sent and received
            // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            //   for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            //     interface_fe_function[side][interface_dofs[side][i]] = 
            //       interface_data[side][i];

            if (multiscale_basis_flag)
            {
              // Instead of solving subdomain problems we compute the response using
              // basis
              solution_star = 0;
              // unsigned int j = 0;
              for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  solution_star.sadd(
                    1.0,
                    interface_fe_function[face][interface_dofs[face][j]],
                    multiscale_basis[face][j]);
                  // j += 1;
                }
            }
            else
            {
              // interface_fe_function.block(1) = 0;
              assemble_rhs_star(interface_function,
                                fe_face_values);
              solve_star();
            }
            
            for (unsigned int i = 0; i < solution_star.size(); ++i)
              AssertIsFinite(solution_star[i]);
            interface_inner_product(solution_bar_function,
                                    solution_star_function,
                                    fe_face_values_mortar,
                                    P_mortar);
            
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)     
              {                   
                // Create vector of u\dot n to send
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  if (interface_dofs[face][j] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
                    interface_data_send[face][j] = solution_star_mortar[face][interface_dofs[face][j]];
                  else //monolithic-darcy part (coupled)
                    interface_data_send[face][j] = - solution_star_mortar[face][interface_dofs[face][j]];
                }    
              }

            MPI_Sendrecv(&interface_data_send[side][i],
                          1,
                          MPI_DOUBLE,
                          neighbors[side],
                          this_mpi,  // send tag

                          &interface_data_receive[side][i],
                          1,
                          MPI_DOUBLE,
                          neighbors[side],
                          neighbors[side],  // recv tag

                          mpi_communicator,
                          &mpi_status);

            // MPI_Send(&interface_data_send[side][0],
            //           interface_dofs[side].size(),
            //           MPI_DOUBLE,
            //           neighbors[side],
            //           this_mpi,
            //           mpi_communicator);
            // MPI_Recv(&interface_data_receive[side][0],
            //           interface_dofs[side].size(),
            //           MPI_DOUBLE,
            //           neighbors[side],
            //           neighbors[side],
            //           mpi_communicator,
            //           &mpi_status);

            // Compute Ap and with it compute alpha
            // for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
            //   Ap[side][j] = (interface_data_send[side][j] +
            //                           interface_data_receive[side][j]);
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)
              {
                if (face == side)
                {
                  for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                  {
                    Ap[face][j] = (interface_data_send[face][j] +
                                    interface_data_receive[face][j]);
                  }
                }
                else
                {
                  for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                    Ap[face][j] = interface_data_send[face][j];
                }
              }
            // Ap[side][i] = (interface_data_send[side][i] +
            //                           interface_data_receive[side][i]);                          
            // interface_fe_function[side].block(2) = 0;
            // assemble_rhs_star(interface_function,
            //                   fe_face_values,
            //                   fe_face_values_mortar_biot);
            // solve_star();
            // pcout << "Computing jacobi, side = " << side << " , i = " << i 
            //       << " Ap[side][i] = " << Ap[side][i] << std::endl;
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                interface_data[face][j] = interface_fe_function[face][interface_dofs[face][j]];

            // if (side == 1 && this_mpi == 0)
            // {
            //   std::cout << "interface_data on side " << side << " for i = " << i << " is: ";
            //   for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
            //     std::cout << interface_data[side][j] << " ";
            //   std::cout << std::endl;
            // }
            // if (this_mpi == 0 && side == 1)
              // jacobi_scaling[side][i] = abs(Ap[side][i]);
              // jacobi_scaling[side][i] = inner_product_mortar(Ap,
              //                                               interface_data,
              //                                               fe_face_values_mortar);
            // else
              jacobi_scaling[side][i] = Ap[side][i];                                          

            // multiscale_basis[side][i] = solution_star;
            ind += 1;
            if (this_mpi == 1)
            {
              std::cout << "\r jacobi for S<e_i,e_i>, i = " << ind << " computed so far. on side " << side << std::flush;
            }
          }
        }
      }
    }
    else if (preconditioner_type == 4)
    {
      jacobi_scaling.resize(n_faces_per_cell);
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        {
          jacobi_scaling[side].resize(interface_dofs[side].size());
          // reset scaling to 1.0
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            jacobi_scaling[side][i] = 1.0;
        }
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        {
          if (neighbor_type[side] == 1) // Stokes-Biot interface
          {
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              if (interface_dofs[side][i] < n_stress_mortar) // displacement lagrange multiplier test space
                jacobi_scaling[side][i] = 2*lame_mu;
              else // pressure lagrange multiplier test space
                jacobi_scaling[side][i] =  1.0;
            }
          }
          else // Biot-Biot interface
          {
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              if (interface_dofs[side][i] < n_stress_mortar) // displacement lagrange multiplier test space
                jacobi_scaling[side][i] = 2*lame_mu;
              else // pressure lagrange multiplier test space
                jacobi_scaling[side][i] = 1.0;
            }
          }
        }
    }
    else if (preconditioner_type == 5)
    {
      //compute <Se_i,e_i> on the interface where e_i is the
      // i-th interface basis function, and S is the interface
      // operator, to use as a scaling in a jacobi preconditioner 
      // for the interface problem. multiscale basis functions
      // are used to compute the action of the interface operator on e_i
      unsigned int ind = 0;
      jacobi_scaling.resize(n_faces_per_cell);
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        jacobi_scaling[side].resize(interface_dofs[side].size());
        // for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          if (neighbors[side] >= 0)
          {
            // interface_fe_function[side] = 0;
            // multiscale_basis[side][i].reinit(solution_star);
            // multiscale_basis[side][i] = 0;

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
              {
                Ap[face][j] = 0;
                interface_data[face][j] = 0;
              }

            // // interface_fe_function[side]                          = 0;
            // // interface_fe_function[side][interface_dofs[side][i]] = 1.0;
            // interface_data[side][i] = 1.0;

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              interface_fe_function[face] = 0;

            // interface_fe_function[side]                          = 0;
            for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
              interface_fe_function[side][interface_dofs[side][j]] = 1.0;
            

            if (multiscale_basis_flag)
            {
              // Instead of solving subdomain problems we compute the response using
              // basis
              solution_star = 0;
              // unsigned int j = 0;
              for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  solution_star.sadd(
                    1.0,
                    interface_fe_function[face][interface_dofs[face][j]],
                    multiscale_basis[face][j]);
                  // j += 1;
                }
            }
            else
            {
              // interface_fe_function.block(1) = 0;
              assemble_rhs_star(interface_function,
                                fe_face_values);
              solve_star();
            }
            
            for (unsigned int i = 0; i < solution_star.size(); ++i)
              AssertIsFinite(solution_star[i]);
            interface_inner_product(solution_bar_function,
                                    solution_star_function,
                                    fe_face_values_mortar,
                                    P_mortar);
            
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)     
              {                   
                // Create vector of u\dot n to send
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  if (interface_dofs[face][j] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
                    interface_data_send[face][j] = solution_star_mortar[face][interface_dofs[face][j]];
                  else //monolithic-darcy part (coupled)
                    interface_data_send[face][j] = - solution_star_mortar[face][interface_dofs[face][j]];
                }    
              }

            MPI_Sendrecv(interface_data_send[side].data(),
                          interface_data_send[side].size(),
                          MPI_DOUBLE,
                          neighbors[side],
                          this_mpi,  // send tag

                          interface_data_receive[side].data(),
                          interface_data_receive[side].size(),
                          MPI_DOUBLE,
                          neighbors[side],
                          neighbors[side],  // recv tag

                          mpi_communicator,
                          &mpi_status);

          
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)
              {
                if (face == side)
                {
                  for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                  {
                    Ap[face][j] = ((interface_data_send[face][j] +
                                    interface_data_receive[face][j]));
                  }
                }
                // else
                // {
                //   for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                //     Ap[face][j] = interface_data_send[face][j];
                // }
              }
            // Ap[side][i] = (interface_data_send[side][i] +
            //                           interface_data_receive[side][i]);                          
            // interface_fe_function[side].block(2) = 0;
            // assemble_rhs_star(interface_function,
            //                   fe_face_values,
            //                   fe_face_values_mortar_biot);
            // solve_star();
            // pcout << "Computing jacobi, side = " << side << " , i = " << i 
            //       << " Ap[side][i] = " << Ap[side][i] << std::endl;
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                interface_data[face][j] = interface_fe_function[face][interface_dofs[face][j]];

            // if (side == 1 && this_mpi == 0)
            // {
            //   std::cout << "interface_data on side " << side << " for i = " << i << " is: ";
            //   for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
            //     std::cout << interface_data[side][j] << " ";
            //   std::cout << std::endl;
            // }
            // if (this_mpi == 0 && side == 1)
              // jacobi_scaling[side][i] = abs(Ap[side][i]);
              // jacobi_scaling[side][i] = inner_product_mortar(Ap,
              //                                               interface_data,
              //                                               fe_face_values_mortar);
            // else
              // jacobi_scaling[side][i] = Ap[side][i];                                          
            double scaling = 1.0;
            scaling = inner_product_mortar(Ap,Ap,fe_face_values_mortar)
                                            / inner_product_mortar(interface_data,interface_data,fe_face_values_mortar);
            // scaling = std::exp(scaling);
            for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
              jacobi_scaling[side][j] = scaling;
            // multiscale_basis[side][i] = solution_star;
            ind += 1;
            if (this_mpi == 3)
            {
              std::cout << "this_mpi = " << this_mpi << ", jacobi for [" << side << "] =" << scaling << std::endl;
            }
          }
        }
      }
    }
    else if (preconditioner_type == 6)
    {
      //compute <Se_i,e_i> on the interface where e_i is the
      // i-th interface basis function, and S is the interface
      // operator, to use as a scaling in a jacobi preconditioner 
      // for the interface problem. multiscale basis functions
      // are used to compute the action of the interface operator on e_i
      unsigned int ind = 0;
      jacobi_scaling.resize(n_faces_per_cell);
      // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          jacobi_scaling[side].resize(interface_dofs[side].size());
        // for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          // if (neighbors[side] >= 0)
          {

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
              {
                Ap[face][j] = 0;
                interface_data[face][j] = 0;
              }

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              interface_fe_function[face] = 0;


            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
              for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
                interface_fe_function[side][interface_dofs[side][j]] = 1.0;
            

            if (multiscale_basis_flag)
            {
              // Instead of solving subdomain problems we compute the response using
              // basis
              solution_star = 0;
              // unsigned int j = 0;
              for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  solution_star.sadd(
                    1.0,
                    interface_fe_function[face][interface_dofs[face][j]],
                    multiscale_basis[face][j]);
                  // j += 1;
                }
            }
            else
            {
              // interface_fe_function.block(1) = 0;
              assemble_rhs_star(interface_function,
                                fe_face_values);
              solve_star();
            }
            
            for (unsigned int i = 0; i < solution_star.size(); ++i)
              AssertIsFinite(solution_star[i]);
            interface_inner_product(solution_bar_function,
                                    solution_star_function,
                                    fe_face_values_mortar,
                                    P_mortar);
            
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)     
              {                   
                // Create vector of u\dot n to send
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  if (interface_dofs[face][j] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
                    interface_data_send[face][j] = solution_star_mortar[face][interface_dofs[face][j]];
                  else //monolithic-darcy part (coupled)
                    interface_data_send[face][j] = - solution_star_mortar[face][interface_dofs[face][j]];
                }    
              

                MPI_Sendrecv(interface_data_send[face].data(),
                              interface_data_send[face].size(),
                              MPI_DOUBLE,
                              neighbors[face],
                              this_mpi,  // send tag

                              interface_data_receive[face].data(),
                              interface_data_receive[face].size(),
                              MPI_DOUBLE,
                              neighbors[face],
                              neighbors[face],  // recv tag

                              mpi_communicator,
                              &mpi_status);

          
                
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  Ap[face][j] = ((interface_data_send[face][j] +
                                  interface_data_receive[face][j]));
                }
                    
              }
            
            

            std::vector<std::vector<double>> Ap_face(n_faces_per_cell);
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)     
                Ap_face[face].resize(interface_dofs[face].size());    

            std::vector<double> scaling(n_faces_per_cell, 1.0);

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)
              {
                // reset Ap_face and interface_data for the face of interest
                for (unsigned int side = 0 ; side < n_faces_per_cell; ++side)
                  for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
                  {
                    Ap_face[side][j] = 0;
                    interface_data[side][j] = 0;
                  }
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  Ap_face[face][j] = Ap[face][j];
                  interface_data[face][j] = interface_fe_function[face][interface_dofs[face][j]];
                }

                scaling[face] = sqrt(inner_product_mortar(Ap_face,Ap_face,fe_face_values_mortar)
                                            / inner_product_mortar(interface_data,interface_data,fe_face_values_mortar));
                // scaling[face] = std::exp(scaling[face]);
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                  jacobi_scaling[face][j] = scaling[face];
              }
            
            ind += 1;
            if (this_mpi == 3)
            {
              for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                std::cout << "this_mpi = " << this_mpi << ", jacobi for [" << face << "] =" << scaling[face] << std::endl;
            }
          }
        }
      }
    }
    else if (preconditioner_type == 7)
    {
      //compute <Se_i,e_i> on the interface where e_i is the
      // i-th interface basis function, and S is the interface
      // operator, to use as a scaling in a jacobi preconditioner 
      // for the interface problem. multiscale basis functions
      // are used to compute the action of the interface operator on e_i
      unsigned int ind = 0;
      jacobi_scaling.resize(n_faces_per_cell);
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        jacobi_scaling[side].resize(interface_dofs[side].size());
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          jacobi_scaling[side][i] = 1.0;
        {
          if (neighbors[side] >= 0)
          {

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
              {
                Ap[face][j] = 0;
                interface_data[face][j] = 0;
              }

            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              interface_fe_function[face] = 0;

            // interface_fe_function[side]                          = 0;
            for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
              interface_fe_function[side][interface_dofs[side][j]] = 1.0;
            

            if (multiscale_basis_flag)
            {
              // Instead of solving subdomain problems we compute the response using
              // basis
              solution_star = 0;
              // unsigned int j = 0;
              for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  solution_star.sadd(
                    1.0,
                    interface_fe_function[face][interface_dofs[face][j]],
                    multiscale_basis[face][j]);
                  // j += 1;
                }
            }
            else
            {
              // interface_fe_function.block(1) = 0;
              assemble_rhs_star(interface_function,
                                fe_face_values);
              solve_star();
            }
            
            for (unsigned int i = 0; i < solution_star.size(); ++i)
              AssertIsFinite(solution_star[i]);
            interface_inner_product(solution_bar_function,
                                    solution_star_function,
                                    fe_face_values_mortar,
                                    P_mortar);
            
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)     
              {                   
                // Create vector of u\dot n to send
                for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                {
                  if (interface_dofs[face][j] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
                    interface_data_send[face][j] = solution_star_mortar[face][interface_dofs[face][j]];
                  else //monolithic-darcy part (coupled)
                    interface_data_send[face][j] = - solution_star_mortar[face][interface_dofs[face][j]];
                }    
              }

            MPI_Sendrecv(interface_data_send[side].data(),
                          interface_data_send[side].size(),
                          MPI_DOUBLE,
                          neighbors[side],
                          this_mpi,  // send tag

                          interface_data_receive[side].data(),
                          interface_data_receive[side].size(),
                          MPI_DOUBLE,
                          neighbors[side],
                          neighbors[side],  // recv tag

                          mpi_communicator,
                          &mpi_status);

          
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)
              {
                if (face == side)
                {
                  for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                  {
                    Ap[face][j] = std::log10(abs((interface_data_send[face][j] +
                                    interface_data_receive[face][j])));
                  }
                }
              }
            
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              for (unsigned int j = 0; j < interface_dofs[face].size(); ++j)
                interface_data[face][j] = interface_fe_function[face][interface_dofs[face][j]];

                                                 
            // double scaling = 1.0;
            // scaling = inner_product_mortar(Ap,Ap,fe_face_values_mortar)
            //                                 / inner_product_mortar(interface_data,interface_data,fe_face_values_mortar);
            // scaling = std::exp(scaling);
            for (unsigned int j = 0; j < interface_dofs[side].size(); ++j)
              jacobi_scaling[side][j] = std::pow(5.0, (Ap[side][j]));
            // multiscale_basis[side][i] = solution_star;
            ind += 1;
            // if (this_mpi == 3)
            // {
            //   std::cout << "this_mpi = " << this_mpi << ", jacobi for [" << side << "] =" << scaling << std::endl;
            // }
          }
        }
      }
    }
    // if (this_mpi == 3)
    // {
    //   // std::cout << "Average eigenvalue: " << avg_eigenvalue << std::endl;
    //   for (unsigned int side = 0; side < jacobi_scaling.size(); ++side)
    //     if (neighbors[side] >= 0)
    //     {
    //       std::cout << "side " << side << std::endl;
    //       for (unsigned int i = 0; i < jacobi_scaling[side].size(); ++i)
    //         std::cout << "Jacobi scaling for dof: " << interface_dofs[side][i] << " : " << jacobi_scaling[side][i] << std::endl;
    //     }
    // }
  }

  template <int dim>
  void
  MixedBiotProblem<dim>::compute_multiscale_basis(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                                    FEFaceValues<dim>                &fe_face_values,
                                                    std::vector<Projector::Projector<dim>> &P_mortar)
  {
    unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    // Quadrature<dim - 1> quad;
    // quad = QGauss<dim - 1>(qdegree);
    multiscale_basis.resize(n_faces_per_cell);
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      multiscale_basis[side].resize(interface_dofs[side].size());

    if (this_mpi == 1)
    {
      std::cout << "Computing multiscale basis functions...\n"
            << "interface_dofs_total.size() = " << interface_dofs_total.size() << std::endl;
    }

    unsigned int ind = 0;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
      {
        if (neighbors[side] >= 0)
        {
          // interface_fe_function[side] = 0;
          multiscale_basis[side][i].reinit(solution_star);
          multiscale_basis[side][i] = 0;

          for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            interface_fe_function[face] = 0;

          // interface_fe_function[side]                          = 0;
          interface_fe_function[side][interface_dofs[side][i]] = 1.0;
          

          // interface_fe_function[side].block(2) = 0;
          assemble_rhs_star(interface_function,
                            fe_face_values);
          solve_star();

          multiscale_basis[side][i] = solution_star;
          ind += 1;
          if (this_mpi == 1)
          {
            std::cout << "\r ..." << ind
                  << " basis functions computed so far." << std::flush;
          }
        }
      }
    }
    if (this_mpi == 1)
    {
      std::cout << std::endl;
    }
    MPI_Barrier(mpi_communicator_biot);
  }
    
  // MixedBiotProblem::solvers
  template <int dim>
  void MixedBiotProblem<dim>::solve_bar ()
  {
    // TimerOutput::Scope t(computing_timer, "Solve bar");
    A_direct.initialize (system_matrix);
    A_direct.vmult(solution_bar, system_rhs_bar);
    constraints_external_bar.distribute(solution_bar);
    pcout << "Solve bar completed." << std::endl;

    if (std::isnan(solution_bar.l2_norm()))
      std::cout << "Warning: NaN values in solution_bar!" << std::endl;
    
    // solution_bar *= avg_eigenvalue;
    // solution_bar.print(std::cout);
  }

  template <int dim>
  void MixedBiotProblem<dim>::solve_star ()
  {
    // TimerOutput::Scope t(computing_timer, "Solve star");
    constraints_external_star.condense(system_rhs_star);
    A_direct.vmult (solution_star, system_rhs_star);
    constraints_external_star.distribute(solution_star);
    // solution_star *= avg_eigenvalue; 
    if (std::isnan(solution_star.l2_norm()))
      std::cout << "Warning: NaN values in solution_star!" << std::endl;
  }

  //Functions for GMRES:-------------------

  //finding the l2 norm of a std::vector<double> vector
  template <int dim>
  double
  MixedBiotProblem<dim>::vect_norm(std::vector<double> v)
  {
  	double result = 0;
  	for(unsigned int i=0; i<v.size(); ++i){
  		result+= v[i]*v[i];
  	}
  	return sqrt(result);
  }

  //Calculating the given rotation matrix
  template <int dim>
  void
  MixedBiotProblem<dim>::givens_rotation(double v1, double v2, double &cs, double &sn)
  {
  	if(abs(v1)<1e-25){
  		cs=0;
  		sn=1;
  	}
  	else{
  		double t = sqrt(v1*v1 + v2*v2);
  		cs = abs(v1)/t;
  		sn=cs*v2/v1;
  	}
  }

  //Applying givens rotation to H column
  template <int dim>
  void
  MixedBiotProblem<dim>::apply_givens_rotation(std::vector<double> &h, 
                                                 std::vector<double> &cs, 
                                                 std::vector<double> &sn,
                                                 unsigned int k_iteration)
  {
	  int k=k_iteration;
  	assert(h.size()>k+1); //size should be k+2
  	double temp;
  	for( int i=0; i<=k-1; ++i){
  		temp= cs[i]* h[i]+ sn[i]*h[i+1];
  		h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1];
  		h[i] = temp;
  	}
  	assert(h.size()==k+2);
  	//update the next sin cos values for rotation
  	double cs_k=0, sn_k=0;
  	 givens_rotation(h[k],h[k+1],cs_k,sn_k);


  	 //Eliminate H(i+1,i)
  	 h[k] = cs_k*h[k] + sn_k*h[k+1];
  	 h[k+1] = 0.0;
  	 //adding cs_k and sn_k as cs(k) and sn(k)
  	 cs.push_back(cs_k);
  	 sn.push_back(sn_k);
  }

  //solving Ry = beta[0:k] for y (with back substitution)
  //where k = # iterations = k_counter + 1
  //and R is upper triangular matrix obtained from
  //applying givens rotations to H
  template <int dim>
  void
  MixedBiotProblem<dim>::back_solve(const std::vector<std::vector<double>> &H, 
                                      const std::vector<double>              &beta, 
                                      std::vector<double>                    &y,
                                      const unsigned int                     &k)
  {
  	//  int k = beta.size()-1;
  	assert(y.size()==beta.size()-1);
    //initializing reset y to zero
  	for(int i=0; i<y.size();i++)
  		y[i]=0;
  	for(int i=k; i>=0; i--)
    {
  		y[i]= beta[i]/H[i][i];
      if (k == 0)
        y[0] = beta[0]/H[0][0];
      else
        for(int j = i+1; j<=k-1; j++)
        {
          y[i]-= H[j][i]*y[j]/H[i][i];
        }
  	}
  }

  /*
     Interface operator function
     takes in lambda and returns A*lambda
     for Biot-Biot interface
  */
  template <int dim>
  void
  MixedBiotProblem<dim>::interface_operator(std::unique_ptr<FEFunction> &solution_bar_function,
                                            std::unique_ptr<FEFunction> &solution_star_function,
                                            std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                            std::vector<std::vector<double>>   &interface_data,
                                            std::vector<std::vector<double>> &interface_data_send,
                                            std::vector<std::vector<double>> &interface_data_receive,
                                            FEFaceValues<dim>                &fe_face_values,
                                            FEFaceValues<dim>                &fe_face_values_mortar,
                                            std::vector<Projector::Projector<dim>> &P_mortar,
                                            std::vector<std::vector<double>> &Ap)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = 
            interface_data[side][i];

    if (multiscale_basis_flag)
    {
      // Instead of solving subdomain problems we compute the response using
      // basis
      solution_star = 0;
      // unsigned int j = 0;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          solution_star.sadd(
            1.0,
            interface_fe_function[side][interface_dofs[side][i]],
            multiscale_basis[side][i]);
          // j += 1;
        }
    }
    else
    {
      // interface_fe_function.block(1) = 0;
      assemble_rhs_star(interface_function,
                        fe_face_values);
      solve_star();
    }        
    // // interface_fe_function.block(1) = 0;
    // assemble_rhs_star(interface_function,
    //                   fe_face_values);
    // solve_star();

    //for computing the inner products
    //in the mortar space
    interface_inner_product(solution_bar_function,
                            solution_star_function,
                            fe_face_values_mortar,
                            P_mortar);
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        // if (neighbor_type[side] == 1 && mortar_type < 2) // Biot-Stokes interface
        // {
        //   for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        //   {
        //     if (interface_dofs[side][i] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
        //       interface_data_send[side][i] = solution_star_mortar[side][interface_dofs[side][i]]
        //                                        * get_normal_direction<dim>(side);
        //     else //monolithic-darcy part (coupled)
        //       interface_data_send[side][i] = - solution_star_mortar[side][interface_dofs[side][i]]
        //                                        * get_normal_direction<dim>(side);
        //   }    
        // }
        // else // Biot-Biot interface or Biot-Stokes interface with Q(k) or DQ(k) mortar
        // {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            if (interface_dofs[side][i] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
              interface_data_send[side][i] = solution_star_mortar[side][interface_dofs[side][i]];
            else //monolithic-darcy part (coupled)
              interface_data_send[side][i] = - solution_star_mortar[side][interface_dofs[side][i]];
          }    
        // }   

        MPI_Sendrecv(interface_data_send[side].data(),
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      this_mpi,  // send tag

                      interface_data_receive[side].data(),
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      neighbors[side],  // recv tag

                      mpi_communicator,
                      &mpi_status);

        // Compute Ap and with it compute alpha
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          Ap[side][i] = (interface_data_send[side][i] +
                          interface_data_receive[side][i]);
        }
        if (preconditioner_type != 0) //Jacobi scaling
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            Ap[side][i] /= jacobi_scaling[side][i];
      }
  }

  // computing the residual over all sides
  template <int dim>
  void
  MixedBiotProblem<dim>::compute_residual(std::unique_ptr<FEFunction> &solution_bar_function,
                                          std::unique_ptr<FEFunction> &solution_star_function,
                                          std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                          FEFaceValues<dim>                &fe_face_values,
                                          FEFaceValues<dim>                &fe_face_values_mortar,
                                          std::vector<std::vector<double>> &interface_data,
                                          std::vector<Projector::Projector<dim>> &P_mortar,
                                          std::vector<std::vector<double>> &r,
                                          double                           &r_norm,
                                          std::vector<std::vector<double>> &r_b,
                                          double                           &b_norm)
  {
    // double initial_val = 0.0;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    std::vector<double>     r_norm_side(n_faces_per_cell,0);
    std::vector<double>     r_b_norm_side(n_faces_per_cell,0);

    //compute Ax from x = initial_val
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = 
            interface_data[side][i];
    
    if (multiscale_basis_flag)
    {
      // Instead of solving subdomain problems we compute the response using
      // basis
      solution_star = 0;
      // unsigned int j = 0;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          solution_star.sadd(
            1.0,
            interface_fe_function[side][interface_dofs[side][i]],
            multiscale_basis[side][i]);
          // j += 1;
        }
    }
    else
    {
      assemble_rhs_star(interface_function,
                        fe_face_values);
      solve_star();
    }        
    // // interface_fe_function.block(1) = 0;
    // assemble_rhs_star(interface_function,
    //                   fe_face_values);
    // solve_star();

    //for computing rhs b for
    // Ax = b in interface problem below
    // solve_bar();

    //for computing the inner products
    //in the mortar space
    interface_inner_product(solution_bar_function,
                            solution_star_function,
                            fe_face_values_mortar,
                            P_mortar);
    
    //compute residual r = b - A*x
    for (unsigned side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      { 
        std::vector<double> r_b_receive_buffer(r[side].size());
        std::vector<double> r_receive_buffer(r[side].size());

        // if (neighbor_type[side] == 1 && mortar_type < 2) // Biot-Stokes interface
        // {
        //   for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        //   {
        //     if (interface_dofs[side][i] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
        //     {
        //       r_b[side][i] = - solution_bar_mortar[side][interface_dofs[side][i]]
        //                         * get_normal_direction<dim>(side);
        //       r[side][i] = - solution_bar_mortar[side][interface_dofs[side][i]]
        //                       * get_normal_direction<dim>(side)
        //                   - solution_star_mortar[side][interface_dofs[side][i]]
        //                       * get_normal_direction<dim>(side);     
        //     }
        //     else //monolithic-darcy part (coupled)
        //     {
        //       r_b[side][i] = solution_bar_mortar[side][interface_dofs[side][i]]
        //                       * get_normal_direction<dim>(side);
        //       r[side][i] =  solution_bar_mortar[side][interface_dofs[side][i]]
        //                       * get_normal_direction<dim>(side)
        //                   + solution_star_mortar[side][interface_dofs[side][i]]
        //                       * get_normal_direction<dim>(side);
        //     }       
        //   }        
        // }
        // else // Biot-Biot interface or Biot-Stokes interface with Q(k) or DQ(k) mortar
        // {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            if (interface_dofs[side][i] < n_stress_mortar) //monolithic-Elascitiy part (coupled)
            {
              r_b[side][i] = - solution_bar_mortar[side][interface_dofs[side][i]];
              r[side][i] = - solution_bar_mortar[side][interface_dofs[side][i]]
                          - solution_star_mortar[side][interface_dofs[side][i]];     
            }
            else //monolithic-darcy part (coupled)
            {
              r_b[side][i] = solution_bar_mortar[side][interface_dofs[side][i]];
              r[side][i] =  solution_bar_mortar[side][interface_dofs[side][i]]
                          + solution_star_mortar[side][interface_dofs[side][i]];
            }       
          }         
        // }

        MPI_Sendrecv(r_b[side].data(),
                    r_b[side].size(),
                    MPI_DOUBLE,
                    neighbors[side],
                    this_mpi,                 // send tag

                    r_b_receive_buffer.data(),
                    r_b_receive_buffer.size(),
                    MPI_DOUBLE,
                    neighbors[side],
                    neighbors[side],          // recv tag (sender rank)

                    mpi_communicator,
                    &mpi_status);

        MPI_Sendrecv(r[side].data(),
                    r[side].size(),
                    MPI_DOUBLE,
                    neighbors[side],
                    this_mpi,                 // send tag

                    r_receive_buffer.data(),
                    r_receive_buffer.size(),
                    MPI_DOUBLE,
                    neighbors[side],
                    neighbors[side],          // recv tag (sender rank)

                    mpi_communicator,
                    &mpi_status);

        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            r[side][i] += r_receive_buffer[i];
            r_b[side][i] += r_b_receive_buffer[i];
          }
        if (preconditioner_type != 0) //Jacobi scaling
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            r[side][i] /= jacobi_scaling[side][i];
            r_b[side][i] /= jacobi_scaling[side][i];
          }
      }
      r_norm = inner_product_mortar(r, r, fe_face_values_mortar);
      b_norm = inner_product_mortar(r_b, r_b, fe_face_values_mortar);

    //Calculating r-norm(same as b-norm)-----
    double r_norm_buffer =0;
    double b_norm_buffer =0;
    MPI_Allreduce(&b_norm,
                  &b_norm_buffer,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);
    MPI_Allreduce(&r_norm,
                  &r_norm_buffer,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);
    b_norm = sqrt(b_norm_buffer);
    r_norm = sqrt(r_norm_buffer);
  }

  // compute inner product in mortar space
  // <A*lambda, lambda>
  template <int dim>
  void
  MixedBiotProblem<dim>::interface_inner_product(std::unique_ptr<FEFunction> &solution_bar_function,
                                                 std::unique_ptr<FEFunction> &solution_star_function,
                                                 FEFaceValues<dim> &fe_face_values_mortar,
                                                 std::vector<Projector::Projector<dim>> &P_mortar)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    unsigned int side = 0;

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        solution_bar_mortar[side] = 0;
        solution_star_mortar[side] = 0;
        solution_bar_mortar_temp[side] = 0;
        solution_star_mortar_temp[side] = 0;
      }

    const unsigned int n_face_q_points = fe_face_values_mortar.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe_mortar.dofs_per_cell;

    Vector<double>                       local_rhs_bar(dofs_per_cell);
    Vector<double>                       local_rhs_star(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_eta(dofs_per_cell, Tensor<1, dim>());

    //first index for dim components, 0, 1: vel, 2: pressure
    //second for value at each quad point
    std::vector<std::vector<double>> bar_values_flux(dim, std::vector<double>(n_face_q_points, 0.0));
    std::vector<std::vector<double>> star_values_flux(dim, std::vector<double>(n_face_q_points, 0.0));
    std::vector<std::vector<std::vector<double>>> bar_values_stress(dim, 
                                                    std::vector<std::vector<double>>(dim,
                                                    std::vector<double>(n_face_q_points, 0.0)));
    std::vector<std::vector<std::vector<double>>> star_values_stress(dim, 
                                                    std::vector<std::vector<double>>(dim,
                                                    std::vector<double>(n_face_q_points, 0.0)));
    

    if (mortar_type < 2)
    {
      std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

      for (unsigned int d=0; d<dim; ++d)
      {
        const FEValuesExtractors::Vector tmp_stress(d*dim);
        stresses[d].first_vector_component = tmp_stress.first_vector_component;
      }
      const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));

      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
      {
        // local_rhs_bar  = 0;
        // local_rhs_star = 0;
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) &&
            (cell->face(face_n)->boundary_id() > 0) && (cell->face(face_n)->boundary_id() < 7)  )
          {
            //Need to reorder faces deal.ii iterators 
            //use a different ordering from freefem, 
            //they are ordered component wise 
            fe_face_values_mortar.reinit(cell, face_n);
            side = cell->face(face_n)->boundary_id() - 1;
            for (unsigned int s = 0; s < dim; ++s)
            {
              solution_bar_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                bar_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);
              solution_star_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                star_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);                                  
            }
            for (unsigned int s = 0 ; s < dim*dim; ++s)
            {
              solution_bar_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                bar_values_stress[s/dim][s%dim], s);
              solution_star_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                star_values_stress[s/dim][s%dim], s);
            }
            for (unsigned int q_point = 0; q_point < n_face_q_points;
                  ++q_point)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                // if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                //                 local_dof_indices[k]) != interface_dofs[side].end())
                // {
                  for (unsigned int s = 0; s < dim; ++s)
                    phi_eta[k][s] = fe_face_values_mortar[stresses[s]].value (k, q_point)
                                  * fe_face_values_mortar.normal_vector(q_point)
                                  * get_normal_direction<dim>(side);
                  phi_p[k] = fe_face_values_mortar[velocity].value (k, q_point)
                              * fe_face_values_mortar.normal_vector(q_point)
                              * get_normal_direction<dim>(side);
                // }
                // else
                // {
                //   for (unsigned int s = 0; s < dim; ++s)
                //     phi_eta[k][s] = 0.0;
                //   phi_p[k] = 0.0;
                // }
              }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                                local_dof_indices[i]) != interface_dofs[side].end())
                {
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    for (unsigned int l = 0; l < dim; ++l)
                    {
                      solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                        (phi_eta[i][s] * 
                          bar_values_stress[s][l][q_point] * 
                          fe_face_values_mortar.normal_vector(q_point)[l]) //  (phi_eta(x_q) * sigma_bar * n) 
                        * fe_face_values_mortar.JxW(q_point);    // dx

                      solution_star_mortar_temp[side](local_dof_indices[i]) +=
                        (phi_eta[i][s] * 
                          star_values_stress[s][l][q_point] * 
                          fe_face_values_mortar.normal_vector(q_point)[l]) // (phi_eta(x_q) * sigma_star * n) 
                        * fe_face_values_mortar.JxW(q_point);    // dx
                    }
                  }
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                      (phi_p[i] * 
                        bar_values_flux[s][q_point] * 
                        fe_face_values_mortar.normal_vector(q_point)[s])   // phi_p(x_q)* u_bar * n 
                      * fe_face_values_mortar.JxW(q_point);    // dx

                    solution_star_mortar_temp[side](local_dof_indices[i]) +=
                      (phi_p[i] * 
                        star_values_flux[s][q_point] * 
                        fe_face_values_mortar.normal_vector(q_point)[s])   // phi_p(x_q)* u_star * n 
                        * fe_face_values_mortar.JxW(q_point);    // dx
                  }
                }
              }
            }
          }
      }
    }
    else
    {
      const FEValuesExtractors::Vector disp_mortar(0);
      const FEValuesExtractors::Scalar press_mortar (2*dim+ 0.5*dim*(dim-1));

      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) &&
            (cell->face(face_n)->boundary_id() > 0) && (cell->face(face_n)->boundary_id() < 7)  )
          {
            //Need to reorder faces deal.ii iterators 
            //use a different ordering from freefem, 
            //they are ordered component wise 
            fe_face_values_mortar.reinit(cell, face_n);
            side = cell->face(face_n)->boundary_id() - 1;
            for (unsigned int s = 0; s < dim; ++s)
            {
              solution_bar_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                bar_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);
              solution_star_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                star_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);                                  
            }
            for (unsigned int s = 0 ; s < dim*dim; ++s)
            {
              solution_bar_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                bar_values_stress[s/dim][s%dim], s);
              solution_star_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                                star_values_stress[s/dim][s%dim], s);
            }
            for (unsigned int q_point = 0; q_point < n_face_q_points;
                  ++q_point)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                // if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                //                 local_dof_indices[k]) != interface_dofs[side].end())
                // {
                  phi_eta[k] = fe_face_values_mortar[disp_mortar].value (k, q_point);
                  phi_p[k] = fe_face_values_mortar[press_mortar].value (k, q_point);
                // }
                // else
                //   {
                //     phi_eta[k] = 0.0;
                //     phi_p[k] = 0.0;
                //   }
              }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                                local_dof_indices[i]) != interface_dofs[side].end())
                {
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    for (unsigned int l = 0; l < dim; ++l)
                    {
                      solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                        (phi_eta[i][s] * 
                          bar_values_stress[s][l][q_point] * 
                          fe_face_values_mortar.normal_vector(q_point)[l]) //  (phi_eta(x_q) * sigma_bar * n) 
                        * fe_face_values_mortar.JxW(q_point);    // dx

                      solution_star_mortar_temp[side](local_dof_indices[i]) +=
                        (phi_eta[i][s] * 
                          star_values_stress[s][l][q_point] * 
                          fe_face_values_mortar.normal_vector(q_point)[l]) // (phi_eta(x_q) * sigma_star * n) 
                        * fe_face_values_mortar.JxW(q_point);    // dx
                    }
                  }
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                      (phi_p[i] * 
                        bar_values_flux[s][q_point] * 
                        fe_face_values_mortar.normal_vector(q_point)[s])   // phi_p(x_q)* u_bar * n 
                      * fe_face_values_mortar.JxW(q_point);    // dx

                    solution_star_mortar_temp[side](local_dof_indices[i]) +=
                      (phi_p[i] * 
                        star_values_flux[s][q_point] * 
                        fe_face_values_mortar.normal_vector(q_point)[s])   // phi_p(x_q)* u_star * n 
                        * fe_face_values_mortar.JxW(q_point);    // dx
                  }
                }
              }
            }
          }
      }

    }
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        project_mortar<dim>(P_mortar[side],
                      dof_handler_mortar,
                      solution_bar_mortar_temp[side],
                      quad,
                      constraints_mortar,
                      neighbors,
                      dof_handler_mortar,
                      solution_bar_mortar[side],
                      interface_dofs[side],
                      side);
        project_mortar<dim>(P_mortar[side],
                      dof_handler_mortar,
                      solution_star_mortar_temp[side],
                      quad,
                      constraints_mortar,
                      neighbors,
                      dof_handler_mortar,
                      solution_star_mortar[side],
                      interface_dofs[side],
                      side);   
        
        // solution_bar_mortar[side] = solution_bar_mortar_temp[side];
        // solution_star_mortar[side] = solution_star_mortar_temp[side];
        // solution_star_mortar[side]*= 1/avg_eigenvalue;
      }           
  }

  // compute inner product in mortar space
  // <lambda, mu> on a particular face
  template <int dim>
  double
  MixedBiotProblem<dim>::inner_product_mortar(std::vector<std::vector<double>> &v_1,
                                              std::vector<std::vector<double>> &v_2,
                                              FEFaceValues<dim> &fe_face_values_mortar)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double inner_product = 0.0;

    const unsigned int n_face_q_points = fe_face_values_mortar.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe_mortar.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_eta(dofs_per_cell, Tensor<1, dim>());

    std::vector<BlockVector<double>> vec_1(n_faces_per_cell, BlockVector<double>(mortar_vec));
    std::vector<BlockVector<double>> vec_2(n_faces_per_cell, BlockVector<double>(mortar_vec));
    for(unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      vec_1[side] = 0;
      vec_2[side] = 0;
    }
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          vec_1[side](interface_dofs[side][i]) = v_1[side][i];
          vec_2[side](interface_dofs[side][i]) = v_2[side][i];
        }
      }

    if (mortar_type < 2)
    {
      //first index for dim components, 0, 1: vel, 2: pressure
      //second for value at each quad point
      std::vector<Tensor<1, dim>> interface_values_v1_flux(n_face_q_points, Tensor<1, dim>());
      std::vector<Tensor<1, dim>> interface_values_v2_flux(n_face_q_points, Tensor<1, dim>());

      std::vector<std::vector<Tensor<1, dim>>> interface_values_v1_stress(dim, 
                                                std::vector<Tensor<1, dim>>(n_face_q_points, Tensor<1, dim>()));
      std::vector<std::vector<Tensor<1, dim>>> interface_values_v2_stress(dim, 
                                                std::vector<Tensor<1, dim>>(n_face_q_points, Tensor<1, dim>()));

      std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

      for (unsigned int d=0; d<dim; ++d)
      {
        const FEValuesExtractors::Vector tmp_stress(d*dim);
        stresses[d].first_vector_component = tmp_stress.first_vector_component;
      }

      const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));

      unsigned int side = 0;
      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) &&
            (cell->face(face_n)->boundary_id() > 0) && (cell->face(face_n)->boundary_id() < 7) )
          {
            //Need to reorder faces deal.ii iterators 
            //use a different ordering from freefem, 
            //they are ordered component wise 
            fe_face_values_mortar.reinit(cell, face_n);
            side = cell->face(face_n)->boundary_id() - 1;
            
            fe_face_values_mortar[velocity].get_function_values(
              vec_1[side], interface_values_v1_flux);
            fe_face_values_mortar[velocity].get_function_values(
              vec_2[side], interface_values_v2_flux);                                 
            
            for (unsigned int s = 0 ; s < dim; ++s)
            {
              fe_face_values_mortar[stresses[s]].get_function_values(
                vec_1[side], interface_values_v1_stress[s]);
              fe_face_values_mortar[stresses[s]].get_function_values(
                vec_2[side], interface_values_v2_stress[s]);
            }
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
              inner_product += 
                (interface_values_v1_flux[q_point] * fe_face_values_mortar.normal_vector(q_point)) *
                (interface_values_v2_flux[q_point] * fe_face_values_mortar.normal_vector(q_point)) // (u1 * u2)
                * fe_face_values_mortar.JxW(q_point);    // dx
              for (unsigned int s = 0; s < dim; ++s)
                inner_product += 
                  (interface_values_v1_stress[s][q_point] * fe_face_values_mortar.normal_vector(q_point) *
                  interface_values_v2_stress[s][q_point] * fe_face_values_mortar.normal_vector(q_point)) // (sigma1.n * sigma2.n)
                  * fe_face_values_mortar.JxW(q_point);    // dx
            }
              
          }
          
      }
    }
    else
    {
      //first index for dim components, 0, 1: vel, 2: pressure
      //second for value at each quad point
      std::vector<double> interface_values_v1_press(n_face_q_points, 0.0);
      std::vector<double> interface_values_v2_press(n_face_q_points, 0.0);

      std::vector<Tensor<1, dim>> interface_values_v1_disp(n_face_q_points, Tensor<1, dim>());
      std::vector<Tensor<1, dim>> interface_values_v2_disp(n_face_q_points, Tensor<1, dim>());

      std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

      const FEValuesExtractors::Vector disp (0);
      const FEValuesExtractors::Scalar press (2*dim + 0.5*dim*(dim-1));

      unsigned int side = 0;
      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) &&
            (cell->face(face_n)->boundary_id() > 0) && (cell->face(face_n)->boundary_id() < 7) )
          {
            //Need to reorder faces deal.ii iterators 
            //use a different ordering from freefem, 
            //they are ordered component wise 
            fe_face_values_mortar.reinit(cell, face_n);
            side = cell->face(face_n)->boundary_id() - 1;
            
            fe_face_values_mortar[press].get_function_values(
              vec_1[side], interface_values_v1_press);
            fe_face_values_mortar[press].get_function_values(
              vec_2[side], interface_values_v2_press);                                 
            
            fe_face_values_mortar[disp].get_function_values(
                  vec_1[side], interface_values_v1_disp);
              fe_face_values_mortar[disp].get_function_values(
                vec_2[side], interface_values_v2_disp);
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
              inner_product += 
                (interface_values_v1_press[q_point] *
                interface_values_v2_press[q_point] ) // (lambda_p_1 * lambda_p_2)
                * fe_face_values_mortar.JxW(q_point);    // dx
              
                inner_product += 
                  (interface_values_v1_disp[q_point] *
                  interface_values_v2_disp[q_point]) // (lambda_d_1 * lambda_d_2)
                  * fe_face_values_mortar.JxW(q_point);    // dx
            }
              
          }
      }
    }
    return inner_product;
  }

  // Arnoldi algorithm
  template <int dim>
  void
  MixedBiotProblem<dim>::arnoldi(std::unique_ptr<FEFunction> &solution_bar_function,
                                std::unique_ptr<FEFunction> &solution_star_function,
                                std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                unsigned int                      &k_counter, 
                                std::vector<std::vector<double>> &interface_data,
                                std::vector<std::vector<double>> &interface_data_send,
                                std::vector<std::vector<double>> &interface_data_receive,
                                FEFaceValues<dim>                &fe_face_values,
                                FEFaceValues<dim>                &fe_face_values_mortar,
                                std::vector<Projector::Projector<dim>> &P_mortar,
                                std::vector<std::vector<double>> &Ap,
                                std::vector<std::vector<double>> &q,
                                std::vector<std::vector<std::vector<double>>>	&Q_side,
                                std::vector<std::vector<double>>	            &H)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    //interface data will be given as Q_side[side][k_counter];
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        interface_data[side]=Q_side[side][k_counter];
    
    //////------solving the  star problem to find AQ(k)---------------------
    interface_operator(solution_bar_function,
                      solution_star_function,
                      interface_function,
                      interface_data,
                      interface_data_send,
                      interface_data_receive,
                      fe_face_values,
                      fe_face_values_mortar,
                      P_mortar,
                      Ap);

    //defing q  to push_back to Q (Arnoldi algorithm)
    //defing h  to push_back to H (Arnoldi algorithm)
    std::vector<double> h(k_counter+2,0);

    std::vector<std::vector<double>> Q_tmp(n_faces_per_cell);
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        Q_tmp[side].resize(interface_dofs[side].size(), 0.0);

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      if (neighbors[side] >= 0)
      {
        q[side].resize(Ap[side].size(),0);
        assert(Ap[side].size()==Q_side[side][k_counter].size());
        q[side] = Ap[side];
      }
    }
    for(unsigned int i=0; i<=k_counter; ++i)
    {
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        for (unsigned int k = 0; k < q[side].size(); ++k)
          Q_tmp[side][k]=Q_side[side][i][k];
      h[i] = inner_product_mortar(Q_tmp, q, fe_face_values_mortar);

      // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      //   if (neighbors[side] >= 0)
      //     for(unsigned int j=0; j<q[side].size();++j)
      //       h[i]+=q[side][j]*Q_side[side][i][j];
    } //////-----------end of loop over side, q is calculated as AQ[][k] and Arnoldi Algorithm continued-------------------------------
    
    //Arnoldi Algorithm continued
    //combining summing h[i] over all subdomains
    std::vector<double> h_buffer(k_counter+2,0);

    MPI_Allreduce(&h[0],
                  &h_buffer[0],
                  k_counter+2,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    h=h_buffer;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for(unsigned int i=0; i<=k_counter; ++i)
          for(unsigned int j=0; j<q[side].size();++j){
            q[side][j]-=h[i]*Q_side[side][i][j];
          }//end first loop for arnoldi algorithm

    // -------------------------------------------------------------
    // Second CGS pass: reorthogonalization
    // -------------------------------------------------------------
    std::vector<double> h2(k_counter + 2, 0.0);

    for (unsigned int i = 0; i <= k_counter; ++i)
    {
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
          for (unsigned int k = 0; k < q[side].size(); ++k)
            Q_tmp[side][k] = Q_side[side][i][k];

      // q here is already the result of the FIRST projection
      h2[i] = inner_product_mortar(Q_tmp,
                                  q,
                                  fe_face_values_mortar);
    }

    // Sum second-pass projection coefficients over MPI ranks
    std::vector<double> h2_buffer(k_counter + 2, 0.0);

    MPI_Allreduce(&h2[0],
                  &h2_buffer[0],
                  k_counter + 2,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    h2 = h2_buffer;


    // Remove the remaining components in the existing Krylov basis
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i <= k_counter; ++i)
          for (unsigned int j = 0; j < q[side].size(); ++j)
            q[side][j] -= h2[i] * Q_side[side][i][j];

    // The Hessenberg coefficient is the sum of both projections
    for (unsigned int i = 0; i <= k_counter; ++i)
      h[i] += h2[i];
    // -----------------------------------------------------------


    double h_dummy = 0;

    //calculating h(k+1)=norm(q) as summation over side,subdomains norm_squared(q[side])
    // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    //   if (neighbors[side] >= 0)
    //     h_dummy+=vect_norm(q[side])*vect_norm(q[side]);

    h_dummy = inner_product_mortar(q, q, fe_face_values_mortar);
    
    double h_k_buffer=0;

    MPI_Allreduce(&h_dummy,
                  &h_k_buffer,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    h[k_counter+1]=sqrt(h_k_buffer);
              

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        for(unsigned int i=0;i<q[side].size();++i)
          q[side][i]/=h[k_counter+1];
        //Pushing back q[side] h to Q  as Q(k+1)
        Q_side[side].push_back(q[side]);
      }

    //Pushing back  h to H as H(k)}
    H.push_back(h);
    //---end of Arnoldi Algorithm
  }

  //local GMRES function.
  template <int dim>
  void
  MixedBiotProblem<dim>::local_gmres(unsigned int &cycle)
  {
    pcout << "Starting GMRES solver for interface problem..." << std::endl;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data(n_faces_per_cell);
    // double initial_val = 0.0; //initial guess for gmres
    interface_fe_function.resize(n_faces_per_cell);
    solution_bar_mortar.resize(n_faces_per_cell);
    solution_star_mortar.resize(n_faces_per_cell);
    solution_bar_mortar_temp.resize(n_faces_per_cell);
    solution_star_mortar_temp.resize(n_faces_per_cell);

    std::vector<Projector::Projector <dim>> P_mortar(n_faces_per_cell, 
                                                  Projector::Projector <dim>(false));

    std::vector<std::vector<double>> lambda(n_faces_per_cell);

    //GMRES structures and parameters
    std::vector<double>	sn;
    std::vector<double>	cs;
    std::vector<std::vector<double>>	H;
    // std::vector<double> error_iter_side(n_faces_per_cell); //saves error in each iteration
    std::vector<double> e_all_iter; //error will be saved here after each iteration
    std::vector<double>	Beta; //beta for each side
    double combined_error_iter =0; //sum of error_iter_side

    std::vector<std::vector<double>> r(n_faces_per_cell); //to be deleted probably: p?
    std::vector<std::vector<double>> r_b(n_faces_per_cell); //RHS
    std::vector<std::vector<double>> Ap(n_faces_per_cell);
    std::vector<std::vector<std::vector<double>>>	Q_side(n_faces_per_cell) ;
    

    //defing q  to push_back to Q (reused in Arnoldi algorithm)
    std::vector<std::vector<double>> q(n_faces_per_cell);

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        interface_data_receive[side].resize(interface_dofs[side].size(),
                                            0);
        interface_data_send[side].resize(interface_dofs[side].size(), 0);
        interface_data[side].resize(interface_dofs[side].size(), initial_val);

        interface_fe_function[side].reinit(mortar_vec);

        solution_bar_mortar[side].reinit(mortar_vec);
        solution_star_mortar[side].reinit(mortar_vec);

        solution_bar_mortar_temp[side].reinit(mortar_vec);
        solution_star_mortar_temp[side].reinit(mortar_vec);

        Ap[side].resize(interface_dofs[side].size(), 0);
        lambda[side].resize(interface_dofs[side].size(), initial_val);

        q[side].resize(interface_dofs[side].size());
        r[side].resize(interface_dofs[side].size(), 0);
        r_b[side].resize(interface_dofs[side].size(), 0);
      }

    std::vector<std::unique_ptr<FEFunction>> interface_function;
    std::unique_ptr<FEFunction> solution_bar_function;
    std::unique_ptr<FEFunction> solution_star_function;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      interface_function.emplace_back(
        std::make_unique<FEFunction>(dof_handler_mortar, interface_fe_function[side]));
    }
    solution_bar_function = std::make_unique<FEFunction>(dof_handler, solution_bar);
    solution_star_function = std::make_unique<FEFunction>(dof_handler, solution_star);

    // Extra for projections from mortar to fine grid and RHS assembly
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);


    // dealii::AffineConstraints<double>   constraints;
    FEFaceValues<dim> fe_face_values(fe,
                                    quad,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
    FEFaceValues<dim> fe_face_values_mortar(fe_mortar,
                                            quad,
                                            update_values | update_normal_vectors |
                                              update_quadrature_points |
                                              update_JxW_values);

    if (multiscale_basis_flag == true)
    {
      compute_multiscale_basis(interface_function,
                              fe_face_values,
                              P_mortar);
    }                                          
    if (preconditioner_type != 0)
      preconditioner(solution_bar_function,
                    solution_star_function,
                    interface_function,
                    interface_data,
                    interface_data_send,
                    interface_data_receive,
                    fe_face_values,
                    fe_face_values_mortar,
                    P_mortar,
                    Ap);
    //Calculating r-norm(same as b-norm)-----
    double r_norm =0;
    double b_norm =0;
    //computing the residual r=b-Ax
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        std::fill(interface_data[side].begin(), interface_data[side].end(), initial_val);
        interface_fe_function[side] = 0;
        std::fill(interface_data_send[side].begin(), interface_data_send[side].end(), 0.0);
        std::fill(interface_data_receive[side].begin(), interface_data_receive[side].end(), 0.0);
        std::fill(Ap[side].begin(), Ap[side].end(), 0.0);
      }
    compute_residual(solution_bar_function,
                     solution_star_function,
                     interface_function,
                     fe_face_values,
                     fe_face_values_mortar,
                     interface_data,
                     P_mortar,
                     r,
                     r_norm,
                     r_b,
                     b_norm);
    //end -----------of calculating r-norm and b_norm------------------

    //Making the first element of matrix Q[side] same as r[side]/r_norm
    for(unsigned int side=0; side<n_faces_per_cell;++side)
      if (neighbors[side] >= 0)
      {
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          q[side][i]= r[side][i]/r_norm;
        //adding q[side] as first element of Q[side]
        Q_side[side].push_back(q[side]);
      }

    //end----------- of caluclating first element of Q[side]-----------------
    e_all_iter.push_back(r_norm);
    Beta.push_back(r_norm);


    unsigned int k_counter = 0; //same as the count of the iteration
    gmres_iteration = 0;
    std::vector<double> y;
    while (k_counter < maxiter)
    {
      gmres_iteration++;
      //Performing the Arnoldi algorithm
      arnoldi(solution_bar_function,
              solution_star_function,
              interface_function,
              k_counter,
              interface_data,
              interface_data_send,
              interface_data_receive,
              fe_face_values,
              fe_face_values_mortar,
              P_mortar,
              Ap,
              q,
              Q_side,
              H);
      
      //Eliminating the last element in H ith row and updating the rotation matrix.
      apply_givens_rotation(H[k_counter],
                            cs,
                            sn,
                            k_counter);
      //Updating the residual vector
      Beta.push_back(-sn[k_counter]*Beta[k_counter]);
      Beta[k_counter]*=cs[k_counter];

      //Combining error at kth iteration
      combined_error_iter=abs(Beta[k_counter+1])/b_norm;


      //saving the combined error at each iteration
      e_all_iter.push_back(combined_error_iter);
    
      if (preconditioner_type == 0)
        residual = combined_error_iter;
      else // recompute the actual residual
      {
        //Calculating the result from H ,Q_side and Beta
        //Finding y which has size k_counter using back sove function
        y.resize(k_counter+1,0);
        assert(Beta.size()==k_counter+2); // gives error if exceed maxiter, do break if size is too much and going to exceed maxiter
        back_solve(H,Beta,y,k_counter);

        // reset lambda to zero, x = 0, lambda = x + Q*y
        for (int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              lambda[side][i] = initial_val;
        //updating X(lambda) to get the final lambda value before solving the final star problem
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            for (unsigned int i = 0; i < interface_data[side].size(); ++i)
              for(unsigned int j=0; j<=k_counter; ++j)
                lambda[side][i] += Q_side[side][j][i]*y[j];
        
        // do not scale while computing the residual for the exit criterion, 
        // otherwise it will be wrong. We can still use preconditioner to 
        // compute the residual but we should not scale it and just use the 
        // norm of r to compute the residual for exit criterion. This is 
        // because the preconditioner changes the norm and we want to check 
        // the actual residual in the original norm for convergence.
        unsigned int tmp_preconditioner_type = preconditioner_type;
        preconditioner_type = 0; 
        compute_residual(solution_bar_function,
                        solution_star_function,
                        interface_function,
                        fe_face_values,
                        fe_face_values_mortar,
                        lambda,
                        P_mortar,
                        r,
                        r_norm,
                        r_b,
                        b_norm);
        residual = r_norm/b_norm; // relative residual for exit criterion
        preconditioner_type = tmp_preconditioner_type;
      }

      pcout << "\r  ..." << gmres_iteration
            << " iterations completed, (residual = " << residual
            << ")..." << std::flush;
      // Exit criterion
      if (residual < tol)
      {
        pcout << "\n  GMRES converges in " << gmres_iteration << " iterations!\n";
        break;
      }
      else if(k_counter>maxiter-2)
      {
        pcout << "\n  GMRES doesn't converge after  " << k_counter + 1 << " iterations!\n";
        break;
      }
      else if(k_counter == interface_dofs_size-1)
      {
        pcout << "\n  GMRES doesn't converge after " << interface_dofs_size << " iterations!\n";
        break;
      }


      
      //maxing interface_data_receive and send zero so it can be used is solving for Ap(or A*Q([k_counter]).
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        interface_data_receive[side].resize(interface_dofs[side].size(), 0);
        interface_data_send[side].resize(interface_dofs[side].size(), 0);
      }

      Ap.resize(n_faces_per_cell);
      k_counter++;
    }//end of the while loop(k_counter<max iteration)

    if (preconditioner_type == 0)
    {
      //Calculating the result from H ,Q_side and Beta
      //Finding y which has size k_counter using back sove function
      // std::vector<double> 
      y.resize(k_counter+1,0);
      // assert(Beta.size()==k_counter+2); // gives error if exceed maxiter, do break if size is too much and going to exceed maxiter
      back_solve(H,Beta,y,k_counter);

      //updating X(lambda) to get the final lambda value 
      //lambda = x_initial + Q*y before solving the final star problem
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
          for (unsigned int i = 0; i < interface_data[side].size(); ++i)
            for(unsigned int j=0; j<=k_counter; ++j)
              lambda[side][i] += Q_side[side][j][i]*y[j];
    }

    //we can replace lambda here and just add interface_data(skip one step below)
    interface_data = lambda;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = 
            interface_data[side][i];

    if (multiscale_basis_flag)
    {
      // Instead of solving subdomain problems we compute the response using
      // basis
      solution_star = 0;
      // unsigned int j = 0;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          solution_star.sadd(
            1.0,
            interface_fe_function[side][interface_dofs[side][i]],
            multiscale_basis[side][i]);
          // j += 1;
        }
    }
    else
    {
      assemble_rhs_star(interface_function,
                        fe_face_values);
      solve_star();
    }        
    // assemble_rhs_star(interface_function,
    //                   fe_face_values);
    // solve_star();
    solution.reinit(solution_bar);
    solution = solution_bar;
    solution.sadd(1.0, solution_star);

    pcout<<"finished local_gmres"<<"\n";
  }


  template <int dim>
  void
  MixedBiotProblem<dim>::output_dof_results(const unsigned int cycle) const
  {
    std::ofstream mesh_file("../output/gnuplot_data/mesh-"
      +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
          +".gnuplot");
    GridOut().write_gnuplot(triangulation, mesh_file);
  }

  template <int dim>
  void
  MixedBiotProblem<dim>::output_results(const unsigned int cycle)
  {
    BlockVector<double> exact_solution;
    exact_solution.reinit(solution);

    AffineConstraints<double> constraints;
    // fill constraints as appropriate
    constraints.close();

    QTrapezoid<1>      q_trapez;
    QIterated<dim>  quadrature(q_trapez,degree+2);

    ExactSolution<dim> exact_solution_function;
    prm_fxn.enter_subsection("Exact solution Darcy");
    exact_solution_function.exact_solution_darcy_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution stress");
    exact_solution_function.exact_solution_stress_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution displacement");
    exact_solution_function.exact_solution_displacement_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution rotation");
    exact_solution_function.exact_solution_rotation_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    VectorTools::project(dof_handler,
                        constraints,
                        quadrature,
                        exact_solution_function,
                        exact_solution);
    
    // Error in FE space
    BlockVector<double> error = solution;
    error -= exact_solution;
                        
    std::vector<std::string> solution_names;
    switch(dim)
    {
      case 2:
        solution_names.push_back ("s11");
        solution_names.push_back ("s12");
        solution_names.push_back ("s21");
        solution_names.push_back ("s22");
        solution_names.push_back ("d1");
        solution_names.push_back ("d2");
        solution_names.push_back ("r");
        solution_names.push_back ("u1");
        solution_names.push_back ("u2");
        solution_names.push_back ("p");
        break;

      case 3:
        solution_names.push_back ("s11");
        solution_names.push_back ("s12");
        solution_names.push_back ("s13");
        solution_names.push_back ("s21");
        solution_names.push_back ("s22");
        solution_names.push_back ("s23");
        solution_names.push_back ("s31");
        solution_names.push_back ("s32");
        solution_names.push_back ("s33");
        solution_names.push_back ("d1");
        solution_names.push_back ("d2");
        solution_names.push_back ("d3");
        solution_names.push_back ("r1");
        solution_names.push_back ("r2");
        solution_names.push_back ("r3");
        solution_names.push_back ("u1");
        solution_names.push_back ("u2");
        solution_names.push_back ("u3");
        solution_names.push_back ("p");
        break;

      default:
      Assert(false, ExcNotImplemented());
    }

    // Components interpretation of the mechanics solution (vector^dim - vector - rotation)
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim*dim+dim, DataComponentInterpretation::component_is_part_of_vector);
    switch (dim)
    {
      case 2:
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        break;

      case 3:
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
        break;

      default:
        Assert(false, ExcNotImplemented());
        break;
    }
    
    // Components interpretation of the flow solution (vector - scalar)
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    
    data_out.add_data_vector(solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
    data_out.build_patches();

    std::ofstream output(
      "../output/paraview_data/sol_dd-" + Utilities::int_to_string(this_mpi)
      + "_" + Utilities::int_to_string(cycle, 2) 
      + ".vtu");
    data_out.write_vtu(output);
  }


  template <int dim>
  void MixedBiotProblem<dim>::compute_errors(const unsigned int &cycle) 
  {
    const unsigned int total_dim = static_cast<unsigned int>(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1);

    const ComponentSelectFunction<dim> stress_mask(std::make_pair(0,dim*dim), total_dim);
    const ComponentSelectFunction<dim> displacement_mask(std::make_pair(dim*dim,dim*dim+dim), total_dim);
    const ComponentSelectFunction<dim> rotation_mask(std::make_pair(dim*dim+dim,dim*dim+dim+0.5*dim*(dim-1)), total_dim);

    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(dim*dim+dim+0.5*dim*(dim-1), dim*dim+dim+0.5*dim*(dim-1)+dim), total_dim);
    const ComponentSelectFunction<dim> pressure_mask(static_cast<unsigned int>(dim*dim+dim+0.5*dim*(dim-1)+dim), total_dim);
  
    ExactSolution<dim> exact_solution;
    prm_fxn.enter_subsection("Exact solution Darcy");
    exact_solution.exact_solution_darcy_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution stress");
    exact_solution.exact_solution_stress_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution displacement");
    exact_solution.exact_solution_displacement_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution rotation");
    exact_solution.exact_solution_rotation_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    prm_fxn.enter_subsection("Exact gradient Darcy Velocity");
    exact_solution.exact_solution_darcy_vel_grad_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact gradient stress");
    exact_solution.exact_solution_stress_grad_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    // Vectors to temporarily store cellwise errros
    Vector<double> cellwise_errors (triangulation.n_active_cells());
    Vector<double> cellwise_norms (triangulation.n_active_cells());

    QTrapezoid<1>      q_trapez;
    QIterated<dim>  quadrature(q_trapez,degree+2);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution.size());

    // Pressure error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::L2_norm,
                                        &pressure_mask);
    const double 
    p_l2_error = VectorTools::compute_global_error(triangulation,
                                                  cellwise_errors,
                                                  VectorTools::L2_norm);

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::L2_norm,
                                        &pressure_mask);
    const double 
    p_l2_norm = VectorTools::compute_global_error(triangulation,
                                                  cellwise_norms,
                                                  VectorTools::L2_norm);
    double p_l2_error_total, p_l2_norm_total;
    MPI_Allreduce(&p_l2_error, //sending this data
                  &p_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&p_l2_norm, //sending this data
                  &p_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);

    // Velocity L2 error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::L2_norm,
                                        &velocity_mask);
    const double 
    u_l2_error = VectorTools::compute_global_error(triangulation,
                                                  cellwise_errors,
                                                  VectorTools::L2_norm);
                                            
    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::L2_norm,
                                        &velocity_mask);
    const double 
    u_l2_norm = VectorTools::compute_global_error(triangulation,
                                                  cellwise_norms,
                                                  VectorTools::L2_norm);
    double u_l2_error_total, u_l2_norm_total;
    MPI_Allreduce(&u_l2_error, //sending this data
                  &u_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&u_l2_norm, //sending this data
                  &u_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);                                              

    // Velocity H(div) error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::Hdiv_seminorm,
                                        &velocity_mask);
    const double 
    u_hdiv_error = VectorTools::compute_global_error(triangulation,
                                                    cellwise_errors,
                                                    VectorTools::L2_norm);
                                            
    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::Hdiv_seminorm,
                                        &velocity_mask);
    const double 
    u_hdiv_norm = VectorTools::compute_global_error(triangulation,
                                                  cellwise_norms,
                                                  VectorTools::L2_norm);  
    double u_hdiv_error_total, u_hdiv_norm_total;
    MPI_Allreduce(&u_hdiv_error, //sending this data
                  &u_hdiv_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&u_hdiv_norm, //sending this data
                  &u_hdiv_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);                                                                                          

    // Displacement L2 error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::L2_norm,
                                        &displacement_mask);
    const double 
    d_l2_error = VectorTools::compute_global_error(triangulation,
                                                  cellwise_errors,
                                                  VectorTools::L2_norm);

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::L2_norm,
                                        &displacement_mask);
    const double 
    d_l2_norm = VectorTools::compute_global_error(triangulation,
                                                  cellwise_norms,
                                                  VectorTools::L2_norm);
    double d_l2_error_total, d_l2_norm_total;
    MPI_Allreduce(&d_l2_error, //sending this data
                  &d_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&d_l2_norm, //sending this data
                  &d_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);

    // Rotation L2 error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::L2_norm,
                                        &rotation_mask);
    const double 
    r_l2_error = VectorTools::compute_global_error(triangulation,
                                                  cellwise_errors,
                                                  VectorTools::L2_norm);

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::L2_norm,
                                        &rotation_mask);
    const double 
    r_l2_norm = VectorTools::compute_global_error(triangulation,
                                                  cellwise_norms,
                                                  VectorTools::L2_norm);                                              
    double r_l2_error_total, r_l2_norm_total;
    MPI_Allreduce(&r_l2_error, //sending this data
                  &r_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&r_l2_norm, //sending this data
                  &r_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);

    // Stress L2 error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::L2_norm,
                                        &stress_mask);
    const double 
    s_l2_error = VectorTools::compute_global_error(triangulation,
                                                  cellwise_errors,
                                                  VectorTools::L2_norm);

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::L2_norm,
                                        &stress_mask);
    const double 
    s_l2_norm = VectorTools::compute_global_error(triangulation,
                                                  cellwise_norms,
                                                  VectorTools::L2_norm);   
    double s_l2_error_total, s_l2_norm_total;
    MPI_Allreduce(&s_l2_error, //sending this data
                  &s_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&s_l2_norm, //sending this data
                  &s_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);

    // Stress H(div) error and norm                                              
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                        cellwise_errors, quadrature,
                                        VectorTools::Hdiv_seminorm,
                                        &stress_mask);
    const double
    s_hdiv_error = VectorTools::compute_global_error(triangulation,
                                                  cellwise_errors,
                                                  VectorTools::L2_norm);
                                               
    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                        cellwise_norms, quadrature,
                                        VectorTools::Hdiv_seminorm,
                                        &stress_mask);                                              
    const double
    s_hdiv_norm = VectorTools::compute_global_error(triangulation,
                                                    cellwise_norms,
                                                    VectorTools::L2_norm);
    double s_hdiv_error_total, s_hdiv_norm_total;
    MPI_Allreduce(&s_hdiv_error, //sending this data
                  &s_hdiv_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot); //communicator
    MPI_Allreduce(&s_hdiv_norm, //sending this data
                  &s_hdiv_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_biot);

    const unsigned int n_active_cells = triangulation.n_active_cells();
    unsigned int n_dofs = dof_handler.n_dofs();
    if (n_processes > 1)
      n_dofs = interface_dofs_size;

    // find the max cell width h
    // first find local subdomain celll width h_local

    // first get an iterator to the min element
    auto it = std::min_element(initial_refinement.begin(), initial_refinement.end());
    unsigned int h_local = 0;

    if (it != initial_refinement.end()) // check if the vector is not empty
      h_local = (*it) * std::pow(2, cycle); // get the value of the minimum element, scale according to domain size
    else
      AssertThrow(false, ExcMessage("Error: initial_refinement vector is empty."));
    
    MPI_Allreduce(&h_local, //sending this data
                  &refinement, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_UNSIGNED, //type of each element
                  MPI_MIN, //min of all elements received
                  mpi_communicator); //communicator

    convergence_table_darcy_l2.add_value("cells", n_active_cells);
    convergence_table_darcy_l2.add_value("h", "1/"+Utilities::int_to_string(refinement));
    convergence_table_darcy_l2.add_value("dofs", n_dofs);
    convergence_table_darcy_l2.add_value("||u-u_h||_L2", u_l2_error/u_l2_norm);
    convergence_table_darcy_l2.add_value("||p-p_h||_L2", p_l2_error/p_l2_norm);
    convergence_table_darcy_l2.add_value("gmres_iter", gmres_iteration);
    convergence_table_darcy_l2.add_value("residual", residual);

    convergence_table_elast_l2.add_value("cells", n_active_cells);
    convergence_table_elast_l2.add_value("h", "1/"+Utilities::int_to_string(refinement));
    convergence_table_elast_l2.add_value("dofs", n_dofs);
    convergence_table_elast_l2.add_value("||d-d_h||_L2", d_l2_error/d_l2_norm);
    convergence_table_elast_l2.add_value("||r-r_h||_L2", r_l2_error/r_l2_norm);
    convergence_table_elast_l2.add_value("||s-s_h||_L2", s_l2_error/s_l2_norm);
    convergence_table_elast_l2.add_value("gmres_iter", gmres_iteration);
    // convergence_table_elast_l2.add_value("residual", residual);
    
    convergence_table_hdiv.add_value("cells", n_active_cells);
    convergence_table_hdiv.add_value("h", "1/"+Utilities::int_to_string(refinement));
    convergence_table_hdiv.add_value("dofs", n_dofs);
    convergence_table_hdiv.add_value("||u-u_h||_Hdiv", u_hdiv_error/u_hdiv_norm);
    convergence_table_hdiv.add_value("||s-s_h||_Hdiv", s_hdiv_error/s_hdiv_norm);
    convergence_table_hdiv.add_value("gmres_iter", gmres_iteration);
    // convergence_table_hdiv.add_value("residual", residual);

    // convergence_table_darcy_l2_total.add_value("cells", n_active_cells);
    convergence_table_darcy_l2_total.add_value("h", "1/"+Utilities::int_to_string(refinement));
    // convergence_table_darcy_l2_total.add_value("dofs", n_dofs);
    // convergence_table_darcy_l2_total.add_value("gmres_iter", gmres_iteration);
    convergence_table_darcy_l2_total.add_value("||p-p_h||_L2", p_l2_error_total/p_l2_norm_total);
    convergence_table_darcy_l2_total.add_value("||u-u_h||_L2", u_l2_error_total/u_l2_norm_total);
    convergence_table_darcy_l2_total.add_value("||u-u_h||_Hdiv", u_hdiv_error_total/u_hdiv_norm_total);
    convergence_table_darcy_l2_total.add_value("||s-s_h||_L2", s_l2_error_total/s_l2_norm_total);
    convergence_table_darcy_l2_total.add_value("||s-s_h||_Hdiv", s_hdiv_error_total/s_hdiv_norm_total);

    // convergence_table_darcy_l2_total.add_value("gmres_iter", gmres_iteration);
    // convergence_table_darcy_l2_total.add_value("residual", residual);

    // convergence_table_elast_l2_total.add_value("cells", n_active_cells);
    convergence_table_elast_l2_total.add_value("h", "1/"+Utilities::int_to_string(refinement));
    convergence_table_elast_l2_total.add_value("dofs", n_dofs);
    convergence_table_elast_l2_total.add_value("gmres_iter", gmres_iteration);
    convergence_table_elast_l2_total.add_value("residual", residual);
    convergence_table_elast_l2_total.add_value("||d-d_h||_L2", d_l2_error_total/d_l2_norm_total);
    convergence_table_elast_l2_total.add_value("||r-r_h||_L2", r_l2_error_total/r_l2_norm_total);
    // convergence_table_elast_l2_total.add_value("||s-s_h||_L2", s_l2_error_total/s_l2_norm_total);
    
    // convergence_table_hdiv_total.add_value("cells", n_active_cells);
    // convergence_table_hdiv_total.add_value("h", "1/"+Utilities::int_to_string(refinement));
    // convergence_table_hdiv_total.add_value("dofs", n_dofs);
    // convergence_table_hdiv_total.add_value("||u-u_h||_Hdiv", u_hdiv_error_total/u_hdiv_norm_total);
    // convergence_table_hdiv_total.add_value("||s-s_h||_Hdiv", s_hdiv_error_total/s_hdiv_norm_total);
    // convergence_table_hdiv_total.add_value("gmres_iter", gmres_iteration);
    // convergence_table_hdiv_total.add_value("residual", residual);
    // std::cout << "the residual is " << residual << "\n";
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_individual_convg_table_darcy_l2()
  {
    convergence_table_darcy_l2.set_precision("residual", 3);
    convergence_table_darcy_l2.set_precision("||u-u_h||_L2", 3);
    convergence_table_darcy_l2.set_precision("||p-p_h||_L2", 3);
    

    convergence_table_darcy_l2.set_scientific("residual", true);
    convergence_table_darcy_l2.set_scientific("||u-u_h||_L2", true);
    convergence_table_darcy_l2.set_scientific("||p-p_h||_L2", true);
    // convergence_table_l2.set_scientific("||d-d_h||_L2", true);
    // convergence_table_l2.set_scientific("||r-r_h||_L2", true);
    // convergence_table_l2.set_scientific("||s-s_h||_L2", true);

    // convergence_table_darcy_l2.set_tex_caption("cells", "\\ cells");
    // convergence_table_darcy_l2.set_tex_caption("dofs", "\\# dofs");
    // convergence_table_darcy_l2.set_tex_caption("gmres_iter", "iter");
    convergence_table_darcy_l2.set_tex_caption("||u-u_h||_L2", "$||u_p-u_{ph}||_{L^2}$");
    convergence_table_darcy_l2.set_tex_caption("||p-p_h||_L2", "$||p_p-p_{ph}||_{L^2}$");
    convergence_table_darcy_l2.set_tex_caption("residual", "residual");
    // convergence_table_l2.set_tex_caption("||d-d_h||_L2", "$||\\eta_p-\\eta_{ph}||_{L^2}$");
    // convergence_table_l2.set_tex_caption("||r-r_h||_L2", "$||\\gamma-\\gamma_{h}||_{L^2}$");
    // convergence_table_l2.set_tex_caption("||s-s_h||_L2", "$||\\sigma_p-\\sigma_{ph}||_{L^2}$");

    convergence_table_darcy_l2.evaluate_convergence_rates("||u-u_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);
    convergence_table_darcy_l2.evaluate_convergence_rates("||p-p_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);
    // convergence_table_l2.evaluate_convergence_rates("||d-d_h||_L2",
    //                                           ConvergenceTable::reduction_rate_log2);
    // convergence_table_l2.evaluate_convergence_rates("||r-r_h||_L2",
    //                                           ConvergenceTable::reduction_rate_log2);                                          
    // convergence_table_l2.evaluate_convergence_rates("||s-s_h||_L2",
    //                                           ConvergenceTable::reduction_rate_log2);

    std::string conv_filename = "../output/convg_tables/darcy_table_individual_l2" + std::string(".tex");

    if (this_mpi_biot == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes_biot; ++i)
    {
      MPI_Barrier(mpi_communicator_biot);
      if (this_mpi_biot == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table_darcy_l2.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table_darcy_l2.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    MPI_Barrier(mpi_communicator_biot);
    if (this_mpi_biot == 0)
    {
        std::ofstream table_file(conv_filename, std::ios::app);
        table_file << "\n\\end{document}\n";
    }
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_individual_convg_table_elast_l2()
  {
    // convergence_table_elast_l2.set_precision("residual", 3);
    convergence_table_elast_l2.set_precision("||d-d_h||_L2", 3);
    convergence_table_elast_l2.set_precision("||r-r_h||_L2", 3);
    convergence_table_elast_l2.set_precision("||s-s_h||_L2", 3);

    // convergence_table_elast_l2.set_scientific("residual", true);
    // convergence_table_l2.set_scientific("||u-u_h||_L2", true);
    // convergence_table_l2.set_scientific("||p-p_h||_L2", true);
    convergence_table_elast_l2.set_scientific("||d-d_h||_L2", true);
    convergence_table_elast_l2.set_scientific("||r-r_h||_L2", true);
    convergence_table_elast_l2.set_scientific("||s-s_h||_L2", true);

    convergence_table_elast_l2.set_tex_caption("cells", "\\ cells");
    convergence_table_elast_l2.set_tex_caption("dofs", "\\# dofs");
    // convergence_table_elast_l2.set_tex_caption("residual", "residual");
    // convergence_table_l2.set_tex_caption("||u-u_h||_L2", "$||u_p-u_{ph}||_{L^2}$");
    // convergence_table_l2.set_tex_caption("||p-p_h||_L2", "$||p_p-p_{ph}||_{L^2}$");
    convergence_table_elast_l2.set_tex_caption("||d-d_h||_L2", "$||\\eta_p-\\eta_{ph}||_{L^2}$");
    convergence_table_elast_l2.set_tex_caption("||r-r_h||_L2", "$||\\gamma-\\gamma_{h}||_{L^2}$");
    convergence_table_elast_l2.set_tex_caption("||s-s_h||_L2", "$||\\sigma_p-\\sigma_{ph}||_{L^2}$");
    convergence_table_elast_l2.set_tex_caption("gmres_iter", "iter");

    // convergence_table_l2.evaluate_convergence_rates("||u-u_h||_L2",
    //                                           ConvergenceTable::reduction_rate_log2);
    // convergence_table_l2.evaluate_convergence_rates("||p-p_h||_L2",
    //                                           ConvergenceTable::reduction_rate_log2);
    convergence_table_elast_l2.evaluate_convergence_rates("||d-d_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);
    convergence_table_elast_l2.evaluate_convergence_rates("||r-r_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);                                          
    convergence_table_elast_l2.evaluate_convergence_rates("||s-s_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);

    std::string conv_filename = "../output/convg_tables/elast_table_individual_l2" + std::string(".tex");

    if (this_mpi_biot == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes_biot; ++i)
    {
      MPI_Barrier(mpi_communicator_biot);
      if (this_mpi_biot == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table_elast_l2.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table_elast_l2.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    MPI_Barrier(mpi_communicator_biot);
    if (this_mpi_biot == 0)
    {
        std::ofstream table_file(conv_filename, std::ios::app);
        table_file << "\n\\end{document}\n";
    }
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_individual_convg_table_hdiv()
  {
    // convergence_table_hdiv.set_precision("residual", 3);
    convergence_table_hdiv.set_precision("||u-u_h||_Hdiv", 3);
    convergence_table_hdiv.set_precision("||s-s_h||_Hdiv", 3);

    // convergence_table_hdiv.set_scientific("residual", true);
    convergence_table_hdiv.set_scientific("||u-u_h||_Hdiv", true);
    convergence_table_hdiv.set_scientific("||s-s_h||_Hdiv", true);

    convergence_table_hdiv.set_tex_caption("cells", "\\ cells");
    convergence_table_hdiv.set_tex_caption("dofs", "\\# dofs");
    // convergence_table_hdiv.set_tex_caption("residual", "residual");
    convergence_table_hdiv.set_tex_caption("||u-u_h||_Hdiv", "$||u_p-u_{ph}||_{H(div)}$");
    convergence_table_hdiv.set_tex_caption("||s-s_h||_Hdiv", "$||\\sigma_p-\\sigma_{ph}||_{H(div)}$");
    convergence_table_hdiv.set_tex_caption("gmres_iter", "iter");

    convergence_table_hdiv.evaluate_convergence_rates("||u-u_h||_Hdiv",
                                              ConvergenceTable::reduction_rate_log2);
    convergence_table_hdiv.evaluate_convergence_rates("||s-s_h||_Hdiv",
                                              ConvergenceTable::reduction_rate_log2);

    
    std::string conv_filename = "../output/convg_tables/biot_table_individual_hdiv" + std::string(".tex");

    if (this_mpi_biot == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes_biot; ++i)
    {
      MPI_Barrier(mpi_communicator_biot);
      if (this_mpi_biot == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table_hdiv.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table_hdiv.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    MPI_Barrier(mpi_communicator_biot);
    if (this_mpi_biot == 0)
    {
        std::ofstream table_file(conv_filename, std::ios::app);
        table_file << "\n\\end{document}\n";
    }
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_total_convg_table_darcy_l2()
  {
    // only one process prints this data
    if (this_mpi_biot == 0)
    {
      // convergence_table_darcy_l2_total.set_precision("residual", 3);
      convergence_table_darcy_l2_total.set_precision("||p-p_h||_L2", 2);
      convergence_table_darcy_l2_total.set_precision("||u-u_h||_L2", 2);
      convergence_table_darcy_l2_total.set_precision("||u-u_h||_Hdiv", 2);
      convergence_table_darcy_l2_total.set_precision("||s-s_h||_L2", 2);
      convergence_table_darcy_l2_total.set_precision("||s-s_h||_Hdiv", 2);

      // convergence_table_darcy_l2_total.set_scientific("residual", true);
      convergence_table_darcy_l2_total.set_scientific("||p-p_h||_L2", true);
      convergence_table_darcy_l2_total.set_scientific("||u-u_h||_L2", true);
      convergence_table_darcy_l2_total.set_scientific("||u-u_h||_Hdiv", true);
      convergence_table_darcy_l2_total.set_scientific("||s-s_h||_L2", true);
      convergence_table_darcy_l2_total.set_scientific("||s-s_h||_Hdiv", true);
      // convergence_table_l2.set_scientific("||d-d_h||_L2", true);
      // convergence_table_l2.set_scientific("||r-r_h||_L2", true);
      // convergence_table_l2.set_scientific("||s-s_h||_L2", true);

      // convergence_table_darcy_l2_total.set_tex_caption("cells", "\\ cells");
      // convergence_table_darcy_l2_total.set_tex_caption("dofs", "dofs");
      // convergence_table_darcy_l2_total.set_tex_caption("gmres_iter", "iter");
      // convergence_table_darcy_l2_total.set_tex_caption("residual", "residual");
      convergence_table_darcy_l2_total.set_tex_caption("||p-p_h||_L2", "$||p_p-p_{ph}||_{\\Omega_p}$");
      convergence_table_darcy_l2_total.set_tex_caption("||u-u_h||_L2", "$||u_p-u_{ph}||_{\\Omega_p}$");
      convergence_table_darcy_l2_total.set_tex_caption("||u-u_h||_Hdiv", "$||u_p-u_{ph}||_{\\mathbf{H}(\\text{div}, \\Omega_p)}$");
      convergence_table_darcy_l2_total.set_tex_caption("||s-s_h||_L2", "$||\\sigma_p-\\sigma_{ph}||_{\\Omega_p}$");
      convergence_table_darcy_l2_total.set_tex_caption("||s-s_h||_Hdiv", "$||\\sigma_p-\\sigma_{ph}||_{\\mathbf{H}(\\text{div}, \\Omega_p)}$");
      // convergence_table_l2.set_tex_caption("||d-d_h||_L2", "$||\\eta_p-\\eta_{ph}||_{L^2}$");
      // convergence_table_l2.set_tex_caption("||r-r_h||_L2", "$||\\gamma-\\gamma_{h}||_{L^2}$");
      // convergence_table_l2.set_tex_caption("||s-s_h||_L2", "$||\\sigma_p-\\sigma_{ph}||_{L^2}$");

      // convergence_table_darcy_l2_total.evaluate_convergence_rates("gmres_iter",
      //                                           ConvergenceTable::reduction_rate_log2);
      convergence_table_darcy_l2_total.evaluate_convergence_rates("||p-p_h||_L2",
                                                ConvergenceTable::reduction_rate_log2);
      convergence_table_darcy_l2_total.evaluate_convergence_rates("||u-u_h||_L2",
                                                ConvergenceTable::reduction_rate_log2);
      convergence_table_darcy_l2_total.evaluate_convergence_rates("||u-u_h||_Hdiv",
                                                ConvergenceTable::reduction_rate_log2);
      convergence_table_darcy_l2_total.evaluate_convergence_rates("||s-s_h||_L2",
                                                ConvergenceTable::reduction_rate_log2);                                          
      convergence_table_darcy_l2_total.evaluate_convergence_rates("||s-s_h||_Hdiv",
                                                ConvergenceTable::reduction_rate_log2);                                          
      // convergence_table_l2.evaluate_convergence_rates("||d-d_h||_L2",
      //                                           ConvergenceTable::reduction_rate_log2);
      // convergence_table_l2.evaluate_convergence_rates("||r-r_h||_L2",
      //                                           ConvergenceTable::reduction_rate_log2);                                          
      // convergence_table_l2.evaluate_convergence_rates("||s-s_h||_L2",
      //                                           ConvergenceTable::reduction_rate_log2);

      std::string conv_filename = "../output/convg_table_total/darcy_table_total_l2" + std::string(".tex");
      
      std::cout << "total error\n" << std::endl;
      convergence_table_darcy_l2_total.write_text(std::cout);
      std::ofstream table_file(conv_filename, std::ios::out);
      table_file << "\\documentclass[10pt]{report}\n"
                 << "\\usepackage{float}\n"
                 << "\\usepackage[margin=0.7in]{geometry}\n\n"
                 << "\\begin{document}\n\n";
      
      convergence_table_darcy_l2_total.write_tex(table_file, false);
      table_file << "\n\\end{document}\n";
      table_file.close();
    }
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_total_convg_table_elast_l2()
  {
    if (this_mpi_biot == 0)
    {
      convergence_table_elast_l2_total.set_precision("residual", 3);
      convergence_table_elast_l2_total.set_precision("||d-d_h||_L2", 2);
      convergence_table_elast_l2_total.set_precision("||r-r_h||_L2", 2);
      // convergence_table_elast_l2_total.set_precision("||s-s_h||_L2", 2);

      convergence_table_elast_l2_total.set_scientific("residual", true);
      // convergence_table_l2.set_scientific("||u-u_h||_L2", true);
      // convergence_table_l2.set_scientific("||p-p_h||_L2", true);
      convergence_table_elast_l2_total.set_scientific("||d-d_h||_L2", true);
      convergence_table_elast_l2_total.set_scientific("||r-r_h||_L2", true);
      // convergence_table_elast_l2_total.set_scientific("||s-s_h||_L2", true);

      // convergence_table_elast_l2_total.set_tex_caption("cells", "\\ cells");
      convergence_table_elast_l2_total.set_tex_caption("dofs", "\\# dofs");
      convergence_table_elast_l2_total.set_tex_caption("residual", "residual");
      // convergence_table_l2.set_tex_caption("||u-u_h||_L2", "$||u_p-u_{ph}||_{L^2}$");
      // convergence_table_l2.set_tex_caption("||p-p_h||_L2", "$||p_p-p_{ph}||_{L^2}$");
      convergence_table_elast_l2_total.set_tex_caption("||d-d_h||_L2", "$||\\eta_p-\\eta_{ph}||_{\\Omega_p}$");
      convergence_table_elast_l2_total.set_tex_caption("||r-r_h||_L2", "$||\\gamma-\\gamma_{h}||_{\\Omega_p}$");
      // convergence_table_elast_l2_total.set_tex_caption("||s-s_h||_L2", "$||\\sigma_p-\\sigma_{ph}||_{\\Omega_p}$");
      convergence_table_elast_l2_total.set_tex_caption("gmres_iter", "iter");

      // convergence_table_l2.evaluate_convergence_rates("||u-u_h||_L2",
      //                                           ConvergenceTable::reduction_rate_log2);
      // convergence_table_l2.evaluate_convergence_rates("||p-p_h||_L2",
      //                                           ConvergenceTable::reduction_rate_log2);
      convergence_table_elast_l2_total.evaluate_convergence_rates("||d-d_h||_L2",
                                                ConvergenceTable::reduction_rate_log2);
      convergence_table_elast_l2_total.evaluate_convergence_rates("||r-r_h||_L2",
                                                ConvergenceTable::reduction_rate_log2);   
       convergence_table_elast_l2_total.evaluate_convergence_rates("gmres_iter",
                                                ConvergenceTable::reduction_rate_log2);                                                                                   
      // convergence_table_elast_l2_total.evaluate_convergence_rates("||s-s_h||_L2",
      //                                           ConvergenceTable::reduction_rate_log2);

      std::string conv_filename = "../output/convg_table_total/elast_table_total_l2" + std::string(".tex");

      std::cout << "total error\n" << std::endl;
      convergence_table_elast_l2_total.write_text(std::cout);
      std::ofstream table_file(conv_filename, std::ios::out);
      table_file << "\\documentclass[10pt]{report}\n"
                 << "\\usepackage{float}\n"
                 << "\\usepackage[margin=0.7in]{geometry}\n\n"
                 << "\\begin{document}\n\n";
      
      convergence_table_elast_l2_total.write_tex(table_file, false);
      table_file << "\n\\end{document}\n";
      table_file.close();
    }
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_total_convg_table_hdiv()
  {
    if (this_mpi_biot == 0)
    {
      // convergence_table_hdiv.set_precision("residual", 3);
      convergence_table_hdiv_total.set_precision("||u-u_h||_Hdiv", 3);
      convergence_table_hdiv_total.set_precision("||s-s_h||_Hdiv", 3);

      // convergence_table_hdiv.set_scientific("residual", true);
      convergence_table_hdiv_total.set_scientific("||u-u_h||_Hdiv", true);
      convergence_table_hdiv_total.set_scientific("||s-s_h||_Hdiv", true);

      // convergence_table_hdiv_total.set_tex_caption("cells", "\\ cells");
      // convergence_table_hdiv_total.set_tex_caption("dofs", "\\# dofs");
      // convergence_table_hdiv.set_tex_caption("residual", "residual");
      convergence_table_hdiv_total.set_tex_caption("||u-u_h||_Hdiv", "$||u_p-u_{ph}||_{H(div)}$");
      convergence_table_hdiv_total.set_tex_caption("||s-s_h||_Hdiv", "$||\\sigma_p-\\sigma_{ph}||_{H(div)}$");
      // convergence_table_hdiv_total.set_tex_caption("gmres_iter", "iter");

      convergence_table_hdiv_total.evaluate_convergence_rates("||u-u_h||_Hdiv",
                                                ConvergenceTable::reduction_rate_log2);
      convergence_table_hdiv_total.evaluate_convergence_rates("||s-s_h||_Hdiv",
                                                ConvergenceTable::reduction_rate_log2);                                        

      
      std::string conv_filename = "../output/convg_table_total/biot_table_total_hdiv" + std::string(".tex");
      
      std::cout << "total error\n" << std::endl;
      convergence_table_hdiv_total.write_text(std::cout);
      std::ofstream table_file(conv_filename, std::ios::out);
      table_file << "\\documentclass[10pt]{report}\n"
                 << "\\usepackage{float}\n"
                 << "\\usepackage[margin=0.7in]{geometry}\n\n"
                 << "\\begin{document}\n\n";
      
      convergence_table_hdiv_total.write_tex(table_file, false);
      table_file << "\n\\end{document}\n";
      table_file.close();
    }
  }

  // MixedBiotProblem::run
  template <int dim>
  void MixedBiotProblem<dim>::run (unsigned int cycle)
  {
    cycle_number = cycle;
    Assert(initial_refinement.size() == 
              dim, ExcDimensionMismatch(initial_refinement.size(), dim));
    Assert(initial_refinement_mortar.size() == 
              dim, ExcDimensionMismatch(initial_refinement_mortar.size(), dim));

    // if (mortar_flag)
    //   {
    //     Assert(n_processes > 1,
    //            ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
    //     Assert(reps.size() >= n_processes + 1,
    //            ExcMessage("Some of the mesh parameters were not provided"));
    //   }
    
    if (cycle == 0)
    {
      // Dimensions of the domain (unit hypercube)
      std::vector<double> subdomain_dimensions(dim);
      for (unsigned int d = 1; d < dim; ++d)
        subdomain_dimensions[d] = 2.0 / double(n_domains[d]);

      subdomain_dimensions[0] = 2.0 / double(n_domains[0]);
      
      get_subdomain_coordinates<dim>(
        this_mpi, n_domains, subdomain_dimensions, p1, p2);

      // if (mortar_flag)
      //   GridGenerator::subdivided_hyper_rectangle(triangulation,
      //                                             reps[this_mpi],
      //                                             p1,
      //                                             p2);
      // else
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                initial_refinement,
                                                p1,
                                                p2);

      // if (mortar_flag)
        GridGenerator::subdivided_hyper_rectangle(triangulation_mortar,
                                                  initial_refinement_mortar,
                                                  p1,
                                                  p2);
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
      triangulation_mortar.refine_global();

      
      // triangulation_mortar.refine_global();
    }

    // for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell; ++side)
    //   {
    //     std::cout << "Neighbor type on side " << side
    //           << " : " << neighbor_type[side] << std::endl;
    //   }
    
    setup_system();
      
    get_interface_dofs();

    assemble_system();

    solve_bar();
    if(n_processes == 1)
      solution = solution_bar;
  }
  template class MixedBiotProblem<2>;
  // template class MixedBiotProblem<3>;
}
