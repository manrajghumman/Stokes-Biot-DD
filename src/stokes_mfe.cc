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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_bdm.h>

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
#include "../inc/stokes_mfe.h"
#include "../inc/data_stokes.h"
#include "../inc/stokes_parameter_reader.h"
#include "../inc/utilities.h"


namespace mfe_stokes
{
  using namespace dealii;
  using namespace dd_utilities;

  // MixedStokesBiotDD class constructor
  template <int dim>
  MixedStokesProblem<dim>::MixedStokesProblem(
    ParameterHandler &prm,
    unsigned int &degree,
    unsigned int &degree_biot,
    std::vector<unsigned int> n_domains,
    const unsigned int &subdomain_type,
    std::vector<int> &neighbors,
    std::vector<unsigned int> &neighbor_type,
    const bool mortar_flag,
    const unsigned int mortar_degree,
    const bool cont_mortar_flag)
    : prm(prm)
    , degree(degree)
    // , qdegree(quad_degree)
    , n_domains(n_domains)
    , subdomain_type(subdomain_type)
    , stokes_neighbor(false)
    , biot_neighbor(false)
    // , external_bdry_ids(external_bdry_ids)
    , mortar_flag(mortar_flag)
    , mortar_degree(mortar_degree)
    , cont_mortar_flag((mortar_flag == 1 && mortar_degree == 0) ? false: cont_mortar_flag) // if mortar_degree is 0, use RT<dim> (0)
    , mpi_communicator(MPI_COMM_WORLD)
    // , tol(tol)
    // , reps(reps)
    , neighbors(neighbors)
    , neighbor_type(neighbor_type)
    , P_coarse2fine(false)
    , P_fine2coarse(false)
    , gmres_iteration(0)
    , fe(FE_Q<dim>(degree+1),//fe for velocity
         dim,
         FE_Q<dim>(degree),//fe for pressure
         1)
    , dof_handler(triangulation)
    , fe_biot_stress(FE_BDM<dim>(degree_biot), dim,
              FE_Nothing<dim>(), 1)
    , fe_darcy_vel(FE_BDM<dim>(degree_biot), 1,
               FE_Nothing<dim>(), 1)
    , dof_handler_biot_stress(triangulation)
    , dof_handler_darcy_vel(triangulation)
    // , dof_handler_mortar(triangulation_mortar)
    // ,fe_mortar([mortar_degree, cont_mortar_flag]() -> FESystem<dim> {
    //   if (mortar_degree > 0 && cont_mortar_flag)
    //     return FESystem<dim>(FE_Q<dim>(mortar_degree), dim, FE_Nothing<dim>(), 1);
    //   else if (mortar_degree == 0 || !cont_mortar_flag)
    //     return FESystem<dim>(FE_RaviartThomas<dim>(mortar_degree), 1, FE_Nothing<dim>(), 1);
    //   else
    //     return FESystem<dim>(FE_Nothing<dim>(), dim, FE_Nothing<dim>(), 1);
    // }()) // Conditional initialization using a lambda function
    , this_mpi (Utilities::MPI::this_mpi_process(mpi_communicator))
    , n_processes (Utilities::MPI::n_mpi_processes(mpi_communicator))
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    // , computing_timer(mpi_communicator,
    //                   std::cout,
    //                   TimerOutput::summary,
    //                   TimerOutput::wall_times)
  {
    interpret_parameters();
    StokesParameterReader param(prm_fxn);
    param.read_parameters("../test function/stokes.prm");
  }


  // MixedStokesProblem::make_grid_and_dofs
  template <int dim>
  void
  MixedStokesProblem<dim>::setup_system()
  {
    // TimerOutput::Scope t(computing_timer, "make_grid_and_dofs");
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

    pcout << "Stokes Part:\n boundary condition:";
    for (unsigned int i = 0; i < external_bdry_ids.size(); ++i)
    {
      if (external_bdry_ids[i] == 0)
        external_bdry_ids[i] = 0;
      else 
        external_bdry_ids[i] = 7;
      if (external_bdry_ids[i] == 7)
        pcout << " N";
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

    // dofs_per_component.size() = dim x dim + dim + dim
    unsigned int n_u = 0, n_p = 0;
    // Enable for dim = 2, 3 right now only works for dim = 2
    for (unsigned int i = 0; i < dim; ++i)
      {
        n_u += dofs_per_component[i];
      }
    n_p = dofs_per_component[dim];

    n_velocity_interface = n_u;

    /**
     * @brief Constructs a BlockDynamicSparsityPattern object with 2x2 blocks.
     *
     * This object is used to store the sparsity pattern of a block matrix,
     * where each block can have a different sparsity pattern. The constructor
     * initializes a 2x2 block structure.
     */
    BlockDynamicSparsityPattern dsp(2, 2);

    dsp.block(0, 0).reinit(n_u, n_u);
    dsp.block(0, 1).reinit(n_u, n_p);
    dsp.block(1, 0).reinit(n_p, n_u);
    dsp.block(1, 1).reinit(n_p, n_p);
    dsp.collect_sizes();

    DoFTools::make_sparsity_pattern(dof_handler, dsp);//->this takes too much memory
    // in moderately-sized 3d problems
      
      sparsity_pattern.copy_from(dsp);

    // Initialize system matrix
    system_matrix.reinit(sparsity_pattern);
    system_matrix_star.reinit(sparsity_pattern);

    // Reinit solution and RHS vectors
    solution_bar_stokes.reinit(2);
    solution_bar_stokes.block(0).reinit(n_u);
    solution_bar_stokes.block(1).reinit(n_p);
    solution_bar_stokes.collect_sizes();
    solution_bar_stokes = 0;

    // Reinit solution and RHS vectors
    solution_star_stokes.reinit(solution_bar_stokes);
    solution_star_stokes = 0;

    system_rhs_bar_stokes.reinit(solution_bar_stokes);
    system_rhs_bar_stokes = 0;

    system_rhs_star_stokes.reinit(solution_bar_stokes);
    system_rhs_star_stokes = 0;

    pcout << "Stokes part:\n N velocity dofs: " << n_u
    <<" | N pressure dofs: "<< n_p
    << std::endl;

  }

  // MixedStokesProblemDD - setup system for biot
  template <int dim>
  void
  MixedStokesProblem<dim>::setup_system_biot()
  {

    // write the dof count for stress in biot side!!
    dof_handler_biot_stress.distribute_dofs(fe_biot_stress);
    DoFRenumbering::component_wise(dof_handler_biot_stress);

    n_biot_stress=0;
    unsigned int n_biot_stress_extra = 0;
    std::vector<types::global_dof_index> dofs_per_component_stress =
            DoFTools::count_dofs_per_fe_component (dof_handler_biot_stress);

    for (unsigned int i = 0; i < dim; ++i)
      n_biot_stress += dofs_per_component_stress[i*dim];

    pcout << " N stress dofs: " << n_biot_stress << std::endl;
    
    // write the dof count for velocity in biot side!!
    dof_handler_darcy_vel.distribute_dofs(fe_darcy_vel);
    DoFRenumbering::component_wise(dof_handler_darcy_vel);

    n_darcy_vel = 0; 
    unsigned int n_darcy_vel_extra = 0;
    std::vector<types::global_dof_index> dofs_per_component_darcy_vel =
            DoFTools::count_dofs_per_fe_component (dof_handler_darcy_vel);
    
    n_darcy_vel = dofs_per_component_darcy_vel[0];
    n_darcy_vel_extra = dofs_per_component_darcy_vel[dim];

    pcout << " N darcy velocity dofs: " << n_darcy_vel << std::endl;

    // Reinit Biot stress vector 
    biot_stress_vec.reinit(2);
    biot_stress_vec.block(0).reinit(n_biot_stress);
    biot_stress_vec.block(1).reinit(n_biot_stress_extra);
    biot_stress_vec.collect_sizes();
    biot_stress_vec = 0;

    // Reinit Darcy velocity vector 
    darcy_velocity_vec.reinit(2);
    darcy_velocity_vec.block(0).reinit(n_darcy_vel);
    darcy_velocity_vec.block(1).reinit(n_darcy_vel_extra);
    darcy_velocity_vec.collect_sizes();
    darcy_velocity_vec = 0;
  }

  // MixedStokesProblemDD - initialize the interface data structure
  template <int dim>
  void
  MixedStokesProblem<dim>::get_interface_dofs()
  {
    // TimerOutput::Scope t(computing_timer, "Get interface DoFs");
    interface_dofs.resize(GeometryInfo<dim>::faces_per_cell,
                          std::vector<unsigned int>());
    std::vector<types::global_dof_index> local_face_dof_indices;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    unsigned int side = 0;

    local_face_dof_indices.resize(fe.dofs_per_face);

    for (const auto &cell : dof_handler.active_cell_iterators())
      for (unsigned int face_n = 0; face_n < n_faces_per_cell; ++face_n)
        if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0 && 
                cell->face(face_n)->boundary_id() !=7))
        {
          cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
          side = cell->face(face_n)->boundary_id() - 1;
          if (neighbor_type[side] == subdomain_type) // Stokes-Stokes interface
            for (auto el : local_face_dof_indices)
              if (el < n_velocity_interface){
                // adding a check to not include the dofs at corners twice
                if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                                                  == interface_dofs[side].end()){
                                                                      interface_dofs[side].push_back(el);
                                                                      interface_dofs_total.push_back(el);                                                    
                }
              }
         }
    
    if (biot_neighbor) // Stokes-Biot interface
    {
      // Get interface dofs for stress dofs in biot side
      std::vector<types::global_dof_index> local_face_dof_indices_biot_stress;
      local_face_dof_indices_biot_stress.resize(fe_biot_stress.dofs_per_face);
      typename DoFHandler<dim>::active_cell_iterator cell_stress, endc_stress;
      cell_stress = dof_handler_biot_stress.begin_active(), endc_stress = dof_handler_biot_stress.end();
      side = 0;
      for (; cell_stress != endc_stress; ++cell_stress)
      {
        for (unsigned int face_n = 0; face_n < n_faces_per_cell;
              ++face_n)
        {
          if (cell_stress->at_boundary(face_n) &&
                (cell_stress->face(face_n)->boundary_id() != 0 && 
                  cell_stress->face(face_n)->boundary_id() !=7))
          {
            cell_stress->face(face_n)->get_dof_indices(local_face_dof_indices_biot_stress, 0);
            side = cell_stress->face(face_n)->boundary_id() - 1;
            if (neighbor_type[side] == 2) // Stokes-Biot interface
              for (auto el : local_face_dof_indices_biot_stress)
                if (el < n_biot_stress){
                  // there are no corner dofs in BDM, so no need to check for duplicates
                  interface_dofs[side].push_back(el);
                  interface_dofs_total.push_back(el);                                                    
                }
          }
        }
      }

      // Get interface dofs for darcy vel dofs in biot side
      std::vector<types::global_dof_index> local_face_dof_indices_darcy_vel;
      local_face_dof_indices_darcy_vel.resize(fe_darcy_vel.dofs_per_face);
      typename DoFHandler<dim>::active_cell_iterator cell_vel, endc_vel;
      cell_vel = dof_handler_darcy_vel.begin_active(), endc_vel = dof_handler_darcy_vel.end();
      side = 0;
      for (; cell_vel != endc_vel; ++cell_vel)
        for (unsigned int face_n = 0; face_n < n_faces_per_cell;
              ++face_n)
          if (cell_vel->at_boundary(face_n) &&
                (cell_vel->face(face_n)->boundary_id() != 0 && 
                  cell_vel->face(face_n)->boundary_id() !=7))
          {
            cell_vel->face(face_n)->get_dof_indices(local_face_dof_indices_darcy_vel, 0);
            side = cell_vel->face(face_n)->boundary_id() - 1;
            if (neighbor_type[side] == 2) // Stokes-Biot interface
              for (auto el : local_face_dof_indices_darcy_vel)
                if (el < n_darcy_vel){
                  // there are no corner dofs in BDM, so no need to check for duplicates
                  interface_dofs[side].push_back(el+n_biot_stress); // offset by n_biot_stress
                  interface_dofs_total.push_back(el+n_biot_stress);                                                    
                }
          }
    }
    for (unsigned int i = 0; i < n_faces_per_cell; ++i)
    {
      if (neighbors[i] >= 0)
      {
        std::cout << "Stokes Part:\n Number of interface dofs on face " << i
              << " : " << interface_dofs[i].size() << std::endl;
      }
    }

  }


  // MixedStokesProblemDD - assemble_system
  template <int dim>
  void
  MixedStokesProblem<dim>::assemble_system()
  {
    // TimerOutput::Scope t(computing_timer, "Assemble system");
    system_matrix         = 0;
    system_rhs_bar_stokes = 0;
    const double gamma    = 100;

    QGauss<dim> quadrature_formula(degree+2);
    QGauss<dim - 1> face_quadrature_formula(degree+2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors | update_gradients |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();


    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_face_dof_indices;
    local_face_dof_indices.resize(fe.dofs_per_face);

    const RightHandSide<dim>    right_hand_side;
    const RightHandSideG<dim>    right_hand_side_g;
    std::vector<Tensor<1, dim>> rhs_f_values(n_q_points, Tensor<1, dim>());
    StressTensor_Exact<dim> stress_tensor;
    BoundaryValues<dim> boundary_values;

    std::vector<double> rhs_g_values(n_q_points);
    std::vector<Tensor<2,dim>> stress_tensor_values(n_face_q_points, Tensor<2,dim>());

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    

    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>>          grad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      local_matrix                = 0;
      local_rhs                   = 0;
      double h_K                  = cell->diameter(); // std::pow(2, 0.5);      

      right_hand_side.value_list(fe_values.get_quadrature_points(),
                                  rhs_f_values);
      right_hand_side_g.value_list(fe_values.get_quadrature_points(),
                                  rhs_g_values);                                       

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          { //compute at one place decrease computations
            symgrad_phi_u[k] =
              fe_values[velocities].symmetric_gradient(k, q);
            div_phi_u[k] = fe_values[velocities].divergence(k, q);
            phi_u[k]     = fe_values[velocities].value(k, q);
            phi_p[k]     = fe_values[pressure].value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) // for test functions
          {
            for (unsigned int j = 0; j <= i; ++j)
              {
                local_matrix(i, j) +=
                  (2 * (symgrad_phi_u[j] * symgrad_phi_u[i]) // 2* (D(u):D(v))
                  - phi_p[j] * div_phi_u[i]                 // - (p, div(v))
                  - div_phi_u[j] * phi_p[i])                // - (div(u), w)
                  * fe_values.JxW(q);                        // dx
              }
            local_rhs(i) += phi_u[i]            
                            * rhs_f_values[q]     // (F, v)
                            * fe_values.JxW(q)
                            -phi_p[i]            
                            * rhs_g_values[q]     // -(G, w)
                            * fe_values.JxW(q); // dx
          }
      }
      
      //Entering boundary integrals
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary())
        { 
          fe_face_values.reinit(cell, face);
          if (face->boundary_id() == 7) //Entering Neumann Condition
          {
            stress_tensor.value_list(fe_face_values.get_quadrature_points(), 
                                stress_tensor_values);  
            for (unsigned int q_point = 0; q_point < n_face_q_points;
                  ++q_point)
              {
                for (int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_u[k]     = fe_face_values[velocities].value(k, q_point);
                }
                const Tensor<1,dim> neumann_value =
                  (stress_tensor_values[q_point] *
                    fe_face_values.normal_vector(q_point));

                for (int i = 0; i < dofs_per_cell; ++i)
                  local_rhs(i) +=
                    (phi_u[i] *                      
                      neumann_value) *                // + (gn, v)
                      fe_face_values.JxW(q_point);    // dx
              }
          }//Ending Neumann Condition
          if (face->boundary_id() == 0)//Entering Dirichlet Condition
          {
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              Tensor<1,dim> gd; 
              Vector<double> tmp(dim);
              boundary_values.vector_value(fe_face_values.get_quadrature_points()[q], tmp);
              for (unsigned int i = 0; i < dim; ++i)
                  gd[i] = tmp[i];
              for (int k = 0; k < dofs_per_cell; ++k)
              {
                symgrad_phi_u[k] =
                  fe_face_values[velocities].symmetric_gradient(k, q);
                phi_u[k]     = fe_face_values[velocities].value(k, q);
                grad_phi_u[k]     = fe_face_values[velocities].gradient(k, q);
                phi_p[k]     = fe_face_values[pressure].value(k, q);
              }
              for (int i = 0; i < dofs_per_cell; ++i)
              {
                for (int j = 0; j <= i; ++j)
                {
                  local_matrix(i,j) += (-2 * (symgrad_phi_u[j] 
                                      * fe_face_values.normal_vector(q))
                                      * phi_u[i] // -2 <D(u).n, v>
                                      -2 * phi_u[j]
                                      * (symgrad_phi_u[i] 
                                        * fe_face_values.normal_vector(q)) // -2 <u, D(v).n>
                                        + gamma * phi_u[j]
                                        * phi_u[i] / h_K // + gamma <u,v> / h_k
                                        + phi_p[j]
                                        * (fe_face_values.normal_vector(q)
                                        * phi_u[i]) // + <p, v.n>
                                        + phi_u[j] 
                                        * fe_face_values.normal_vector(q)
                                        * phi_p[i]) // - <u.n, w>
                                        * fe_face_values.JxW(q);
                }
                local_rhs(i) += (-2 * gd * (symgrad_phi_u[i] 
                              * fe_face_values.normal_vector(q)) // -2 <gd, D(v).n>
                              + gamma * gd * phi_u [i] / h_K // gamma * <gd, v>/h
                              + gd * fe_face_values.normal_vector(q)
                              * phi_p[i]) // - <gd.n, w>
                              * fe_face_values.JxW(q);
              }
            }
          }//Ending Dirichlet Condition
        }
      }


      //Matrix is symmetric copying from other half
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            local_matrix(i, j) = local_matrix(j, i);

      // Add local contributions to global system
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              local_matrix(i, j));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_bar_stokes(local_dof_indices[i]) += local_rhs(i);
    }
    
  }


  // MixedStokesProblem - assemble RHS of star problems
  template <int dim>
  void
  MixedStokesProblem<dim>::assemble_rhs_star(
    FEFaceValues<dim> &fe_face_values)
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS star stokes");
  }


  // MixedStokesProblemDD::solvers
  template <int dim>
  void MixedStokesProblem<dim>::solve_bar()
  {
      // TimerOutput::Scope t(computing_timer, "Solve bar stokes");
    
  }


  template <int dim>
  void MixedStokesProblem<dim>::solve_star()
  {
    // TimerOutput::Scope t(computing_timer, "Solve star");
    
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
  MixedStokesProblem<dim>::run(unsigned int cycle)
  {
    // tolerance = tol;
    // qdegree   = quad_degree;

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
    if(biot_neighbor)
      setup_system_biot();

    // std::cout << "Stokes Part: \nBiot neighbor: " << biot_neighbor << std::endl;
      
    get_interface_dofs();

    assemble_system();
    
    
  }

  template class MixedStokesProblem<2>;
  // template class MixedStokesProblem<3>;
} // namespace mfe_stokes
