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
// #include <deal.II/numerics/fe_field_function.h>
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
    ParameterHandler           &prm,
    unsigned int               &degree,
    unsigned int               &degree_biot,
    std::vector<unsigned int>   n_domains,
    const unsigned int         &subdomain_type,
    std::vector<int>           &neighbors,
    std::vector<unsigned int>  &neighbor_type,
    MPI_Comm                   mpi_communicator_stokes,
    const unsigned int         mortar_degree,
    const bool                 cont_mortar_flag)
    : mpi_communicator(MPI_COMM_WORLD)
    , mpi_communicator_stokes(mpi_communicator_stokes)
    , this_mpi (Utilities::MPI::this_mpi_process(mpi_communicator))
    , n_processes (Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_stokes (Utilities::MPI::this_mpi_process(mpi_communicator_stokes))
    , n_processes_stokes (Utilities::MPI::n_mpi_processes(mpi_communicator_stokes))
    , prm(prm)
    , degree(degree)
    , subdomain_type(subdomain_type)
    , n_domains(n_domains)
    , stokes_neighbor(false)
    , biot_neighbor(false)
    , mortar_degree(mortar_degree)
    , cont_mortar_flag(cont_mortar_flag) // if mortar_degree is 0, use RT<dim> (0)
    // , tol(tol)
    // , reps(reps)
    , neighbors(neighbors)
    , neighbor_type(neighbor_type)
    , P_stokes2biot(false)
    , P_biot2stokes(false)
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
    , dof_handler_biot_stress(triangulation_mortar)
    , dof_handler_darcy_vel(triangulation_mortar)
    , dof_handler_mortar(triangulation_mortar)
    ,fe_mortar([mortar_degree, cont_mortar_flag]() -> FESystem<dim> {
      if (cont_mortar_flag)
        return FESystem<dim>(FE_Q<dim>(mortar_degree), dim, FE_Nothing<dim>(), 1);
      else if (!cont_mortar_flag)
        return FESystem<dim>(FE_RaviartThomas<dim>(mortar_degree), 1, FE_Nothing<dim>(), 1);
      else
      {
        AssertThrow(false, ExcMessage("Invalid mortar element configuration."));
        return FESystem<dim>(FE_Nothing<dim>(), dim, FE_Nothing<dim>(), 1);
      }
    }()) // Conditional initialization using a lambda function
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator_stokes) == 0))
    // , computing_timer(mpi_communicator,
    //                   std::cout,
    //                   TimerOutput::summary,
    //                   TimerOutput::wall_times)
  {
    interpret_parameters();
    StokesParameterReader param(prm_fxn);
    param.read_parameters("../test function/matrix assembly/stokes.prm");
    pcout << "cont_mortar_flag: " << cont_mortar_flag
          << ", mortar_degree: " << mortar_degree 
          << ", degree: " << degree << std::endl;
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
    faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell, 0);

    // Label interface faces and count how many of them there are per interface
    mark_interface_faces<dim>(triangulation, neighbors, p1, p2, faces_on_interface);
    // if (mortar_flag)
    mark_interface_faces(
      triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);
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
        if (face->center()[dim - 1] == 1)
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
    solution_bar.reinit(2);
    solution_bar.block(0).reinit(n_u);
    solution_bar.block(1).reinit(n_p);
    solution_bar.collect_sizes();
    solution_bar = 0;

    // Reinit solution and RHS vectors
    solution_star.reinit(solution_bar);
    solution_star = 0;

    system_rhs_bar.reinit(solution_bar);
    system_rhs_bar = 0;

    system_rhs_star.reinit(solution_bar);
    system_rhs_star = 0;

    pcout << "Stokes part:\n N velocity dofs: " << n_u
          <<" | N pressure dofs: "<< n_p
          << std::endl;

    dof_handler_mortar.distribute_dofs(fe_mortar);
    DoFRenumbering::component_wise(dof_handler_mortar);

    std::vector<types::global_dof_index> dofs_per_component_mortar =
            DoFTools::count_dofs_per_fe_component (dof_handler_mortar);
    unsigned int n_u_mortar = 0, n_p_mortar = 0;
    // Enable for dim = 2, 3 right now only works for dim = 2
    if (cont_mortar_flag && mortar_degree > 0)
      for (unsigned int i = 0; i < dim; ++i)
        n_u_mortar += dofs_per_component_mortar[i];
    else if (!cont_mortar_flag) // the case of RT elements
      n_u_mortar = dofs_per_component_mortar[0];

    n_p_mortar = dofs_per_component_mortar[dim];

    n_velocity_interface = n_u_mortar;
    pcout << "Stokes part:\n N velocity mortar dofs: " << n_u_mortar
          <<" | N pressure mortar dofs: "<< n_p_mortar
          << std::endl;

    mortar_vec.reinit(2);
    mortar_vec.block(0).reinit(n_u_mortar);
    mortar_vec.block(1).reinit(n_p_mortar);
    mortar_vec.collect_sizes();
    mortar_vec = 0;

    // solution_bar_mortar.reinit(mortar_vec);
    // solution_star_mortar.reinit(mortar_vec);
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
    interface_dofs.clear();
    interface_dofs_total.clear();
    interface_dofs_size = 0;
    interface_dofs.resize(GeometryInfo<dim>::faces_per_cell,
                          std::vector<unsigned int>());
    std::vector<types::global_dof_index> local_face_dof_indices;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    unsigned int side = 0;

    local_face_dof_indices.resize(fe_mortar.dofs_per_face);

    for (const auto &cell : dof_handler_mortar.active_cell_iterators())
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
                                                                  == interface_dofs[side].end())
                {
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
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      if (neighbors[side] >= 0 && this_mpi == 0)
      {
        std::cout << "Stokes Part:\n Number of interface dofs on face " << side
              << " : " << interface_dofs[side].size() << ", interface dofs: ";
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          std::cout << interface_dofs[side][i] << " ";
        std::cout << std::endl;
      }
    }
    compute_interface_dofs_size<dim>(interface_dofs_total, 
                                     mpi_communicator, 
                                     this_mpi, 
                                     interface_dofs_size);
    // if (this_mpi == 0)
    // {
    //   std::cout << "interface_dofs: " << std::endl;
    //   for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    //   {
    //     std::cout << "side " << side << ": ";
    //     for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
    //       std::cout << interface_dofs[side][i] << " ";
    //     std::cout << std::endl;
    //   }
    // }
  }


  // MixedStokesProblemDD - assemble_system
  template <int dim>
  void
  MixedStokesProblem<dim>::assemble_system()
  {
    // TimerOutput::Scope t(computing_timer, "Assemble system");
    system_matrix         = 0;
    system_rhs_bar        = 0;

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

    //for Dirichlet BC from velocity u = u_0
    Functions::ParsedFunction<dim> boundary_values_dir(dim);
    prm_fxn.enter_subsection("BC Dirichlet");
    boundary_values_dir.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    //for F = -div sigma
    Functions::ParsedFunction<dim> rhs_vec(dim);
    prm_fxn.enter_subsection("RHS vector");
    rhs_vec.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    Tensor<1, dim> rhs_vec_val;

    //for G = -div u
    Functions::ParsedFunction<dim> rhs_scalar(1);
    prm_fxn.enter_subsection("RHS scalar");
    rhs_scalar.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    //for Neumann BC from stress tensor sigma 
    StressTensor_Exact<dim> stress_tensor(prm_fxn);
    // Functions::ParsedFunction<dim> stress_tensor_fxn(dim*dim);
    prm_fxn.enter_subsection("BC Neumann");
    stress_tensor.stress_function.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    
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

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dim; ++i)
          rhs_vec_val[i] = rhs_vec.value(fe_values.quadrature_point(q), i);

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
                            * rhs_vec_val     // (f_f, v)
                            * fe_values.JxW(q)
                            -phi_p[i]            
                            * rhs_scalar.value(fe_values.quadrature_point(q))     // -(q, w)
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
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_u[k]     = fe_face_values[velocities].value(k, q_point);
                }
                const Tensor<1,dim> neumann_value =
                  (stress_tensor_values[q_point] *
                    fe_face_values.normal_vector(q_point));

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
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
              // Vector<double> tmp(dim);
              // boundary_values_dir.vector_value(fe_face_values.get_quadrature_points()[q], tmp);
              for (unsigned int i = 0; i < dim; ++i)
                gd[i] = boundary_values_dir.value(fe_face_values.get_quadrature_points()[q], i);
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                symgrad_phi_u[k] =
                  fe_face_values[velocities].symmetric_gradient(k, q);
                phi_u[k]     = fe_face_values[velocities].value(k, q);
                grad_phi_u[k]     = fe_face_values[velocities].gradient(k, q);
                phi_p[k]     = fe_face_values[pressure].value(k, q);
              }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j <= i; ++j)
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
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            local_matrix(i, j) = local_matrix(j, i);
      
      // Add local contributions to global system
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        system_rhs_bar(local_dof_indices[i]) += local_rhs(i);
        
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            local_matrix(i, j));
      }
    }
  }


  // MixedStokesProblem - assemble RHS of star problems
  template <int dim>
  void
  MixedStokesProblem<dim>::assemble_rhs_star(
    std::vector<std::unique_ptr<FEFunction>> &interface_function,
    FEFaceValues<dim> &fe_face_values)
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS star stokes");
    system_rhs_star = 0;

    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;

    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    // std::vector<Tensor<1, dim>>          interface_values(n_face_q_points, Tensor<1, dim>());
    std::vector<std::vector<double>>     interface_values(dim, std::vector<double>(n_face_q_points, 0.0));//first index for dim components, 0, 1: vel, 2: pressure
    std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);                                            //second for value at each quad point
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    int side;
    // if (this_mpi == 1)
    //   std::cout << "Assembling RHS star for this_mpi = " << this_mpi << std::endl;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      local_rhs                   = 0;
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
      {
        if (cell->at_boundary(face_n) &&
          (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7)  )
          {
            //Need to reorder faces deal.ii iterators 
            //use a different ordering from freefem ... only works for 2D
            // Assert(dim == 2,
            //         ExcMessage("This function is only implemented for dim = 2."));
            fe_face_values.reinit(cell, face_n);
            // if (face_n == 0)
            //   side = 3;
            // else if (face_n == 1)
            //   side = 1;
            // else if (face_n == 2)
            //   side = 0;
            // else if (face_n == 3)
            //   side = 2;
            // else if (face_n == 4)
            //   side = 4; // only for 3D
            // else if (face_n == 5)
            //   side = 5; // only for 3D
            side = cell->face(face_n)->boundary_id() - 1;
            // std::cout << "interface_fe_function[side].block(0).size(): " << interface_fe_function[side].size() << " this_mpi, side, face_n = " << this_mpi << ", " << side << ", " << face_n << std::endl;
            // std::cout << "interface_fe_function[side].block(0).size(): " << interface_fe_function[side].size() << " this_mpi, side = " << this_mpi << ", " << side << std::endl;
            // std::cout << "\ninterface_values = " << interface_values.size() << std::endl;
            // fe_face_values[velocities].get_function_values(
            //   interface_fe_function[side], interface_values);
            // interface_function.vector_value_list(fe_face_values.get_quadrature_points(),
            //                              interface_values);
            for (unsigned int s = 0; s < dim; ++s)
              interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                    interface_values[s], s);
            for (unsigned int q_point = 0; q_point < n_face_q_points;
                ++q_point)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                phi_u[k]  = fe_face_values[velocities].value(k, q_point); 
              }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int s = 0; s < dim; ++s)
                  local_rhs(i) +=
                    - (phi_u[i][s] * get_normal_direction<dim>(cell->face(face_n)->boundary_id()-1)*  // phi_v_i(x_q)
                      interface_values[s][q_point]) *    // Tn_i = lambda
                      fe_face_values.JxW(q_point);    // dx
              }
            }
          }  
      }

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_star(local_dof_indices[i]) += local_rhs(i);        
    }
  }


  // MixedStokesProblemDD::solvers
  template <int dim>
  void MixedStokesProblem<dim>::solve_bar()
  {
      // TimerOutput::Scope t(computing_timer, "Solve bar stokes");
    A_direct.initialize (system_matrix);
    A_direct.vmult(solution_bar, system_rhs_bar);
  }


  template <int dim>
  void MixedStokesProblem<dim>::solve_star()
  {
    // TimerOutput::Scope t(computing_timer, "Solve star");
    A_direct.vmult (solution_star, system_rhs_star);
  }

  //Functions for GMRES:-------------------

  //finding the l2 norm of a std::vector<double> vector
  template <int dim>
  double
  MixedStokesProblem<dim>::vect_norm(std::vector<double> v)
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
  MixedStokesProblem<dim>::givens_rotation(double v1, double v2, double &cs, double &sn)
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
  MixedStokesProblem<dim>::apply_givens_rotation(std::vector<double> &h, 
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
  MixedStokesProblem<dim>::back_solve(const std::vector<std::vector<double>> &H, 
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
  		for(int j = i+1; j<=k-1; j++)
      {
  			y[i]-= H[j][i]*y[j]/H[i][i];
  		}
  	}
  }

  /*
     Interface operator function
     takes in lambda and returns A*lambda
     currently only for Stokes-Stokes interface
  */
  template <int dim>
  void
  MixedStokesProblem<dim>::interface_operator(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                              std::unique_ptr<FEFunction>       &solution_bar_function,
                                              std::unique_ptr<FEFunction>       &solution_star_function,
                                              std::vector<std::vector<double>> &interface_data,
                                              std::vector<std::vector<double>> &interface_data_send,
                                              std::vector<std::vector<double>> &interface_data_receive,
                                              FEFaceValues<dim>                &fe_face_values,
                                              FEFaceValues<dim>                &fe_face_values_mortar,
                                              std::vector<std::vector<double>> &Ap)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = 
          interface_data[side][i];

    // interface_fe_function.block(1) = 0;
    assemble_rhs_star(interface_function,
                      fe_face_values);
    solve_star();

    interface_inner_product(solution_bar_function,
                            solution_star_function,
                            fe_face_values_mortar);
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        // Create vector of u\dot n to send
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          interface_data_send[side][i] = solution_star_mortar[side][interface_dofs[side][i]];

        MPI_Sendrecv(interface_data_send[side].data(),
                      static_cast<int>(interface_dofs[side].size()),
                      MPI_DOUBLE,
                      neighbors[side],
                      this_mpi,  // send tag

                      interface_data_receive[side].data(),
                      static_cast<int>(interface_dofs[side].size()),
                      MPI_DOUBLE,
                      neighbors[side],
                      neighbors[side],  // recv tag

                      mpi_communicator,
                      &mpi_status);

        // Compute Ap and with it compute alpha
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          Ap[side][i] = - (interface_data_send[side][i] +
                          interface_data_receive[side][i]);
        }
      }
  }

  // computing the residual over all sides
  template <int dim>
  void
  MixedStokesProblem<dim>::compute_residual(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                            std::unique_ptr<FEFunction>       &solution_bar_function,
                                            std::unique_ptr<FEFunction>       &solution_star_function,
                                            FEFaceValues<dim>                &fe_face_values,
                                            FEFaceValues<dim>                &fe_face_values_mortar,
                                            std::vector<std::vector<double>> &interface_data,
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

    // interface_fe_function.block(1) = 0;
    assemble_rhs_star(interface_function,
                      fe_face_values);
    solve_star();

    //for computing rhs b for
    // Ax = b in interface problem below
    solve_bar();

    //for computing the inner products
    //in the mortar space
    interface_inner_product(solution_bar_function,
                            solution_star_function,
                            fe_face_values_mortar);
    //compute residual r = b - A*x
    for (unsigned side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        std::vector<double> r_b_receive_buffer(r[side].size());
        std::vector<double> r_receive_buffer(r[side].size());

        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          r_b[side][i] = solution_bar_mortar[side][interface_dofs[side][i]];
          r[side][i] =  solution_bar_mortar[side][interface_dofs[side][i]]+
                        solution_star_mortar[side][interface_dofs[side][i]];
        }

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
        r_norm_side[side] = vect_norm(r[side]);
        r_b_norm_side[side] = vect_norm(r_b[side]);
      }

    //Calculating r-norm(same as b-norm)-----
    // double r_norm =0;
    for(unsigned int side=0; side<n_faces_per_cell;++side)
      if (neighbors[side] >= 0)
      {
        r_norm+=r_norm_side[side]*r_norm_side[side];
        b_norm+=r_b_norm_side[side]*r_b_norm_side[side];
      }
    double r_norm_buffer =0;
    double b_norm_buffer =0;
    //Using MPI to get the global r-norm and b-norm
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
    //end -----------of calculating r-norm and b_norm------------------
  }

  // compute inner product in mortar space
  template <int dim>
  void
  MixedStokesProblem<dim>::interface_inner_product(std::unique_ptr<FEFunction>      &solution_bar_function,
                                                  std::unique_ptr<FEFunction>       &solution_star_function,
                                                  FEFaceValues<dim>                 &fe_face_values_mortar)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        solution_bar_mortar[side] = 0;
        solution_star_mortar[side] = 0;
      }
    
    const unsigned int n_face_q_points = fe_face_values_mortar.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe_mortar.dofs_per_cell;

    Vector<double>                       local_rhs_bar(dofs_per_cell);
    Vector<double>                       local_rhs_star(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    std::vector<std::vector<unsigned int>> interface_dofs_count(GeometryInfo<dim>::faces_per_cell);

    //first index for dim components, 0, 1: vel, 2: pressure
    //second for value at each quad point
    std::vector<std::vector<double>> bar_values(dim, std::vector<double>(n_face_q_points, 0.0));
    std::vector<std::vector<double>> star_values(dim, std::vector<double>(n_face_q_points, 0.0));
    std::vector<Tensor<1, dim>>      phi_mu(dofs_per_cell);

    unsigned int side = 0;

    for (const auto &cell : dof_handler_mortar.active_cell_iterators())
    {
      local_rhs_bar  = 0;
      local_rhs_star = 0;
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
      {
        if (cell->at_boundary(face_n) &&
          (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7)  )
        {
          //Need to reorder faces deal.ii iterators 
          //use a different ordering from freefem, 
          //they are ordered component wise 
          fe_face_values_mortar.reinit(cell, face_n);
          side = cell->face(face_n)->boundary_id() - 1;
          for (unsigned int s = 0; s < dim; ++s)
          {
            solution_bar_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                      bar_values[s], s);
            solution_star_function->value_list(fe_face_values_mortar.get_quadrature_points(),
                                      star_values[s], s);
          }

          for (unsigned int q_point = 0; q_point < n_face_q_points;
              ++q_point)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                             local_dof_indices[k]) != interface_dofs[side].end())
              {             
                // if (std::find (interface_dofs_count[side].begin(), interface_dofs_count[side].end(),
                //               local_dof_indices[k]) == interface_dofs_count[side].end()) 
                // {            
                phi_mu[k]  = fe_face_values_mortar[velocities].value(k, q_point); 
                // interface_dofs_count[side].push_back(local_dof_indices[k]);
                // }
              }
              else
                phi_mu[k] = 0;
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                             local_dof_indices[i]) != interface_dofs[side].end())
              {   
                for (unsigned int s = 0; s < dim; ++s)
                {
                  solution_bar_mortar[side](local_dof_indices[i]) +=
                    (phi_mu[i][s] * get_normal_direction<dim>(side) *  // mu(x_q)
                      bar_values[s][q_point]) *    // u_bar
                      fe_face_values_mortar.JxW(q_point);    // dx

                  solution_star_mortar[side](local_dof_indices[i]) +=
                    (phi_mu[i][s] * get_normal_direction<dim>(side) *  // mu(x_q)
                      star_values[s][q_point]) *    // u_star
                      fe_face_values_mortar.JxW(q_point);    // dx
                }
              }
            }
          }
        }
      }
      // for (unsigned int i = 0; i < dofs_per_cell; ++i)
      // {
      //   solution_bar_mortar(local_dof_indices[i]) += local_rhs_bar(i);        
      //   solution_star_mortar(local_dof_indices[i]) += local_rhs_star(i);
      // }
    }
    // if (this_mpi == 0)
    // {
    //   std::cout << "interface_dofs_count: " << std::endl;
    //   for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    //   {
    //     std::cout << "side " << side << ": ";
    //     for (unsigned int i = 0; i < interface_dofs_count[side].size(); ++i)
    //       std::cout << interface_dofs_count[side][i] << " ";
    //     std::cout << std::endl;
    //   }
    // }
    // if (this_mpi == 0)
    // {
    //   std::cout << "interface_dofs: " << std::endl;
    //   for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    //   {
    //     std::cout << "side " << side << ": ";
    //     for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
    //       std::cout << interface_dofs[side][i] << " ";
    //     std::cout << std::endl;
    //   }
    // }
    // solution_bar_mortar.block(0).print(std::cout);
    // solution_star_mortar.print(std::cout);
  }

  // Arnoldi algorithm
  template <int dim>
  void
  MixedStokesProblem<dim>::arnoldi(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                    std::unique_ptr<FEFunction>       &solution_bar_function,
                                    std::unique_ptr<FEFunction>       &solution_star_function,
                                    unsigned int                      &k_counter, 
                                    std::vector<std::vector<double>> &interface_data,
                                    std::vector<std::vector<double>> &interface_data_send,
                                    std::vector<std::vector<double>> &interface_data_receive,
                                    FEFaceValues<dim>                &fe_face_values,
                                    FEFaceValues<dim>                &fe_face_values_mortar,
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
    interface_operator(interface_function,
                        solution_bar_function,
                        solution_star_function,
                        interface_data,
                        interface_data_send,
                        interface_data_receive,
                        fe_face_values,
                        fe_face_values_mortar,
                        Ap);

    //defing q  to push_back to Q (Arnoldi algorithm)
    //defing h  to push_back to H (Arnoldi algorithm)
    /*
        let k = iter count = k_counter + 1
        since c++ index starts from 0
        we start k_counter from 0. Now
        q[side].size() = k, where k = iter count
        h.size() = k + 1
        we use the extra space in h to store h(k+1) = norm(q)
    */
    std::vector<double> h(k_counter+2,0);


    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      if (neighbors[side] >= 0)
      {
        q[side].resize(Ap[side].size(),0);
        assert(Ap[side].size()==Q_side[side][k_counter].size());
        q[side] = Ap[side];
        for(unsigned int i=0; i<=k_counter; ++i)
          for(unsigned int j=0; j<q[side].size();++j)
            h[i]+=q[side][j]*Q_side[side][i][j];
      }
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
    double h_dummy = 0;

    //calculating h(k+1)=norm(q) as summation over side,subdomains norm_squared(q[side])
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        h_dummy+=vect_norm(q[side])*vect_norm(q[side]);
    
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
  MixedStokesProblem<dim>::local_gmres(unsigned int &cycle)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data(n_faces_per_cell);
    interface_fe_function.resize(n_faces_per_cell);
    solution_bar_mortar.resize(n_faces_per_cell);
    solution_star_mortar.resize(n_faces_per_cell);
    double initial_val = 0.0; //initial guess for gmres
    std::vector<std::vector<double>> lambda(n_faces_per_cell);

    // Extra for projections from mortar to fine grid and RHS assembly
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);
    // std::vector<BlockVector<double>> in_vec_lambda(n_faces_per_cell, BlockVector<double>(mortar_vec));
    // std::vector<Functions::FEFieldFunction<dim, BlockVector<double>>>
    //       interface_function;
    // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    // {
    //   interface_function.push_back(
    //     Functions::FEFieldFunction<dim, BlockVector<double>>
    //                               (dof_handler, in_vec_lambda[side]));
    // }
    // Functions::FEFieldFunction<dim, BlockVector<double>> 
    //       stokes_vector(dof_handler, in_vec_stokes);


    // dealii::AffineConstraints<double>   constraints;
    FEFaceValues<dim> fe_face_values_mortar(fe_mortar,
                                          quad,
                                          update_values | update_normal_vectors |
                                            update_quadrature_points |
                                            update_JxW_values);
     FEFaceValues<dim> fe_face_values(fe,
                                    quad,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);

    //GMRES structures and parameters
    std::vector<double>	sn;
    std::vector<double>	cs;
    std::vector<std::vector<double>>	H;
  //      std::vector<double> error_iter_side(n_faces_per_cell); //saves error in each iteration
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

        Ap[side].resize(interface_dofs[side].size(), 0);
        lambda[side].resize(interface_dofs[side].size(), initial_val);

        q[side].resize(interface_dofs[side].size());
        r[side].resize(interface_dofs[side].size(), 0);
        r_b[side].resize(interface_dofs[side].size(), 0);
      }

    // std::vector<BlockVector<double>> in_vec_lambda(n_faces_per_cell, BlockVector<double>(mortar_vec));
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
    //Calculating r-norm(same as b-norm)-----
    double r_norm =0;
    double b_norm =0;
    //computing the residual r=b-Ax
    compute_residual(interface_function,
                     solution_bar_function,
                     solution_star_function,
                     fe_face_values,
                     fe_face_values_mortar,
                     interface_data,
                     r,
                     r_norm,
                     r_b,
                     b_norm);
    //end -----------of calculating r-norm and b_norm------------------

    //Making the first element of matrix Q[side] same as r_side[side/r_norm
    for(unsigned int side=0; side<n_faces_per_cell;++side)
      if (neighbors[side] >= 0)
      {
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          q[side][i]= r[side][i]/r_norm ;
        //adding q[side] as first element of Q[side]
        Q_side[side].push_back(q[side]);
      }

    //end----------- of calculating first element of Q[side]-----------------
    e_all_iter.push_back(r_norm);
    Beta.push_back(r_norm);


    unsigned int k_counter = 0; //same as the count of the iteration
    gmres_iteration = 0;
    while (k_counter < maxiter)
    {
      gmres_iteration++;
      //Performing the Arnoldi algorithm
      arnoldi(interface_function,
              solution_bar_function,
              solution_star_function,
              k_counter,
              interface_data,
              interface_data_send,
              interface_data_receive,
              fe_face_values,
              fe_face_values_mortar,
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
      
      residual = combined_error_iter;

      pcout << "\r  ..." << gmres_iteration
            << " iterations completed, (residual = " << combined_error_iter
            << ")..." << std::flush;
      // Exit criterion
      if (combined_error_iter < tol)
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

    //Calculating the result from H ,Q_side and Beta
    //Finding y which has size k_counter using back sove function
    std::vector<double> y(k_counter+1,0);
    // std::cout << "\n this_mpi: " << this_mpi << " size of Beta: " << Beta.size() << " k_counter: " << k_counter << "\n";
    // assert(Beta.size()==k_counter+2); // gives error if exceed maxiter, do break if size is too much and going to exceed maxiter
    back_solve(H,Beta,y,k_counter);
    
    //updating X(lambda) to get the final lambda value 
    //lambda = x_initial + Q*y before solving the final star problem
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_data[side].size(); ++i)
          for(unsigned int j=0; j<=k_counter; ++j)
            lambda[side][i] += Q_side[side][j][i]*y[j];


    //we can replace lambda here and just add interface_data(skip one step below)
    interface_data = lambda;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = interface_data[side][i];

    assemble_rhs_star(interface_function,
                      fe_face_values);
    solve_star();
    solution.reinit(solution_bar);
    solution = solution_bar;
    solution.sadd(1.0, solution_star);
    pcout<<"finished local_gmres"<<"\n";
  }

  template <int dim>
  void 
  MixedStokesProblem<dim>::write_dof_locations(const std::string &  filename) const
  {
    const FEValuesExtractors::Vector velocities(0);
    std::map<types::global_dof_index, Point<dim>> dof_location_map =
      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler_mortar,
                                          fe.component_mask(velocities));

    std::ofstream dof_location_file(filename);
    DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                 dof_location_map);
  }

  template <int dim>
  void
  MixedStokesProblem<dim>::output_dof_results(const unsigned int cycle) const
  {
    std::ofstream mesh_file("../output/gnuplot_data/mesh-"
      +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
          +".gnuplot");
    GridOut().write_gnuplot(triangulation_mortar, mesh_file);

    if (cont_mortar_flag)
      write_dof_locations("../output/gnuplot_data/dof_loc-"
        +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
        +".gnuplot");
  }

  template <int dim>
  void
  MixedStokesProblem<dim>::output_results(const unsigned int cycle) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

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
      + ".vtk");
    data_out.write_vtk(output);
  }

  template <int dim>
  void MixedStokesProblem<dim>::compute_errors(const unsigned int &cycle) 
  {
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);
    const ComponentSelectFunction<dim> v_x_mask(0, dim + 1);
    const ComponentSelectFunction<dim> v_y_mask(1, dim + 1);

    ExactSolution<dim> exact_solution;
    prm_fxn.enter_subsection("Exact solution");
    exact_solution.exact_solution_val_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    prm_fxn.enter_subsection("Exact gradient");
    exact_solution.exact_solution_grad_val_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();

    Vector<double> cellwise_errors(triangulation.n_active_cells());
    Vector<double> cellwise_norms(triangulation.n_active_cells());

    QTrapezoid<1>  q_trapez;
    QIterated<dim> quadrature(q_trapez, degree + 2);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution.size());

    // Pressure L2 error and norm
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    p_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norms,
                                        VectorTools::L2_norm);     
                                        
    double p_l2_error_total, p_l2_norm_total;
    MPI_Allreduce(&p_l2_error, //sending this data
                  &p_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator
    MPI_Allreduce(&p_l2_norm, //sending this data
                  &p_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator                                   
    
    // Velocity L2 error and norm    
    cellwise_errors = 0;
    cellwise_norms = 0;                                
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norms,
                                        VectorTools::L2_norm);
                                        
    double u_l2_error_total, u_l2_norm_total;
    MPI_Allreduce(&u_l2_error, //sending this data
                  &u_l2_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator
    MPI_Allreduce(&u_l2_norm, //sending this data
                  &u_l2_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator                                          

    // Velocity H1 semi-norms
    cellwise_errors = 0;
    cellwise_norms = 0;
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &velocity_mask);
    
    const double u_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &velocity_mask);
    
    const double u_H1_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norms,
                                        VectorTools::L2_norm);            
                                        
    double u_H1_error_total, u_H1_norm_total;
    MPI_Allreduce(&u_H1_error, //sending this data
                  &u_H1_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator
    MPI_Allreduce(&u_H1_norm, //sending this data
                  &u_H1_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator                                          

    // Pressure H1 semi-norm
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &pressure_mask);
    const double p_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &pressure_mask);
    const double p_H1_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norms,
                                        VectorTools::L2_norm); 
                                        
    double p_H1_error_total, p_H1_norm_total;
    MPI_Allreduce(&p_H1_error, //sending this data
                  &p_H1_error_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator
    MPI_Allreduce(&p_H1_norm, //sending this data
                  &p_H1_norm_total, //receiving the result here
                  1, //number of elements in alpha and alpha_buffer = 1+1
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator_stokes); //communicator                                          
    
    const unsigned int n_active_cells = triangulation.n_active_cells();
    unsigned int n_dofs = dof_handler.n_dofs();
    if (n_processes > 1)
      n_dofs = interface_dofs_size;

    convergence_table_l2.add_value("cells", n_active_cells);
    convergence_table_l2.add_value("h", "1/"+Utilities::int_to_string(std::pow(2,cycle)));
    convergence_table_l2.add_value("dofs", n_dofs);
    convergence_table_l2.add_value("u_L2", u_l2_error/u_l2_norm);
    convergence_table_l2.add_value("p_L2", p_l2_error/p_l2_norm);
    convergence_table_l2.add_value("gmres_iter", gmres_iteration);
    convergence_table_l2.add_value("residual", residual);

    convergence_table_h1.add_value("cells", n_active_cells);
    convergence_table_h1.add_value("h", "1/"+Utilities::int_to_string(std::pow(2,cycle)));
    convergence_table_h1.add_value("dofs", n_dofs);
    convergence_table_h1.add_value("u_H1", u_H1_error/u_H1_norm);
    convergence_table_h1.add_value("p_H1", p_H1_error/p_H1_norm);
    // convergence_table_h1.add_value("ux_H1", ux_H1_error);
    // convergence_table_h1.add_value("uy_H1", uy_H1_error);
    // convergence_table_h1.add_value("gmres_iter", gmres_iteration);

    convergence_table_l2_total.add_value("cells", n_active_cells);
    convergence_table_l2_total.add_value("h", "1/"+Utilities::int_to_string(std::pow(2,cycle)));
    convergence_table_l2_total.add_value("dofs", n_dofs);
    convergence_table_l2_total.add_value("u_L2", u_l2_error_total/u_l2_norm_total);
    convergence_table_l2_total.add_value("p_L2", p_l2_error_total/p_l2_norm_total);
    convergence_table_l2_total.add_value("gmres_iter", gmres_iteration);
    convergence_table_l2_total.add_value("residual", residual);

    convergence_table_h1_total.add_value("cells", n_active_cells);
    convergence_table_h1_total.add_value("h", "1/"+Utilities::int_to_string(std::pow(2,cycle)));
    convergence_table_h1_total.add_value("dofs", n_dofs);
    convergence_table_h1_total.add_value("u_H1", u_H1_error_total/u_H1_norm_total);
    convergence_table_h1_total.add_value("p_H1", p_H1_error_total/p_H1_norm_total);
  }

  template <int dim>
  void MixedStokesProblem<dim>::print_individual_convg_table_l2()
  {
    convergence_table_l2.set_precision("residual", 3);
    convergence_table_l2.set_precision("u_L2", 3);
    convergence_table_l2.set_precision("p_L2", 3);

    convergence_table_l2.set_scientific("residual", true);
    convergence_table_l2.set_scientific("u_L2", true);
    convergence_table_l2.set_scientific("p_L2", true);

    convergence_table_l2.set_tex_caption("cells", "\\ cells");
    convergence_table_l2.set_tex_caption("dofs", "\\# dofs");
    convergence_table_l2.set_tex_caption("residual", "residual");
    convergence_table_l2.set_tex_caption("gmres_iter", "iter");
    convergence_table_l2.set_tex_caption("u_L2", "$||u_f-u_{fh}||_{L^2}$");
    convergence_table_l2.set_tex_caption("p_L2", "$||p_f-p_{fh}||_{L^2}$");

    convergence_table_l2.evaluate_convergence_rates(
          "u_L2", ConvergenceTable::reduction_rate_log2);
    convergence_table_l2.evaluate_convergence_rates(
          "p_L2", ConvergenceTable::reduction_rate_log2);

    std::string conv_filename = "../output/convg_tables/stokes_table_individual_l2" + std::string(".tex");

    if (this_mpi_stokes == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes_stokes; ++i)
    {
      MPI_Barrier(mpi_communicator_stokes);
      if (this_mpi_stokes == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table_l2.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table_l2.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    MPI_Barrier(mpi_communicator_stokes);
    if (this_mpi_stokes == 0)
    {
        std::ofstream table_file(conv_filename, std::ios::app);
        table_file << "\n\\end{document}\n";
    }
  }

  template <int dim>
  void
  MixedStokesProblem<dim>::print_individual_convg_table_h1()
  {
    // convergence_table_h1.set_precision("residual", 3);
    convergence_table_h1.set_precision("u_H1", 3);
    convergence_table_h1.set_precision("p_H1", 3);
    // convergence_table_h1.set_precision("ux_H1", 3);
    // convergence_table_h1.set_precision("uy_H1", 3);

    // convergence_table_h1.set_scientific("residual", true);
    convergence_table_h1.set_scientific("h", true);
    convergence_table_h1.set_scientific("u_H1", true);
    convergence_table_h1.set_scientific("p_H1", true);
    // convergence_table_h1.set_scientific("ux_H1", true);
    // convergence_table_h1.set_scientific("uy_H1", true);

    convergence_table_h1.set_tex_caption("cells", "\\ cells");
    convergence_table_h1.set_tex_caption("dofs", "\\# dofs");
    // convergence_table_h1.set_tex_caption("residual", "residual");
    convergence_table_h1.set_tex_caption("u_H1", "$||u_f-u_{fh}||_{H^1}$");
    convergence_table_h1.set_tex_caption("p_H1", "$||p_f-p_{fh}||_{H^1}$");
    // convergence_table_h1.set_tex_caption("ux_H1", "$H^1$-error (ux)");
    // convergence_table_h1.set_tex_caption("uy_H1", "$H^1$-error (uy)");
    // convergence_table_h1.set_tex_caption("gmres_iter", "gmres iter");

    convergence_table_h1.evaluate_convergence_rates(
          "u_H1", ConvergenceTable::reduction_rate_log2);
    convergence_table_h1.evaluate_convergence_rates(
          "p_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table_h1.evaluate_convergence_rates(
    //       "ux_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table_h1.evaluate_convergence_rates(
    //       "uy_H1", ConvergenceTable::reduction_rate_log2);

    std::string conv_filename = "../output/convg_tables/table_individual_h1" + std::string(".tex");

    if (this_mpi == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes_stokes; ++i)
    {
      MPI_Barrier(mpi_communicator_stokes);
      if (this_mpi_stokes == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table_h1.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table_h1.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    // MPI_Barrier(mpi_communicator);
    if (this_mpi_stokes == 0)
    {
      std::ofstream table_file(conv_filename, std::ios::app);
      table_file << "\n\\end{document}\n";
    }
  }

    template <int dim>
  void MixedStokesProblem<dim>::print_total_convg_table_l2()
  {
    if (this_mpi_stokes == 0)
    {
      convergence_table_l2_total.set_precision("residual", 3);
      convergence_table_l2_total.set_precision("u_L2", 3);
      convergence_table_l2_total.set_precision("p_L2", 3);
      convergence_table_l2_total.set_scientific("residual", true);
      convergence_table_l2_total.set_scientific("u_L2", true);
      convergence_table_l2_total.set_scientific("p_L2", true);

      convergence_table_l2_total.set_tex_caption("cells", "\\ cells");
      convergence_table_l2_total.set_tex_caption("dofs", "\\# dofs");
      convergence_table_l2_total.set_tex_caption("residual", "residual");
      convergence_table_l2_total.set_tex_caption("gmres_iter", "iter");
      convergence_table_l2_total.set_tex_caption("u_L2", "$||u_f-u_{fh}||_{L^2}$");
      convergence_table_l2_total.set_tex_caption("p_L2", "$||p_f-p_{fh}||_{L^2}$");

      convergence_table_l2_total.evaluate_convergence_rates(
            "u_L2", ConvergenceTable::reduction_rate_log2);
      convergence_table_l2_total.evaluate_convergence_rates(
            "p_L2", ConvergenceTable::reduction_rate_log2);

      std::string conv_filename = "../output/convg_table_total/stokes_table_total_l2" + std::string(".tex");

      std::cout << "total error\n" << std::endl;
      convergence_table_l2_total.write_text(std::cout);
      std::ofstream table_file(conv_filename, std::ios::out);
      table_file << "\\documentclass[10pt]{report}\n"
                  << "\\usepackage{float}\n"
                  << "\\usepackage[margin=0.7in]{geometry}\n\n"
                  << "\\begin{document}\n\n";
      
      convergence_table_l2_total.write_tex(table_file, false);
      table_file << "\n\\end{document}\n";
      table_file.close();
    }
  }

  template <int dim>
  void
  MixedStokesProblem<dim>::print_total_convg_table_h1()
  {
    if (this_mpi_stokes == 0)
    {
      // convergence_table_h1.set_precision("residual", 3);
      convergence_table_h1_total.set_precision("u_H1", 3);
      convergence_table_h1_total.set_precision("p_H1", 3);
      // convergence_table_h1.set_precision("ux_H1", 3);
      // convergence_table_h1.set_precision("uy_H1", 3);

      // convergence_table_h1.set_scientific("residual", true);
      convergence_table_h1_total.set_scientific("h", true);
      convergence_table_h1_total.set_scientific("u_H1", true);
      convergence_table_h1_total.set_scientific("p_H1", true);
      // convergence_table_h1.set_scientific("ux_H1", true);
      // convergence_table_h1.set_scientific("uy_H1", true);

      convergence_table_h1_total.set_tex_caption("cells", "\\ cells");
      convergence_table_h1_total.set_tex_caption("dofs", "\\# dofs");
      // convergence_table_h1.set_tex_caption("residual", "residual");
      convergence_table_h1_total.set_tex_caption("u_H1", "$||u_f-u_{fh}||_{H^1}$");
      convergence_table_h1_total.set_tex_caption("p_H1", "$||p_f-p_{fh}||_{H^1}$");
      // convergence_table_h1.set_tex_caption("ux_H1", "$H^1$-error (ux)");
      // convergence_table_h1.set_tex_caption("uy_H1", "$H^1$-error (uy)");
      // convergence_table_h1.set_tex_caption("gmres_iter", "gmres iter");

      convergence_table_h1_total.evaluate_convergence_rates(
            "u_H1", ConvergenceTable::reduction_rate_log2);
      convergence_table_h1_total.evaluate_convergence_rates(
            "p_H1", ConvergenceTable::reduction_rate_log2);
      // convergence_table_h1.evaluate_convergence_rates(
      //       "ux_H1", ConvergenceTable::reduction_rate_log2);
      // convergence_table_h1.evaluate_convergence_rates(
      //       "uy_H1", ConvergenceTable::reduction_rate_log2);

      std::string conv_filename = "../output/convg_table_total/stokes_table_total_h1" + std::string(".tex");

      std::cout << "total error\n" << std::endl;
      convergence_table_h1_total.write_text(std::cout);
      std::ofstream table_file(conv_filename, std::ios::out);
      table_file << "\\documentclass[10pt]{report}\n"
                  << "\\usepackage{float}\n"
                  << "\\usepackage[margin=0.7in]{geometry}\n\n"
                  << "\\begin{document}\n\n";
      
      convergence_table_h1_total.write_tex(table_file, false);
      table_file << "\n\\end{document}\n";
      table_file.close();
    }
  }

  // MixedStokesProblemDD::run
  template <int dim>
  void MixedStokesProblem<dim>::run(unsigned int cycle)
  {
    // tolerance = tol;
    // qdegree   = quad_degree;

    Assert(reps[0].size() == dim, ExcDimensionMismatch(reps[0].size(), dim));

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
        subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

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
                                                  reps[0],
                                                  p1,
                                                  p2);

      // if (mortar_flag)
        GridGenerator::subdivided_hyper_rectangle(triangulation_mortar,
                                                  reps[0],
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
      // triangulation_mortar.clear();
      // triangulation_mortar.copy_triangulation(triangulation);
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
    solve_bar();
    if(n_processes == 1)
      solution = solution_bar;
    
  }

  template class MixedStokesProblem<2>;
  // template class MixedStokesProblem<3>;
} // namespace mfe_stokes
