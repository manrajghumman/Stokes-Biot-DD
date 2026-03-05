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
    std::vector<unsigned int>  &n_domains,
    const unsigned int         &subdomain_type,
    std::vector<int>           &neighbors,
    std::vector<unsigned int>  &neighbor_type,
    MPI_Comm                   mpi_communicator_stokes,
    const unsigned int         mortar_degree,
    const unsigned int         mortar_type,
    const unsigned int         mortar_degree_biot,
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
    , mortar_type(mortar_type)
    , mortar_degree_biot(mortar_degree_biot)
    , cont_mortar_flag(cont_mortar_flag) // if mortar_degree is 0, use RT<dim> (0)
    , neighbors(neighbors)
    , neighbor_type(neighbor_type)
    , gmres_iteration(0)
    , dof_handler(triangulation)
    , dof_handler_biot_mortar(triangulation_mortar)                          
    , dof_handler_mortar(triangulation_mortar)
    , fe(FE_Q<dim>(degree+1),//fe for velocity
         dim,
         FE_Q<dim>(degree),//fe for pressure
         1)
    , fe_mortar([mortar_degree, cont_mortar_flag]() -> FESystem<dim> {
      if (cont_mortar_flag)
        return FESystem<dim>(FE_Q<dim>(mortar_degree), dim, FE_Nothing<dim>(), 1);
      else if (!cont_mortar_flag)
        return FESystem<dim>(FE_DGQ<dim>(mortar_degree), dim, FE_Nothing<dim>(), 1);
      else
      {
        AssertThrow(false, ExcMessage("Invalid mortar element configuration."));
        return FESystem<dim>(FE_Nothing<dim>(), dim, FE_Nothing<dim>(), 1);
      }
    }()) // Conditional initialization using a lambda function
    , fe_biot_mortar([mortar_degree_biot, mortar_type]() -> FESystem<dim> {
        if (mortar_type == 0) // BDM(k), k>0; k = 1 is no mortar case
          return FESystem<dim>(FE_BDM<dim>(mortar_degree_biot), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_BDM<dim>(mortar_degree_biot), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else if (mortar_type == 1) // RT(k), k>=0
          return FESystem<dim>(FE_RaviartThomas<dim>(mortar_degree_biot), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_RaviartThomas<dim>(mortar_degree_biot), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else if (mortar_type == 2) // Q(k), k>0
          return FESystem<dim>(FE_Q<dim>(mortar_degree_biot), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_Q<dim>(mortar_degree_biot), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else if (mortar_type == 3) // DQ(k), k>=0
          return FESystem<dim>(FE_DGQ<dim>(mortar_degree_biot), dim,    // displacement lagrange multiplier on mortar
                              FE_Nothing<dim>(), dim,
                              FE_Nothing<dim>(), 0.5*dim*(dim-1), 
                              FE_DGQ<dim>(mortar_degree_biot), 1,    // pressure lagrange multiplier on mortar
                              FE_Nothing<dim>(), 1);
        else
          AssertThrow(false, ExcMessage("Invalid mortar type. Use 0 for no mortar, 1 for BDM, or 2 for RT."));
      }())
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator_stokes) == 0))
    // , computing_timer(mpi_communicator,
    //                   std::cout,
    //                   TimerOutput::summary,
    //                   TimerOutput::wall_times)
  {
    interpret_parameters();
    StokesParameterReader param(prm_fxn);
    param.read_parameters("../test function/dd test/stokes.prm");
    pcout << "inside stokes call, cont_mortar_flag: " << cont_mortar_flag
          << ", mortar_degree: " << mortar_degree 
          << ", degree: " << degree <<
          ", mortar_degree_biot: " << mortar_degree_biot << std::endl;
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
    for (unsigned int i = 0; i < dim; ++i)
      n_u_mortar += dofs_per_component_mortar[i];

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
  }

  // MixedStokesProblemDD - setup system for biot
  template <int dim>
  void
  MixedStokesProblem<dim>::setup_system_biot()
  {
    // write the dof count for biot side!!
    dof_handler_biot_mortar.distribute_dofs(fe_biot_mortar);
    DoFRenumbering::component_wise(dof_handler_biot_mortar);

    std::vector<types::global_dof_index> dofs_per_component_biot_mortar =
            DoFTools::count_dofs_per_fe_component (dof_handler_biot_mortar);

    unsigned int n_s_mortar=0, n_u_mortar=0, n_g_mortar=0, n_z_mortar = 0, n_p_mortar = 0;

    // displacement is normal trace of tensor valued stress and
    // pressure is normal trace of vector valued flux
    if (mortar_type < 2) // BDM or RT mortar
    {
      for (unsigned int i=0; i<dim; ++i)
      {
        n_s_mortar += dofs_per_component_biot_mortar[i*dim];
        n_u_mortar += dofs_per_component_biot_mortar[dim*dim + i];

        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
            n_g_mortar = dofs_per_component_biot_mortar[dim*dim + dim];
        else if (dim == 3)
            n_g_mortar += dofs_per_component_biot_mortar[dim*dim + dim + i];
      }
      n_z_mortar = dofs_per_component_biot_mortar[dim*dim + dim + 0.5*dim*(dim-1)];
      n_p_mortar = dofs_per_component_biot_mortar[dim*dim + dim +0.5*dim*(dim-1) + dim];
    }
    else // displacement is a vector mortar space and pressure is a scalar mortar space
    {
      for (unsigned int i=0; i<dim; ++i)
      {
        n_s_mortar += dofs_per_component_biot_mortar[i];
        n_u_mortar += dofs_per_component_biot_mortar[dim + i];

        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
            n_g_mortar = dofs_per_component_biot_mortar[2*dim];
        else if (dim == 3)
            n_g_mortar += dofs_per_component_biot_mortar[2*dim + i];
      }
      n_z_mortar = dofs_per_component_biot_mortar[2*dim + 0.5*dim*(dim-1)];
      n_p_mortar = dofs_per_component_biot_mortar[2*dim +0.5*dim*(dim-1) + 1];
    }

    n_stress_mortar = n_s_mortar;
    n_disp_mortar = n_u_mortar;
    n_rot_mortar = n_g_mortar;
    n_flux_mortar = n_z_mortar;
    n_pressure_mortar = n_p_mortar;

    if (this_mpi == 1)
    {
      std::cout << "Stokes-Biot mortar part:\n N stress mortar dofs: " << n_s_mortar
                << "\n N displacement mortar dofs: " << n_u_mortar
                << "\n N rotation mortar dofs: " << n_g_mortar
                << "\n N flux mortar dofs: " << n_z_mortar
                << "\n N pressure mortar dofs: " << n_p_mortar << std::endl;
    }

    mortar_vec_biot.reinit(5);
    mortar_vec_biot.block(0).reinit (n_s_mortar);
    mortar_vec_biot.block(1).reinit (n_u_mortar);
    mortar_vec_biot.block(2).reinit (n_g_mortar);
    mortar_vec_biot.block(3).reinit (n_z_mortar);
    mortar_vec_biot.block(4).reinit (n_p_mortar);
    mortar_vec_biot.collect_sizes ();
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

    local_face_dof_indices.resize(fe_mortar.dofs_per_cell);

    for (const auto &cell : dof_handler_mortar.active_cell_iterators())
      for (unsigned int face_n = 0; face_n < n_faces_per_cell; ++face_n)
        if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0 && 
                cell->face(face_n)->boundary_id() !=7))
        {
          // cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
          cell->get_dof_indices(local_face_dof_indices);
          side = cell->face(face_n)->boundary_id() - 1;
          if (neighbor_type[side] == subdomain_type) // Stokes-Stokes interface
          {
            // for (auto el : local_face_dof_indices)
            for (unsigned int i = 0; i < local_face_dof_indices.size(); ++i)
            {
                auto el = local_face_dof_indices[i];
              if (el < n_velocity_interface && fe_mortar.has_support_on_face(i, face_n)){
                // adding a check to not include the dofs at corners twice
                if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                                                  == interface_dofs[side].end())
                {
                  interface_dofs[side].push_back(el);
                  interface_dofs_total.push_back(el);                                                    
                }
              }
            }
          }
        }
    
    if (biot_neighbor) // Stokes-Biot interface
    {
      // Get interface dofs for stress and velocity dofs in biot side
      std::vector<types::global_dof_index> local_face_dof_indices_biot;
      side = 0;
      if (mortar_type < 2) // BDM or RT mortar
      {
        local_face_dof_indices_biot.resize(fe_biot_mortar.dofs_per_face);
        for (auto &cell_biot : dof_handler_biot_mortar.active_cell_iterators())
        {
          for (unsigned int face_n = 0; face_n < n_faces_per_cell;
                ++face_n)
          {
            if (cell_biot->at_boundary(face_n) &&
                  (cell_biot->face(face_n)->boundary_id() != 0 && 
                    cell_biot->face(face_n)->boundary_id() !=7))
            {
              cell_biot->face(face_n)->get_dof_indices(local_face_dof_indices_biot, 0);
              side = cell_biot->face(face_n)->boundary_id() - 1;
              if (neighbor_type[side] == 2) // Stokes-Biot interface
                for (auto el : local_face_dof_indices_biot)
                {
                  // there are no corner dofs in BDM, so no need to check for duplicates
                  interface_dofs[side].push_back(el);
                  interface_dofs_total.push_back(el);                                                    
                }
            }
          }
        }
      }
      else
      {
        local_face_dof_indices_biot.resize(fe_biot_mortar.dofs_per_cell);
        for (auto &cell_biot : dof_handler_biot_mortar.active_cell_iterators())
          for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            if (cell_biot->at_boundary(face) &&
                  (cell_biot->face(face)->boundary_id() != 0 && 
                    cell_biot->face(face)->boundary_id() !=7))
            {
              cell_biot->get_dof_indices(local_face_dof_indices_biot);
              side = cell_biot->face(face)->boundary_id() - 1;
              if (neighbor_type[side] == 2) // Stokes-Biot interface
                for (unsigned int i = 0; i < local_face_dof_indices_biot.size(); ++i)
                  {
                    auto el = local_face_dof_indices_biot[i];
                    if (fe_biot_mortar.has_support_on_face(i, face))
                    {
                      // adding a check to not include the dofs at corners twice
                      if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                                                == interface_dofs[side].end())
                      {
                        interface_dofs[side].push_back(el);
                        interface_dofs_total.push_back(el);                                                    
                      }
                    }
                  }
            }
      }
    }
    if (this_mpi == 1)
    {
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        if (neighbors[side] >= 0)
        {
          if (neighbor_type[side] == 1) // Stokes-Stokes interface  
          {
            std::cout << "Stokes Part:\n Number of interface dofs on face " << side
                  << " : " << interface_dofs[side].size() << ", interface dofs: ";
                  std::cout << std::endl;
          }
          else if (neighbor_type[side] == 2) // Stokes-Biot interface
          {
            std::cout << "Stokes-Biot Part:\n Number of interface dofs on face " << side
                  << " : " << interface_dofs[side].size() << ", interface dofs: ";
          }
          // for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          //   std::cout << interface_dofs[side][i] << " ";
          std::cout << std::endl;
        }
      }
    }
    if (this_mpi == 0 || this_mpi == 2)
    {
      std::cout << "Total interface dofs before function: " << interface_dofs_total.size() << std::endl;
    }
    compute_interface_dofs_size<dim>(interface_dofs_total, 
                                     mpi_communicator, 
                                     this_mpi, 
                                     n_processes,
                                     n_domains,
                                     interface_dofs_size);
    if (this_mpi == 0 || this_mpi == 2)
    {
      std::cout << "Total number of interface dofs: " << interface_dofs_total.size() << std::endl;
      std::cout << "interface_dofs: " << std::endl;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        std::cout << "side " << side << ": ";
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          std::cout << interface_dofs[side][i] << " ";
        std::cout << std::endl;
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
    system_rhs_bar        = 0;
    double alpha = 1.0; // BJS coupling parameter

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

    // to keep track of which face we are on for interface boundary conditions
    // our objects assume clockwise labelling but dealii assumes component-wise 
    //labelling, so we need to convert between the two when applying boundary conditions
    unsigned int side; 
    Tensor<1, dim> tangent_vector; // the tangent vector to a side
    double const_bjs = 0; 
    const K_permeability<dim> k_permeability;
    std::vector<Tensor<2,dim>>  k_permeability_values(n_face_q_points);
                                             
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
          if (face->boundary_id() == 7) //Entering Neumann Condition
          {
            fe_face_values.reinit(cell, face);
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
            fe_face_values.reinit(cell, face);
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
                                        * phi_p[i]) // <u.n, w>
                                        * fe_face_values.JxW(q);
                }
                local_rhs(i) += (-2 * gd * (symgrad_phi_u[i] 
                              * fe_face_values.normal_vector(q)) // -2 <gd, D(v).n>
                              + gamma * gd * phi_u [i] / h_K // gamma * <gd, v>/h
                              + gd * fe_face_values.normal_vector(q)
                              * phi_p[i]) // <gd.n, w>
                              * fe_face_values.JxW(q);
              }
            }
          }//Ending Dirichlet Condition
          if (face->boundary_id() != 0 && face->boundary_id() != 7) //Entering Stokes-Biot interface BJS term
          {
            side = face->boundary_id() - 1;
            if (neighbor_type[side] == 2) // Stokes-Biot interface
            {
              fe_face_values.reinit(cell, face);
              k_permeability.value_list (fe_face_values.get_quadrature_points(), k_permeability_values);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  phi_u[k] = fe_face_values[velocities].value(k, q);

                // computes normal tangent vector in 2D, 
                // assumes counterclockwise ordering of faces in deal.ii
                if (dim == 2) 
                {
                  tangent_vector[0] = - fe_face_values.normal_vector(q)[1] * get_normal_direction<dim>(side);
                  tangent_vector[1] = fe_face_values.normal_vector(q)[0] * get_normal_direction<dim>(side);
                }
                else if (dim == 3)
                {
                  Assert(false, ExcMessage("Tangent vector not implemented for 3D yet"));
                }
                // compute the constant in the BJS condition, from mu*alpha_BJS*sqrt(K_j^-1)
                const_bjs = alpha * 
                            std::sqrt(1/((k_permeability_values[q]*tangent_vector)
                                                      *tangent_vector));
                                                      
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j <= i; ++j)
                  {
                    local_matrix(i,j) += const_bjs * (phi_u[j] * tangent_vector)
                                          * (phi_u[i] * tangent_vector) // alpha*sqrt(K^-1) <u_t, v_t>
                                          * fe_face_values.JxW(q);
                  }
                }
              }
            }
          }
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
    FEFaceValues<dim> &fe_face_values,
    FEFaceValues<dim> &fe_face_values_mortar_biot)
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS star stokes");
    system_rhs_star = 0;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    if (stokes_neighbor)
    {
      const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
      const unsigned int dofs_per_cell   = fe.dofs_per_cell;

      Vector<double>                       local_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const FEValuesExtractors::Vector velocities(0);
      // std::vector<Tensor<1, dim>>          interface_values(n_face_q_points, Tensor<1, dim>());
      std::vector<std::vector<double>>     interface_values(dim, std::vector<double>(n_face_q_points, 0.0));//first index for dim components, 0, 1: vel, 2: pressure
      std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);                                            //second for value at each quad point
      unsigned int side;
      // if (this_mpi == 1)
      //   std::cout << "Assembling RHS star for this_mpi = " << this_mpi << std::endl;
      // unsigned int hit = 1;
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
              //dealii uses component-wise labelling but 
              //our objects use clockwise labelling
              side = cell->face(face_n)->boundary_id() - 1;
              if (neighbor_type[side] == 1) // Stokes-Stokes interface
              {
                fe_face_values.reinit(cell, face_n);

                for (unsigned int s = 0; s < dim; ++s)
                  interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                        interface_values[s], s);

                for (unsigned int s = 0; s < dim; ++s)
                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    AssertIsFinite(interface_values[s][q]);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                    ++q_point)
                {
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]  = fe_face_values[velocities].value(k, q_point)
                                * get_normal_direction<dim>(side); // multiply by normal direction to get correct sign
                  }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int s = 0; s < dim; ++s)
                      local_rhs(i) +=
                        - (phi_u[i][s] *  // phi_v_i(x_q)
                          interface_values[s][q_point]) *    // Tn_i = lambda_f
                          fe_face_values.JxW(q_point);    // dx
                  }
                }
              }
            }  
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs_star(local_dof_indices[i]) += local_rhs(i);        
      }
    }
    if (biot_neighbor)
    {
      double alpha = 1.0; // BJS coupling parameter
      const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      Vector<double>                       local_rhs (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      const FEValuesExtractors::Vector velocities(0);
      std::vector<Tensor<1, dim>>     phi_u(dofs_per_cell);
      double lambda_p = 0.0; // the pressure lagrange multiplier value at the quad point
      Tensor<1, dim> lambda_d; // the displacement lagrange multiplier value at the quad point

      Tensor<1, dim> tangent_vector; // the tangent vector to a side                                                
      double const_bjs = 0; 
      const K_permeability<dim> k_permeability;
      std::vector<Tensor<2,dim>>  k_permeability_values(n_face_q_points);

      unsigned int side;
      

      if (mortar_type < 2) // BDM or RT mortar
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
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7))
            {
              side = cell->face(face_n)->boundary_id() - 1;
              if (neighbor_type[side] == 2) // Stokes-Biot interface
              {
                fe_face_values.reinit (cell, face_n);
                k_permeability.value_list (fe_face_values.get_quadrature_points(), k_permeability_values);
                for (unsigned int s = 0; s < dim; ++s)
                  interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                                      interface_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);

                for (unsigned int s = 0 ; s < dim*dim; ++s)
                  interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                                interface_values_stress[s/dim][s%dim], s);

                for (unsigned int q=0; q<n_face_q_points; ++q)
                {
                  // computes normal tangent vector in 2D, 
                  // assumes counterclockwise ordering of faces in deal.ii
                  if (dim == 2) 
                  {
                    tangent_vector[0] = - fe_face_values.normal_vector(q)[1] * get_normal_direction<dim>(side);
                    tangent_vector[1] = fe_face_values.normal_vector(q)[0] * get_normal_direction<dim>(side);
                  }
                  else if (dim == 3)
                  {
                    Assert(false, ExcMessage("Tangent vector not implemented for 3D yet"));
                  }
                  // compute the constant in the BJS condition, from mu*alpha_BJS*sqrt(K_j^-1)
                  const_bjs = alpha * 
                              std::sqrt(1/((k_permeability_values[q]*tangent_vector)
                                                        *tangent_vector));
                  // compute lambda_p and lambda_d at the quad point  
                  lambda_p = 0;                                    
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    lambda_p += interface_values_flux[s][q] 
                                  * fe_face_values.normal_vector(q)[s];    
                    lambda_d[s] = 0;
                    for (unsigned int l = 0; l < dim; ++l)
                      lambda_d[s] += interface_values_stress[s][l][q] 
                                    * fe_face_values.normal_vector(q)[l]; 
                  }
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]  = fe_face_values[velocities].value(k, q); 
                  }                 
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                    local_rhs(i) += (- (phi_u[i] * fe_face_values.normal_vector(q) 
                                        * lambda_p) // - <lambda_p, v.n>
                                    + const_bjs * (phi_u[i] * tangent_vector)
                                    * (lambda_d * tangent_vector)) 
                                  * fe_face_values.JxW(q); // c_bjs<lambda_d.t_f, v.t_f> term from BJS condition                     
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
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7))
            {
              side = cell->face(face_n)->boundary_id() - 1;
              if (neighbor_type[side] == 2) // Stokes-Biot interface
              {
                fe_face_values.reinit (cell, face_n);
                k_permeability.value_list (fe_face_values.get_quadrature_points(), k_permeability_values);
                interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                                    interface_values_press, 2*dim+0.5*dim*(dim-1));
                
                for (unsigned int s = 0; s < dim; ++s)
                  interface_function[side]->value_list(fe_face_values.get_quadrature_points(),
                                                      interface_values_disp[s], s);
                for (unsigned int q=0; q<n_face_q_points; ++q)
                {
                  // computes normal tangent vector in 2D, 
                  // assumes counterclockwise ordering of faces in deal.ii
                  if (dim == 2) 
                  {
                    tangent_vector[0] = - fe_face_values.normal_vector(q)[1] * get_normal_direction<dim>(side);
                    tangent_vector[1] = fe_face_values.normal_vector(q)[0] * get_normal_direction<dim>(side);
                  }
                  else if (dim == 3)
                  {
                    Assert(false, ExcMessage("Tangent vector not implemented for 3D yet"));
                  }
                  // compute the constant in the BJS condition, from mu*alpha_BJS*sqrt(K_j^-1)
                  const_bjs = alpha * 
                              std::sqrt(1/((k_permeability_values[q]*tangent_vector)
                                                        *tangent_vector));
                  // compute lambda_d at the quad point   
                  // lambda_p at q is just interface_values_press at q, 
                  // since the mortar space is Q(k) or DQ(k)                                 
                  for (unsigned int s = 0; s < dim; ++s)   
                  {
                    lambda_d[s] = 0;
                    lambda_d[s] += interface_values_disp[s][q]; 
                  }
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]  = fe_face_values[velocities].value(k, q);
                  }      
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                    local_rhs(i) += (- (phi_u[i] * fe_face_values.normal_vector(q) 
                                        * interface_values_press[q]) // - <lambda_p, v.n>
                                      + const_bjs * (phi_u[i] * tangent_vector)
                                      * (lambda_d * tangent_vector)) 
                                    * fe_face_values.JxW(q); // c_bjs<lambda_d.t_f, v.t_f> term from BJS condition                               
                  }                                    
                }  
              }
            }
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            system_rhs_star(local_dof_indices[i]) += local_rhs(i);
        }
      }
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
                                              FEFaceValues<dim>                &fe_face_values_mortar_biot,
                                              std::vector<Projector::Projector<dim>> &P_mortar,
                                              std::vector<std::vector<double>> &Ap)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = 
          interface_data[side][i];

    // interface_fe_function.block(1) = 0;
    assemble_rhs_star(interface_function,
                      fe_face_values,
                      fe_face_values_mortar_biot);
    solve_star();

    for (unsigned int i = 0; i < solution_star.size(); ++i)
      AssertIsFinite(solution_star[i]);
    interface_inner_product(interface_function,
                            solution_bar_function,
                            solution_star_function,
                            fe_face_values_mortar,
                            fe_face_values_mortar_biot,
                            P_mortar);
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        // Create vector of u\dot n to send
        if (neighbor_type[side] == 1) //Stokes-Stokes interface
        {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_data_send[side][i] = - solution_star_mortar[side][interface_dofs[side][i]];
        }
        else if (neighbor_type[side] == 2) //Stokes-Biot interface
        {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            if (interface_dofs[side][i] < n_stress_mortar) // displacement lagrange multiplier test space
              interface_data_send[side][i] = solution_star_mortar[side][interface_dofs[side][i]];
            else // pressure lagrange multiplier test space
              interface_data_send[side][i] = - solution_star_mortar[side][interface_dofs[side][i]];
          }
        }

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
          Ap[side][i] = (interface_data_send[side][i] +
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
                                            FEFaceValues<dim>                &fe_face_values_mortar_biot,
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

    // compute Ax from x = initial_val
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
      {
        interface_fe_function[side][interface_dofs[side][i]] = 
          interface_data[side][i];
      }

    assemble_rhs_star(interface_function,
                      fe_face_values,
                      fe_face_values_mortar_biot);
    solve_star();

    //for computing rhs b for
    // Ax = b in interface problem below
    solve_bar();


    //for computing the inner products
    //in the mortar space
    interface_inner_product(interface_function,
                            solution_bar_function,
                            solution_star_function,
                            fe_face_values_mortar,
                            fe_face_values_mortar_biot,
                            P_mortar);

    //compute residual r = b - A*x
    for (unsigned side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        std::vector<double> r_b_receive_buffer(r[side].size());
        std::vector<double> r_receive_buffer(r[side].size());

        if (neighbor_type[side] == 1) //Stokes-Stokes interface
        {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            r_b[side][i] = solution_bar_mortar[side][interface_dofs[side][i]];
            r[side][i] =  solution_bar_mortar[side][interface_dofs[side][i]]
                          + solution_star_mortar[side][interface_dofs[side][i]];
          }
        }
        else if (neighbor_type[side] == 2) //Stokes-Biot interface
        {
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
      }

      r_norm = inner_product_mortar(r, r, fe_face_values_mortar, fe_face_values_mortar_biot);
      b_norm = inner_product_mortar(r_b, r_b, fe_face_values_mortar, fe_face_values_mortar_biot);

    // //Calculating r-norm(same as b-norm)-----
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
  MixedStokesProblem<dim>::interface_inner_product(std::vector<std::unique_ptr<FEFunction>> &interface_function,
                                                  std::unique_ptr<FEFunction>      &solution_bar_function,
                                                  std::unique_ptr<FEFunction>       &solution_star_function,
                                                  FEFaceValues<dim>                 &fe_face_values_mortar,
                                                  FEFaceValues<dim>                 &fe_face_values_mortar_biot,
                                                  std::vector<Projector::Projector<dim>> &P_mortar)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        solution_bar_mortar[side] = 0;
        solution_star_mortar[side] = 0;
        solution_bar_mortar_temp[side] = 0;
        solution_star_mortar_temp[side] = 0;
      }

    // used for projection to mortar space
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);

    if (stokes_neighbor)
    { 
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
            side = cell->face(face_n)->boundary_id() - 1;
            if (neighbor_type[side] == 1) //Stokes-Stokes interface
            {
              fe_face_values_mortar.reinit(cell, face_n);
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
                    phi_mu[k]  = fe_face_values_mortar[velocities].value(k, q_point)
                                * get_normal_direction<dim>(side); // multiply by normal direction to get correct sign
                  }
                  else
                    phi_mu[k] = 0;
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                      (phi_mu[i][s] * bar_values[s][q_point]) // <mu(x_q), u_bar(x_q)>
                        * fe_face_values_mortar.JxW(q_point); // dx

                    solution_star_mortar_temp[side](local_dof_indices[i]) +=
                      (phi_mu[i][s] * star_values[s][q_point])  // <mu(x_q), u_star(x_q)>
                        * fe_face_values_mortar.JxW(q_point);   // dx
                  }
                }
              }
            }
          }
        }
      }

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0 && neighbor_type[side] == 1) //Stokes-Stokes interface
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
        } 
    } 
    if (biot_neighbor) 
    {
      double alpha = 1.0;     
      const unsigned int n_face_q_points = fe_face_values_mortar_biot.get_quadrature().size();
      const unsigned int dofs_per_cell   = fe_biot_mortar.dofs_per_cell;

      Vector<double>                       local_rhs_bar(dofs_per_cell);
      Vector<double>                       local_rhs_star(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<double> phi_p(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_eta(dofs_per_cell, Tensor<1, dim>());

      const K_permeability<dim> k_permeability;
      std::vector<Tensor<2,dim>>  k_permeability_values(n_face_q_points);

      //first index for dim components, 0, 1: vel, 2: pressure
      //second for value at each quad point
      std::vector<std::vector<double>> bar_values(dim, std::vector<double>(n_face_q_points, 0.0));
      std::vector<std::vector<double>> star_values(dim, std::vector<double>(n_face_q_points, 0.0));

      double lambda_p = 0.0; // the pressure lagrange multiplier value at the quad point
      Tensor<1, dim> lambda_d; // the displacement lagrange multiplier value at the quad point

      unsigned int side = 0;
      double const_bjs = 0.0; // the constant in the BJS condition, from mu*alpha_BJS*sqrt(K_j^-1)

      Tensor<1, dim> tangent_vector;

      if (mortar_type < 2) // using BDM or RT for the mortar space
      {
        std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

        for (unsigned int d=0; d<dim; ++d)
        {
          const FEValuesExtractors::Vector tmp_stress(d*dim);
          stresses[d].first_vector_component = tmp_stress.first_vector_component;
        }
        const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));

        std::vector<std::vector<double>> interface_values_flux(dim, std::vector<double> (n_face_q_points));
        std::vector<std::vector<std::vector<double>>> interface_values_stress(dim, 
                                                        std::vector<std::vector<double>>(
                                                        dim, std::vector<double>(n_face_q_points)));

        for (const auto &cell : dof_handler_biot_mortar.active_cell_iterators())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
            if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7)  )
            {
              //Need to reorder faces deal.ii iterators 
              //use a different ordering from freefem, 
              //they are ordered component wise 
              side = cell->face(face_n)->boundary_id() - 1;
              if (neighbor_type[side] == 2) //Stokes-Biot interface
              {
                fe_face_values_mortar_biot.reinit(cell, face_n);
                k_permeability.value_list (fe_face_values_mortar_biot.get_quadrature_points(), k_permeability_values);
                for (unsigned int s = 0; s < dim; ++s)
                {
                  solution_bar_function->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                            bar_values[s], s);
                  solution_star_function->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                            star_values[s], s);
                }
                for (unsigned int s = 0; s < dim; ++s)
                  interface_function[side]->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                                      interface_values_flux[s], dim*dim + dim + 0.5*dim*(dim-1)+s);

                for (unsigned int s = 0 ; s < dim*dim; ++s)
                  interface_function[side]->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                                interface_values_stress[s/dim][s%dim], s);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                      ++q_point)
                {
                  // computes normal tangent vector in 2D, 
                  // assumes counterclockwise ordering of faces in deal.ii
                  if (dim == 2) 
                  {
                    tangent_vector[0] = - fe_face_values_mortar_biot.normal_vector(q_point)[1] 
                                           * get_normal_direction<dim>(side);
                    tangent_vector[1] = fe_face_values_mortar_biot.normal_vector(q_point)[0] 
                                           * get_normal_direction<dim>(side);
                  }
                  else if (dim == 3)
                  {
                    Assert(false, ExcMessage("Tangent vector not implemented for 3D yet"));
                  }
                  // compute the constant in the BJS condition, from mu*alpha_BJS*sqrt(K_j^-1)
                  const_bjs = alpha * 
                              std::sqrt(1/((k_permeability_values[q_point] * tangent_vector)
                                                        * tangent_vector));
                  // compute lambda_p and lambda_d at the quad point
                  lambda_p = 0;
                  for (unsigned int s = 0; s < dim; ++s)
                  {
                    lambda_p += interface_values_flux[s][q_point] 
                                * fe_face_values_mortar_biot.normal_vector(q_point)[s];    
                    lambda_d[s] = 0;
                    for (unsigned int l = 0; l < dim; ++l)
                      lambda_d[s] += interface_values_stress[s][l][q_point] 
                                    * fe_face_values_mortar_biot.normal_vector(q_point)[l]; 
                  }                                                   
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    for (unsigned int s = 0; s < dim; ++s)
                      phi_eta[k][s] = fe_face_values_mortar_biot[stresses[s]].value (k, q_point)
                                    * fe_face_values_mortar_biot.normal_vector(q_point)
                                    * get_normal_direction<dim>(side);

                    phi_p[k] = fe_face_values_mortar_biot[velocity].value (k, q_point)
                              * fe_face_values_mortar_biot.normal_vector(q_point)
                              * get_normal_direction<dim>(side);
                  }
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // contribution from the coupling term in the 
                    // displacement lagrange multiplier test function
                    for (unsigned int s = 0; s < dim; ++s)
                    {
                      solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                        - (const_bjs * bar_values[s][q_point] * tangent_vector[s]
                          * phi_eta[i] * tangent_vector) // - c_bjs<u_bar_f * t_f, mu_d * t_f>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx

                      solution_star_mortar_temp[side](local_dof_indices[i]) +=
                      - (const_bjs * star_values[s][q_point] * tangent_vector[s]
                          * phi_eta[i] * tangent_vector) // - c_bjs<u_star_f * t_f, mu_d * t_f>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx
                    }
                    solution_star_mortar_temp[side][local_dof_indices[i]] +=
                      ((- phi_eta[i] * fe_face_values_mortar_biot.normal_vector(q_point) 
                        * lambda_p) // <phi_eta(x_q) * n_p, lambda_p> the extra negative sign as n_p = -n_f
                        + const_bjs * (lambda_d * tangent_vector) 
                        * (phi_eta[i] * tangent_vector)) // c_bjs<lambda_d * t_f,  phi_eta * t_f>
                      * fe_face_values_mortar_biot.JxW(q_point);    // dx
                    
                    // contribution from the coupling term in the 
                    // pressure lagrange multiplier test function
                    for (unsigned int s = 0; s < dim; ++s)
                    {
                      solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                        ((bar_values[s][q_point] * fe_face_values_mortar_biot.normal_vector(q_point)[s])
                          * phi_p[i])   // <u_bar * n, mu_p>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx

                      solution_star_mortar_temp[side](local_dof_indices[i]) +=
                        ((star_values[s][q_point] * fe_face_values_mortar_biot.normal_vector(q_point)[s])
                          * phi_p[i])   // <u_star * n, mu_p>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx
                    }
                    solution_star_mortar_temp[side][local_dof_indices[i]] +=
                    - ((lambda_d * fe_face_values_mortar_biot.normal_vector(q_point))
                      * phi_p[i]) // <lambda_d * n, mu_p> extra negative sign as n_p = -n_f
                      * fe_face_values_mortar_biot.JxW(q_point);    // dx
                  }
                }
              }
            }
        }
      }
      else // using Q(k) or DQ(k) for mortar space
      {
        FEValuesExtractors::Vector displacement(0);
        FEValuesExtractors::Scalar pressure(2*dim + 0.5*dim*(dim-1));

        std::vector<std::vector<double>> interface_values_disp(dim, std::vector<double>(n_face_q_points));
        std::vector<double> interface_values_press(n_face_q_points);
        
        for (const auto &cell : dof_handler_biot_mortar.active_cell_iterators())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
            if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7)  )
            {              
              //Need to reorder faces deal.ii iterators 
              //use a different ordering from freefem, 
              //they are ordered component wise 
              side = cell->face(face_n)->boundary_id() - 1;
              if (neighbor_type[side] == 2) //Stokes-Biot interface
              {
                fe_face_values_mortar_biot.reinit(cell, face_n);
                k_permeability.value_list (fe_face_values_mortar_biot.get_quadrature_points(), k_permeability_values);
                for (unsigned int s = 0; s < dim; ++s)
                {
                  solution_bar_function->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                            bar_values[s], s);
                  solution_star_function->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                            star_values[s], s);
                }
                for (unsigned int s = 0; s < dim; ++s)
                  interface_function[side]->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                                      interface_values_disp[s], s);
                interface_function[side]->value_list(fe_face_values_mortar_biot.get_quadrature_points(),
                                                    interface_values_press, 2*dim + 0.5*dim*(dim-1));

                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                {
                  // computes normal tangent vector in 2D, 
                  // assumes counterclockwise ordering of faces in deal.ii
                  if (dim == 2) 
                  {
                    tangent_vector[0] = - fe_face_values_mortar_biot.normal_vector(q_point)[1] 
                                           * get_normal_direction<dim>(side);
                    tangent_vector[1] = fe_face_values_mortar_biot.normal_vector(q_point)[0] 
                                         * get_normal_direction<dim>(side);
                  }
                  else if (dim == 3)
                  {
                    Assert(false, ExcMessage("Tangent vector not implemented for 3D yet"));
                  }
                  // compute the constant in the BJS condition, from mu*alpha_BJS*sqrt(K_j^-1)
                  const_bjs = alpha * 
                              std::sqrt(1/((k_permeability_values[q_point]*tangent_vector)
                                                        *tangent_vector));
                  // compute lambda_d at the quad point
                  // lambda_p at q is just interface_values_press at q, 
                  // since the mortar space is Q(k) or DQ(k)                                  
                  for (unsigned int s = 0; s < dim; ++s)   
                  {
                    lambda_d[s] = 0;
                    lambda_d[s] += interface_values_disp[s][q_point]; 
                  }

                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(),
                                local_dof_indices[k]) != interface_dofs[side].end())
                    {
                      phi_eta[k] = fe_face_values_mortar_biot[displacement].value (k, q_point);
                      phi_p[k] = fe_face_values_mortar_biot[pressure].value (k, q_point);
                    }
                    else
                    {
                      phi_eta[k] = 0.0;
                      phi_p[k] = 0.0;
                    }
                  }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // contribution from the coupling term in the 
                    // displacement lagrange multiplier test function
                    for (unsigned int s = 0; s < dim; ++s)
                    {
                      solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                       - (const_bjs * bar_values[s][q_point] * tangent_vector[s]
                          * phi_eta[i] * tangent_vector) // - c_bjs<u_bar_f * t_f, mu_d * t_f>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx

                      solution_star_mortar_temp[side](local_dof_indices[i]) +=
                      - (const_bjs * star_values[s][q_point] * tangent_vector[s]
                          * phi_eta[i] * tangent_vector) // - c_bjs<u_star_f * t_f, mu_d * t_f>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx
                    }
                    solution_star_mortar_temp[side][local_dof_indices[i]] +=
                      ((- phi_eta[i] * fe_face_values_mortar_biot.normal_vector(q_point) 
                        * interface_values_press[q_point]) // <phi_eta(x_q) * n_p * lambda_p> the extra negative sign as n_p = -n_f
                        + const_bjs * (lambda_d * tangent_vector) 
                        * (phi_eta[i] * tangent_vector)) // c_bjs<lambda_d * t_f,  phi_eta * t_f>
                      * fe_face_values_mortar_biot.JxW(q_point);    // dx
                    
                    // contribution from the coupling term in the 
                    // pressure lagrange multiplier test function
                    for (unsigned int s = 0; s < dim; ++s)
                    {
                      solution_bar_mortar_temp[side](local_dof_indices[i]) +=
                        ((bar_values[s][q_point] * fe_face_values_mortar_biot.normal_vector(q_point)[s])
                          * phi_p[i])   // <u_bar * n, mu_p>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx

                      solution_star_mortar_temp[side](local_dof_indices[i]) +=
                        ((star_values[s][q_point] * fe_face_values_mortar_biot.normal_vector(q_point)[s])
                          * phi_p[i])   // <u_star * n, mu_p>
                        * fe_face_values_mortar_biot.JxW(q_point);    // dx
                    }
                    solution_star_mortar_temp[side][local_dof_indices[i]] +=
                    - ((lambda_d * fe_face_values_mortar_biot.normal_vector(q_point))
                      * phi_p[i]) // <lambda_d * n, mu_p> extra negative sign as n_p = -n_f
                      * fe_face_values_mortar_biot.JxW(q_point);    // dx
                  }
                }                                    
              }
            }
        }
      }
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0 && neighbor_type[side] == 2) //Stokes-Biot interface
        {
          project_mortar<dim>(P_mortar[side + n_faces_per_cell],
                              dof_handler_biot_mortar,
                              solution_bar_mortar_temp[side],
                              quad,
                              constraints_mortar,
                              neighbors,
                              dof_handler_biot_mortar,
                              solution_bar_mortar[side],
                              interface_dofs[side],
                              side);
          project_mortar<dim>(P_mortar[side + n_faces_per_cell],
                              dof_handler_biot_mortar,
                              solution_star_mortar_temp[side],
                              quad,
                              constraints_mortar,
                              neighbors,
                              dof_handler_biot_mortar,
                              solution_star_mortar[side],
                              interface_dofs[side],
                              side);   

          // solution_bar_mortar[side] = solution_bar_mortar_temp[side];
          // solution_star_mortar[side] = solution_star_mortar_temp[side];
        } 
    }
  }

  // compute inner product in mortar space
  // <lambda, mu> on a particular face
  template <int dim>
  double
  MixedStokesProblem<dim>::inner_product_mortar(std::vector<std::vector<double>> &v_1,
                                                std::vector<std::vector<double>> &v_2,
                                                FEFaceValues<dim> &fe_face_values_mortar,
                                                FEFaceValues<dim> &fe_face_values_mortar_biot)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    double inner_product = 0.0;

    if (stokes_neighbor)
    {
      const unsigned int n_face_q_points = fe_face_values_mortar.get_quadrature().size();
      const unsigned int dofs_per_cell   = fe_mortar.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());

      //first index for dim components, 0, 1: vel, 2: pressure
      //second for value at each quad point
      std::vector<Tensor<1, dim>> interface_values_v1_flux(n_face_q_points, Tensor<1, dim>());
      std::vector<Tensor<1, dim>> interface_values_v2_flux(n_face_q_points, Tensor<1, dim>());

      const FEValuesExtractors::Vector velocity (0);

      BlockVector<double> vec_1(mortar_vec);
      BlockVector<double> vec_2(mortar_vec);
      vec_1 = 0;
      vec_2 = 0;

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0 && neighbor_type[side] == 1) //Stokes-Stokes interface
        {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            vec_1(interface_dofs[side][i]) = v_1[side][i];
            vec_2(interface_dofs[side][i]) = v_2[side][i];
          }
        }

      unsigned int side = 0;
      for (const auto &cell : dof_handler_mortar.active_cell_iterators())
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          if (cell->at_boundary(face_n) &&
            (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7) )
          {
            //Need to reorder faces deal.ii iterators 
            //use a different ordering from freefem, 
            //they are ordered component wise 
            side = cell->face(face_n)->boundary_id() - 1;

            if (neighbor_type[side] == 1) //Stokes-Stokes interface
            {
              fe_face_values_mortar.reinit(cell, face_n);
              fe_face_values_mortar[velocity].get_function_values(
                vec_1, interface_values_v1_flux);
              fe_face_values_mortar[velocity].get_function_values(
                vec_2, interface_values_v2_flux);                                 
              
              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
              {
                inner_product += 
                  interface_values_v1_flux[q_point] * (interface_values_v2_flux[q_point]) // (u1 * u2)
                  * fe_face_values_mortar.JxW(q_point);    // dx
              }
            }
          }
      }
    }
    if (biot_neighbor)
    {
      const unsigned int n_face_q_points = fe_face_values_mortar_biot.get_quadrature().size();
      const unsigned int dofs_per_cell   = fe_biot_mortar.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      // std::vector<double> phi_p(dofs_per_cell);
      // std::vector<Tensor<1, dim>> phi_eta(dofs_per_cell, Tensor<1, dim>());

      BlockVector<double> vec_1(mortar_vec_biot);
      BlockVector<double> vec_2(mortar_vec_biot);
      vec_1 = 0;
      vec_2 = 0;

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0 && neighbor_type[side] == 2) //Stokes-Biot interface
        {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            vec_1(interface_dofs[side][i]) = v_1[side][i];
            vec_2(interface_dofs[side][i]) = v_2[side][i];
          }
        }
      unsigned int side = 0;

      if (mortar_type < 2) // using BDM or RT for the mortar space
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

        for (const auto &cell : dof_handler_biot_mortar.active_cell_iterators())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
            if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7) )
            {
              //Need to reorder faces deal.ii iterators 
              //use a different ordering from freefem, 
              //they are ordered component wise 
              side = cell->face(face_n)->boundary_id() - 1;
              
              if (neighbor_type[side] == 2) //Stokes-Biot interface
              {
                fe_face_values_mortar_biot.reinit(cell, face_n);
                fe_face_values_mortar_biot[velocity].get_function_values(
                  vec_1, interface_values_v1_flux);
                fe_face_values_mortar_biot[velocity].get_function_values(
                  vec_2, interface_values_v2_flux);                                 
                
                for (unsigned int s = 0 ; s < dim; ++s)
                {
                  fe_face_values_mortar_biot[stresses[s]].get_function_values(
                    vec_1, interface_values_v1_stress[s]);
                  fe_face_values_mortar_biot[stresses[s]].get_function_values(
                    vec_2, interface_values_v2_stress[s]);
                }
                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                {
                  inner_product += 
                    (interface_values_v1_flux[q_point] * fe_face_values_mortar_biot.normal_vector(q_point)) *
                    (interface_values_v2_flux[q_point] * fe_face_values_mortar_biot.normal_vector(q_point)) // (u1 * u2)
                    * fe_face_values_mortar_biot.JxW(q_point);    // dx
                  for (unsigned int s = 0; s < dim; ++s)
                    inner_product += 
                      (interface_values_v1_stress[s][q_point] * fe_face_values_mortar_biot.normal_vector(q_point) *
                      interface_values_v2_stress[s][q_point] * fe_face_values_mortar_biot.normal_vector(q_point)) // (sigma1.n * sigma2.n)
                      * fe_face_values_mortar_biot.JxW(q_point);    // dx
                }
              }
            } 
        }
      }
      else
      {
        std::vector<Tensor<1, dim>> interface_values_v1_disp(n_face_q_points, Tensor<1, dim>());
        std::vector<Tensor<1, dim>> interface_values_v2_disp(n_face_q_points, Tensor<1, dim>());
        std::vector<double> interface_values_v1_press(n_face_q_points);
        std::vector<double> interface_values_v2_press(n_face_q_points);
        const FEValuesExtractors::Vector displacement(0);
        const FEValuesExtractors::Scalar pressure(2*dim + 0.5*dim*(dim-1));

        for (const auto &cell : dof_handler_biot_mortar.active_cell_iterators())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
            if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7) )
            {
              //Need to reorder faces deal.ii iterators 
              //use a different ordering from freefem, 
              //they are ordered component wise 
              side = cell->face(face_n)->boundary_id() - 1;
              
              if (neighbor_type[side] == 2) //Stokes-Biot interface
              {
                fe_face_values_mortar_biot.reinit(cell, face_n);
                fe_face_values_mortar_biot[displacement].get_function_values(
                  vec_1, interface_values_v1_disp);
                fe_face_values_mortar_biot[displacement].get_function_values(
                  vec_2, interface_values_v2_disp);
                fe_face_values_mortar_biot[pressure].get_function_values(
                  vec_1, interface_values_v1_press);
                fe_face_values_mortar_biot[pressure].get_function_values(
                  vec_2, interface_values_v2_press);
                
                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                {
                  inner_product +=
                    (interface_values_v1_disp[q_point] * 
                      interface_values_v2_disp[q_point]) // (lambda_d1 * lambda_d2)
                    * fe_face_values_mortar_biot.JxW(q_point);    // dx
                  inner_product +=
                    (interface_values_v1_press[q_point] * 
                      interface_values_v2_press[q_point]) // (lambda_p1 * lambda_p2)
                    * fe_face_values_mortar_biot.JxW(q_point);    // dx
                }
              }
            }
        }
      }
    }
    
    return inner_product;
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
                                    FEFaceValues<dim>                &fe_face_values_mortar_biot,
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
    interface_operator(interface_function,
                        solution_bar_function,
                        solution_star_function,
                        interface_data,
                        interface_data_send,
                        interface_data_receive,
                        fe_face_values,
                        fe_face_values_mortar,
                        fe_face_values_mortar_biot,
                        P_mortar,
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
        for(unsigned int i=0; i<=k_counter; ++i)
        {
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            for (unsigned int k = 0; k < q[side].size(); ++k)
              Q_tmp[side][k]=Q_side[side][i][k];
          h[i] = inner_product_mortar(Q_tmp, q, fe_face_values_mortar, fe_face_values_mortar_biot);

          // for(unsigned int j=0; j<q[side].size();++j)
          //   h[i]+=q[side][j]*Q_side[side][i][j];
        }
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
    // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    //   if (neighbors[side] >= 0)
    //     h_dummy+=vect_norm(q[side])*vect_norm(q[side]);

    h_dummy = inner_product_mortar(q, q, fe_face_values_mortar, fe_face_values_mortar_biot);
    
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
    solution_bar_mortar_temp.resize(n_faces_per_cell);
    solution_star_mortar_temp.resize(n_faces_per_cell);
    // double initial_val = 1.0; //initial guess for gmres
    std::vector<std::vector<double>> lambda(n_faces_per_cell);

    // [0] will hold factored L2 projection matrix for Stokes-Stokes interface
    // [1] will hold factored L2 projection matrix for Stokes-Biot interface
    std::vector<Projector::Projector<dim>> P_mortar(2*n_faces_per_cell, 
                                                    Projector::Projector<dim>(false));

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
    FEFaceValues<dim> fe_face_values_mortar_biot(fe_biot_mortar,
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

        if (neighbor_type[side] == 1) //Stokes-Stokes interface
        {
          interface_fe_function[side].reinit(mortar_vec);

          solution_bar_mortar[side].reinit(mortar_vec);
          solution_star_mortar[side].reinit(mortar_vec);

          solution_bar_mortar_temp[side].reinit(mortar_vec);
          solution_star_mortar_temp[side].reinit(mortar_vec);
        }
        else if (neighbor_type[side] == 2) //Stokes-Biot interface
        {
          interface_fe_function[side].reinit(mortar_vec_biot);

          solution_bar_mortar[side].reinit(mortar_vec_biot);
          solution_star_mortar[side].reinit(mortar_vec_biot);

          solution_bar_mortar_temp[side].reinit(mortar_vec_biot);
          solution_star_mortar_temp[side].reinit(mortar_vec_biot);
        }

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
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_fe_function[side].size(); ++i)
          AssertIsFinite(interface_fe_function[side][i]);
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      if (neighbor_type[side] == 2) //Stokes-Biot interface
        interface_function.emplace_back(
          std::make_unique<FEFunction>(dof_handler_biot_mortar, interface_fe_function[side]));
      else //Stokes-Stokes interface or otherwise enter a dummy cause you need 4 sides for the loop but only 2 or 3 are used
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
                     fe_face_values_mortar_biot,
                     interface_data,
                     P_mortar,
                     r,
                     r_norm,
                     r_b,
                     b_norm);
    //end -----------of calculating r-norm and b_norm------------------

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < r[side].size(); ++i)
          AssertIsFinite(r[side][i]);

    pcout << "r_norm: " << r_norm << " b_norm: " << b_norm << std::endl;
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
              fe_face_values_mortar_biot,
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


    //we can replace interface_data here and just add lambda(skip one step below)
    interface_data = lambda;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = interface_data[side][i];

    assemble_rhs_star(interface_function,
                      fe_face_values,
                      fe_face_values_mortar_biot);
    solve_star();
    solution.reinit(solution_bar);
    solution = solution_bar;
    solution.sadd(1.0, solution_star);
    pcout<<"finished local_gmres"<<"\n";
  }

  template <int dim>
  void
  MixedStokesProblem<dim>::local_cg()
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data(n_faces_per_cell);
    interface_fe_function.resize(n_faces_per_cell);
    solution_bar_mortar.resize(n_faces_per_cell);
    solution_star_mortar.resize(n_faces_per_cell);
    solution_bar_mortar_temp.resize(n_faces_per_cell);
    solution_star_mortar_temp.resize(n_faces_per_cell);
    // double initial_val = 0.0; //initial guess for gmres
    std::vector<std::vector<double>> lambda(n_faces_per_cell);

    std::vector<Projector::Projector<dim>> P_mortar(2*n_faces_per_cell, 
                                                    Projector::Projector<dim>(false));

    // Extra for projections from mortar to fine grid and RHS assembly
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);

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
    FEFaceValues<dim> fe_face_values_mortar_biot(fe_biot_mortar,
                                          quad,
                                          update_values | update_normal_vectors |
                                            update_quadrature_points |
                                            update_JxW_values);                        

    // CG structures and parameters
    std::vector<double> alpha_side(n_faces_per_cell, 0),
    alpha_side_d(n_faces_per_cell, 0), beta_side(n_faces_per_cell, 0),
    beta_side_d(n_faces_per_cell, 0);
    std::vector<double> alpha(2, 0), beta(2, 0);                                  

    std::vector<std::vector<double>> r(n_faces_per_cell); //to be deleted probably: p?
    std::vector<std::vector<double>> r_b(n_faces_per_cell); //RHS
    std::vector<std::vector<double>> p(n_faces_per_cell); //search direction
    std::vector<std::vector<double>> Ap(n_faces_per_cell); // A*p matrix vector product       
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        interface_data_receive[side].resize(interface_dofs[side].size(),
                                            0);
        interface_data_send[side].resize(interface_dofs[side].size(), 0);
        interface_data[side].resize(interface_dofs[side].size(), initial_val);

        if (neighbor_type[side] == 1) //Stokes-Stokes interface
        {
          interface_fe_function[side].reinit(mortar_vec);

          solution_bar_mortar[side].reinit(mortar_vec);
          solution_star_mortar[side].reinit(mortar_vec);

          solution_bar_mortar_temp[side].reinit(mortar_vec);
          solution_star_mortar_temp[side].reinit(mortar_vec);
        }
        else if (neighbor_type[side] == 2) //Stokes-Biot interface
        {
          interface_fe_function[side].reinit(mortar_vec_biot);

          solution_bar_mortar[side].reinit(mortar_vec_biot);
          solution_star_mortar[side].reinit(mortar_vec_biot);

          solution_bar_mortar_temp[side].reinit(mortar_vec_biot);
          solution_star_mortar_temp[side].reinit(mortar_vec_biot);
        }

        Ap[side].resize(interface_dofs[side].size(), 0);
        lambda[side].resize(interface_dofs[side].size(), initial_val);

        r[side].resize(interface_dofs[side].size(), 0);
        r_b[side].resize(interface_dofs[side].size(), 0);
      }

    std::vector<std::unique_ptr<FEFunction>> interface_function;
    std::unique_ptr<FEFunction> solution_bar_function;
    std::unique_ptr<FEFunction> solution_star_function;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
    {
      if (neighbor_type[side] == 2) //Stokes-Biot interface
        interface_function.emplace_back(
          std::make_unique<FEFunction>(dof_handler_biot_mortar, interface_fe_function[side]));
      else //Stokes-Stokes interface or otherwise enter a dummy cause you need 4 sides for the loop but only 2 or 3 are used
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
                     fe_face_values_mortar_biot,
                     interface_data,
                     P_mortar,
                     r,
                     r_norm,
                     r_b,
                     b_norm);
    //end -----------of calculating r-norm and b_norm------------------

    p = r;

    double normB    = 0;
    double normRold = 0;

    unsigned int iteration_counter = 0;
    gmres_iteration = 0;
    while (iteration_counter < maxiter)
    {
      iteration_counter++;
      interface_data = p;

      // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      //   for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
      //     interface_fe_function[side][interface_dofs[side][i]] = 
      //       interface_data[side][i];

      gmres_iteration++;

      // computing Ap = A*p
      interface_operator(interface_function,
                          solution_bar_function,
                          solution_star_function,
                          interface_data,
                          interface_data_send,
                          interface_data_receive,
                          fe_face_values,
                          fe_face_values_mortar,
                          fe_face_values_mortar_biot,
                          P_mortar,
                          Ap);

      // CG algorithm steps to compute alpha, beta, update r, p, lambda
      // for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      //   if (neighbors[side] >= 0)
      //     {
      //       alpha_side[side]   = 0;
      //       alpha_side_d[side] = 0;
      //       beta_side[side]    = 0;
      //       beta_side_d[side]  = 0;

      //       // From Ap compute alpha
      //       for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
      //       { 
      //         alpha_side[side] += r[side][i] * r[side][i];
      //         alpha_side_d[side] += p[side][i] * Ap[side][i];
      //       }
      //     }

      // // Fancy some lambdas, huh?
      // std::for_each(alpha_side.begin(), alpha_side.end(), [&](double n) {
      //   alpha[0] += n;
      // });
      // std::for_each(alpha_side_d.begin(), alpha_side_d.end(), [&](double n) {
      //   alpha[1] += n;
      // });
      std::vector<double> alpha_buffer(2, 0);

      alpha[0] = inner_product_mortar(r, r, fe_face_values_mortar, fe_face_values_mortar_biot);
      alpha[1] = inner_product_mortar(p, Ap, fe_face_values_mortar, fe_face_values_mortar_biot);

      MPI_Allreduce(&alpha[0], //sending this data
                    &alpha_buffer[0], //receiving the result here
                    2, //number of elements in alpha and alpha_buffer = 1+1
                    MPI_DOUBLE, //type of each element
                    MPI_SUM, //adding all elements received
                    mpi_communicator);


      alpha = alpha_buffer;

      if (gmres_iteration == 1)
        normB = alpha[0];

      normRold = alpha[0];

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        {
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              lambda[side][i] += (alpha[0] * p[side][i]) / alpha[1];
              r[side][i] -= (alpha[0] * Ap[side][i]) / alpha[1];
            }

          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            beta_side[side] += r[side][i] * r[side][i];
        }

      pcout << "\r  ..." << gmres_iteration
            << " iterations completed, (residual = " << fabs(alpha[0] / normB)
            << ")..." << std::flush;
      residual = fabs(alpha[0] / normB);
      
      // residual_vector.push_back(residual);
      // Exit criterion
      if (fabs(alpha[0]) / normB < tol)
        {
          residual = fabs(alpha[0] / normB);
          pcout << "\n  CG converges in " << gmres_iteration << " iterations!\n";
          break;
        }
      
      if (std::isnan(fabs(alpha[0]) / normB))
      {
        residual = fabs(alpha[0] / normB);
        pcout << "\n  residual is not a number! at " << gmres_iteration << " iteration\n";
        break;
      }
      
      if ((fabs(alpha[0]) / normB >= tol) && gmres_iteration == maxiter)
      {
        residual = fabs(alpha[0] / normB);
        pcout << "\n  CG exceeds maxiter = " << gmres_iteration << " iterations!\n";
        break;
      }

      if(iteration_counter == interface_dofs_size)
      {
        pcout << "\n  CG doesn't converge after  " << interface_dofs_size << " iterations!\n";
        break;
      }

      // std::for_each(beta_side.begin(), beta_side.end(), [&](double n) {
      //   beta[0] += n;
      // });

      beta[0] = inner_product_mortar(r, r, fe_face_values_mortar, fe_face_values_mortar_biot);
      double beta_buffer = 0;

      MPI_Allreduce(
        &beta[0], &beta_buffer, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

      beta[0] = beta_buffer;
      beta[1] = normRold;

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      {
        if (neighbors[side] >= 0)
          for (unsigned int i = 0; i < interface_data[side].size(); ++i)
            p[side][i] = r[side][i] + (beta[0] / beta[1]) * p[side][i];

        // interface_data_receive[side].resize(interface_dofs[side].size(), 0);
        // interface_data_send[side].resize(interface_dofs[side].size(), 0);
        std::fill(interface_data_receive[side].begin(), interface_data_receive[side].end(), 0.0);
        std::fill(interface_data_send[side].begin(), interface_data_send[side].end(), 0.0);

        // Ap.resize(n_faces_per_cell);
        std::fill(Ap[side].begin(), Ap[side].end(), 0.0);
      }
    }

    //we can replace interface_data here and just add lambda(skip one step below)
    interface_data = lambda;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        interface_fe_function[side][interface_dofs[side][i]] = interface_data[side][i];

    assemble_rhs_star(interface_function,
                      fe_face_values,
                      fe_face_values_mortar_biot);
    solve_star();
    solution.reinit(solution_bar);
    solution = solution_bar; // u = u_bar
    solution.sadd(1.0, solution_star); // u = 1*u+u_star
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
      convergence_table_l2_total.evaluate_convergence_rates(
            "gmres_iter", ConvergenceTable::reduction_rate_log2);

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
