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
    MPI_Comm mpi_communicator_biot,
    const bool mortar_flag,
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
      stokes_neighbor(false),
      biot_neighbor(false),
      mortar_flag(mortar_flag),
      mortar_degree(mortar_degree),
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
      pcout (std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator_biot) == 0))
      // computing_timer (mpi_communicator,
      //                   pcout,
      //                   TimerOutput::summary,
      //                   TimerOutput::wall_times)
  {
    interpret_parameters();
    BiotParameterReader param(prm_fxn);
    param.read_parameters("../test function/biot.prm");
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
    double s0 = 1.0;
    double alpha = 1.0;         
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
      std::vector<std::vector<Tensor<1,dim>>> phi_s(dofs_per_cell, std::vector<Tensor<1,dim> > (dim));
      std::vector<Tensor<1,dim>> div_phi_s(dofs_per_cell);
      std::vector<Tensor<1,dim>> phi_d(dofs_per_cell);
      std::vector<Tensor<1,rotation_dim>> phi_r(dofs_per_cell);

      Tensor<2,dim> sigma_i, asigma_j, apId_j;
      Tensor<1,rotation_dim> asym_i, asym_j;

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

          for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
            phi_r[k][r_i] = fe_values[rotations[r_i]].value (k, q);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const double mu = lame_parameters_values[q][1];
          const double lambda = lame_parameters_values[q][0];

          // compliance_tensor(phi_s[j], mu, lambda, asigma);
          // compliance_tensor_pressure(phi_p[j], mu, lambda, apId);
          make_asymmetry_tensor(phi_s[i], asym_i);
          make_tensor(phi_s[i], sigma_i);

          for (unsigned int j=0; j<dofs_per_cell; ++j)// i is the test function, multiplied from left
          {
            // make_tensor(phi_s[j], sigma);
            make_asymmetry_tensor(phi_s[j], asym_j);
            compliance_tensor(phi_s[j], mu, lambda, asigma_j);
            compliance_tensor_pressure(phi_p[j], mu, lambda, apId_j);

            local_matrix(i, j) += (phi_u[i] * k_inverse_values[q] * phi_u[j] - phi_p[j] * div_phi_u[i]                 // Darcy law
                                    + div_phi_u[j] * phi_p[i] + s0*phi_p[i]*phi_p[j] + alpha*trace(asigma_j)*phi_p[i]  // Momentum
                                    + alpha * alpha * trace(apId_j)*phi_p[i]
                                    + scalar_product(asigma_j, sigma_i) + alpha*scalar_product(apId_j, sigma_i)        // Mixed elasticity eq-ns
                                    + scalar_product(phi_d[j], div_phi_s[i])  + scalar_product(phi_d[i], div_phi_s[j])
                                    + scalar_product<1, rotation_dim>(phi_r[i], asym_j)
                                    + scalar_product<1, rotation_dim>(phi_r[j], asym_i) )
                                  * fe_values.JxW(q);
          }
          local_rhs(i) += phi_d[i] * rhs_vec_val // (f_p, v_s)
                          + phi_p[i] * rhs_scalar.value(fe_values.quadrature_point(q)) // (q, w_p)
                          * fe_values.JxW(q);
        }
      }

      //Entering boundary integrals
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary())
        { 
          fe_face_values.reinit(cell, face);
          if (face->boundary_id() == 0)//Entering Dirichlet Condition
          {
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              // Get boundary values displacement
              Tensor<1,dim> gd_disp; 
              std::vector<Tensor<1,dim>> tau_n_i(dofs_per_cell);
              for (unsigned int i = 0; i < dim; ++i)
                gd_disp[i] = boundary_values_dir_disp.value(fe_face_values.quadrature_point(q), i);
              
              // Evaluate test functions  
              for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                phi_u[k] = fe_values[velocity].value (k, q);
                for (unsigned int s_i=0; s_i<dim; ++s_i)
                  tau_n_i[k][s_i] = fe_values[stresses[s_i]].value (k, q)
                                    * fe_face_values.normal_vector(q);
              }
          
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                local_rhs(i) += (- phi_u[i] * fe_face_values.normal_vector(q) // -<u.n, g_p^D_p>
                                  * boundary_values_dir_press.value(fe_face_values.quadrature_point(q))
                                  + tau_n_i[i] * gd_disp )     // <tau.n, g_p^D_d>
                                  * fe_face_values.JxW(q); 
                                
              }
            }
          }
        }
      }

      cell->get_dof_indices (local_dof_indices);
      // Add local contributions to global system
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        system_rhs_bar(local_dof_indices[i]) += local_rhs(i);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                            local_dof_indices[j],
                            local_matrix(i,j));
      }
    }
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
    A_direct.initialize (system_matrix);
    A_direct.vmult(solution_bar, system_rhs_bar);
  }

  template <int dim>
  void MixedBiotProblem<dim>::solve_star ()
  {
    // TimerOutput::Scope t(computing_timer, "Solve star");
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
  MixedBiotProblem<dim>::output_results(const unsigned int cycle) const
  {
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
      + ".vtk");
    data_out.write_vtk(output);
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
    // prm_fxn.enter_subsection("Exact solution stress");
    // exact_solution.exact_solution_stress_data.parse_parameters(prm_fxn);
    // prm_fxn.leave_subsection();
    prm_fxn.enter_subsection("Exact solution displacement");
    exact_solution.exact_solution_displacement_data.parse_parameters(prm_fxn);
    prm_fxn.leave_subsection();
    // prm_fxn.enter_subsection("Exact solution rotation");
    // exact_solution.exact_solution_rotation_data.parse_parameters(prm_fxn);
    // prm_fxn.leave_subsection();

    // Vectors to temporarily store cellwise errros
    Vector<double> cellwise_errors (triangulation.n_active_cells());
    Vector<double> cellwise_norms (triangulation.n_active_cells());

    QTrapezoid<1>      q_trapez;
    QIterated<dim>  quadrature(q_trapez,degree+2);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution.size());

    // Pressure error and norm
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

    // Velocity L2 error and norm
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

    // Displacement error and norm
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

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();
    convergence_table_l2.add_value("cells", n_active_cells);
    convergence_table_l2.add_value("h", "1/"+Utilities::int_to_string(std::pow(2,cycle)));
    convergence_table_l2.add_value("dofs", n_dofs);
    convergence_table_l2.add_value("||u-u_h||_L2", u_l2_error/u_l2_norm);
    convergence_table_l2.add_value("||p-p_h||_L2", p_l2_error/p_l2_norm);
    convergence_table_l2.add_value("||d-d_h||_L2", d_l2_error/d_l2_norm);
    convergence_table_l2.add_value("residual", residual);
  }

  template <int dim>
  void MixedBiotProblem<dim>::print_individual_convg_table_l2()
  {
    convergence_table_l2.set_precision("residual", 3);
    convergence_table_l2.set_precision("||u-u_h||_L2", 3);
    convergence_table_l2.set_precision("||p-p_h||_L2", 3);
    convergence_table_l2.set_precision("||d-d_h||_L2", 3);

    convergence_table_l2.set_scientific("residual", true);
    convergence_table_l2.set_scientific("||u-u_h||_L2", true);
    convergence_table_l2.set_scientific("||p-p_h||_L2", true);
    convergence_table_l2.set_scientific("||d-d_h||_L2", true);

    convergence_table_l2.set_tex_caption("cells", "\\ cells");
    convergence_table_l2.set_tex_caption("dofs", "\\# dofs");
    convergence_table_l2.set_tex_caption("residual", "residual");
    convergence_table_l2.set_tex_caption("||u-u_h||_L2", "$||u_p-u_{ph}||_{L^2}$");
    convergence_table_l2.set_tex_caption("||p-p_h||_L2", "$||p_p-p_{ph}||_{L^2}$");
    convergence_table_l2.set_tex_caption("||d-d_h||_L2", "$||u_s-u_{sh}||_{L^2}$");

    convergence_table_l2.evaluate_convergence_rates("||u-u_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);
    convergence_table_l2.evaluate_convergence_rates("||p-p_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);
    convergence_table_l2.evaluate_convergence_rates("||d-d_h||_L2",
                                              ConvergenceTable::reduction_rate_log2);

    std::string conv_filename = "../output/convg_tables/biot_table_individual_l2" + std::string(".tex");

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
        convergence_table_l2.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table_l2.write_tex(table_file, false);
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

    solve_bar();
    if(n_processes == 1)
      solution = solution_bar;
  }
  template class MixedBiotProblem<2>;
  // template class MixedBiotProblem<3>;
}
