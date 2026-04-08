/* ---------------------------------------------------------------------
 * Utilities:
 *  - DD & mortar related - mesh subdivision, mortar projection
 *  - Printing the convergence tables
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 *         Manraj Singh Ghumman, University of Pittsburgh, 2023-2025
 */
#ifndef STOKES_BIOT_DD_UTILITIES_H
#define STOKES_BIOT_DD_UTILITIES_H

#include "projector.h"
#include <map>
#include <deal.II/lac/block_vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/convergence_table.h>

namespace dd_utilities
{
  using namespace dealii;

  // Elasticity problem related utilities
  // create rank-2 tensor
  template <int dim>
  Tensor<2,dim> make_tensor(const std::vector<Tensor<1,dim> > &vec, Tensor<2, dim> &res)
  {
    for(int row=0;row<dim;++row)
      for(int col=0;col<dim;++col)
        res[row][col] = vec[row][col];

    return res;
  }

  // create asymmetry tensor
  template <int dim, int tensor_dim>
  void make_asymmetry_tensor(const std::vector<Tensor<1,dim> > &vec, Tensor<1, tensor_dim> &res)
  {
    Tensor<2,dim> mat;
    make_tensor(vec, mat);

    switch(dim)
    {
      case 2:
        res[0] = mat[0][1] - mat[1][0];
        break;
      case 3:
        res[0] = mat[2][1] - mat[1][2];
        res[1] = mat[0][2] - mat[2][0];
        res[2] = mat[1][0] - mat[0][1];
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // create compliance tensor
  template <int dim>
  void compliance_tensor(const std::vector<Tensor<1,dim> > &vec, const double mu, const double lambda, Tensor<2,dim> &res)
  {
    double trace=0.;
    for (unsigned int i=0;i<dim;++i)
      trace += vec[i][i];

    for (unsigned int row=0;row<dim;++row)
      for (unsigned int col=0;col<dim;++col)
        if (row == col)
          res[row][col] = 1/(2*mu)*(vec[row][col] - lambda/(2*mu + dim*lambda)*trace);
        else
          res[row][col] = 1/(2*mu)*(vec[row][col]);
  }


  // Create compliance tensor for pressure variables
  template <int dim>
  void compliance_tensor_pressure(const double pres, const double mu, const double lambda, Tensor<2,dim> &res)
  {
    for(unsigned int i=0;i<dim;i++)
      res[i][i] = ((2*mu+(dim-2)*lambda)*pres)/((2*mu+dim*lambda)*2*mu);
  }

  // Domain decomposition related utilities
  
  /*
    Function to get the true normal vector orientation 
    (in deal.II they are all positive by default)
    in dd two domains sharing a face will have different
    signs of test functions on the interface but this
    will not show up in the normal vector from deal.II
    so we need to manually set it here
  */
  template <int dim>
  double
  get_normal_direction(const unsigned int &face)
  {
    std::vector<double> normals(6);
    normals[0] = -1;
    normals[1] = 1;
    normals[2] = 1;
    normals[3] = -1;
    normals[4] = -1;
    normals[5] = 1;

    return normals[face];
  }

  // Split the number into two divisors closest to sqrt (in order to partition
  // mesh into subdomains)
  template <int dim>
  void
  find_2_divisors(const unsigned int &n_proc, std::vector<unsigned int> &n_dom)
  {
    double tmp = floor(sqrt(n_proc));
    while (true)
    {
      if (fmod(double(n_proc), tmp) == 0)
      {
        n_dom[0] = n_proc / tmp; // x
        n_dom[1] = tmp;          // y
        break;
      }
      else
      {
        tmp -= 1;
      }
    }
  }

  // Split number into three divisors (Try to make it recursive better)
  template <int dim>
  void
  find_divisors(const unsigned int &n_proc, std::vector<unsigned int> &n_dom)
  {
    Assert(dim == 2 || dim == 3, ExcNotImplemented());

    if (dim == 2)
      find_2_divisors<dim>(n_proc, n_dom);
    else if (dim == 3)
    {
      double tmp = floor(pow(n_proc, 1.0 / 3.0));
      while (true)
      {
        if (fmod(static_cast<double>(n_proc), tmp) == 0)
        {
          std::vector<unsigned int> two_divisors(2);
          unsigned int              two_proc = n_proc / tmp;
          find_2_divisors<dim>(two_proc, two_divisors);

          n_dom[0] = two_divisors[0];
          n_dom[1] = two_divisors[1];
          n_dom[2] = tmp;
          break;
        }
        else
        {
          tmp -= 1;
        }
      }
    }
  }

  // Compute the lower left and upper right coordinate of a block
  template <int dim>
  void
  get_subdomain_coordinates(const unsigned int &             this_mpi,
                            const std::vector<unsigned int> &n_doms,
                            const std::vector<double> &      dims,
                            Point<dim, double> &                     p1,
                            Point<dim, double> &                     p2)
  {
    switch (dim)
      {
        case 2:
          p1[0] = (this_mpi % n_doms[0]) * dims[0];
          p1[1] = dims[1] * (floor(double(this_mpi) / double(n_doms[0])));
          p2[0] = (this_mpi % n_doms[0]) * dims[0] + dims[0];
          p2[1] =
            dims[1] * (floor(double(this_mpi) / double(n_doms[0]))) + dims[1];
          break;

        case 3:
          p1[0] = (this_mpi % n_doms[0]) * dims[0];
          p1[1] = dims[1] * (floor(double(this_mpi % (n_doms[0] * n_doms[1])) /
                                   double(n_doms[0])));
          p1[2] =
            dims[2] * (floor(double(this_mpi) / double(n_doms[0] * n_doms[1])));
          p2[0] = (this_mpi % n_doms[0]) * dims[0] + dims[0];
          p2[1] = dims[1] * (floor(double(this_mpi % (n_doms[0] * n_doms[1])) /
                                   double(n_doms[0]))) +
                  dims[1];
          p2[2] = dims[2] *
                    (floor(double(this_mpi) / double(n_doms[0] * n_doms[1]))) +
                  dims[2];
          break;

        default:
          Assert(false, ExcNotImplemented());
          break;
      }
  }

  // Find neighboring subdomains
  // output: modifies the vector neighbours and outputs the neighbours 
  //         relative to this_mpi
  template <int dim>
  void
  find_neighbors(const unsigned int &             this_mpi,
                 const std::vector<unsigned int> &n_doms,
                 std::vector<std::vector<unsigned int>> & dd_info_matrix,
                 std::vector<int>            &neighbors,
                 std::vector<unsigned int>   &neighbor_type)
  {
    Assert(neighbors.size() == 2.0 * dim,
           ExcDimensionMismatch(neighbors.size(), 2.0 * dim));
    Assert(neighbor_type.size() == 2.0 * dim,
           ExcDimensionMismatch(neighbor_type.size(), 2.0 * dim));
    switch (dim)
      {
        case 2:
          if (n_doms[0] == 1 && n_doms[1] == 1) // only one domain
          {
            neighbors[0] = -1;
            neighbors[1] = -1;
            neighbors[2] = -1;
            neighbors[3] = -1;
          }
          else if (n_doms[0] == 1) // 1 x n setup
          {
            if (this_mpi == 0) // only one line
            {
              neighbors[0] = -1;
              neighbors[1] = -1;
              neighbors[2] = this_mpi + 1;
              neighbors[3] = -1;
            }
            else if (this_mpi == n_doms[0] - 1) // only one line
            {
              neighbors[0] = 0;
              neighbors[1] = -1;
              neighbors[2] = -1;
              neighbors[3] = - 1;
            }
            else
            {
              neighbors[0] = this_mpi - 1;
              neighbors[1] = -1;
              neighbors[2] = this_mpi + 1;
              neighbors[3] = - 1;
            }
          }
          else if (this_mpi % n_doms[0] == 0) // x = a line
          {
            neighbors[0] = this_mpi - n_doms[0];
            neighbors[1] = this_mpi + 1;
            neighbors[2] = this_mpi + n_doms[0];
            neighbors[3] = -1;
          }
          else if ((this_mpi + 1) % n_doms[0] ==
                   0 /*&& (this_mpi + 1) >= n_doms[0]*/) // x = b line
          {
            neighbors[0] = this_mpi - n_doms[0];
            neighbors[1] = -1;
            neighbors[2] = this_mpi + n_doms[0];
            neighbors[3] = this_mpi - 1;
          }
          else
          {
            neighbors[0] = this_mpi - n_doms[0];
            neighbors[1] = this_mpi + 1;
            neighbors[2] = this_mpi + n_doms[0];
            neighbors[3] = this_mpi - 1;
          }
          break;

        case 3:
          if (this_mpi % (n_doms[0] * n_doms[1]) == 0) // corner at origin
            {
              neighbors[0] = -1;
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = -1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi - n_doms[0] + 1) % (n_doms[0] * n_doms[1]) ==
                   0) // corner at x=a
            {
              neighbors[0] = -1;
              neighbors[1] = -1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi + n_doms[0]) % (n_doms[0] * n_doms[1]) ==
                   0) // corner at y=a
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = -1;
              neighbors[3] = -1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi + 1) % (n_doms[0] * n_doms[1]) ==
                   0) // corner at x=y=a
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = -1;
              neighbors[2] = -1;
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if (this_mpi % n_doms[0] == 0) // plane x = a
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = -1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi + 1) % n_doms[0] == 0) // plane x = b
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = -1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if (this_mpi % (n_doms[0] * n_doms[1]) <
                   n_doms[0]) // plane y = a
            {
              neighbors[0] = -1;
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if (this_mpi % (n_doms[0] * n_doms[1]) >
                   n_doms[0] * n_doms[1] - n_doms[0]) // plane y = b
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = -1;
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          break;

        default:
          Assert(false, ExcNotImplemented());
          break;
      }


    for (unsigned int i = 0; i < neighbors.size(); ++i)
    {
      if (neighbors[i] < 0)
        neighbors[i] = -1;
      else if (dim == 2 && neighbors[i] >= int(n_doms[0] * n_doms[1]))
        neighbors[i] = -1;
      else if (dim == 3 &&
                neighbors[i] >= int(n_doms[0] * n_doms[1] * n_doms[2]))
        neighbors[i] = -1;
    }
    
    if (dim == 2) {
      if (neighbors[0] < 0) {neighbor_type[0] = 0;}
      else
        neighbor_type[0] = dd_info_matrix[n_doms[1] - neighbors[0] / n_doms[0] - 1][neighbors[0] % n_doms[0]];
      if (neighbors[1] < 0) {neighbor_type[1] = 0;}
      else
        neighbor_type[1] = dd_info_matrix[n_doms[1] - neighbors[1] / n_doms[0] - 1][neighbors[1] % n_doms[0]];
      if (neighbors[2] < 0) {neighbor_type[2] = 0;}
      else
        neighbor_type[2] = dd_info_matrix[n_doms[1] - neighbors[2] / n_doms[0] - 1][neighbors[2] % n_doms[0]];
      if (neighbors[3] < 0) {neighbor_type[3] = 0;}
      else
        neighbor_type[3] = dd_info_matrix[n_doms[1] - neighbors[3] / n_doms[0] - 1][neighbors[3] % n_doms[0]];
    } else {
      Assert(false, ExcMessage("Dimension = 3 not  neighbors_type not yet supported"));
    }
  }

  template <int dim, typename Number, int spacedim = dim>
  void
  mark_interface_faces(const Triangulation<dim, spacedim> &tria,
                       const std::vector<int> &  neighbors,
                       const Point<dim, double> &        p1,
                       const Point<dim, double> &        p2,
                       std::vector<Number> &     faces_per_interface)
  {
    Assert(faces_per_interface.size() == neighbors.size(),
           ExcDimensionMismatch(faces_per_interface.size(), neighbors.size()));

    // Label boundaries
    // On unit hypercube for example:
    //  1 - plane y=0, 2 - plane x=1, 3 - plane y=1, 4 - plane x=0, 5 - plane
    //  z=0, 6 - plane z=1 for interfaces, 0 - for outside

    typename Triangulation<dim>::cell_iterator cell, endc;
    cell = tria.begin_active(), endc = tria.end();

    for (; cell != endc; ++cell)
      for (unsigned int face_number = 0;
           face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          // If left boundary of the subdomain in 2d or
          if (std::fabs(cell->face(face_number)->center()(0) - p1[0]) < 1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[3] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(4);
                  faces_per_interface[3] += 1;
                }
            }
          // If bottom boundary of the subdomain
          else if (std::fabs(cell->face(face_number)->center()(1) - p1[1]) <
                   1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[0] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(1);
                  faces_per_interface[0] += 1;
                }
            }
          // If right boundary of the subdomain
          else if (std::fabs(cell->face(face_number)->center()(0) - p2[0]) <
                   1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[1] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(2);
                  faces_per_interface[1] += 1;
                }
            }
          // If top boundary of the subdomain
          else if (std::fabs(cell->face(face_number)->center()(1) - p2[1]) <
                   1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[2] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(3);
                  faces_per_interface[2] += 1;
                }
            }
          else if (dim == 3 && std::fabs(cell->face(face_number)->center()(2) -
                                         p1[2]) < 1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[4] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(5);
                  faces_per_interface[4] += 1;
                }
            }
          else if (dim == 3 && std::fabs(cell->face(face_number)->center()(2) -
                                         p2[2]) < 1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[5] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(6);
                  faces_per_interface[5] += 1;
                }
            }
        }
  }

  template <int dim>
  void
  project_mortar(Projector::Projector<dim> &proj,
                 const DoFHandler<dim>     &dof1,
                 BlockVector<double>       &in_vec,
                 const Quadrature<dim - 1> &quad,
                 AffineConstraints<double> &constraints,
                 const std::vector<int>    &neighbors,
                 const DoFHandler<dim>     &dof2,
                 BlockVector<double>       &out_vec,
                 const std::vector<unsigned int> &interface_dofs,
                 const unsigned int &side)
  {
    out_vec = 0;

    Functions::FEFieldFunction<dim, BlockVector<double>>
                                              fe_interface_data(dof1, in_vec);
    std::map<types::global_dof_index, double> boundary_values_velocity;

    std::map<types::boundary_id, const Function<dim, double> *>  boundary_functions_velocity;

    constraints.clear();

    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      if (face == side)
        boundary_functions_velocity[face + 1] = &fe_interface_data;
    
    // std::cout << "n_boundary_dofs = "
    //       << dof2.n_boundary_dofs(boundary_functions_velocity)
    //       << std::endl;
    proj.project_boundary_values(in_vec,
                                 dof2,
                                 boundary_functions_velocity,
                                 interface_dofs,
                                 side,
                                 quad,
                                 constraints);

    constraints.close();
    constraints.distribute(out_vec);
  }

  template <int dim>
  void
  compute_interface_dofs_size(std::vector<unsigned int> &interface_dofs_total,
                              const MPI_Comm      &mpi_communicator,
                              const unsigned int  &this_mpi,
                              const unsigned int  &n_processes,
                              std::vector<unsigned int>   &n_domains,
                              int                 &interface_dofs_size)
  {
    interface_dofs_size = 0;
    int tmp = 0;

    // If dim == 2, we can use MPI_Allreduce to get the total size of the interface dofs
    // If dim == 3, this is not implemented yet
    if (dim == 2)
    {
      unsigned int count = 0;
      unsigned int x, y;
      x = this_mpi % n_domains[0];
      y = this_mpi / n_domains[0];
      if (y == 0)
      {
        if (this_mpi % 2 == 0)
          tmp = 0;
        else
          tmp = interface_dofs_total.size();
      }
      else
      {
        if (this_mpi % 2 == 1)
          tmp = 0;
        else
          tmp = interface_dofs_total.size();
      }

      // std::cout << "this_mpi = " << this_mpi << " tmp = " << tmp << std::endl;
      MPI_Allreduce(&tmp, //sending this data
                    &interface_dofs_size, //receiving the result here
                    1, //number of elements in alpha and alpha_buffer = 1+1
                    MPI_INT, //type of each element
                    MPI_SUM, //adding all elements received
                    mpi_communicator);
    }
    else
      throw std::runtime_error("dim = 3 not yet implemented!");
  }

} // namespace dd_utilities

#endif // STOKES_BIOT_DD_UTILITIES_H
