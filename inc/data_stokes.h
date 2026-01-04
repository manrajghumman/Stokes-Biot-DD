/* ---------------------------------------------------------------------
 * Function representing true solution. (read from parameter file
 *                           and stored for use in error computation)
 * RHS, physical parameters, boundary conditions are directly read from 
 * parameter file and used.
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Singh Ghumman, University of Pittsburgh, 2023 - 2024
 */

#ifndef STOKES_MFE_DATA_H
#define STOKES_MFE_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor_function.h>

namespace mfe_stokes
{
  using namespace dealii;

  template <int dim>
    class ExactSolution : public Function<dim>
  {
    // private:
    // ParameterHandler &prm;
    public:
      Functions::ParsedFunction<dim> exact_solution_val_data;
      Functions::ParsedFunction<dim> exact_solution_grad_val_data;
      ExactSolution(ParameterHandler &);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &  value) const override;
      virtual void vector_gradient(	const Point< dim > & 	p,
                    std::vector< Tensor< 1, dim> > & 	gradients)	const override;
  };

  template <int dim>
  ExactSolution<dim>::ExactSolution(ParameterHandler &) : 
    Function<dim>(dim + 1),
    // prm(param),
    exact_solution_val_data(dim + 1),
    exact_solution_grad_val_data((dim + 1) * dim)
  {}


  template <int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &  values) const
    {
      AssertDimension(values.size(), dim + 1);

      values(0) = exact_solution_val_data.value(p, 0);
      values(1) = exact_solution_val_data.value(p, 1);
      values(2) = exact_solution_val_data.value(p, 2);
    }
  
  template <int dim>
    void ExactSolution<dim>::vector_gradient(const Point< dim > & 	p,
                                      std::vector< Tensor< 1, dim> > & 	gradients) const
    {
      gradients[0][0] = exact_solution_grad_val_data.value(p, 0);
      gradients[0][1] = exact_solution_grad_val_data.value(p, 1);
      gradients[1][0] = exact_solution_grad_val_data.value(p, 2);
      gradients[1][1] = exact_solution_grad_val_data.value(p, 3);
      gradients[2][0] = exact_solution_grad_val_data.value(p, 4);
      gradients[2][1] = exact_solution_grad_val_data.value(p, 5);
    }

  // There is no easy way to take dot product of ParsedFunction with normal vector
  // So we define a TensorFunction for stress tensor
  template <int dim>
  class StressTensor_Exact : public TensorFunction<2, dim>
  {
  public:
    Functions::ParsedFunction<dim> stress_function;
    StressTensor_Exact(ParameterHandler &);
    //for F
    //function to explain the relation y = f(x)
    virtual Tensor<2, dim> value(const Point<dim> &p) const override;
  };

  template <int dim>
  StressTensor_Exact<dim>::StressTensor_Exact(ParameterHandler &)
  : TensorFunction<2, dim>(), stress_function(dim*dim)
  {}

  template <int dim>
  Tensor<2,dim> StressTensor_Exact<dim>::value(const Point<dim> &p) const
  { 
    Tensor<2, dim> return_value;

    return_value [0][0] = stress_function.value(p, 0);
    return_value [0][1] = stress_function.value(p, 1);
    return_value [1][0] = stress_function.value(p, 2);
    return_value [1][1] = stress_function.value(p, 3);

    return return_value;
  }
} // namespace mfe_stokes

#endif // STOKES_MFE_DATA_H
