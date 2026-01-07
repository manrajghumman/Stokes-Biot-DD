/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions and
 * the true solution.
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2018
 */

#ifndef BIOT_MFE_DATA_H
#define BIOT_MFE_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor_function.h>

namespace mfe_biot
{
  using namespace dealii;

  // Lame parameters (lambda and mu)
  template <int dim>
  class LameParameters : public Function<dim>
  {
  public:
      LameParameters ()  : Function<dim>() {}
      virtual void vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const;

      virtual void vector_value_list (const std::vector<Point<dim> > &points,
      std::vector<Vector<double> > &value_list) const;
  };

  template <int dim>
  void LameParameters<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
  {
      Assert(values.size() == 2,
              ExcDimensionMismatch(values.size(),2));
      Assert(dim != 1, ExcNotImplemented());

      double x,y,z;
      x = p[0];
      y = p[1];

      if (dim == 3)
          z = p[2];

      switch (dim)
      {
          case 2:
              values(0) = 1.0;
              values(1) = 1.0;
              break;
          case 3:
              values(0) = 1.0;
              values(1) = 1.0;
              break;
          default:
              Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  void LameParameters<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> > &value_list) const
  {
      Assert(value_list.size() == points.size(),
              ExcDimensionMismatch(value_list.size(), points.size()));

      const unsigned int n_points = points.size();

      for (unsigned int p=0; p<n_points; ++p)
          LameParameters<dim>::vector_value(points[p], value_list[p]);
  }

  // Inverse of permeability tensor
  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                              std::vector<Tensor<2,dim> >    &values) const;
  };

  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                              std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
    {
      values[p].clear ();

      const double x = points[p][0];
      const double y = points[p][1];

      switch (dim)
      {
        case 2:
          values[p][0][0] = 1.0;
          values[p][0][1] = 0.0;
          values[p][1][0] = 0.0;
          values[p][1][1] = 1.0;
          break;
        default:
        Assert(false, ExcMessage("The inverse of permeability tensor for dim != 2 is not provided"));
      }
    }
  }

  // Exact solution
  template <int dim>
  class ExactSolution : public Function<dim>
  {
    public:
    Functions::ParsedFunction<dim> exact_solution_darcy_data;
    Functions::ParsedFunction<dim> exact_solution_stress_data;
    Functions::ParsedFunction<dim> exact_solution_displacement_data;
    Functions::ParsedFunction<dim> exact_solution_rotation_data;
    ExactSolution();

    virtual void vector_value (const Point<dim> &p,
                                Vector<double>   &values) const override;
    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim,double> > &grads) const override;
  };

  template <int dim>
  ExactSolution<dim>::ExactSolution() :
    Function<dim>(static_cast<unsigned int>(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1)),
    exact_solution_darcy_data(dim+1),
    exact_solution_stress_data(dim*dim - 1),
    exact_solution_displacement_data(dim),
    exact_solution_rotation_data(0.5*dim*(dim-1))
  {}

  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    double x = p[0];
    double y = p[1];
    double z;

    if (dim == 3)
        z = p[2];

    const LameParameters<dim> lame_function;
    Vector<double> vec(2);
    lame_function.vector_value(p,vec);

    const double lmbda = vec[0];
    const double mu = vec[1];

    switch (dim)
    {
      case 2:
        values(0) = exact_solution_stress_data.value(p, 0);
        values(1) = exact_solution_stress_data.value(p, 1);
        values(2) = exact_solution_stress_data.value(p, 1);
        values(3) = exact_solution_stress_data.value(p, 2);
        values(4) = exact_solution_displacement_data.value(p, 0);
        values(5) = exact_solution_displacement_data.value(p, 1);
        values(6) = 0.0;
        values(7) = exact_solution_darcy_data.value(p, 0);
        values(8) = exact_solution_darcy_data.value(p, 1);
        values(9) = exact_solution_darcy_data.value(p, 2);
        break;
      case 3:
        values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
        values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
        values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
        values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
        values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
        values(5) = 0;
        values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
        values(7) = 0;
        values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;

        values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
        values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
        values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);

        values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
        values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
        values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  template <int dim>
  void
  ExactSolution<dim>::vector_gradient (const Point<dim> &p,
                                        std::vector<Tensor<1,dim,double> > &grads) const
  {
    double x = p[0];
    double y = p[1];
    double z;

    if (dim == 3)
    z = p[2];

    const LameParameters<dim> lame_function;
    Vector<double> vec(2);
    lame_function.vector_value(p,vec);

    const double lambda = vec[0];
    const double mu = vec[1];
    double t = FunctionTime<double>::get_time();

    int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
    Tensor<1,dim> tmp;
    switch (dim)
    {
      case 2:
        grads[0][0] = lambda*(exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)-exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x)))+mu*exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)*2.0-3.141592653589793*exp(t)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
        grads[0][1] = -lambda*(exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)-exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0)))+mu*exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*2.0+3.141592653589793*exp(t)*sin(x*3.141592653589793)*sin(y*3.141592653589793);

        grads[1][0] = mu*(exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,2.0)*pow(y-1.0,3.0)*1.2E1+cos(x*y)*sin(x)+y*sin(x*y)*cos(x)*2.0+(y*y)*cos(x*y)*sin(x))*(1.0/2.0))*2.0;
        grads[1][1] = mu*(exp(t)*((x*x*x)*(y*y)*-1.2E1+cos(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(x-1.0,2.0)+sin(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0)*2.0)*(1.0/2.0)+exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*(1.0/2.0))*-2.0;

        grads[2][0] = mu*(exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,2.0)*pow(y-1.0,3.0)*1.2E1+cos(x*y)*sin(x)+y*sin(x*y)*cos(x)*2.0+(y*y)*cos(x*y)*sin(x))*(1.0/2.0))*2.0;
        grads[2][1] = mu*(exp(t)*((x*x*x)*(y*y)*-1.2E1+cos(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(x-1.0,2.0)+sin(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0)*2.0)*(1.0/2.0)+exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*(1.0/2.0))*-2.0;

        grads[3][0] = lambda*(exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)-exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x)))-mu*exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*2.0-3.141592653589793*exp(t)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
        grads[3][1] = -lambda*(exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)-exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0)))-mu*exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)*2.0+3.141592653589793*exp(t)*sin(x*3.141592653589793)*sin(y*3.141592653589793);

        grads[7][0] = (3.141592653589793*3.141592653589793)*exp(t)*cos(y*3.141592653589793)*sin(x*3.141592653589793);
        grads[7][1] = (3.141592653589793*3.141592653589793)*exp(t)*cos(x*3.141592653589793)*sin(y*3.141592653589793);

        grads[8][0] = (3.141592653589793*3.141592653589793)*exp(t)*cos(x*3.141592653589793)*sin(y*3.141592653589793);
        grads[8][1] = (3.141592653589793*3.141592653589793)*exp(t)*cos(y*3.141592653589793)*sin(x*3.141592653589793);
        break;
      case 3:
        for (int k=0;k<total_dim;++k)
            grads[k] = tmp;

        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  }
}

#endif //BIOT_MFE_DATA_H
