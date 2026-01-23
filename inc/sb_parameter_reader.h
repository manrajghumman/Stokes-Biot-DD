// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 Manraj Singh Ghumman
//
// This file is part for reading parameters in parameters_stokes.prm.
//
// ---------------------------------------------------------------------

#ifndef STOKES_BIOT_PARAMETER_READER_H
#define STOKES_BIOT_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
// #include <deal.II/base/parsed_function.h>

namespace dd_stokes_biot
{
  using namespace dealii;

class StokesBiotParameterReader : public Subscriptor
  {
  public:
    StokesBiotParameterReader(ParameterHandler &paramhandler) : prm(paramhandler) {}
    inline void read_parameters(const std::string);
  private:
    inline void declare_parameters();
    ParameterHandler &prm;
  };

  inline void StokesBiotParameterReader::declare_parameters()
  {
    prm.declare_entry("dim", "2",
                      Patterns::Integer(),
                      "Spatial dimension (2D or 3D)");
    prm.declare_entry("refinements", "1",
                      Patterns::Integer(),
                      "Number of global mesh refinements");
    prm.declare_entry("initial_grid_2D", "1, 1",
                        Patterns::Anything(),
                        "Initial grid size for 2D (x, y) "
                        "in number of cells per direction");
    prm.declare_entry("initial_grid_3D", "1, 1, 1",
                        Patterns::Anything(),
                        "Initial grid size for 3D (x, y, z) "
                        "in number of cells per direction");
    prm.declare_entry("grid", "1, 1",
                      Patterns::Anything(),
                      "Differentiate Stokes and Biot subdomains");

    prm.enter_subsection("Stokes Parameters");
    {
        prm.declare_entry("degree", "0",
                      Patterns::Integer(),
                      "Finite element degree for Stokes");
        prm.declare_entry("quad_degree", "11",
                      Patterns::Integer(),
                      "Quadrature degree for Stokes");
        prm.declare_entry("continuous_mortar", "yes",
                      Patterns::Selection("yes|no"));
        prm.declare_entry("mortar_degree", "2",
                      Patterns::Integer());
        prm.declare_entry("gamma", "100.0",
                      Patterns::Double(),
                      "Nitsche's penalty parameter for Stokes");
    }
    prm.leave_subsection();

    prm.enter_subsection("Biot Parameters");
    {
        prm.declare_entry("degree", "0",
                      Patterns::Integer(),
                      "Finite element degree for Biot");
        prm.declare_entry("quad_degree", "11",
                      Patterns::Integer(),
                      "Quadrature degree for Biot");
        prm.declare_entry("mortar_type", "1",
                      Patterns::Integer(), "1");
        prm.declare_entry("mortar_degree", "2",
                      Patterns::Integer());
        prm.declare_entry("s_0", "1.0",
                      Patterns::Double(),
                      "Storativity coefficient");
        prm.declare_entry("alpha_BJS", "1.0",
                      Patterns::Double(),
                      "Biot-Willis coefficient");
        prm.declare_entry("lame_mu", "1.0",
                      Patterns::Double(),
                      "Lame's first parameter - mu");
        prm.declare_entry("lame_lambda", "1.0",
                      Patterns::Double(),
                      "Lame's second parameter - lambda");
        prm.declare_entry("Kinv", "1, 0; 0, 1",
                      Patterns::Anything(),
                      "Inverse of Permeability tensor");
    }
    prm.leave_subsection();

    prm.enter_subsection("External Boundary Conditions");
    {
        prm.declare_entry("boundary_2D", "N, D, D, D",
                      Patterns::Anything(),
                    "Boundary types (N=Neumann, D=Dirichlet)");
        prm.declare_entry("boundary_3D", "N, D, D, D, D, D",
                      Patterns::Anything(),
                    "Boundary types (N=Neumann, D=Dirichlet)");
    }
    prm.leave_subsection();

    // prm.enter_subsection("Mortar");
    // {
    //     prm.declare_entry("mortar", "no",
    //                   Patterns::Selection("yes|no"));
    //     prm.declare_entry("continuous_mortar", "yes",
    //                   Patterns::Selection("yes|no"));
    //     prm.declare_entry("mortar_degree", "2",
    //                   Patterns::Integer());
    // }
    // prm.leave_subsection();

    prm.enter_subsection("Solver");
    {
        prm.declare_entry("max_iter", "1000",
                      Patterns::Integer());
        prm.declare_entry("tol", "1e-8",
                      Patterns::Double(0.0));
    }
    prm.leave_subsection();
  }

  inline void StokesBiotParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.parse_input (parameter_file);
  }
}

#endif //STOKES_BIOT_PARAMETER_READER_H
