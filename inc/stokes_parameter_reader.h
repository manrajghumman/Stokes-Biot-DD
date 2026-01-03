// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 Manraj Singh Ghumman
//
// This file is part of mfe_stokes.
//
// ---------------------------------------------------------------------

#ifndef STOKES_PARAMETER_READER_H
#define STOKES_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

namespace mfe_stokes
{
  using namespace dealii;

  class StokesParameterReader : public Subscriptor
  {
    public:
      StokesParameterReader (ParameterHandler &prm) : prm(prm) {}
      inline void read_parameters(const std::string &filename);
    private:
      inline void declare_parameters();
      ParameterHandler &prm;
  };

  inline void StokesParameterReader::declare_parameters()
  {
    prm.enter_subsection("RHS vector");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("RHS scalar");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,1);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("BC Dirichlet");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,4);
      prm.set("Function expression", "0; 0; 0; 0");
    }
    prm.leave_subsection();
  }

  
  inline void StokesParameterReader::read_parameters(const std::string &filename)
  {
    declare_parameters();
    prm.parse_input (filename);
  }
}//namespace mfe_stokes
#endif // STOKES_PARAMETER_READER_H