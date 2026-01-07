// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 Manraj Singh Ghumman
//
// This file is part of mfe_biot.
//
// ---------------------------------------------------------------------

#ifndef BIOT_PARAMETER_READER_H
#define BIOT_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

namespace mfe_biot
{
  using namespace dealii;

  class BiotParameterReader : public Subscriptor
  {
    public:
      BiotParameterReader (ParameterHandler &prm) : prm(prm) {}
      inline void read_parameters(const std::string &filename);
    private:
      inline void declare_parameters();
      ParameterHandler &prm;
  };

  inline void BiotParameterReader::declare_parameters()
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

    prm.enter_subsection("BC Dirichlet Displacement");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("BC Dirichlet Pressure");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,1);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    // Exact solution values Darcy
    prm.enter_subsection("Exact solution Darcy");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    // Exact solution, gradient values Darcy
    prm.enter_subsection("Exact gradient Darcy");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,6);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    // Exact solution values Elast 
    prm.enter_subsection("Exact solution stress");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();
    prm.enter_subsection("Exact solution displacement");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();
    prm.enter_subsection("Exact solution rotation");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    // Exact solution, gradient values  Elast
    prm.enter_subsection("Exact gradient stress");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,8);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();
    prm.enter_subsection("Exact gradient displacement");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,4);
      prm.set("Function expression", "0; 0; 0; 0");
    }
    prm.leave_subsection();
    prm.enter_subsection("Exact gradient rotation");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,8);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();
  }

  inline void BiotParameterReader::read_parameters(const std::string &filename)
  {
    declare_parameters();
    prm.parse_input (filename);
  }
}//namespace mfe_biot
#endif // BIOT_PARAMETER_READER_H