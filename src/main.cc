/* ---------------------------------------------------------------------
 * This program implements multiscale mortar mixed finite element
 * method for the multiphysics Stokes-Biot system of equations.
 *
 * This implementation allows for non-matching grids by utilizing
 * the mortar finite element space on the interface. To speed things
 * up a little, the multiscale stress basis is also available for
 * the cases when the mortar grid is much coarser than the subdomain
 * ones.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Northwestern University, 2024
 *         Manraj Ghumman, University of Pittsburgh, 2025
 * based on Eldar Khattatov's Elasticity DD implementation from 2017.
 */
#include <cstdlib>
// Utilities, data, etc..
#include "../inc/sb_dd.h"
#include "../inc/sb_parameter_reader.h"

// Main function is simple here
int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace dd_stokes_biot;

      // Initialize MPI via dealii wrapper with 1 thread per MPI process
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // read the biot parameters from parameters.prm and update them in
      // the test function biot.prm files

      const MPI_Comm mpi_communicator = MPI_COMM_WORLD;
      const unsigned int this_mpi_process =
        Utilities::MPI::this_mpi_process(mpi_communicator);

      if (this_mpi_process == 0)
      {
        const int status =
          std::system("python3 \"../test function/update_test_function_constants.py\" "
              "\"../parameters_sb.prm\" "
              "\"../test function/dd test/biot.prm\" "
              "\"../test function/dd test/stokes.prm\"");

        AssertThrow(status == 0,
                    ExcMessage("Failed to update Function constants in biot.prm/stokes.prm."));
      }

      MPI_Barrier(mpi_communicator);


      // Ask for parameters from parameter reader
      ParameterHandler prm;
      StokesBiotParameterReader parameter_reader(prm);
      parameter_reader.read_parameters("../parameters_sb.prm");

      switch (prm.get_integer("dim"))
      {
        case 2:
        {
          MixedStokesBiotProblem<2> stokes_biot(prm);

          stokes_biot.run();
          break;
        }
        case 3:
        {
          AssertThrow(false, ExcMessage("3D implementation is not yet available."));
          break;
        }
        default:
          throw std::runtime_error("Dimension must be either 2 or 3.");
      }

    } 


  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}
