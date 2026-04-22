#ifndef STOKES_PROBLEM_TEMPLATES_H
#define STOKES_PROBLEM_TEMPLATES_H

#include "stokes_mfe.h"
#include <fstream>

namespace mfe_stokes
{
	using namespace dealii;
	
	template <int dim>
	MixedStokesProblem<dim>::StressPostprocessor::StressPostprocessor()
		: DataPostprocessorTensor<dim>("fluid_stress",
										update_values | update_gradients)
	{}

	template <int dim>
	void MixedStokesProblem<dim>::StressPostprocessor::evaluate_vector_field(
	const DataPostprocessorInputs::Vector<dim> &inputs,
	std::vector<Vector<double>> &computed_quantities) const
	{
		const unsigned int n_points = inputs.solution_values.size();
		AssertDimension(computed_quantities.size(), n_points);

		for (unsigned int q = 0; q < n_points; ++q)
		{
			AssertDimension(computed_quantities[q].size(), dim * dim);

			const double p = inputs.solution_values[q][dim];

			Tensor<2, dim> grad_u;
			for (unsigned int i = 0; i < dim; ++i)
				for (unsigned int j = 0; j < dim; ++j)
				grad_u[i][j] = inputs.solution_gradients[q][i][j];

			Tensor<2, dim> sigma;
			for (unsigned int i = 0; i < dim; ++i)
				for (unsigned int j = 0; j < dim; ++j)
				sigma[i][j] = grad_u[i][j] + grad_u[j][i];

			for (unsigned int d = 0; d < dim; ++d)
				sigma[d][d] -= p;

			unsigned int c = 0;
			for (unsigned int i = 0; i < dim; ++i)
				for (unsigned int j = 0; j < dim; ++j, ++c)
					computed_quantities[q](c) = sigma[i][j];
		}
	}

	// template <int dim>
	// void MixedStokesProblem<dim>::output_results(const unsigned int cycle) const
	// {
	// 	DataOut<dim> data_out;
	// 	data_out.attach_dof_handler(dof_handler);

	// 	std::vector<std::string> solution_names(dim, "velocity");
	// 	solution_names.emplace_back("pressure");

	// 	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	// 			interpretation(dim,
	// 									DataComponentInterpretation::component_is_part_of_vector);
	// 	interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	// 	data_out.add_data_vector(solution,
	// 													solution_names,
	// 													DataOut<dim>::type_dof_data,
	// 													interpretation);

	// 	StressPostprocessor stress_postprocessor(mu);
	// 	data_out.add_data_vector(solution, stress_postprocessor);

	// 	data_out.build_patches();

	// 	std::ofstream out("solution-" + Utilities::int_to_string(cycle, 2) + ".vtu");
	// 	data_out.write_vtu(out);
	// }
}

#endif