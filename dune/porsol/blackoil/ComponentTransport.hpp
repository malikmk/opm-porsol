/*
  Copyright 2010 SINTEF ICT, Applied Mathematics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_COMPONENTTRANSPORT_HEADER_INCLUDED
#define OPM_COMPONENTTRANSPORT_HEADER_INCLUDED

#include <tr1/array>
#include <vector>
#include <dune/porsol/blackoil/fluid/BlackoilDefs.hpp>
#include <dune/porsol/blackoil/BlackoilFluid.hpp>
#include <dune/common/param/ParameterGroup.hpp>

namespace Opm
{


template <class Grid, class Rock, class Fluid>
class ExplicitCompositionalTransport : public BlackoilDefs
{
public:
    void init(const Dune::parameter::ParameterGroup& param)
    {
    }

    void transport(const Grid& grid,
                   const Rock& rock,
                   const Fluid& fluid,
                   const PhaseVec& external_pressure,
                   const CompVec& external_composition,
                   const std::vector<double>& face_flux,
                   const std::vector<PhaseVec>& cell_pressure,
                   const std::vector<PhaseVec>& face_pressure,
                   const double dt,
                   std::vector<CompVec>& cell_z)
    {
        int num_cells = grid.numCells();
        std::vector<CompVec> comp_change;
        std::vector<double> cell_outflux;
        std::vector<double> cell_max_ff_deriv;
        double cur_time = 0.0;
        while (cur_time < dt) {
            updateFluidProperties(grid, fluid, cell_pressure, face_pressure, cell_z, external_pressure, external_composition);
            computeChange(grid, face_flux, comp_change, cell_outflux, cell_max_ff_deriv);
            double min_time = 1e100;
            for (int cell = 0; cell < num_cells; ++cell) {
                double time = (rock.porosity(cell)*grid.cellVolume(cell))/(cell_outflux[cell]*cell_max_ff_deriv[cell]);
                min_time = std::min(time, min_time);
            }
            min_time *= 0.49; // Semi-random CFL factor... \TODO rigorize
            double step_time = dt - cur_time;
            if (min_time < step_time) {
                step_time = min_time;
                cur_time += min_time;
            } else {
                cur_time = dt;
            }
            std::cout << "Taking step in explicit transport solver: " << step_time << std::endl;
            for (int cell = 0; cell < num_cells; ++cell) {
                comp_change[cell] *= (step_time/rock.porosity(cell));
                cell_z[cell] += comp_change[cell];
            }
        }
    }


private: // Data
    typename Fluid::FluidData fluid_data_;
    PhaseVec bdy_saturation_;
    PhaseVec bdy_fractional_flow_;
    std::tr1::array<CompVec, numPhases> bdy_comp_in_phase_;
    PhaseVec bdy_relperm_;
    PhaseVec bdy_viscosity_;

private: // Methods

    void updateFluidProperties(const Grid& grid,
                               const Fluid& fluid,
                               const std::vector<PhaseVec>& cell_pressure,
                               const std::vector<PhaseVec>& face_pressure,
                               const std::vector<CompVec>& cell_z,
                               const PhaseVec& external_pressure,
                               const CompVec& external_composition)
    {
        fluid_data_.compute(grid, fluid, cell_pressure, face_pressure, cell_z, external_composition);

        BlackoilFluid::FluidState state = fluid.computeState(external_pressure, external_composition);
        bdy_saturation_ = state.saturation_;
        double total_mobility = 0.0;
        for (int phase = 0; phase < numPhases; ++phase) {
            total_mobility += state.mobility_[phase];
        }
        bdy_fractional_flow_ = state.mobility_;
        bdy_fractional_flow_ /= total_mobility;
        std::copy(state.phase_to_comp_, state.phase_to_comp_ + numComponents*numPhases,
                  &bdy_comp_in_phase_[0][0]);
        bdy_relperm_ = state.relperm_;
        bdy_viscosity_ = state.viscosity_;
    }



    void computeChange(const Grid& grid,
                       const std::vector<double>& face_flux,
                       std::vector<CompVec>& comp_change,
                       std::vector<double>& cell_outflux,
                       std::vector<double>& cell_max_ff_deriv)
    {
        comp_change.clear();
        CompVec zero(0.0);
        comp_change.resize(grid.numCells(), zero);
        cell_outflux.clear();
        cell_outflux.resize(grid.numCells(), 0.0);
        cell_max_ff_deriv.clear();
        cell_max_ff_deriv.resize(grid.numCells(), 0.0);
        for (int face = 0; face < grid.numFaces(); ++face) {
            // Set up needed quantities.
            int c0 = grid.faceCell(face, 0);
            int c1 = grid.faceCell(face, 1);
            int upwind_cell = (face_flux[face] > 0.0) ? c0 : c1;
            int downwind_cell = (face_flux[face] > 0.0) ? c1 : c0;
            PhaseVec upwind_sat = upwind_cell < 0 ? bdy_saturation_ : fluid_data_.saturation[upwind_cell];
            PhaseVec upwind_relperm = upwind_cell < 0 ? bdy_relperm_ : fluid_data_.rel_perm[upwind_cell];
            PhaseVec upwind_viscosity = upwind_cell < 0 ? bdy_viscosity_ : fluid_data_.viscosity[upwind_cell];
            PhaseVec upwind_ff = upwind_cell < 0 ? bdy_fractional_flow_ : fluid_data_.frac_flow[upwind_cell];
            PhaseVec phase_flux(upwind_ff);
            phase_flux *= face_flux[face];
            CompVec change(0.0);

            // Estimate max derivative of ff.
            double face_max_ff_deriv = 0.0;
            if (downwind_cell >= 0) { // Only contribution on inflow and internal faces.
                // Evaluating all functions at upwind viscosity.
                PhaseVec downwind_mob(0.0);
                double downwind_totmob = 0.0;
                for (int phase = 0; phase < numPhases; ++phase) {
                    downwind_mob[phase] = fluid_data_.rel_perm[downwind_cell][phase]/upwind_viscosity[phase];
                    downwind_totmob += downwind_mob[phase];
                }
                PhaseVec downwind_ff = downwind_mob;
                downwind_ff /= downwind_totmob;
                PhaseVec ff_diff = upwind_ff;
                ff_diff -= downwind_ff;
                for (int phase = 0; phase < numPhases; ++phase) {
                    if (std::fabs(ff_diff[phase]) > 1e-14) {
                        double ff_deriv = ff_diff[phase]/(upwind_sat[phase] - fluid_data_.saturation[downwind_cell][phase]);
                        ASSERT(ff_deriv >= 0.0);
                        face_max_ff_deriv = std::max(face_max_ff_deriv, ff_deriv);
                    }
                }
            }

            // Compute z change.
            for (int phase = 0; phase < numPhases; ++phase) {
                CompVec z_in_phase = bdy_comp_in_phase_[phase];
                if (upwind_cell >= 0) {
                    for (int comp = 0; comp < numComponents; ++comp) {
                        z_in_phase[comp] = fluid_data_.cellA[numPhases*numComponents*upwind_cell + numComponents*phase + comp];
                    }
                }
                z_in_phase *= phase_flux[phase];
                change += z_in_phase;
            }

            // Update output variables.
            if (upwind_cell >= 0) {
                cell_outflux[upwind_cell] += std::fabs(face_flux[face]);
            }
            if (c0 >= 0) {
                comp_change[c0] -= change;
                cell_max_ff_deriv[c0] = std::max(cell_max_ff_deriv[c0], face_max_ff_deriv);
            }
            if (c1 >= 0) {
                comp_change[c1] += change;
                cell_max_ff_deriv[c1] = std::max(cell_max_ff_deriv[c1], face_max_ff_deriv);
            }
        }
    }


};


} // namespace Opm


#endif // OPM_COMPONENTTRANSPORT_HEADER_INCLUDED
