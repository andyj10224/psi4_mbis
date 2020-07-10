/*
 * @BEGIN LICENSE
 *
 * mbis2 by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2019 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */


 /**
    PSI4 MBIS: Implements the MBIS Charge Partitioning Scheme as described in the paper:

    Minimal Basis Iterative Stockholder: Atoms in Molecules for Force- Field Development
    (J. Chem. Theory Comput. 2016, 12, 3894−3912, DOI: 10.1021/acs.jctc.6b00456) By: Toon Verstraelen et al.

    @author Andy Jiang, Zachary L. Glick, C. David Sherrill
    @version 1.0 07/09/2020
*/


#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libfock/v.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/exception.h"
#include <vector>
#include <chrono>

namespace psi{ namespace mbis2 {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "MBIS2"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        options.add_bool("DO_TEI", true);
    }

    return true;
}

//The initial guess for the number of electrons in each shell (2, 8, 8, 18, etc...) of an atom
int get_nai(int z, int m) {
    if (z <= 2) {
        return z;
    }

    else if (z <= 10) {
        if (m == 1) {
            return 2;
        }
        else {
            return z - 2;
        }
    }

    else if (z <= 18) {
        if (m == 1) {
            return 2;
        }
        else if (m == 2) {
            return 8;
        }
        else {
            return z - 10;
        }
    }

    else if (z <= 36) {
        if (m == 1) {
            return 2;
        }
        else if (m == 2) {
            return 8;
        }
        else if (m == 3) {
            return 8;
        }
        else {
            return z - 18;
        }
    }
}

//The number of shells of a neutral atom given it's atomic number (for elements lighter than Krypton)
int get_mA(int atomic_num) {
    if (atomic_num <= 2) {
        return 1;
    }
    else if (atomic_num <= 10) {
        return 2;
    }
    else if (atomic_num <= 18) {
        return 3;
    }
    else if (atomic_num <= 36) {
        return 4;
    }
}

//The distance between a point (x, y, z) and the center of an atom (Rx, Ry, Rz)
double distance(double x, double y, double z, double Rx, double Ry, double Rz) {
    return sqrt(pow((x-Rx), 2) + pow((y-Ry), 2) + pow((z-Rz), 2));
}

//Proatomic density of a specific shell of an atom  (Equation 7 in Verstraelen et al.)
double rho_ai_o(double n, double sigma, double distance) {

    if (fabs(sigma) < 1.0e-16) {
        return 0.0;
    }

    return n * exp(-distance / sigma) / (pow(sigma, 3) * 8 * M_PI);

}

//The proatomic denisty of an atom in a molecule (Equation 6 in Verstraelen et al.)
double rho_a_o(int atom, const std::vector<std::vector<double>> &n, const std::vector<std::vector<double>> &s, const std::vector<std::vector<double>> &distances, int point_num) {

    double sum = 0.0;

    for (int m = 0; m < 4; m++) {
        sum += rho_ai_o(n[atom][m], s[atom][m], distances[atom][point_num]);
    }

    return sum;
}

//Fills arrays consisting of all of nuclei of every atom in a molecule
void atomic_positions(SharedMolecule mol, int num_atoms, std::vector<double> &Rxs, std::vector<double> &Rys, std::vector<double> &Rzs) {

    for (int atom = 0; atom < num_atoms; atom++) {
        Rxs[atom] = mol->fx(atom);
        Rys[atom] = mol->fy(atom);
        Rzs[atom] = mol->fz(atom);
    }

}

//The sum of all proatomic densities in a molecule (Equation 5 in Verstraelen et al.)
double rho_o(int num_atoms, const std::vector<std::vector<double>> &n, const std::vector<std::vector<double>> &s, const std::vector<std::vector<double>> &distances, int point_num) {
    double sum = 0.0;

    for (int atom = 0; atom < num_atoms; atom++) {
        sum += rho_a_o(atom, n, s, distances, point_num);
    }

    return sum;
}

extern "C" PSI_API
SharedWavefunction mbis2(SharedWavefunction ref_wfn, Options& options) {

    timer_on("MBIS");

    int print = options.get_int("PRINT");

    //a_0 = Bohr Radius, max_iter = max number of iterations for MBIS
    double a_0 = 1.0;
    int max_iter = 100;

    //nbf = number of basis functions, multiplicity = multiplicity of molecule
    SharedMolecule mol = ref_wfn->molecule();
    int nbf = ref_wfn->basisset()->nbf();
    int multiplicity = mol->multiplicity();

    std::shared_ptr<SuperFunctional> funct = (std::shared_ptr<SuperFunctional>) (new SuperFunctional());
    std::shared_ptr<PSIO> psio = (std::shared_ptr<PSIO>) (new PSIO());

    //The grid used that yields the grid points
    DFTGrid grid = DFTGrid(mol, ref_wfn->basisset(), options);

    //Total number of points in the grid
    int total_points = grid.npoints();

    //Used to calculate molecular electron density
    std::shared_ptr<PointFunctions> point_func = NULL;

    //Point Function for RHF Procedures
    if (multiplicity == 1) {
        point_func = (std::shared_ptr<PointFunctions>) (new RKSFunctions(ref_wfn->basisset(), total_points, nbf));
    }

    //Point Function for UHF Procedures
    else {
        point_func = (std::shared_ptr<PointFunctions>) (new UKSFunctions(ref_wfn->basisset(), total_points, nbf));
    }

    std::vector<std::shared_ptr<BlockOPoints>> blocks = grid.blocks();

    grid.print();

    //Alpha and Beta Density matrices (Atomic Orbital)
    SharedMatrix Da = ref_wfn->Da_subset("AO");
    SharedMatrix Db = ref_wfn->Db_subset("AO");


    if (multiplicity == 1) {
        point_func->set_pointers(Da);
    }

    else {
        point_func->set_pointers(Da, Db);
    }

    //Number of blocks in the grid
    int num_blocks = blocks.size();

    //Keeps track of the number of points already calculated after each block iteration
    int running_points = 0;

    //Will later represent the values of x, y, z, weights, and rho (molecular electron density) at each grid point
    std::vector<double> x_points(total_points, 0.0);
    std::vector<double> y_points(total_points, 0.0);
    std::vector<double> z_points(total_points, 0.0);
    std::vector<double> weights(total_points, 0.0);
    std::vector<double> rho(total_points, 0.0);

    auto start = std::chrono::steady_clock::now();
    psi::outfile->Printf("STARTING THE CREATION OF RHO VECTOR\n\n");

    //Initializes values of x, y, z, weights, and rho at each grid point
    for (int b = 0; b < num_blocks; b++) {
        std::shared_ptr<BlockOPoints> block = blocks[b];
        point_func->set_ansatz(0);
        point_func->set_max_functions(block->local_nbf());
        point_func->set_max_points(block->npoints());
        point_func->compute_points(block);
        std::size_t num_points = block->npoints();

        SharedVector rho_block = point_func->point_values()["RHO_A"];

        if (multiplicity != 1) {
            rho_block->add(point_func->point_values()["RHO_B"]);
        }

        double* x = block->x();
        double* y = block->y();
        double* z = block->z();
        double* w = block->w();

        for (int p = 0; p < num_points; p++) {
            x_points[running_points + p] = x[p];
            y_points[running_points + p] = y[p];
            z_points[running_points + p] = z[p];
            weights[running_points + p] = w[p];

            rho[running_points + p] = rho_block->get(p);
        }

        running_points += num_points;
    }

    auto end1 = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_time = end1 - start;
    psi::outfile->Printf("FINISHED BUILDING RHO VECTOR, TIME: %16.8f\n\n", elapsed_time.count());

    //Checkes to make sure the total number of electrons matches as calculated by the molecular electron density function matches what is in the molecule
    double num_electrons = 0.0;
    for (int i = 0; i < total_points; i++) {

        num_electrons += weights[i] * rho[i];

    }

    psi::outfile->Printf("Num Electrons in Molecule: %16.8f\n\n", num_electrons);

    auto end2 = std::chrono::steady_clock::now();
    elapsed_time = end2 - end1;
    psi::outfile->Printf("FINISHED CALCULATING ELECTRONS, TIME: %16.8f\n\n", elapsed_time.count());

    //Number of atoms in the molecule
    int num_atoms = mol->natom();

    //Represents an array of N_ai, and Sigma_ai (number of atoms by 4, max number of shells) as described by equations 18 and 19 in Verstraelen et al.
    std::vector<std::vector<double>> Nai(num_atoms, std::vector<double>(4, 0.0));
    std::vector<std::vector<double>> Sai(num_atoms, std::vector<double>(4, 0.0));

    //Stores values of N_ai and sigma_ai before N_ai and sigma_ai are updated
    std::vector<std::vector<double>> Nai_temp(num_atoms, std::vector<double>(4, 0.0));
    std::vector<std::vector<double>> Sai_temp(num_atoms, std::vector<double>(4, 0.0));

    //Stores x, y, and z positions of all nuclei in an atom
    std::vector<double> Rxs(num_atoms, 0.0);
    std::vector<double> Rys(num_atoms, 0.0);
    std::vector<double> Rzs(num_atoms, 0.0);
    atomic_positions(mol, num_atoms, Rxs, Rys, Rzs);

    //Stores the distances of all points on the grid to the nuclei of every atom
    std::vector<std::vector<double>> distances(num_atoms, std::vector<double>(total_points, 0.0));

    //Stores the x, y, and z components of the vector that points from the nuclei of every atom to every grid point
    std::vector<std::vector<std::vector<double>>> xyz_components(num_atoms, std::vector<std::vector<double>>(total_points, std::vector<double>(3, 0.0)));

    for (int i = 0; i < num_atoms; i++) {
        for (int j = 0; j < total_points; j++) {
            double x = x_points[j];
            double y = y_points[j];
            double z = z_points[j];
            distances[i][j] = distance(x, y, z, Rxs[i], Rys[i], Rzs[i]);

            xyz_components[i][j][0] = x - Rxs[i];
            xyz_components[i][j][1] = y - Rys[i];
            xyz_components[i][j][2] = z - Rzs[i];
        }
    }

    //Stores the values of rho_o at every point and rho_a_not for every point for every atom
    std::vector<double> rho_o_points(total_points, 0.0);
    std::vector<std::vector<double>> rho_a_o_points(num_atoms, std::vector<double>(total_points, 0.0));

    std::vector<double> electron_populations(num_atoms, 0.0);

    psi::outfile->Printf("======> STARTING MBIS ITERATIONS <=====\n\n");
    auto end3 = std::chrono::steady_clock::now();

    int iter = 1;

    //represents the number of shells of an atom and the atomic number of an atom through the iterations
    int m_A, atomic_num;

    //Whether the MBIS Calculation converged
    bool is_converged = false;

    while (iter <= max_iter) {

        //represents the convergence criteria integral, the number of atoms that meet the criteria, and the electron population of each atom, respectively
        double integral;
        int count = 0;
        double population;

        psi::outfile->Printf("Iteration %d\n", iter);

        //Initializes Nai and Sai for the first guess
        if (iter == 1) {
            for (int atom = 0; atom < num_atoms; atom++) {
                atomic_num = round(mol->Z(atom));
                m_A = get_mA(atomic_num);

                for (int m = 0; m < m_A; m++) {
                    Nai[atom][m] = (double) get_nai(atomic_num, m+1);
                    Nai_temp[atom][m] = Nai[atom][m];
			        if (m_A == 1) {
				        Sai[atom][m] = a_0/(2.0*atomic_num);
			        }
			        else {
                        Sai[atom][m] = a_0/(2.0*pow((double)atomic_num, (1.0 - m/(m_A - 1.0))));
			        }
                    Sai_temp[atom][m] = Sai[atom][m];
                }
            }
        }

        //Intializes the initial values of rho_o and rho_a_o for every point and every atom
        for (int point = 0; point < total_points; point++) {
            rho_o_points[point] = 0.0;
            for (int atom = 0; atom < num_atoms; atom++) {
                rho_a_o_points[atom][point] = rho_a_o(atom, Nai, Sai, distances, point);
                rho_o_points[point] += rho_a_o_points[atom][point];
            }
        }

        //Calculates the new values of Nai and Sai for every atom
        for (int atom = 0; atom < num_atoms; atom++) {

            atomic_num = round(mol->Z(atom));
            m_A = get_mA(atomic_num);

            for (int m = 0; m < m_A; m++) {

                double sum_n = 0.0;
                double sum_s = 0.0;

                for (int point = 0; point < total_points; point++) {
                    double w = weights[point];

                    if (fabs(w) > 0.0) {
                        double rho_p = rho[point];
                        if (fabs(rho_p) > 0.0) {
                            double rho_o_point = rho_o_points[point];
                            if (fabs(rho_o_point) > 0.0) {
                                double rho_ai_o_point = rho_ai_o(Nai_temp[atom][m], Sai_temp[atom][m], distances[atom][point]);
                                if (fabs(rho_ai_o_point) > 0.0) {
                                    sum_n += w * rho_p * rho_ai_o_point/rho_o_point;
                                    sum_s += w * distances[atom][point] * rho_p * rho_ai_o_point/rho_o_point;
                                }
                            }
                        }
                    }
                }

                Nai_temp[atom][m] = sum_n;
                Sai_temp[atom][m] = sum_s/(3*Nai_temp[atom][m]);

            }
        }

        //Checks for convergence (Equation 20 in Verstraelen et al.), 1.0e-8 instead of 1.0e-16 should be sufficient for our purposes
        for (int atom = 0; atom < num_atoms; atom++) {

            integral = 0.0;

            for (int point = 0; point < total_points; point++) {
                double w = weights[point];

                if (fabs(w) > 0.0) {
                    integral += w * pow((rho_a_o(atom, Nai_temp, Sai_temp, distances, point) - rho_a_o_points[atom][point]), 2);
                }
            }

            if (integral < 1.0e-8) {
                count += 1;
            }

        }

        //Calculates the electron population for each atom for every iteration
        for (int atom = 0; atom < num_atoms; atom++) {
            population = 0.0;
            for (int m = 0; m < 4; m++) {
                Nai[atom][m] = Nai_temp[atom][m];
                Sai[atom][m] = Sai_temp[atom][m];

                population += Nai[atom][m];

            }
            psi::outfile->Printf("Atom %d, Population %16.8f\n", atom+1, population);
            if (count == num_atoms) {
                electron_populations[atom] = population;
            }
        }

        //Convergence is achieved if every atom meets the convergence criteria
        if (count == num_atoms) {
            psi::outfile->Printf("\n\nMBIS Converged, Buy Andy an ice cream!!!\n\n");
            is_converged = true;
            break;
        }

        iter += 1;

    }

    //Throws exception if no convergence in MBIS
    if (!is_converged) {
        throw PsiException("MBIS failed to converge in " + std::to_string(max_iter) + " iterations. Tell Andy to take a break!", "plugin.cc", 475);
    }


    auto end4 = std::chrono::steady_clock::now();
    elapsed_time = end4 - end3;
    psi::outfile->Printf("FINISHED ITERATIONS, TIME: %16.8f\n\n", elapsed_time.count());

    //Final calculation for rho_o_points and rho_a_o_points after MBIS has converged
    for (int point = 0; point < total_points; point++) {
        rho_o_points[point] = 0.0;
        for (int atom = 0; atom < num_atoms; atom++) {
            rho_a_o_points[atom][point] = rho_a_o(atom, Nai, Sai, distances, point);
            rho_o_points[point] += rho_a_o_points[atom][point];
        }
    }

    //Represents rho_a for every atom, as defined in Equation 5 of Verstraelen et al.
    std::vector<std::vector<double>> rho_a(num_atoms, std::vector<double>(total_points, 0.0));

    for (int i = 0; i < num_atoms; i++) {
        for (int j = 0; j < total_points; j++) {
            if (fabs(rho_o_points[j]) > 0.0) {
                rho_a[i][j] = -rho[j] * rho_a_o_points[i][j] / rho_o_points[j];
            }
        }
    }

    //Kronecker Delta
    std::vector<std::vector<double>> k_delta = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    //Maps 0, 1, 2 to X, Y, Z coordinates, respectively
    std::map<int, char> cartesian;

    cartesian[0] = 'X';
    cartesian[1] = 'Y';
    cartesian[2] = 'Z';

    psi::outfile->Printf("======> CALCULATING MULTIPOLES <======\n\n");

    psi::outfile->Printf("======> CARTESIAN MULTIPOLES <======\n\n");

    //Stores the monopoles (charge) on every atom
    std::vector<double> monopoles(num_atoms, 0.0);

    for (int atom = 0; atom < num_atoms; atom++) {
        monopoles[atom] = round(mol->Z(atom)) - electron_populations[atom];
        psi::outfile->Printf("Atom %d, Charge %16.8f\n", atom+1, monopoles[atom]);
    }

    psi::outfile->Printf("\n");

    //Stores the cartesian multipoles of each atom (Horton definition)
    std::vector<std::vector<double>> dipoles(num_atoms, std::vector<double>(3, 0.0));
    std::vector<std::vector<std::vector<double>>> quadrupoles(num_atoms, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0)));
    std::vector<std::vector<std::vector<std::vector<double>>>> octopoles(num_atoms, std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0))));

    //Stores the cartesian multipoles of each atom (Defnition given by Anthony Stone in "The Theory of Intermolecular Forces, Second Edition")
    std::vector<std::vector<std::vector<double>>> stone_quadrupoles(num_atoms, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0)));
    std::vector<std::vector<std::vector<std::vector<double>>>> stone_octopoles(num_atoms, std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0))));

    //Calculates the Cartesian Dipoles
    for (int atom = 0; atom < num_atoms; atom++) {
        for (int i = 0; i < 3; i++) {
            for (int point = 0; point < total_points; point++) {
                dipoles[atom][i] += weights[point] * rho_a[atom][point] * xyz_components[atom][point][i];
            }
            psi::outfile->Printf("Atom %d, Dipole %c, %16.8f\n", atom+1, cartesian[i], dipoles[atom][i]);
        }
    }
    psi::outfile->Printf("\n");

    //Calculates the Cartesian Quadrupoles
    for (int atom = 0; atom < num_atoms; atom++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i > j) {
                    quadrupoles[atom][i][j] = quadrupoles[atom][j][i];
                    stone_quadrupoles[atom][i][j] = stone_quadrupoles[atom][i][j];
                }

                else {
                    for (int point = 0; point < total_points; point++) {
                        quadrupoles[atom][i][j] += weights[point] * rho_a[atom][point] * (xyz_components[atom][point][i] * xyz_components[atom][point][j]);
                        if (i == j) {
                            stone_quadrupoles[atom][i][j] += weights[point] * rho_a[atom][point] * (1.5 * xyz_components[atom][point][i] * xyz_components[atom][point][j] - 0.5 * pow(distances[atom][point], 2));
                        }
                        else {
                            stone_quadrupoles[atom][i][j] += weights[point] * rho_a[atom][point] * (1.5 * xyz_components[atom][point][i] * xyz_components[atom][point][j]);
                        }
                    }
                }

                if (i <= j) {
                    psi::outfile->Printf("Atom %d, Quadrupole %c%c, %16.8f\n", atom+1, cartesian[i], cartesian[j], quadrupoles[atom][i][j]);
                }
            }
        }
    }
    psi::outfile->Printf("\n");

    //Calculates the Cartesian Octopoles
    for (int atom = 0; atom < num_atoms; atom++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (k >= j && j >= i) {
                        for (int point = 0; point < total_points; point++) {
                            octopoles[atom][i][j][k] += weights[point] * rho_a[atom][point] * (xyz_components[atom][point][i] * xyz_components[atom][point][j] * xyz_components[atom][point][k]);

                            stone_octopoles[atom][i][j][k] += weights[point] * rho_a[atom][point] * (2.5 * xyz_components[atom][point][i] * xyz_components[atom][point][j] * xyz_components[atom][point][k] - 0.5 * pow(distances[atom][point], 2) * (k_delta[j][k] * xyz_components[atom][point][i] + k_delta[i][k] * xyz_components[atom][point][j] + k_delta[i][j] * xyz_components[atom][point][k]));

                        }

                        double rep = octopoles[atom][i][j][k];
                        double stone_rep = stone_octopoles[atom][i][j][k];

                        octopoles[atom][j][i][k] = rep;
                        octopoles[atom][j][k][i] = rep;
                        octopoles[atom][i][k][j] = rep;
                        octopoles[atom][k][i][j] = rep;
                        octopoles[atom][k][j][i] = rep;

                        stone_octopoles[atom][j][i][k] = stone_rep;
                        stone_octopoles[atom][j][k][i] = stone_rep;
                        stone_octopoles[atom][i][k][j] = stone_rep;
                        stone_octopoles[atom][k][i][j] = stone_rep;
                        stone_octopoles[atom][k][j][i] = stone_rep;


                        psi::outfile->Printf("Atom %d, Octopole %c%c%c, %16.8f\n", atom+1, cartesian[i], cartesian[j], cartesian[k], octopoles[atom][i][j][k]);
                    }
                }
            }
        }
    }

    psi::outfile->Printf("\n\n");

    psi::outfile->Printf("======> SPHERICAL MULTIPOLES <======\n\n");

    //Stores and Calculates the Spherical Multipoles for Each Atom (Horton)
    std::vector<std::vector<double>> s_dipoles(num_atoms, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> s_quadrupoles(num_atoms, std::vector<double>(5, 0.0));
    std::vector<std::vector<double>> s_octopoles(num_atoms, std::vector<double>(7, 0.0));

    for (int atom = 0; atom < num_atoms; atom++) {
        s_dipoles[atom][0] = dipoles[atom][2];
        psi::outfile->Printf("Atom %d, Spherical Dipole %c%c%c, %16.8f\n", atom+1, '1', '0', ' ', s_dipoles[atom][0]);
        s_dipoles[atom][1] = dipoles[atom][0];
        psi::outfile->Printf("Atom %d, Spherical Dipole %c%c%c, %16.8f\n", atom+1, '1', '1', 'c', s_dipoles[atom][1]);
        s_dipoles[atom][2] = dipoles[atom][1];
        psi::outfile->Printf("Atom %d, Spherical Dipole %c%c%c, %16.8f\n", atom+1, '1', '1', 's', s_dipoles[atom][2]);

        s_quadrupoles[atom][0] = stone_quadrupoles[atom][2][2];
        psi::outfile->Printf("Atom %d, Spherical Quadrupole %c%c%c, %16.8f\n", atom+1, '2', '0', ' ', s_quadrupoles[atom][0]);
        s_quadrupoles[atom][1] = 2.0/sqrt(3.0) * stone_quadrupoles[atom][0][2];
        psi::outfile->Printf("Atom %d, Spherical Quadrupole %c%c%c, %16.8f\n", atom+1, '2', '1', 'c', s_quadrupoles[atom][1]);
        s_quadrupoles[atom][2] = 2.0/sqrt(3.0) * stone_quadrupoles[atom][1][2];
        psi::outfile->Printf("Atom %d, Spherical Quadrupole %c%c%c, %16.8f\n", atom+1, '2', '1', 's', s_quadrupoles[atom][2]);
        s_quadrupoles[atom][3] = 1.0/sqrt(3.0) * (stone_quadrupoles[atom][0][0] - stone_quadrupoles[atom][1][1]);
        psi::outfile->Printf("Atom %d, Spherical Quadrupole %c%c%c, %16.8f\n", atom+1, '2', '2', 'c', s_quadrupoles[atom][3]);
        s_quadrupoles[atom][4] = 2.0/sqrt(3.0) * stone_quadrupoles[atom][0][1];
        psi::outfile->Printf("Atom %d, Spherical Quadrupole %c%c%c, %16.8f\n", atom+1, '2', '2', 's', s_quadrupoles[atom][4]);

        s_octopoles[atom][0] = stone_octopoles[atom][2][2][2];
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '0', ' ', s_octopoles[atom][0]);
        s_octopoles[atom][1] = sqrt(1.5) * stone_octopoles[atom][0][2][2];
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '1', 'c', s_octopoles[atom][1]);
        s_octopoles[atom][2] = sqrt(1.5) * stone_octopoles[atom][1][2][2];
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '1', 's', s_octopoles[atom][2]);
        s_octopoles[atom][3] = sqrt(0.6) * (stone_octopoles[atom][0][0][2] - stone_octopoles[atom][1][1][2]);
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '2', 'c', s_octopoles[atom][3]);
        s_octopoles[atom][4] = 2.0 * sqrt(0.6) * (stone_octopoles[atom][0][1][2]);
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '2', 's', s_octopoles[atom][4]);
        s_octopoles[atom][5] = sqrt(0.1) * (stone_octopoles[atom][0][0][0] - 3.0*stone_octopoles[atom][0][1][1]);
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '3', 'c', s_octopoles[atom][5]);
        s_octopoles[atom][6] = sqrt(0.1) * (3.0*stone_octopoles[atom][0][0][1] - stone_octopoles[atom][1][1][1]);
        psi::outfile->Printf("Atom %d, Spherical Octopole %c%c%c, %16.8f\n", atom+1, '3', '3', 's', s_octopoles[atom][6]);

        psi::outfile->Printf("\n");
    }

    timer_off("MBIS");

    return ref_wfn;
}


}}
