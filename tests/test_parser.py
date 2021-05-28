#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import logging
import numpy as np

from nomad.datamodel import EntryArchive
from nomad.units import ureg as units

from lobsterparser import LobsterParser

e = (1 * units.e).to_base_units().magnitude
eV = (1 * units.e).to_base_units().magnitude


@pytest.fixture
def parser():
    return LobsterParser()


def A_to_m(value):
    return (value * units.angstrom).to_base_units().magnitude


def eV_to_J(value):
    return (value * units.eV).to_base_units().magnitude


# default pytest.approx settings are abs=1e-12, rel=1e-6 so it doesn't work for small numbers
# use the default just for comparison with zero
def approx(value):
    return pytest.approx(value, abs=0, rel=1e-6)


def test_Fe(parser):
    """
    Tests spin-polarized Fe calculation with LOBSTER 4.0.0
    """

    archive = EntryArchive()
    parser.parse('tests/Fe/lobsterout', archive, logging)

    run = archive.section_run[0]
    assert run.program_name == "LOBSTER"
    assert run.run_clean_end is True
    assert run.program_version == "4.0.0"
    assert run.time_run_wall_start.magnitude == 1619687985

    assert len(run.section_single_configuration_calculation) == 1
    scc = run.section_single_configuration_calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 2
    assert scc.x_lobster_abs_total_spilling[0] == approx(8.02)
    assert scc.x_lobster_abs_total_spilling[1] == approx(8.96)
    assert len(scc.x_lobster_abs_charge_spilling) == 2
    assert scc.x_lobster_abs_charge_spilling[0] == approx(2.97)
    assert scc.x_lobster_abs_charge_spilling[1] == approx(8.5)

    method = run.section_method
    assert len(method) == 1
    assert method[0].x_lobster_code == "VASP"
    assert method[0].basis_set == "pbeVaspFit2015"

    # ICOHPLIST.lobster
    cohp = scc.x_lobster_section_cohp
    assert cohp.x_lobster_number_of_cohp_pairs == 20
    assert len(cohp.x_lobster_cohp_atom1_labels) == 20
    assert cohp.x_lobster_cohp_atom1_labels[19] == "Fe2"
    assert len(cohp.x_lobster_cohp_atom2_labels) == 20
    assert cohp.x_lobster_cohp_atom1_labels[3] == "Fe1"
    assert len(cohp.x_lobster_cohp_distances) == 20
    assert cohp.x_lobster_cohp_distances[0].magnitude == approx(
        A_to_m(2.831775))
    assert cohp.x_lobster_cohp_distances[13].magnitude == approx(
        A_to_m(2.45239))
    assert cohp.x_lobster_cohp_distances[19].magnitude == approx(
        A_to_m(2.831775))
    assert np.shape(cohp.x_lobster_cohp_translations) == (20, 3)
    assert all([a == b for a, b in zip(
        cohp.x_lobster_cohp_translations[0], [0, 0, -1])])
    assert all([a == b for a, b in zip(
        cohp.x_lobster_cohp_translations[13], [0, 0, 0])])
    assert all([a == b for a, b in zip(
        cohp.x_lobster_cohp_translations[19], [0, 0, 1])])
    assert np.shape(cohp.x_lobster_integrated_cohp_at_fermi_level) == (2, 20)
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.08672))
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 19].magnitude == approx(
        eV_to_J(-0.08672))
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[1, 19].magnitude == approx(
        eV_to_J(-0.16529))
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[1, 7].magnitude == approx(
        eV_to_J(-0.48790))

    # COHPCAR.lobster
    assert len(cohp.x_lobster_cohp_energies) == 201
    assert cohp.x_lobster_cohp_energies[0].magnitude == approx(eV_to_J(-10.06030))
    assert cohp.x_lobster_cohp_energies[200].magnitude == approx(eV_to_J(3.00503))
    assert np.shape(cohp.x_lobster_average_cohp_values) == (2, 201)
    assert cohp.x_lobster_average_cohp_values[0][196] == approx(0.02406)
    assert cohp.x_lobster_average_cohp_values[1][200] == approx(0.01816)
    assert np.shape(cohp.x_lobster_average_integrated_cohp_values) == (2, 201)
    assert cohp.x_lobster_average_integrated_cohp_values[0][200].magnitude == approx(
        eV_to_J(-0.06616))
    assert cohp.x_lobster_average_integrated_cohp_values[1][200].magnitude == approx(
        eV_to_J(-0.02265))
    assert np.shape(cohp.x_lobster_cohp_values) == (20, 2, 201)
    assert cohp.x_lobster_cohp_values[10][1][200] == approx(0.02291)
    assert cohp.x_lobster_cohp_values[19][0][200] == approx(0.01439)
    assert np.shape(cohp.x_lobster_integrated_cohp_values) == (20, 2, 201)
    assert cohp.x_lobster_integrated_cohp_values[10][0][200].magnitude == approx(
        eV_to_J(-0.12881))
    assert cohp.x_lobster_integrated_cohp_values[19][1][200].magnitude == approx(
        eV_to_J(-0.06876))

    # ICOOPLIST.lobster
    coop = scc.x_lobster_section_coop
    assert coop.x_lobster_number_of_coop_pairs == 20
    assert len(coop.x_lobster_coop_atom1_labels) == 20
    assert coop.x_lobster_coop_atom1_labels[19] == "Fe2"
    assert len(coop.x_lobster_coop_atom2_labels) == 20
    assert coop.x_lobster_coop_atom1_labels[3] == "Fe1"
    assert len(coop.x_lobster_coop_distances) == 20
    assert coop.x_lobster_coop_distances[0].magnitude == approx(
        A_to_m(2.831775))
    assert coop.x_lobster_coop_distances[13].magnitude == approx(
        A_to_m(2.45239))
    assert coop.x_lobster_coop_distances[19].magnitude == approx(
        A_to_m(2.831775))
    assert np.shape(coop.x_lobster_coop_translations) == (20, 3)
    assert all([a == b for a, b in zip(
        coop.x_lobster_coop_translations[0], [0, 0, -1])])
    assert all([a == b for a, b in zip(
        coop.x_lobster_coop_translations[13], [0, 0, 0])])
    assert all([a == b for a, b in zip(
        coop.x_lobster_coop_translations[19], [0, 0, 1])])
    assert np.shape(coop.x_lobster_integrated_coop_at_fermi_level) == (2, 20)
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.06882))
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 19].magnitude == approx(
        eV_to_J(-0.06882))
    assert coop.x_lobster_integrated_coop_at_fermi_level[1, 19].magnitude == approx(
        eV_to_J(-0.11268))
    assert coop.x_lobster_integrated_coop_at_fermi_level[1, 7].magnitude == approx(
        eV_to_J(-0.05179))

    # COOPCAR.lobster
    assert len(coop.x_lobster_coop_energies) == 201
    assert coop.x_lobster_coop_energies[0].magnitude == approx(eV_to_J(-10.06030))
    assert coop.x_lobster_coop_energies[200].magnitude == approx(eV_to_J(3.00503))
    assert np.shape(coop.x_lobster_average_coop_values) == (2, 201)
    assert coop.x_lobster_average_coop_values[0][196] == approx(-0.04773)
    assert coop.x_lobster_average_coop_values[1][200] == approx(-0.04542)
    assert np.shape(coop.x_lobster_average_integrated_coop_values) == (2, 201)
    assert coop.x_lobster_average_integrated_coop_values[0][200].magnitude == approx(
        eV_to_J(-0.12265))
    assert coop.x_lobster_average_integrated_coop_values[1][200].magnitude == approx(
        eV_to_J(-0.14690))
    assert np.shape(coop.x_lobster_coop_values) == (20, 2, 201)
    assert coop.x_lobster_coop_values[3][1][200] == approx(-0.01346)
    assert coop.x_lobster_coop_values[0][0][200] == approx(-0.04542)
    assert np.shape(coop.x_lobster_integrated_coop_values) == (20, 2, 201)
    assert coop.x_lobster_integrated_coop_values[10][0][199].magnitude == approx(
        eV_to_J(-0.07360))
    assert coop.x_lobster_integrated_coop_values[19][1][200].magnitude == approx(
        eV_to_J(-0.13041))

    # CHARGE.lobster
    atomic_multipoles = scc.section_atomic_multipoles
    assert len(atomic_multipoles) == 2
    mulliken = atomic_multipoles[0]
    assert mulliken.atomic_multipole_kind == "mulliken"
    assert mulliken.number_of_lm_atomic_multipoles == 1
    assert np.shape(mulliken.atomic_multipole_lm) == (1, 2)
    assert all([a == b for a, b in zip(
        mulliken.atomic_multipole_lm[0], [0, 0])])
    assert np.shape(mulliken.atomic_multipole_values) == (1, 2)
    assert mulliken.atomic_multipole_values[0][0] == pytest.approx(0.0 * e, abs=1e-6)
    assert mulliken.atomic_multipole_values[0][1] == pytest.approx(0.0 * e, abs=1e-6)

    loewdin = atomic_multipoles[1]
    assert loewdin.atomic_multipole_kind == "loewdin"
    assert loewdin.number_of_lm_atomic_multipoles == 1
    assert np.shape(loewdin.atomic_multipole_lm) == (1, 2)
    assert all([a == b for a, b in zip(
        loewdin.atomic_multipole_lm[0], [0, 0])])
    assert np.shape(loewdin.atomic_multipole_values) == (1, 2)
    assert loewdin.atomic_multipole_values[0][0] == pytest.approx(0.0 * e, abs=1e-6)
    assert loewdin.atomic_multipole_values[0][1] == pytest.approx(0.0 * e, abs=1e-6)

    # DOSCAR.lobster total and integrated DOS
    assert len(scc.section_dos) == 1
    dos = scc.section_dos[0]
    assert dos.dos_kind == 'electronic'
    assert dos.number_of_dos_values == 201
    assert len(dos.dos_energies) == 201
    assert dos.dos_energies[0].magnitude == approx(eV_to_J(-10.06030))
    assert dos.dos_energies[16].magnitude == approx(eV_to_J(-9.01508))
    assert dos.dos_energies[200].magnitude == approx(eV_to_J(3.00503))
    assert np.shape(dos.dos_values) == (2, 201)
    assert dos.dos_values[0][6] == pytest.approx(0.0, abs=1e-30)
    assert dos.dos_values[0][200] == approx(0.26779 / eV)
    assert dos.dos_values[1][195] == approx(0.37457 / eV)
    assert np.shape(dos.dos_integrated_values) == (2, 201)
    assert dos.dos_integrated_values[0][10] == approx(0.0 + 18)
    assert dos.dos_integrated_values[0][188] == approx(11.07792 + 18)
    assert dos.dos_integrated_values[1][200] == approx(10.75031 + 18)

    # DOSCAR.lobster atom and lm-projected dos
    assert len(scc.x_lobster_section_atom_projected_dos) == 2
    ados1 = scc.x_lobster_section_atom_projected_dos[0]
    ados2 = scc.x_lobster_section_atom_projected_dos[1]
    ados1.x_lobster_atom_projected_dos_atom_index == 1
    ados2.x_lobster_atom_projected_dos_atom_index == 2
    assert ados2.x_lobster_number_of_dos_values == 201
    assert len(ados2.x_lobster_atom_projected_dos_energies) == 201
    assert ados2.x_lobster_atom_projected_dos_energies[0].magnitude == approx(
        eV_to_J(-10.06030))
    assert ados2.x_lobster_atom_projected_dos_energies[16].magnitude == approx(
        eV_to_J(-9.01508))
    assert ados1.x_lobster_atom_projected_dos_energies[200].magnitude == approx(
        eV_to_J(3.00503))
    assert ados2.x_lobster_atom_projected_dos_m_kind == 'real_orbital'
    assert ados2.x_lobster_number_of_lm_atom_projected_dos == 6
    assert np.shape(ados2.x_lobster_atom_projected_dos_lm) == (6, 2)
    assert all([a[0] == b[0] and a[1] == b[1] for a, b in zip(
        ados1.x_lobster_atom_projected_dos_lm,
        [[0, 0], [2, 3], [2, 2], [2, 0], [2, 1], [2, 4]])])
    assert all([a[0] == b[0] and a[1] == b[1] for a, b in zip(
        ados2.x_lobster_atom_projected_dos_lm,
        [[0, 0], [2, 3], [2, 2], [2, 0], [2, 1], [2, 4]])])
    assert np.shape(ados1.x_lobster_atom_projected_dos_values_lm) == (6, 2, 201)
    assert np.shape(ados2.x_lobster_atom_projected_dos_values_lm) == (6, 2, 201)
    assert ados1.x_lobster_atom_projected_dos_values_lm[2, 1, 190] == approx(
        0.21304 / eV)
    assert ados2.x_lobster_atom_projected_dos_values_lm[5, 0, 200] == approx(
        0.00784 / eV)
    assert ados2.x_lobster_atom_projected_dos_values_lm[0, 1, 35] == approx(
        0.01522 / eV)


def test_NaCl(parser):
    """
    Test non-spin-polarized NaCl calculation with LOBSTER 3.2.0
    """

    archive = EntryArchive()
    parser.parse('tests/NaCl/lobsterout', archive, logging)

    run = archive.section_run[0]
    assert run.program_name == "LOBSTER"
    assert run.run_clean_end is True
    assert run.program_version == "3.2.0"
    assert run.time_run_wall_start.magnitude == 1619713048

    assert len(run.section_single_configuration_calculation) == 1
    scc = run.section_single_configuration_calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 1
    assert scc.x_lobster_abs_total_spilling[0] == approx(9.29)
    assert len(scc.x_lobster_abs_charge_spilling) == 1
    assert scc.x_lobster_abs_charge_spilling[0] == approx(0.58)

    method = run.section_method
    assert len(method) == 1
    assert method[0].x_lobster_code == "VASP"
    assert method[0].basis_set == "pbeVaspFit2015"

    # ICOHPLIST.lobster
    cohp = scc.x_lobster_section_cohp
    assert cohp.x_lobster_number_of_cohp_pairs == 72
    assert len(cohp.x_lobster_cohp_atom1_labels) == 72
    assert cohp.x_lobster_cohp_atom1_labels[71] == "Cl7"
    assert len(cohp.x_lobster_cohp_atom2_labels) == 72
    assert cohp.x_lobster_cohp_atom2_labels[43] == "Cl6"
    assert len(cohp.x_lobster_cohp_distances) == 72
    assert cohp.x_lobster_cohp_distances[0].magnitude == approx(A_to_m(3.99586))
    assert cohp.x_lobster_cohp_distances[47].magnitude == approx(A_to_m(2.82550))
    assert cohp.x_lobster_cohp_distances[71].magnitude == approx(A_to_m(3.99586))
    assert np.shape(cohp.x_lobster_cohp_translations) == (72, 3)
    assert all([a == b for a, b in zip(
        cohp.x_lobster_cohp_translations[0], [-1, 0, 0])])
    assert all([a == b for a, b in zip(
        cohp.x_lobster_cohp_translations[54], [0, -1, 0])])
    assert all([a == b for a, b in zip(
        cohp.x_lobster_cohp_translations[71], [0, 1, 0])])
    assert np.shape(cohp.x_lobster_integrated_cohp_at_fermi_level) == (1, 72)
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.02652))
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 71].magnitude == approx(
        eV_to_J(-0.02925))

    # COHPCAR.lobster
    assert len(cohp.x_lobster_cohp_energies) == 201
    assert cohp.x_lobster_cohp_energies[0].magnitude == approx(eV_to_J(-12.02261))
    assert cohp.x_lobster_cohp_energies[200].magnitude == approx(eV_to_J(2.55025))
    assert np.shape(cohp.x_lobster_average_cohp_values) == (1, 201)
    assert cohp.x_lobster_average_cohp_values[0][0] == pytest.approx(0.0)
    assert cohp.x_lobster_average_cohp_values[0][151] == approx(-0.03162)
    assert np.shape(cohp.x_lobster_average_integrated_cohp_values) == (1, 201)
    assert cohp.x_lobster_average_integrated_cohp_values[0][0].magnitude == approx(
        eV_to_J(-0.15834))
    assert cohp.x_lobster_average_integrated_cohp_values[0][200].magnitude == approx(
        eV_to_J(-0.24310))
    assert np.shape(cohp.x_lobster_cohp_values) == (72, 1, 201)
    assert cohp.x_lobster_cohp_values[1][0][200] == pytest.approx(0.0)
    assert cohp.x_lobster_cohp_values[71][0][140] == approx(-0.00403)
    assert np.shape(cohp.x_lobster_integrated_cohp_values) == (72, 1, 201)
    assert cohp.x_lobster_integrated_cohp_values[2][0][200].magnitude == approx(
        eV_to_J(-0.02652))
    assert cohp.x_lobster_integrated_cohp_values[67][0][199].magnitude == approx(
        eV_to_J(-0.04137))

    # ICOOPLIST.lobster
    coop = scc.x_lobster_section_coop
    assert coop.x_lobster_number_of_coop_pairs == 72
    assert len(coop.x_lobster_coop_atom1_labels) == 72
    assert coop.x_lobster_coop_atom1_labels[71] == "Cl7"
    assert len(coop.x_lobster_coop_atom2_labels) == 72
    assert coop.x_lobster_coop_atom2_labels[0] == "Na2"
    assert len(coop.x_lobster_coop_distances) == 72
    assert coop.x_lobster_coop_distances[0].magnitude == approx(A_to_m(3.99586))
    assert coop.x_lobster_coop_distances[12].magnitude == approx(A_to_m(2.82550))
    assert coop.x_lobster_coop_distances[71].magnitude == approx(A_to_m(3.99586))
    assert np.shape(coop.x_lobster_coop_translations) == (72, 3)
    assert all([a == b for a, b in zip(
        coop.x_lobster_coop_translations[0], [-1, 0, 0])])
    assert all([a == b for a, b in zip(
        coop.x_lobster_coop_translations[13], [0, 1, 0])])
    assert all([a == b for a, b in zip(
        coop.x_lobster_coop_translations[71], [0, 1, 0])])
    assert np.shape(coop.x_lobster_integrated_coop_at_fermi_level) == (1, 72)
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.00519))
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 71].magnitude == approx(
        eV_to_J(-0.00580))

    # COOPCAR.lobster
    assert len(coop.x_lobster_coop_energies) == 201
    assert coop.x_lobster_coop_energies[0].magnitude == approx(eV_to_J(-12.02261))
    assert coop.x_lobster_coop_energies[200].magnitude == approx(eV_to_J(2.55025))
    assert np.shape(coop.x_lobster_average_coop_values) == (1, 201)
    assert coop.x_lobster_average_coop_values[0][0] == pytest.approx(0.0)
    assert coop.x_lobster_average_coop_values[0][145] == approx(0.03178)
    assert np.shape(coop.x_lobster_average_integrated_coop_values) == (1, 201)
    assert coop.x_lobster_average_integrated_coop_values[0][0].magnitude == approx(
        eV_to_J(0.00368))
    assert coop.x_lobster_average_integrated_coop_values[0][200].magnitude == approx(
        eV_to_J(0.00682))
    assert np.shape(coop.x_lobster_coop_values) == (72, 1, 201)
    assert coop.x_lobster_coop_values[1][0][200] == pytest.approx(0.0)
    assert coop.x_lobster_coop_values[71][0][143] == approx(0.01862)
    assert np.shape(coop.x_lobster_integrated_coop_values) == (72, 1, 201)
    assert coop.x_lobster_integrated_coop_values[2][0][200].magnitude == approx(
        eV_to_J(-0.00519))
    assert coop.x_lobster_integrated_coop_values[71][0][199].magnitude == approx(
        eV_to_J(-0.00580))

    # CHARGE.lobster
    atomic_multipoles = scc.section_atomic_multipoles
    assert len(atomic_multipoles) == 2
    mulliken = atomic_multipoles[0]
    assert mulliken.atomic_multipole_kind == "mulliken"
    assert mulliken.number_of_lm_atomic_multipoles == 1
    assert np.shape(mulliken.atomic_multipole_lm) == (1, 2)
    assert all([a == b for a, b in zip(
        mulliken.atomic_multipole_lm[0], [0, 0])])
    assert np.shape(mulliken.atomic_multipole_values) == (1, 8)
    # here the approx is not really working (changing the 0.78 to for example
    # 10 makes the test still pass)
    assert mulliken.atomic_multipole_values[0][0] == approx(0.78 * e)
    assert mulliken.atomic_multipole_values[0][7] == approx(-0.78 * e)

    loewdin = atomic_multipoles[1]
    assert loewdin.atomic_multipole_kind == "loewdin"
    assert loewdin.number_of_lm_atomic_multipoles == 1
    assert np.shape(loewdin.atomic_multipole_lm) == (1, 2)
    assert all([a == b for a, b in zip(
        loewdin.atomic_multipole_lm[0], [0, 0])])
    assert np.shape(loewdin.atomic_multipole_values) == (1, 8)
    assert loewdin.atomic_multipole_values[0][0] == approx(0.67 * e)
    assert loewdin.atomic_multipole_values[0][7] == approx(-0.67 * e)

    # DOSCAR.lobster total and integrated DOS
    assert len(scc.section_dos) == 1
    dos = scc.section_dos[0]
    assert dos.dos_kind == 'electronic'
    assert dos.number_of_dos_values == 201
    assert len(dos.dos_energies) == 201
    assert dos.dos_energies[0].magnitude == approx(eV_to_J(-12.02261))
    assert dos.dos_energies[25].magnitude == approx(eV_to_J(-10.20101))
    assert dos.dos_energies[200].magnitude == approx(eV_to_J(2.55025))
    assert np.shape(dos.dos_values) == (1, 201)
    assert dos.dos_values[0][6] == pytest.approx(0.0, abs=1e-30)
    assert dos.dos_values[0][162] == approx(20.24722 / eV)
    assert dos.dos_values[0][200] == pytest.approx(0.0, abs=1e-30)
    assert np.shape(dos.dos_integrated_values) == (1, 201)
    assert dos.dos_integrated_values[0][10] == approx(7.99998 + 80)
    assert dos.dos_integrated_values[0][160] == approx(27.09225 + 80)
    assert dos.dos_integrated_values[0][200] == approx(31.99992 + 80)

    # DOSCAR.lobster atom and lm-projected dos
    assert len(scc.x_lobster_section_atom_projected_dos) == 8
    ados1 = scc.x_lobster_section_atom_projected_dos[0]
    ados8 = scc.x_lobster_section_atom_projected_dos[7]
    ados1.x_lobster_atom_projected_dos_atom_index == 1
    ados8.x_lobster_atom_projected_dos_atom_index == 8
    assert ados8.x_lobster_number_of_dos_values == 201
    assert len(ados8.x_lobster_atom_projected_dos_energies) == 201
    assert ados1.x_lobster_atom_projected_dos_energies[0].magnitude == approx(
        eV_to_J(-12.02261))
    assert ados8.x_lobster_atom_projected_dos_energies[25].magnitude == approx(
        eV_to_J(-10.20101))
    assert ados8.x_lobster_atom_projected_dos_energies[200].magnitude == approx(
        eV_to_J(2.55025))
    assert ados8.x_lobster_atom_projected_dos_m_kind == 'real_orbital'
    assert ados1.x_lobster_number_of_lm_atom_projected_dos == 1
    assert ados8.x_lobster_number_of_lm_atom_projected_dos == 4
    assert np.shape(ados1.x_lobster_atom_projected_dos_lm) == (1, 2)
    assert np.shape(ados8.x_lobster_atom_projected_dos_lm) == (4, 2)
    assert all([a[0] == b[0] and a[1] == b[1] for a, b in zip(
        ados1.x_lobster_atom_projected_dos_lm,
        [[0, 0]])])
    assert all([a[0] == b[0] and a[1] == b[1] for a, b in zip(
        ados8.x_lobster_atom_projected_dos_lm,
        [[0, 0], [1, 2], [1, 0], [1, 1]])])
    assert np.shape(ados1.x_lobster_atom_projected_dos_values_lm) == (1, 1, 201)
    assert np.shape(ados8.x_lobster_atom_projected_dos_values_lm) == (4, 1, 201)
    assert ados1.x_lobster_atom_projected_dos_values_lm[0, 0, 190] == pytest.approx(
        0.0, abs=1e-30)
    assert ados8.x_lobster_atom_projected_dos_values_lm[3, 0, 141] == approx(
        0.32251 / eV)
    assert ados8.x_lobster_atom_projected_dos_values_lm[0, 0, 152] == approx(
        0.00337 / eV)


def test_HfV(parser):
    """
    Test non-spin-polarized HfV2 calculation with LOBSTER 2.0.0,
    it has different ICOHPLIST.lobster and ICOOPLIST.lobster scheme.
    Also test backup structure parsing when no CONTCAR is present.
    """

    archive = EntryArchive()
    parser.parse('tests/HfV2/lobsterout', archive, logging)

    run = archive.section_run[0]
    assert run.program_name == "LOBSTER"
    assert run.run_clean_end is True
    assert run.program_version == "2.0.0"

    assert len(run.section_single_configuration_calculation) == 1
    scc = run.section_single_configuration_calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 1
    assert scc.x_lobster_abs_total_spilling[0] == approx(4.39)
    assert len(scc.x_lobster_abs_charge_spilling) == 1
    assert scc.x_lobster_abs_charge_spilling[0] == approx(2.21)

    # backup partial system parsing
    system = run.section_system
    assert len(system) == 1
    assert len(system[0].atom_species) == 12
    assert all([a == b for a, b in zip(system[0].atom_species,
               [72, 72, 72, 72, 23, 23, 23, 23, 23, 23, 23, 23])])
    assert all([a == b for a, b in zip(system[0].configuration_periodic_dimensions,
               [True, True, True])])

    # method
    method = run.section_method
    assert method[0].basis_set == "Koga"

    # ICOHPLIST.lobster
    cohp = scc.x_lobster_section_cohp
    assert cohp.x_lobster_number_of_cohp_pairs == 56
    assert len(cohp.x_lobster_cohp_atom1_labels) == 56
    assert cohp.x_lobster_cohp_atom1_labels[41] == "V6"
    assert len(cohp.x_lobster_cohp_atom2_labels) == 56
    assert cohp.x_lobster_cohp_atom2_labels[16] == "V9"
    assert len(cohp.x_lobster_cohp_distances) == 56
    assert cohp.x_lobster_cohp_distances[0].magnitude == approx(A_to_m(3.17294))
    assert cohp.x_lobster_cohp_distances[47].magnitude == approx(A_to_m(2.60684))
    assert cohp.x_lobster_cohp_distances[55].magnitude == approx(A_to_m(2.55809))
    assert cohp.x_lobster_cohp_translations is None
    assert len(cohp.x_lobster_cohp_number_of_bonds) == 56
    assert cohp.x_lobster_cohp_number_of_bonds[0] == 2
    assert cohp.x_lobster_cohp_number_of_bonds[53] == 1
    assert np.shape(cohp.x_lobster_integrated_cohp_at_fermi_level) == (1, 56)
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-1.72125))
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 55].magnitude == approx(
        eV_to_J(-1.62412))

    # ICOOPLIST.lobster
    coop = scc.x_lobster_section_coop
    assert coop.x_lobster_number_of_coop_pairs == 56
    assert len(coop.x_lobster_coop_atom1_labels) == 56
    assert coop.x_lobster_coop_atom1_labels[41] == "V6"
    assert len(coop.x_lobster_coop_atom2_labels) == 56
    assert coop.x_lobster_coop_atom2_labels[11] == "Hf4"
    assert len(coop.x_lobster_coop_distances) == 56
    assert coop.x_lobster_coop_distances[0].magnitude == approx(A_to_m(3.17294))
    assert coop.x_lobster_coop_distances[47].magnitude == approx(A_to_m(2.60684))
    assert coop.x_lobster_coop_distances[55].magnitude == approx(A_to_m(2.55809))
    assert coop.x_lobster_coop_translations is None
    assert len(coop.x_lobster_coop_number_of_bonds) == 56
    assert coop.x_lobster_coop_number_of_bonds[0] == 2
    assert coop.x_lobster_coop_number_of_bonds[53] == 1
    assert np.shape(coop.x_lobster_integrated_coop_at_fermi_level) == (1, 56)
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.46493))
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 55].magnitude == approx(
        eV_to_J(-0.50035))


def test_failed_case(parser):
    """
    Check that we also handle gracefully a case where the lobster ends very early.
    Here it is because of a wrong CONTCAR.
    """

    archive = EntryArchive()
    parser.parse('tests/failed_case/lobsterout', archive, logging)

    run = archive.section_run[0]
    assert run.run_clean_end is False
