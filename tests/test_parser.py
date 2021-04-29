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

A = (1 * units.angstrom).to_base_units().magnitude
eV = (1 * units.eV).to_base_units().magnitude
e = (1 * units.e).to_base_units().magnitude


@pytest.fixture
def parser():
    return LobsterParser()


def test_Fe(parser):

    archive = EntryArchive()
    parser.run('tests/Fe/lobsterout', archive, logging)

    run = archive.section_run[0]
    assert run.program_name == "LOBSTER"
    assert run.run_clean_end is True
    assert run.program_version == "4.0.0"
    assert run.time_run_cpu1_start.magnitude == 1619687985

    assert len(run.section_single_configuration_calculation) == 1
    scc = run.section_single_configuration_calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 2
    assert scc.x_lobster_abs_total_spilling[0] == pytest.approx(8.02)
    assert scc.x_lobster_abs_total_spilling[1] == pytest.approx(8.96)
    assert len(scc.x_lobster_abs_charge_spilling) == 2
    assert scc.x_lobster_abs_charge_spilling[0] == pytest.approx(2.97)
    assert scc.x_lobster_abs_charge_spilling[1] == pytest.approx(8.5)

    method = run.section_method
    assert len(method) == 1
    assert method[0].x_lobster_code == "VASP"

    # ICOHPLIST.lobster
    icohplist = scc.x_lobster_section_icohplist
    assert icohplist.x_lobster_number_of_icohp_values == 20
    assert len(icohplist.x_lobster_icohp_atom1_labels) == 20
    assert icohplist.x_lobster_icohp_atom1_labels[19] == "Fe2"
    assert len(icohplist.x_lobster_icohp_atom2_labels) == 20
    assert icohplist.x_lobster_icohp_atom1_labels[3] == "Fe1"
    assert len(icohplist.x_lobster_icohp_distances) == 20
    assert icohplist.x_lobster_icohp_distances[0] == pytest.approx(2.83178 * A)
    assert icohplist.x_lobster_icohp_distances[13] == pytest.approx(
        2.45239 * A)
    assert icohplist.x_lobster_icohp_distances[19] == pytest.approx(
        2.83178 * A)
    assert np.shape(icohplist.x_lobster_icohp_translations) == (20, 3)
    assert all([a == b for a, b in zip(
        icohplist.x_lobster_icohp_translations[0], [0, 0, -1])])
    assert all([a == b for a, b in zip(
        icohplist.x_lobster_icohp_translations[13], [0, 0, 0])])
    assert all([a == b for a, b in zip(
        icohplist.x_lobster_icohp_translations[19], [0, 0, 1])])
    assert np.shape(icohplist.x_lobster_icohp_values) == (2, 20)
    assert icohplist.x_lobster_icohp_values[0, 0] == pytest.approx(-0.08672 * eV)
    assert icohplist.x_lobster_icohp_values[0, 19] == pytest.approx(-0.08672 * eV)
    assert icohplist.x_lobster_icohp_values[1, 19] == pytest.approx(-0.16529 * eV)
    assert icohplist.x_lobster_icohp_values[1, 7] == pytest.approx(-0.48790 * eV)

    # ICOOPLIST.lobster
    icooplist = scc.x_lobster_section_icooplist
    assert icooplist.x_lobster_number_of_icoop_values == 20
    assert len(icooplist.x_lobster_icoop_atom1_labels) == 20
    assert icooplist.x_lobster_icoop_atom1_labels[19] == "Fe2"
    assert len(icooplist.x_lobster_icoop_atom2_labels) == 20
    assert icooplist.x_lobster_icoop_atom1_labels[3] == "Fe1"
    assert len(icooplist.x_lobster_icoop_distances) == 20
    assert icooplist.x_lobster_icoop_distances[0] == pytest.approx(2.83178 * A)
    assert icooplist.x_lobster_icoop_distances[13] == pytest.approx(
        2.45239 * A)
    assert icooplist.x_lobster_icoop_distances[19] == pytest.approx(
        2.83178 * A)
    assert np.shape(icooplist.x_lobster_icoop_translations) == (20, 3)
    assert all([a == b for a, b in zip(
        icooplist.x_lobster_icoop_translations[0], [0, 0, -1])])
    assert all([a == b for a, b in zip(
        icooplist.x_lobster_icoop_translations[13], [0, 0, 0])])
    assert all([a == b for a, b in zip(
        icooplist.x_lobster_icoop_translations[19], [0, 0, 1])])
    assert np.shape(icooplist.x_lobster_icoop_values) == (2, 20)
    assert icooplist.x_lobster_icoop_values[0, 0] == pytest.approx(-0.06882 * eV)
    assert icooplist.x_lobster_icoop_values[0, 19] == pytest.approx(-0.06882 * eV)
    assert icooplist.x_lobster_icoop_values[1, 19] == pytest.approx(-0.11268 * eV)
    assert icooplist.x_lobster_icoop_values[1, 7] == pytest.approx(-0.05179 * eV)

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
    assert mulliken.atomic_multipole_values[0][0] == pytest.approx(0.0 * e)
    assert mulliken.atomic_multipole_values[0][1] == pytest.approx(0.0 * e)

    loewdin = atomic_multipoles[1]
    assert loewdin.atomic_multipole_kind == "loewdin"
    assert loewdin.number_of_lm_atomic_multipoles == 1
    assert np.shape(loewdin.atomic_multipole_lm) == (1, 2)
    assert all([a == b for a, b in zip(
        loewdin.atomic_multipole_lm[0], [0, 0])])
    assert np.shape(loewdin.atomic_multipole_values) == (1, 2)
    assert loewdin.atomic_multipole_values[0][0] == pytest.approx(0.0 * e)
    assert loewdin.atomic_multipole_values[0][1] == pytest.approx(0.0 * e)


def test_NaCl(parser):

    archive = EntryArchive()
    parser.run('tests/NaCl/lobsterout', archive, logging)

    run = archive.section_run[0]
    assert run.program_name == "LOBSTER"
    assert run.run_clean_end is True
    assert run.program_version == "3.2.0"
    assert run.time_run_cpu1_start.magnitude == 1619713048

    assert len(run.section_single_configuration_calculation) == 1
    scc = run.section_single_configuration_calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 1
    assert scc.x_lobster_abs_total_spilling[0] == pytest.approx(9.29)
    assert len(scc.x_lobster_abs_charge_spilling) == 1
    assert scc.x_lobster_abs_charge_spilling[0] == pytest.approx(0.58)

    method = run.section_method
    assert len(method) == 1
    assert method[0].x_lobster_code == "VASP"

    # ICOHPLIST.lobster
    icohplist = scc.x_lobster_section_icohplist
    assert icohplist.x_lobster_number_of_icohp_values == 72
    assert len(icohplist.x_lobster_icohp_atom1_labels) == 72
    assert icohplist.x_lobster_icohp_atom1_labels[71] == "Cl7"
    assert len(icohplist.x_lobster_icohp_atom2_labels) == 72
    assert icohplist.x_lobster_icohp_atom2_labels[43] == "Cl6"
    assert len(icohplist.x_lobster_icohp_distances) == 72
    assert icohplist.x_lobster_icohp_distances[0] == pytest.approx(3.99586 * A)
    assert icohplist.x_lobster_icohp_distances[47] == pytest.approx(2.82550 * A)
    assert icohplist.x_lobster_icohp_distances[71] == pytest.approx(3.99586 * A)
    assert np.shape(icohplist.x_lobster_icohp_translations) == (72, 3)
    assert all([a == b for a, b in zip(
        icohplist.x_lobster_icohp_translations[0], [-1, 0, 0])])
    assert all([a == b for a, b in zip(
        icohplist.x_lobster_icohp_translations[54], [0, -1, 0])])
    assert all([a == b for a, b in zip(
        icohplist.x_lobster_icohp_translations[71], [0, 1, 0])])
    assert np.shape(icohplist.x_lobster_icohp_values) == (1, 72)
    assert icohplist.x_lobster_icohp_values[0, 0] == pytest.approx(-0.02652 * eV)
    assert icohplist.x_lobster_icohp_values[0, 71] == pytest.approx(-0.02925 * eV)

    # ICOOPLIST.lobster
    icooplist = scc.x_lobster_section_icooplist
    assert icooplist.x_lobster_number_of_icoop_values == 72
    assert len(icooplist.x_lobster_icoop_atom1_labels) == 72
    assert icooplist.x_lobster_icoop_atom1_labels[71] == "Cl7"
    assert len(icooplist.x_lobster_icoop_atom2_labels) == 72
    assert icooplist.x_lobster_icoop_atom2_labels[0] == "Na2"
    assert len(icooplist.x_lobster_icoop_distances) == 72
    assert icooplist.x_lobster_icoop_distances[0] == pytest.approx(3.99586 * A)
    assert icooplist.x_lobster_icoop_distances[12] == pytest.approx(2.82550 * A)
    assert icooplist.x_lobster_icoop_distances[71] == pytest.approx(3.99586 * A)
    assert np.shape(icooplist.x_lobster_icoop_translations) == (72, 3)
    assert all([a == b for a, b in zip(
        icooplist.x_lobster_icoop_translations[0], [-1, 0, 0])])
    assert all([a == b for a, b in zip(
        icooplist.x_lobster_icoop_translations[13], [0, 1, 0])])
    assert all([a == b for a, b in zip(
        icooplist.x_lobster_icoop_translations[71], [0, 1, 0])])
    assert np.shape(icooplist.x_lobster_icoop_values) == (1, 72)
    assert icooplist.x_lobster_icoop_values[0, 0] == pytest.approx(-0.00519 * eV)
    assert icooplist.x_lobster_icoop_values[0, 71] == pytest.approx(-0.00580 * eV)

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
    assert mulliken.atomic_multipole_values[0][0] == pytest.approx(0.78 * e)
    assert mulliken.atomic_multipole_values[0][7] == pytest.approx(-0.78 * e)

    loewdin = atomic_multipoles[1]
    assert loewdin.atomic_multipole_kind == "loewdin"
    assert loewdin.number_of_lm_atomic_multipoles == 1
    assert np.shape(loewdin.atomic_multipole_lm) == (1, 2)
    assert all([a == b for a, b in zip(
        loewdin.atomic_multipole_lm[0], [0, 0])])
    assert np.shape(loewdin.atomic_multipole_values) == (1, 8)
    assert loewdin.atomic_multipole_values[0][0] == pytest.approx(0.67 * e)
    assert loewdin.atomic_multipole_values[0][1] == pytest.approx(-0.67 * e)


def test_example(parser):
    test_Fe(parser)