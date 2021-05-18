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

import datetime
import numpy as np
import ase.io
from os import path

from nomad.datamodel import EntryArchive
from nomad.parsing import FairdiParser
from nomad.units import ureg as units
from nomad.datamodel.metainfo.public import section_run as Run
from nomad.datamodel.metainfo.public import section_system as System
from nomad.datamodel.metainfo.public import section_method as Method
from nomad.datamodel.metainfo.public import section_single_configuration_calculation as SCC
from nomad.datamodel.metainfo.public import section_atomic_multipoles, section_dos

from nomad.parsing.file_parser import UnstructuredTextFileParser, Quantity

from .metainfo.lobster import x_lobster_section_cohp, x_lobster_section_coop, \
    x_lobster_section_atom_projected_dos

'''
This is a LOBSTER code parser.
'''

e = (1 * units.e).to_base_units().magnitude
eV = (1 * units.eV).to_base_units().magnitude


def parse_ICOXPLIST(fname, scc, method):

    def icoxp_line_split(string):
        tmp = string.split()
        # LOBSTER version 3 and above
        if len(tmp) == 8:
            return [tmp[1], tmp[2], float(tmp[3]), [int(tmp[4]),
                    int(tmp[5]), int(tmp[6])], float(tmp[7])]
        # LOBSTER versions below 3
        elif len(tmp) == 6:
            return [tmp[1], tmp[2], float(tmp[3]), float(tmp[4]), int(tmp[5])]

    icoxplist_parser = UnstructuredTextFileParser(quantities=[
        Quantity('icoxpslist_for_spin', r'\s*CO[OH]P.*spin\s*\d\s*([^#]+[-\d\.]+)',
                 repeats=True,
                 sub_parser=UnstructuredTextFileParser(quantities=[
                     Quantity('line',
                              # LOBSTER version 3 and above
                              r'(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\d]+\s+[-\d]+\s+[-\d]+\s+[-\.\d]+\s*)|'
                              # LOBSTER versions below 3
                              r'(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\.\d]+\s+[\d]+\s*)',
                              repeats=True, str_operation=icoxp_line_split)])
                 )
    ])

    if not path.isfile(fname):
        return
    icoxplist_parser.mainfile = fname
    icoxplist_parser.parse()

    icoxp = []
    for spin, icoxplist in enumerate(icoxplist_parser.get('icoxpslist_for_spin')):

        lines = icoxplist.get('line')
        if lines is None:
            break
        if type(lines[0][4]) is int:
            a1, a2, distances, tmp, bonds = zip(*lines)
        else:
            a1, a2, distances, v, tmp = zip(*lines)
        icoxp.append(0)
        icoxp[-1] = list(tmp)
        if spin == 0:
            if method == 'o':
                section = scc.m_create(x_lobster_section_coop)
            elif method == 'h':
                section = scc.m_create(x_lobster_section_cohp)

            setattr(section, "x_lobster_number_of_co{}p_pairs".format(
                method), len(list(a1)))
            setattr(section, "x_lobster_co{}p_atom1_labels".format(
                method), list(a1))
            setattr(section, "x_lobster_co{}p_atom2_labels".format(
                method), list(a2))
            setattr(section, "x_lobster_co{}p_distances".format(
                method), np.array(distances) * units.angstrom)

            # version specific entries
            if 'v' in locals():
                setattr(section, "x_lobster_co{}p_translations".format(
                    method), list(v))
            if 'bonds' in locals():
                setattr(section, "x_lobster_co{}p_number_of_bonds".format(
                    method), list(bonds))

    if len(icoxp) > 0:
        setattr(section, "x_lobster_integrated_co{}p_at_fermi_level".format(
            method), np.array(icoxp) * units.eV)


def parse_COXPCAR(fname, scc, method, logger):
    coxpcar_parser = UnstructuredTextFileParser(quantities=[
        Quantity('coxp_pairs', r'No\.\d+:(\w{1,2}\d+)->(\w{1,2}\d+)\(([\d\.]+)\)\s*?',
                 repeats=True),
        Quantity('coxp_lines', r'\n\s*(-*\d+\.\d+(?:[ \t]+-*\d+\.\d+)+)',
                 repeats=True)
    ])

    if not path.isfile(fname):
        return
    coxpcar_parser.mainfile = fname
    coxpcar_parser.parse()

    if method == 'o':
        if not scc.x_lobster_section_coop:
            section = scc.m_create(x_lobster_section_coop)
        else:
            section = scc.x_lobster_section_coop
    elif method == 'h':
        if not scc.x_lobster_section_cohp:
            section = scc.m_create(x_lobster_section_cohp)
        else:
            section = scc.x_lobster_section_cohp

    pairs = coxpcar_parser.get('coxp_pairs')
    a1, a2, distances = zip(*pairs)
    number_of_pairs = len(list(a1))

    setattr(section, "x_lobster_number_of_co{}p_pairs".format(
        method), number_of_pairs)
    setattr(section, "x_lobster_co{}p_atom1_labels".format(
        method), list(a1))
    setattr(section, "x_lobster_co{}p_atom2_labels".format(
        method), list(a2))
    setattr(section, "x_lobster_co{}p_distances".format(
        method), np.array(distances) * units.angstrom)

    coxp_lines = coxpcar_parser.get('coxp_lines')
    coxp_lines = list(zip(*coxp_lines))

    setattr(section, "x_lobster_number_of_co{}p_values".format(
        method), len(coxp_lines[0]))
    setattr(section, "x_lobster_co{}p_energies".format(
        method), np.array(coxp_lines[0]) * units.eV)

    if len(coxp_lines) == 2 * number_of_pairs + 3:
        coxp = [[x] for x in coxp_lines[3::2]]
        icoxp = [[x] for x in coxp_lines[4::2]]
        acoxp = [coxp_lines[1]]
        aicoxp = [coxp_lines[2]]
    elif len(coxp_lines) == 4 * number_of_pairs + 5:
        coxp = [x for x in zip(coxp_lines[5:number_of_pairs * 2 + 4:2],
                coxp_lines[number_of_pairs * 2 + 5: 4 * number_of_pairs + 4:2])]
        icoxp = [x for x in zip(coxp_lines[6:number_of_pairs * 2 + 5:2],
                 coxp_lines[number_of_pairs * 2 + 6: 4 * number_of_pairs + 5:2])]
        acoxp = [coxp_lines[1], coxp_lines[3]]
        aicoxp = [coxp_lines[2], coxp_lines[4]]
    else:
        logger.warning('Unexpected number of columns {} '
                       'in CO{}PCAR.lobster.'.format(len(coxp_lines),
                                                     method.upper()))
        return

    # FIXME: correct magnitude?
    setattr(section, "x_lobster_co{}p_values".format(
        method), np.array(coxp))
    setattr(section, "x_lobster_average_co{}p_values".format(
        method), np.array(acoxp))
    setattr(section, "x_lobster_integrated_co{}p_values".format(
        method), np.array(icoxp) * units.eV)
    setattr(section, "x_lobster_average_integrated_co{}p_values".format(
        method), np.array(aicoxp) * units.eV)
    setattr(section, "x_lobster_integrated_co{}p_values".format(
        method), np.array(icoxp) * units.eV)


def parse_CHARGE(fname, scc):
    charge_parser = UnstructuredTextFileParser(quantities=[
        Quantity(
            'charges', r'\s*\d+\s+[A-Za-z]{1,2}\s+([-\d\.]+)\s+([-\d\.]+)\s*', repeats=True)
    ])

    if not path.isfile(fname):
        return
    charge_parser.mainfile = fname
    charge_parser.parse()

    charges = charge_parser.get('charges')
    if charges is not None:
        scc.m_create(section_atomic_multipoles)
        scc.m_create(section_atomic_multipoles)
        scc.section_atomic_multipoles[0].atomic_multipole_kind = "mulliken"
        scc.section_atomic_multipoles[0].atomic_multipole_m_kind = "integrated"
        scc.section_atomic_multipoles[0].atomic_multipole_lm = np.array([[0, 0]])
        scc.section_atomic_multipoles[0].number_of_lm_atomic_multipoles = 1
        # FIXME: the mulliken charge has an obvious unit,
        # but section_atomic_multipoles.atomic_multipole_values is unitless currently
        scc.section_atomic_multipoles[0].atomic_multipole_values = np.array(
            [list(zip(*charges))[0]]) * e
        # FIXME: Loewdin charge might not be allowed here according to the wiki?
        scc.section_atomic_multipoles[1].atomic_multipole_kind = "loewdin"
        scc.section_atomic_multipoles[1].atomic_multipole_m_kind = "integrated"
        scc.section_atomic_multipoles[1].atomic_multipole_lm = np.array([[0, 0]])
        scc.section_atomic_multipoles[1].number_of_lm_atomic_multipoles = 1
        scc.section_atomic_multipoles[1].atomic_multipole_values = np.array(
            [list(zip(*charges))[1]]) * e


def parse_DOSCAR(fname, run, logger):

    def parse_species(run, atomic_numbers):
        """
        If we don't have any structure from the underlying DFT code, we can
        at least figure out what atoms we have in the structure. The best place
        to get this info from is the DOSCAR.lobster
        """

        if not run.section_system:
            system = run.m_create(System)
            system.atom_species = atomic_numbers
            system.configuration_periodic_dimensions = [True, True, True]

    def translate_lm(lm):
        lm_dictionary = {
            's': [0, 0],
            'p_z': [1, 0],
            'p_x': [1, 1],
            'p_y': [1, 2],
            'd_z^2': [2, 0],
            'd_xz': [2, 1],
            'd_yz': [2, 2],
            'd_xy': [2, 3],
            'd_x^2-y^2': [2, 4],
            'z^3': [3, 0],
            'xz^2': [3, 1],
            'yz^2': [3, 2],
            'xyz': [3, 3],
            'z(x^2-y^2)': [3, 4],
            'x(x^2-3y^2)': [3, 5],
            'y(3x^2-y^2)': [3, 6],
        }
        return lm_dictionary.get(lm[1:])

    if not path.isfile(fname):
        return

    energies = []
    dos_values = []
    integral_dos = []
    atom_projected_dos_values = []
    atom_index = 0
    atomic_numbers = []
    lms = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0:
                n_atoms = int(line.split()[0])
            if i == 1:
                cell_volume = float(line.split()[0]) * units.angstrom**3
            if i == 5:
                n_dos = int(line.split()[2])
            if 'Z=' in line:
                atom_index += 1
                atom_projected_dos_values.append([])
                lms.append((line.split(';')[-1]).split())
                atomic_numbers.append(int(line.split(';')[-2].split('=')[1]))
                continue
            if i > 5:
                line = [float(x) for x in line.split()]
                if atom_index == 0:
                    energies.append(line[0])
                    if len(line) == 3:
                        dos_values.append([line[1]])
                        integral_dos.append([line[2]])
                    elif len(line) == 5:
                        dos_values.append([line[1], line[2]])
                        integral_dos.append([line[3], line[4]])
                else:
                    atom_projected_dos_values[-1].append(line[1:])

    parse_species(run, atomic_numbers)

    if len(dos_values) == n_dos:
        dos = run.section_single_configuration_calculation[0].m_create(section_dos)
        dos.dos_kind = 'electronic'
        dos.number_of_dos_values = n_dos
        dos.dos_energies = energies * units.eV
        dos.dos_values = np.array(list(zip(*dos_values))) / eV
        # FIXME: it is not clear if we should do this or if this is done
        # by normalizer (LOBSTER energies are already normalized)
        dos.dos_energies_normalized = energies * units.eV
        dos.dos_values_normalized = dos.dos_values / cell_volume.to_base_units().magnitude / n_atoms
        # FIXME: the usage of other parsers and definition of dos_integrated_values
        # is inconsistent, recheck later when it is cleared, follow the definition
        # for now (add the core electrons)
        n_electrons = sum(atomic_numbers)
        index = (np.abs(energies)).argmin()
        # integrated dos at the Fermi level should be the number of electrons
        n_valence_electrons = int(round(sum(integral_dos[index])))
        n_core_electrons = n_electrons - n_valence_electrons
        dos.dos_integrated_values = np.array(list(zip(*integral_dos))) + n_core_electrons / len(integral_dos[0])
    else:
        logger.warning('Unable to parse total dos from DOSCAR.lobster, \
                            it doesn\'t contain enough dos values')
        return

    for i, pdos in enumerate(atom_projected_dos_values):
        if len(pdos) == n_dos:
            section_pdos = run.section_single_configuration_calculation[0].m_create(
                x_lobster_section_atom_projected_dos)
            section_pdos.x_lobster_number_of_dos_values = n_dos
            # FIXME: should the atoms be indexed from 0 or 1?
            section_pdos.x_lobster_atom_projected_dos_atom_index = i + 1
            section_pdos.x_lobster_atom_projected_dos_m_kind = 'real_orbital'
            section_pdos.x_lobster_number_of_lm_atom_projected_dos = len(lms[i])
            section_pdos.x_lobster_atom_projected_dos_energies = energies * units.eV
            section_pdos.x_lobster_atom_projected_dos_lm = [translate_lm(lm) for lm in lms[i]]
            if len(lms[i]) == len(pdos[0]):
                # we have the same lm-projections for spin up and dn
                section_pdos.x_lobster_atom_projected_dos_values_lm = np.array(
                    [[lmdos] for lmdos in zip(*pdos)]) / eV
            elif len(lms[i]) * 2 == len(pdos[0]):
                pdos_up = list(zip(*pdos))[0::2]
                pdos_dn = list(zip(*pdos))[1::2]
                section_pdos.x_lobster_atom_projected_dos_values_lm = np.array(
                    [[a, b] for a, b in zip(pdos_up, pdos_dn)]) / eV
            else:
                logger.warning('Unexpected number of columns in DOSCAR.lobster')
                return
        else:
            logger.warning('Unable to parse atom lm-projected dos from DOSCAR.lobster, \
                            it doesn\'t contain enough dos values')


mainfile_parser = UnstructuredTextFileParser(quantities=[
    Quantity('program_version', r'^LOBSTER\s*v([\d\.]+)\s*', repeats=False),
    Quantity('datetime', r'starting on host \S* on (\d{4}-\d\d-\d\d\sat\s\d\d:\d\d:\d\d)\s[A-Z]{3,4}',
             repeats=False),
    Quantity('x_lobster_code',
             r'detecting used PAW program... (.*)', repeats=False),
    Quantity('x_lobster_basis',
             r'setting up local basis functions...\s*((?:[a-zA-Z]{1,2}\s+\(.+\)(?:\s+\d\S+)+\s+)+)',
             repeats=False,
             sub_parser=UnstructuredTextFileParser(quantities=[
                 Quantity('x_lobster_basis_species',
                          r'([a-zA-Z]+){1,2}\s+\((.+)\)((?:\s+\d\S+)+)\s+', repeats=True)
             ])),
    Quantity('spilling', r'((?:spillings|abs. tot)[\s\S]*?charge\s*spilling:\s*\d+\.\d+%)',
             repeats=True,
             sub_parser=UnstructuredTextFileParser(quantities=[
                 Quantity('abs_total_spilling',
                          r'abs.\s*total\s*spilling:\s*(\d+\.\d+)%', repeats=False),
                 Quantity('abs_charge_spilling',
                          r'abs.\s*charge\s*spilling:\s*(\d+\.\d+)%', repeats=False)
             ])),
    Quantity('finished', r'finished in (\d)', repeats=False),
])


class LobsterParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/lobster', code_name='LOBSTER',
            code_homepage='http://schmeling.ac.rwth-aachen.de/cohp/',
            mainfile_name_re=r'.*lobsterout$',
            mainfile_contents_re=(r'^LOBSTER\s*v[\d\.]+.*'),
        )

    def parse(self, mainfile: str, archive: EntryArchive, logger):
        mainfile_parser.mainfile = mainfile
        mainfile_path = path.dirname(mainfile)
        mainfile_parser.parse()

        run = archive.m_create(Run)

        run.program_name = 'LOBSTER'
        run.program_version = str(mainfile_parser.get('program_version'))
        # FIXME: There is a timezone info present as well, but datetime support for timezones
        # is bad and it doesn't support some timezones (for example CEST).
        # That leads to test failures, so ignore it for now.
        date = datetime.datetime.strptime(' '.join(mainfile_parser.get('datetime')),
                                          '%Y-%m-%d at %H:%M:%S') - datetime.datetime(1970, 1, 1)
        run.time_run_wall_start = date.total_seconds()
        code = mainfile_parser.get('x_lobster_code')

        # parse structure
        if code == 'VASP':
            try:
                structure = ase.io.read(mainfile_path + '/CONTCAR', format="vasp")
            except FileNotFoundError:
                logger.warning('Unable to parse structure info, no CONTCAR detected')
        else:
            logger.warning('Parsing of {} structure is not supported'.format(code))
        if 'structure' in locals():
            system = run.m_create(System)
            system.lattice_vectors = structure.get_cell() * units.angstrom
            system.atom_labels = structure.get_chemical_symbols()
            system.configuration_periodic_dimensions = structure.get_pbc()
            system.atom_positions = structure.get_positions() * units.angstrom

        if mainfile_parser.get('finished') is not None:
            run.run_clean_end = True
        else:
            run.run_clean_end = False

        scc = run.m_create(SCC)
        method = run.m_create(Method)
        scc.single_configuration_to_calculation_method_ref = method

        spilling = mainfile_parser.get('spilling')
        if spilling is not None:
            method.number_of_spin_channels = len(spilling)
            total_spilling = []
            charge_spilling = []
            for s in spilling:
                total_spilling.append(s.get('abs_total_spilling'))
                charge_spilling.append(s.get('abs_charge_spilling'))
            scc.x_lobster_abs_total_spilling = np.array(total_spilling)
            scc.x_lobster_abs_charge_spilling = np.array(charge_spilling)

        method_keys = [
            'x_lobster_code'
        ]

        for key in method_keys:
            val = mainfile_parser.get(key)
            if val is not None:
                setattr(method, key, val)

        val = mainfile_parser.get('x_lobster_basis').get('x_lobster_basis_species')
        if val is not None:
            method.basis_set = val[0][1]

        parse_ICOXPLIST(mainfile_path + '/ICOHPLIST.lobster', scc, 'h')
        parse_ICOXPLIST(mainfile_path + '/ICOOPLIST.lobster', scc, 'o')

        parse_COXPCAR(mainfile_path + '/COHPCAR.lobster', scc, 'h', logger)
        parse_COXPCAR(mainfile_path + '/COOPCAR.lobster', scc, 'o', logger)

        parse_CHARGE(mainfile_path + '/CHARGE.lobster', scc)

        parse_DOSCAR(mainfile_path + '/DOSCAR.lobster', run, logger)

        if run.section_system:
            scc.single_configuration_calculation_to_system_ref = run.section_system[0]
