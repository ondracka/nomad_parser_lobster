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

import numpy as np

from nomad.metainfo import Section, Quantity, MSection, SubSection, SectionProxy
from nomad.datamodel.metainfo import public
from nomad.metainfo.legacy import LegacyDefinition


class section_single_configuration_calculation(public.section_single_configuration_calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_lobster_abs_total_spilling = Quantity(
        type=float,
        shape=['number_of_spin_channels'],
        description='''
        Absolute total spilling (in all levels)
        when projecting from the original wave functions into the local basis.
        ''')

    x_lobster_abs_charge_spilling = Quantity(
        type=float,
        shape=['number_of_spin_channels'],
        description='''
        Absolute total spilling of density (in occupied levels)
        when projecting from the original wave functions into the local basis.
        ''')

    x_lobster_section_icohplist = SubSection(
        sub_section=SectionProxy('x_lobster_section_icohplist'))

    x_lobster_section_icooplist = SubSection(
        sub_section=SectionProxy('x_lobster_section_icooplist'))


class section_method(public.section_method):

    m_def = Section(validate=False, extends_base_section=True)

    x_lobster_code = Quantity(
        type=str,
        description='''
        Used PAW program
        ''')


class x_lobster_section_icohplist(MSection):
    m_def = Section(validate=False)

    x_lobster_number_of_icohp_values = Quantity(
        type=int,
        description='''
        Number of calculated iCOHPs
        ''')

    x_lobster_icohp_atom1_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_icohp_values'],
        description='''
        Species and indices of the first atom for which is the specific iCOHP calculated
        ''')

    x_lobster_icohp_atom2_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_icohp_values'],
        description='''
        Species and indices of the second atom for which is the specific iCOHP calculated
        ''')

    x_lobster_icohp_distances = Quantity(
        type=float,
        shape=['x_lobster_number_of_icohp_values'],
        description='''
        Distance between the atom pair for which is the specific iCOHP calculated
        ''')

    x_lobster_icohp_translations = Quantity(
        type=np.dtype(np.int8),
        shape=['x_lobster_number_of_icohp_values', 3],
        description='''
        Vector connecting the unit-cell of the first atom with the one of the second atom
        ''')

    x_lobster_icohp_values = Quantity(
        type=np.dtype(np.float32),
        shape=['number_of_spin_channels', 'x_lobster_number_of_icohp_values'],
        description='''
        Calculated iCOHPs
        ''')


class x_lobster_section_icooplist(MSection):
    m_def = Section(validate=False)

    x_lobster_number_of_icoop_values = Quantity(
        type=int,
        description='''
        Number of calculated iCOOPs
        ''')

    x_lobster_icoop_atom1_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_icoop_values'],
        description='''
        Species and indices of the first atom for which is the specific iCOOP calculated
        ''')

    x_lobster_icoop_atom2_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_icoop_values'],
        description='''
        Species and indices of the second atom for which is the specific iCOOP calculated
        ''')

    x_lobster_icoop_distances = Quantity(
        type=float,
        shape=['x_lobster_number_of_icoop_values'],
        description='''
        Distance between the atom pair for which is the specific iCOOP calculated
        ''')

    x_lobster_icoop_translations = Quantity(
        type=np.dtype(np.int8),
        shape=['x_lobster_number_of_icoop_values', 3],
        description='''
        Vector connecting the unit-cell of the first atom with the one of the second atom
        ''')

    x_lobster_icoop_values = Quantity(
        type=np.dtype(np.float32),
        shape=['number_of_spin_channels', 'x_lobster_number_of_icoop_values'],
        description='''
        Calculated iCOOPs
        ''')
