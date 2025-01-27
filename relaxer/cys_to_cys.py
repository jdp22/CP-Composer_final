#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import io
import logging
import numpy as np
import pdbfixer
import openmm
from openmm.app import Modeller
from openmm import app as openmm_app
from openmm import unit
ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

from .base import ForceFieldMinimizer


class ForceFieldMinimizerCys(ForceFieldMinimizer):

    def _fix_cyclic(self, fixer, cyclic_chains, cyclic_opts):

        assert cyclic_opts is not None, f'cyclic_opts should not be None, but list of pairs ((chain_id, res_pos), (chain_id, res_pos))'
        
        all_cyc_cys = {}
        for resid1, resid2 in cyclic_opts:
            all_cyc_cys[resid1] = 1
            all_cyc_cys[resid2] = 1

        # remove hydrogen on the sulfer
        modeller = Modeller(fixer.topology, fixer.positions)
        for chain in modeller.topology.chains():
            if chain.id not in cyclic_chains: continue
            atoms_to_remove = []
            for i, res in enumerate(chain.residues()):
                if (chain.id, i) not in all_cyc_cys:
                    continue
                for atom in res.atoms():
                    if atom.name == 'HG':
                        atoms_to_remove.append(atom)
            modeller.delete(atoms_to_remove)
        
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        
        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        pdb_fixed = out_handle.getvalue()

        new_fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_fixed))

        resid2sg = {}
        for chain in new_fixer.topology.chains():
            if chain.id not in cyclic_chains: continue
            for i, residue in enumerate(chain.residues()):
                if residue.name != 'CYS': continue
                resid = (chain.id, i)
                for atom in residue.atoms():
                    if atom.name == 'SG': resid2sg[resid] = atom
        
        connects = []
        for res1, res2 in cyclic_opts:
            sg1, sg2 = resid2sg[res1], resid2sg[res2]
            connects.append('CONECT' + str(sg1.id).rjust(5) + str(sg2.id).rjust(5))
            connects.append('CONECT' + str(sg2.id).rjust(5) + str(sg1.id).rjust(5))

        pdb_fixed = self._add_connects(pdb_fixed, connects)
        return pdb_fixed, connects


if __name__ == '__main__':
    import sys
    force_field = ForceFieldMinimizerCys()
    force_field(sys.argv[1], sys.argv[2], cyclic_chains=['B'], cyclic_opts=[(('B', 1), ('B', 4))]) # starts from 0, the i-th residue