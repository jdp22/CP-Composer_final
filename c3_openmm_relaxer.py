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


class ForceFieldMinimizer(object):

    def __init__(self, stiffness=10.0, max_iterations=0, tolerance=2.39*unit.kilocalories_per_mole, platform='CUDA'):
        super().__init__()
        self.stiffness = stiffness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        assert platform in ('CUDA', 'CPU')
        self.platform = platform

    def _fix(self, pdb_str, cyclic_chains, cyclic_opts):
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens()

        if cyclic_chains is not None:
            pdb_fixed, connects = self._fix_cyclic(fixer, cyclic_chains, cyclic_opts)
        else:
            out_handle = io.StringIO()
            openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
            pdb_fixed = out_handle.getvalue()
            connects = []

        return pdb_fixed, connects
    
    def _fix_cyclic(self, fixer, cyclic_chains, cyclic_opts):
        modeller = Modeller(fixer.topology, fixer.positions)
        for chain in modeller.topology.chains():
            if chain.id not in cyclic_chains: continue
            atoms_to_remove = []
            for i, res in enumerate(chain.residues()):
                if i == 0:
                    for atom in res.atoms():
                        if atom.name == 'H2' or atom.name == 'H3':
                            atoms_to_remove.append(atom)
                elif i == len(chain) - 1:
                    for atom in res.atoms():
                        if atom.name == 'OXT': atoms_to_remove.append(atom)
            modeller.delete(atoms_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        pdb_fixed = out_handle.getvalue()

        new_fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_fixed))
        connects = []
        for chain in new_fixer.topology.chains():
            if chain.id not in cyclic_chains: continue
            n_term, c_term = None, None
            for i, res in enumerate(chain.residues()):
                if i == 0:
                    for atom in res.atoms():
                        if atom.name == 'N': n_term = atom.id
                elif i == len(chain) - 1:
                    for atom in res.atoms():
                        if atom.name == 'C': c_term = int(atom.id)
            connects.append('CONECT' + str(n_term).rjust(5) + str(c_term).rjust(5))
            connects.append('CONECT' + str(c_term).rjust(5) + str(n_term).rjust(5))
        
        pdb_fixed = self._add_connects(pdb_fixed, connects)
        return pdb_fixed, connects

    def _get_pdb_string(self, topology, positions):
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, positions, f, keepIds=True)
            return f.getvalue()
        
    def _minimize(self, pdb_str):
        pdb = openmm_app.PDBFile(io.StringIO(pdb_str))

        force_field = openmm_app.ForceField("charmm36.xml") # referring to http://docs.openmm.org/latest/userguide/application/02_running_sims.html
        constraints = openmm_app.HBonds
        system = force_field.createSystem(pdb.topology, constraints=constraints)

        # Add constraints to non-generated regions
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", self.stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        
        for i, a in enumerate(pdb.topology.atoms()):
            if a.element.name != 'hydrogen':
                force.addParticle(i, pdb.positions[i])
                
        system.addForce(force)

        # Set up the integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName("CUDA")
        simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Perform minimization
        ret = {}
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        simulation.minimizeEnergy(maxIterations=self.max_iterations, tolerance=self.tolerance)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        ret["min_pdb"] = self._get_pdb_string(simulation.topology, state.getPositions())

        return ret['min_pdb'], ret
    
    def _add_energy_remarks(self, pdb_str, ret):
        pdb_lines = pdb_str.splitlines()
        pdb_lines.insert(1, "REMARK   1  FINAL ENERGY:   {:.3f} KCAL/MOL".format(ret['efinal']))
        pdb_lines.insert(1, "REMARK   1  INITIAL ENERGY: {:.3f} KCAL/MOL".format(ret['einit']))
        return "\n".join(pdb_lines)

    def _add_connects(self, pdb_str, connects):
        exist_connects = [l for l in pdb_str.split('\n') if 'CONECT' in l]
        connects = [c for c in connects if c not in exist_connects]
        # add CONECT records at the end
        pdb_str = pdb_str.strip().strip('END').strip()
        pdb_str = pdb_str.split('\n')
        pdb_str = pdb_str + connects + ['END\n']
        pdb_str = '\n'.join(pdb_str)
        return pdb_str

    def __call__(self, pdb_str, out_path, return_info=True, cyclic_chains=None, cyclic_opts=None):
        if '\n' not in pdb_str and pdb_str.lower().endswith(".pdb"):
            with open(pdb_str) as f:
                pdb_str = f.read()

        pdb_fixed, connects = self._fix(pdb_str, cyclic_chains, cyclic_opts)
        pdb_min, ret = self._minimize(pdb_fixed)
        pdb_min = self._add_connects(pdb_min, connects)
        pdb_min = self._add_energy_remarks(pdb_min, ret)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        with open(out_path, 'w') as f:
            f.write(pdb_min)
        if return_info:
            return pdb_min, ret
        else:
            return pdb_min


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
    # force_field(sys.argv[1], sys.argv[2], cyclic_chains=['B'], cyclic_opts=[(('B', 1), ('B', 4))]) # starts from 0, the i-th residue
    force_field(sys.argv[1], sys.argv[2], cyclic_chains=['B'], cyclic_opts=[(('B', 2), ('B', 5)),(('B', 6), ('B', 10)),(('B', 11), ('B', 14))])