#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
import io
import logging
import numpy as np
import pdbfixer
import openmm
from openmm import Vec3
from openmm.app import Modeller, PDBFile
from openmm import app as openmm_app
from openmm import unit
ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

from .base import ForceFieldMinimizer

TMB_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'custom', '1_3_5_TMB_with_H.pdb'
))


# from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


# have been validated with kabsch from RefineGNN
def kabsch(a, b):
    # find optimal rotation matrix to transform a into b
    # a, b are both [N, 3]
    # a_aligned = aR + t
    a, b = np.array(a), np.array(b)
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    # a_aligned = np.dot(a_c, rotation)
    # t = b_mean - np.mean(a_aligned, axis=0)
    # a_aligned += t
    t = b_mean - np.dot(a_mean, rotation)
    a_aligned = np.dot(a, rotation) + t

    return a_aligned, rotation, t


def _get_H3_from_tmb(topology):
    names = ['H13', 'H23', 'H33']
    atoms = {}
    for chain in topology.chains():
        for res in chain.residues():
            for atom in res.atoms():
                if atom.name in names:
                    atoms[atom.name] = atom
    return [atoms[name] for name in names]


def _dummy_tmb(sg_coords, chain_id):
    tmb = PDBFile(open(TMB_PATH, 'r'))
    pos = tmb.getPositions(asNumpy=True)
    pos = np.array(pos)
    # pos = pos - np.mean(pos, axis=0) + center

    # align H13, H23, H33 to SG1, SG2, SG3
    h3_coords = []
    for atom in _get_H3_from_tmb(tmb.topology):
        coord = tmb.positions[atom.index]
        h3_coords.append(np.array([coord.x, coord.y, coord.z]))
    
    _, rotation, t = kabsch(h3_coords, sg_coords)
    pos = np.dot(pos, rotation) + t

    tmb.positions = openmm.unit.quantity.Quantity([Vec3(x[0], x[1], x[2]) for x in pos], unit=openmm.unit.nanometer)

    modeller = Modeller(tmb.topology, tmb.positions)
    for chain in modeller.topology.chains(): chain.id = chain_id

    atoms_to_remove = _get_H3_from_tmb(modeller.topology)
    modeller.delete(atoms_to_remove)

    return modeller.topology, modeller.positions


def _reorganize_connects(existing, new):
    connect_dict = {}
    for line in existing + new:
        line = re.split(r'\s+', line.strip())
        src, dsts = line[1], line[2:]
        if src not in connect_dict: connect_dict[src] = []
        connect_dict[src].extend(dsts)
    connects = []
    for key in sorted([int(i) for i in connect_dict]):
        key = str(key)
        s = 'CONECT' + key.rjust(5)
        for d in sorted([int(j) for j in set(connect_dict[key])]):
            s += str(d).rjust(5)
        connects.append(s)
    return connects


class ForceFieldMinimizerBicycle(ForceFieldMinimizer):

    def _fix_cyclic(self, fixer, cyclic_chains, cyclic_opts):

        assert cyclic_opts is not None, f'cyclic_opts should not be None, but list of pairs ((chain_id, res_pos), (chain_id, res_pos))'
        
        all_cyc = {}
        for resid1, resid2, resid3 in cyclic_opts:
            all_cyc[resid1] = 1
            all_cyc[resid2] = 1
            all_cyc[resid3] = 1

        # remove hydrogen on the sulfer and record SG positions
        resid2sgpos = {}
        modeller = Modeller(fixer.topology, fixer.positions)
        for chain in modeller.topology.chains():
            if chain.id not in cyclic_chains: continue
            atoms_to_remove = []
            for i, res in enumerate(chain.residues()):
                resid = (chain.id, i)
                if resid not in all_cyc:
                    continue
                for atom in res.atoms():
                    if atom.name == 'HG':
                        atoms_to_remove.append(atom)
                    elif atom.name == 'SG':
                        coord = modeller.positions[atom.index]
                        resid2sgpos[resid] = np.array([coord.x, coord.y, coord.z])

            modeller.delete(atoms_to_remove)

        # add 1,3,5-trimethylbenezene with CH2
        for resids in cyclic_opts:
            # center = np.mean([resid2sgpos[resid] for resid in resids], axis=0)
            # topo, position = _dummy_tmb(center, resids[0][0])
            topo, position = _dummy_tmb(np.array([resid2sgpos[resid] for resid in resids]), resids[0][0])
            modeller.add(topo, position)

        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        
        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        pdb_fixed = out_handle.getvalue()

        new_fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_fixed))

        resid2sg, tmb_carbons = {}, []
        for chain in new_fixer.topology.chains():
            if chain.id not in cyclic_chains: continue
            for i, residue in enumerate(chain.residues()):
                if residue.name == 'CYS':
                    resid = (chain.id, i)
                    for atom in residue.atoms():
                        if atom.name == 'SG': resid2sg[resid] = atom
                elif residue.name == 'TMB':
                    carbons = []
                    for atom in residue.atoms():
                        if atom.name in ['CM1', 'CM2', 'CM3']:
                            carbons.append(atom)
                    assert len(carbons) == 3
                    tmb_carbons.append(carbons)
        
        connects = []

        for i, (res1, res2, res3) in enumerate(cyclic_opts):
            sg1, sg2, sg3 = resid2sg[res1], resid2sg[res2], resid2sg[res3]
            c1, c2, c3 = tmb_carbons[i]
            for sg, c in zip([sg1, sg2, sg3], [c1, c2, c3]):
                # clear wrong S-S bonds automatically identified by pdbfixer
                pattern = rf"^CONECT {sg.id}\b.*(?:\n|$)"
                pdb_fixed = re.sub(pattern, "", pdb_fixed, flags=re.MULTILINE)
                connects.append('CONECT' + str(sg.id).rjust(5) + str(c.id).rjust(5))
                connects.append('CONECT' + str(c.id).rjust(5) + str(sg.id).rjust(5))

        # reorganize CONECT record
        pattern = r'^CONECT\b.*(?:\n|$)'
        exist_connects = re.findall(pattern, pdb_fixed, flags=re.MULTILINE)
        pdb_fixed = re.sub(pattern, "", pdb_fixed, flags=re.MULTILINE)

        connects = _reorganize_connects(exist_connects, connects)
        
        pdb_fixed = self._add_connects(pdb_fixed, connects)
        # print(pdb_fixed)
        return pdb_fixed, connects


if __name__ == '__main__':
    import sys
    force_field = ForceFieldMinimizerBicycle()
    force_field(sys.argv[1], sys.argv[2], cyclic_chains=['B'], cyclic_opts=[(('B', 0), ('B', 7), ('B', 12))]) # starts from 0, the i-th residue