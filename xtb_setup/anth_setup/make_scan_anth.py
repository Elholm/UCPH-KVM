#!/groups/kemi/koerstz/anaconda3/envs/quantum_ml/bin/python

import pickle
import pandas as pd
import numpy as np
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append("/home/jelholm/opt/tQMC/QMC")
from qmmol import QMMol
from qmconf import QMConf
import myio.io_ts_scan_xtb as ts_scan_xtb
from calculator.xtb import xTB

def create_xcontrol(mol,xcontrol_name):
    '''
    #Find the pairs
    smart_mol1 = Chem.MolFromSmarts('[#6]1-[#6]-[#6]-[#6]-1')
    smart_mol2 = Chem.MolFromSmarts('[#6]1-[#6]-[#6]-1')
    
    sub1 = np.array(mol.GetSubstructMatches(smart_mol1))
    sub2 = np.array(mol.GetSubstructMatches(smart_mol2))
    
    highlightAtomList = []
    for i in range(len(sub2)):
        for y in sub2[i]:
            if y in sub1[0]:
                highlightAtomList.append(int(y))
    atom_pairs = [highlightAtomList[i * 2:(i + 1) * 2] for i in range((len(highlightAtomList)) // 2 )] 
    '''
    # NEW TRY TO FIND ANTHRACENE ATOM PAIRS. GOOD LUCK. THANKS.
    smart_mol_1 = Chem.MolFromSmarts('[#6]1-[#6]-[#6]:[#6]-[#6]-[#6]-[#6]:[#6]-1')
    sub_1 = np.array(mol.GetSubstructMatches(smart_mol_1))
    bad = []
    for x in sub_1[0]:
        for y in sub_1:
            if x not in y:
                bad.append(x)
    del_list = []
    for x in range(len(sub_1[0])):
        if sub_1[0][x] in bad:
            del_list.append(x)
    new_sub_1 = np.delete(sub_1[0],del_list)

    
    #Write the xcontrol file
    fo = open(xcontrol_name, "w")
    
    fo.write("$constrain\n")
    fo.write(" force constant=0.5\n")
    #fo.write(" distance: "+str(atom_pairs[0][0]+1)+", "+str(atom_pairs[0][1]+1)+", auto\n")
    #fo.write(" distance: "+str(atom_pairs[1][0]+1)+", "+str(atom_pairs[1][1]+1)+", auto\n")
    fo.write(" distance: "+str(new_sub_1[0]+1)+", "+str(new_sub_1[1]+1)+", auto\n")
    fo.write(" distance: "+str(new_sub_1[2]+1)+", "+str(new_sub_1[3]+1)+", auto\n")
    fo.write("$scan\n")
    fo.write(" mode=concerted\n")
    fo.write(" 1: 1.5, 3.5, 30\n")
    fo.write(" 2: 1.5, 3.5, 30\n")
    fo.write("$end\n")
    
    fo.close()
    
    return


def ts_scan(prod, name, chrg, mult, xcontrol_name):
    """ scan for transition state """
    
    charged = True # hard coded for mogens
    
    # create conformers
    ts_qmmol = QMMol()
    ts_qmmol.add_conformer(prod.write_xyz(to_file=False), fmt='xyz', label=name, charged_fragments=charged, set_initial=True)
    
    xtb_params = {'method': 'gfn2',
                  'opt': 'opt',
                  'cpus': 1,
                  'input': '../' + str(xcontrol_name)}

    ts_qmmol.calc = xTB(parameters=xtb_params)
    ts_conf = ts_qmmol.conformers[0]

    #ts_conf.conf_calculate(quantities=['energy', 'structure'], keep_files=True)
    #ts_conf.conf_calculate(quantities=['energy'], keep_files=True)
    ts_conf.conf_calculate(quantities=['energy', 'ts_guess'], keep_files=True)


if __name__ == '__main__':

    df = pd.read_pickle(sys.argv[1])
    charge = 0
    mult = 1
    compound_list = list()

    for x in df.itertuples():
        name = str(x.reac.label.split('_r')[0])
        mol = x.prod.get_rdkit_mol()

        xcontrol_name = name+"_xcontrol"
        create_xcontrol(mol, xcontrol_name)

        n = name + "_ts"
        
        ts_scan(x.prod, n, charge, mult, xcontrol_name)
        
        with open(n+"_xtbscan.log", 'r') as out:
           output = out.read() 
        
        print(output)
        ts_qmmol = QMMol()
        
        xyz_file = ts_scan_xtb.read_ts_guess_structure2(output)
        ts_qmmol.add_conformer(input_mol=xyz_file, fmt='xyz', label=n, charge=charge, multiplicity=mult, read_freq=False, charged_fragments=True)
        ts_conf = ts_qmmol.conformers[0]

        ts_conf.results['energy'] = ts_scan_xtb.read_ts_guess_energy(output)[0]
        
        #TRYING TO GET DATA FOR THE TWO MONOMERS IN CLOSE VICINITY
        monomers_qmmol = QMMol()
        monomers_xyz = ts_scan_xtb.read_monos_guess_structure2(output)
        monomers_qmmol.add_conformer(input_mol=monomers_xyz, fmt='xyz', label=f'{name}_monos', charge=charge, multiplicity=mult, read_freq=False, charged_fragments=True)
        monomers_conf = monomers_qmmol.conformers[0]
        monomers_conf.results['energy'] = ts_scan_xtb.read_monos_guess_energy(output)[0]



        #print(ts_conf.results['energy'])
        reac_qmconf = x.reac
        prod_qmconf = x.prod
        ts_qmconf = ts_conf
        storage = x.storage*2625.5 #convert from Hartree to kJ/mol
        tbr = (ts_conf.results['energy']-x.prod.results['energy'])*2625.5 #convert from Hartree to kJ/mol

        compound_list.append({'rep': x.rep,
                              'reac': reac_qmconf,
                              'prod': prod_qmconf,
                              'scan_ts': ts_qmconf,
                              'storage': storage,
                              'tbr': tbr,
                              'monos': monomers_conf})
        
        #Clean up
        os.remove(str(n) + ".out")
        #os.remove(str(n) + "_xtbscan.log")
        os.remove(str(n) + ".xyz")


    structures = compound_list
    data = pd.DataFrame(structures)
    data.to_pickle(sys.argv[1].split('/')[0].split('.')[0] + '_new.pkl')
    #data.to_pickle(sys.argv[1].split('.')[0] + '_new.pkl') #IF running on frontend this should be used!
