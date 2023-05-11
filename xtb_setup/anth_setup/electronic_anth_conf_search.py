#!/groups/kemi/koerstz/anaconda3/envs/quantum_ml/bin/python
import sys
import pandas as pd
import copy

from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append("/home/jelholm/opt/tQMC/QMC")
#sys.path.append("/groups/kemi/koerstz/opt/QMC/QMC") #this works
from qmmol import QMMol
from qmconf import QMConf
from calculator.xtb import xTB
from calculator.orca import ORCA
from calculator.gaussian import Gaussian

from conformers.create_conformers import RotatableBonds


def map_atoms(reactant, product):
    """Returns new order in product """

    reac = copy.deepcopy(reactant)
    prod = copy.deepcopy(product)

    Chem.Kekulize(reac,clearAromaticFlags=True)
    Chem.Kekulize(prod,clearAromaticFlags=True)
     
    reac = change_mol(reac)
    prod = change_mol(prod)
    
    # Break Bond in reactant, in order to compare to product.
    smarts_bond = Chem.MolFromSmarts('[CX4;H0;R]-[CX4;H1;R]')
    atom_idx = list(reac.GetSubstructMatch(smarts_bond))
    
    if len(atom_idx) != 0:
        bond = reac.GetBondBetweenAtoms(atom_idx[0], atom_idx[1])
        broken_bond_reac = Chem.FragmentOnBonds(reac, [bond.GetIdx()], addDummies=False)
        
        # find new atom order for product
        prod_order = prod.GetSubstructMatch(broken_bond_reac)
    
    else:
        prod_order = prod.GetSubstructMatch(reac)
    
    return prod_order


def change_mol(mol):
    """ Change molecule to only compare conectivity """

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            bond.SetBondType(Chem.BondType.SINGLE)

    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(0)

    return mol


def reorder_product(reac, prod):
    """ change atom order of product, to match reactant  """
    new_product_order = map_atoms(reac, prod)
    reordered_product = Chem.RenumberAtoms(prod, new_product_order)

    return reordered_product


def reactant2product(reac_smi,sub1="H",sub2="H"):
    """ create prodruct from reactant """
    sub1_s = f"({sub1})"
    sub2_s = f"({sub2})"
    #Current smarts is for anthracene single to dimer but without substituents
    smarts = f"[c:1]1[c:2][c:3]2[c:4]{sub1_s if sub1_s != '(H)' else ''}[c:5]3[c:6][c:7][c:8][c:9][c:10]3[c:11]{sub2_s if sub2_s != '(H)' else ''}[c:12]2[c:13][c:14]1>>[c:1]1[c:2][c:3]2[C:4]3{sub1_s if sub1_s != '(H)' else ''}[c:5]4[c:6][c:7][c:8][c:9][c:10]4[C:11]{sub2_s if sub2_s != '(H)' else ''}([C:15]5{sub1_s if sub1_s != '(H)' else ''}[c:16]6[c:17][c:18][c:19][c:20][c:21]6[C:22]3{sub2_s if sub2_s != '(H)' else ''}[c:23]7[c:24][c:25][c:26][c:27][c:28]57)[c:12]2[c:13][c:14]1"
    #smarts = f"[c:1]1[c:2][c:3]2[c:4][c:5]3[c:6][c:7][c:8][c:9][c:10]3[c:11][c:12]2[c:13][c:14]1>>[c:1]1[c:2][c:3]2[C:4]3{sub1_s if sub1_s != '(H)' else ''}[c:5]4[c:6][c:7][c:8][c:9][c:10]4[C:11]([C:15]5{sub1_s if sub1_s != '(H)' else ''}[c:16]6[c:17][c:18][c:19][c:20][c:21]6[C:22]3[c:23]7[c:24][c:25][c:26][c:27][c:28]57)[c:12]2[c:13][c:14]1"
    print(smarts)
    smarts_old = "[C:1]12[C:2]=[C:3][C:4]([C:5]=[C:6]1)[C:7][C:8][C:9]2>>[C:1]12[C:2]3[C:3]4[C:4]([C:5]4[C:6]13)[C:7][C:8][C:9]2"
    smarts_bnd = "[C:1]12[C:2]=[C:3][C:4]([C:5][C:6][C:7]1)[C:8]=[C:9]2>>[C:1]12[C:2]3[C:3]4[C:4]([C:5][C:6][C:7]1)[C:8]4[C:9]23"
    smarts_bod = "[C:1]12[C:2]=[C:3][C:4]([C:5][C:6]1)[C:7]=[C:8]2>>[C:1]12[C:2]3[C:3]4[C:4]([C:5][C:6]1)[C:7]4[C:8]23"
    #smarts = "[C:1]12[C:2]=[C:3][C:4]([C:5][C:6]1)[C:7]=[C:8]2>>[C:1]1[C:2][C:3]2[C:4]3[C:5]4[C:6]1[C:7]4[C:8]23" 
    #above is smarts from bendte
    #smarts = "[C:1]1[C:2]2[C:3]=[C:4][C:5]1[C:6]=[C:7]2>>[C:1]1[C:2]2[C:3]3[C:4]4[C:5]1[C:6]4[C:7]23"
    #above is the old smarts for NBD to QC
    __rxn__ = AllChem.ReactionFromSmarts(smarts)
    print("getting mol...")
    # create reactant mol
    reac_mol =  Chem.MolFromSmiles(reac_smi)
    print("running reaction...")
    prod_mol = __rxn__.RunReactants((reac_mol,))[0][0]
    #prod_smi = Chem.MolToSmiles(prod_mol)
    print("getting prod_smi...")
    prod_smi = Chem.MolToSmiles(Chem.RemoveHs(prod_mol))
    print(prod_smi)
    reac_smi = Chem.MolToSmiles(Chem.MolFromSmiles(reac_smi))

    return reac_smi, prod_smi


def gs_conformer_search(name, rdkit_conf, chrg, mult, cpus):
    """ ground state conformer search """
    
    charged = True # hard coded for mogens
    print("creating conformers") 
    # create conformers
    qmmol = QMMol()
    qmmol.add_conformer(rdkit_conf, fmt='rdkit', label=name,
                       charged_fragments=charged, set_initial=True)
    print("finding rot bond")
    # find #-- rot bond.
    mol = qmmol.get_rdkit_mol()
    triple_smart = '[c]:[c]-[CH0]#[CH0]'
    try:
        extra_rot_bond = int(len(mol.GetSubstructMatches(Chem.MolFromSmarts(triple_smart))) / 2)
    except:
        print("no extra rot bond")
    print("creating new conformers")
    # print compute number of conformers to find
    rot_bonds = len(RotatableBonds(qmmol.initial_conformer.get_rdkit_mol())) + extra_rot_bond
    num_confs = 5 + 5*rot_bonds
    print(f"num_confs is: {num_confs}")
    try:
        qmmol.create_random_conformers(threads=cpus, num_confs=num_confs)
    except:
        print("could not create random conformers")
    xtb_params = {'method': 'gfn2',
                  'opt': 'tight',
                  'cpus': 1}#,
                  #'alpb': 'toluene'}
    print("beginning xTB opt")
    qmmol.calc = xTB(parameters=xtb_params)
    qmmol.optimize(num_procs=cpus, keep_files=False)
    print("xtb opt done!")
    # Get most stable conformer. If most stable conformer
    # not identical to initial conf try second lowest.
    initial_smi = Chem.MolToSmiles(Chem.RemoveHs(qmmol.initial_conformer.get_rdkit_mol()),isomericSmiles=False)
    #initial_smi = Chem.MolToSmiles(Chem.RemoveStereochemistry(Chem.RemoveHs(qmmol.initial_conformer.get_rdkit_mol())))
    print(f"initial smi found: {initial_smi}")
    low_energy_conf = qmmol.nlowest(1)[0]
    print("low_energy_conf found")
    conf_smi = Chem.MolToSmiles(Chem.RemoveHs(low_energy_conf.get_rdkit_mol()),isomericSmiles=False)
    #conf_smi = Chem.MolToSmiles(Chem.RemoveStereochemistry(Chem.RemoveHs(low_energy_conf.get_rdkit_mol())))
    print(f"conf_smi foundi: {conf_smi}")
    i = 1
    while initial_smi != conf_smi:
        low_energy_conf = qmmol.nlowest(i+1)[-1]
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(low_energy_conf.get_rdkit_mol()),isomericSmiles=False)
        #conf_smi = Chem.MolToSmiles(Chem.RemoveStereochemistry(Chem.RemoveHs(low_energy_conf.get_rdkit_mol())))
        i += 1
        
        if len(qmmol.conformers) < i:
            sys.exit('no conformers match the initial input')

    return low_energy_conf


def gs_mogens(name, smi, chrg, mult, cps, sub1):
    """GS conformers search given a smiles string  """
    
    reac_smi, prod_smi = reactant2product(smi, sub1)
    print(f"{reac_smi} {prod_smi}") 
    reac_mol = Chem.AddHs(Chem.MolFromSmiles(reac_smi))
    try:
        prod_mol = reorder_product(reac_mol, Chem.AddHs(Chem.MolFromSmiles(prod_smi))) #THIS IS THE ACTUAL PLACE THAT FAILS!
        print("reorder succesful")
    except:
        prod_mol = Chem.MolFromSmiles(prod_smi)
        Chem.SanitizeMol(prod_mol)
        prod_mol = Chem.AddHs(Chem.MolFromSmiles(prod_smi))    

    for i, comp in enumerate([(reac_mol, "_r"), (prod_mol, "_p")]):
        mol, p_r = comp
        
        AllChem.EmbedMolecule(mol)
        rdkit_conf = mol.GetConformer()

        # create mol for QMMol using rdkit
        n = name + p_r
        
        if p_r == '_r':
            try:
                reac_qmconf = gs_conformer_search(n, rdkit_conf, chrg, mult, cps)
            except:
                print("Original reac qmconf failed, thus algo turns to no reorder")
                count = 0
                reac_qmconf = None
                while reac_qmconf is None and count < 40:
                    try:
                        reac_qmconf = gs_retry(chrg, mult, cps, name, reac_smi)
                    except:
                        count += 1
                        print("Failed retry:",count)
                        pass
        if p_r == '_p':
            try:
                prod_qmconf = gs_conformer_search(n, rdkit_conf, chrg, mult, cps)
            except:
                print("Original prod qmconf failed, thus algo turns to no reorder")
                count = 0
                prod_qmconf = None
                while prod_qmconf is None and count < 40:
                    try:
                        prod_qmconf = gs_retry(chrg, mult, cps, name, prod_smi)
                    except:
                        count += 1
                        print("Failed retry:",count)
                        pass
    storage = prod_qmconf.results['energy'] - 2*reac_qmconf.results['energy']

    return reac_qmconf, prod_qmconf, storage



def gs_retry(chrg, mult, cps, name, prod_smi):
    
    mol = Chem.AddHs(Chem.MolFromSmiles(prod_smi))
    AllChem.EmbedMolecule(mol)
    rdkit_conf = mol.GetConformer()
    n = name + '_p'
    return gs_conformer_search(n, rdkit_conf, chrg, mult, cps)



def ts_search(gs_dict):
    """ Perform ts scan of the bond getting broken"""
    pass


def ts_test(test_qmconf):
    """ Automatically test TS if it correct """
    pass 


if __name__ == '__main__':
    
    import pandas as pd
    import sys
    
    ### Kør 1
    ##test_smi = 'ClC1=CC=CC=C1C#CC2=CC3C=CC2C3' # one extra
    ##test_smi = 'ClC1=CC=CC=C1C#CC2=CC3C=C(C#CC4=CC=CC=C4)C2C3' # 2 extra
    #test_smi = 'CN(C)c1ccc(C2=C(C#N)[C@@H]3C=C[C@H]2C3)cc1'

    ##reac_smi, prod_smi = reactant2product(Chem.MolToSmiles(Chem.MolFromSmiles(test_smi),kekuleSmiles=True))
    #reac_smi, prod_smi = reactant2product(test_smi)
    #print(reac_smi, prod_smi)
    
    ##NBD = Chem.AddHs(Chem.MolFromSmiles(reac_smi))
    ##NBD.SetProp("_Name","NBD")
    ##AllChem.EmbedMolecule(NBD)
    ##print(Chem.MolToMolBlock(NBD),file=open('NBD.mol','w+'))
    
    ##QC = Chem.AddHs(Chem.MolFromSmiles(prod_smi))
    ##QC.SetProp("_Name","QC")
    ##AllChem.EmbedMolecule(QC)
    ##print(Chem.MolToMolBlock(QC),file=open('QC.mol','w+'))

    #try:
    #    reac, prod, storage = gs_mogens('test', reac_smi, 0, 1, 4)
    #    print(storage*2625.5) #kJ/mol
    #except:
    #    print("something went wrong!")
    #    pass

    
    
    ### Kør mange
    cpus = 2

    data = pd.read_csv(sys.argv[1])
    
    # find storage energy
    compound_list = list()
    for idx, compound in data.iterrows():
        try: 
            reac_qmconf, prod_qmconf, storage = gs_mogens(str(compound.comp_name),
                                                          compound.smiles,
                                                          compound.charge,
                                                          compound.multiplicity,
                                                          cpus, str(compound.sub1))

            print(str(compound.comp_name),compound.smiles,compound.charge,compound.multiplicity,cpus) 
            compound_list.append({'rep': str(compound.comp_name),
                                  'reac': reac_qmconf,
                                  'prod': prod_qmconf,
                                  'storage': storage})
        except:
            pass
    ## find ts
    #with Pool(cpus) as pool:
    #    structures = pool.map(ts_search, compound_list)
    
    structures = compound_list
    data = pd.DataFrame(structures)
    data.to_pickle(sys.argv[1].split('/')[0].split('.')[0] + '.pkl')
    #data.to_pickle(sys.argv[1].split('/')[1].split('.')[0] + '.pkl') #works with submit_python
    #data.to_pickle(sys.argv[1].split('.')[0] + '.pkl') #IF running on frontend this should be used!
