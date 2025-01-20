from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score
from rdkit.Chem import QED, Crippen, MolFromSmiles, rdmolops, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import networkx as nx
import os.path as op
import math
#from rdkit.six.moves import cPickle
import _pickle as cPickle
#from rdkit.six import iteritems
from rdkit import Chem
import pickle
import numpy as np

import sys
import os
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Fingerprints import FingerprintMols

def compute_rmse(gt, pred):
    return mean_squared_error(gt, pred, squared=False)

def compute_r2score(gt, pred):
    return r2_score(gt, pred)

def compute_roc_auc(gt, pred):
    return roc_auc_score(gt, pred)

def check_valid(smiles_list):
    total_num = len(smiles_list)
    empty_num = smiles_list.count("")
    return 1 - empty_num / float(total_num)

def check_unique(smiles_list):
    total_num = len(smiles_list)
    smiles_set = set(smiles_list)
    if "" in smiles_set:
        smiles_set.remove("")
    return len(smiles_set) / float(total_num)

def check_nolvelty(gen_smiles, train_smiles):
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel*100./len(gen_smiles)
    return novel_ratio

_fscores = None
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = cPickle.load(gzip.open('%s.pkl.gz'%name))
    outDict = {}
    for i in _fscores:
        for j in range(1,len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol,ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead,nSpiro

def calculateScore(m):
    if _fscores is None: readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,2)  #<- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId,v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp,-4)*v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m,includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads,nSpiro=numBridgeheadsAndSpiro(m,ri)
    nMacrocycles=0
    for x in ri.AtomRings():
        if len(x)>8: nMacrocycles+=1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters+1)
    spiroPenalty = math.log10(nSpiro+1)
    bridgePenalty = math.log10(nBridgeheads+1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0: macrocyclePenalty = math.log10(2)

    score2 = 0. -sizePenalty -stereoPenalty -spiroPenalty -bridgePenalty -macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
      score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.: sascore = 8. + math.log(sascore+1.-9.)
    if sascore > 10.: sascore = 10.0
    elif sascore < 1.: sascore = 1.0 

    return sascore

def compute_plogp(mol):

    #mol = MolFromSmiles(smiles_string)
    #logp = (Crippen.MolLogP(mol) - np.mean(logP_values)) / np.std(logP_values)
    logp = Crippen.MolLogP(mol)
    #SA_score = (-sascorer.calculateScore(mol) - np.mean(SA_scores)) / np.std(SA_scores)
    SA_score = -calculateScore(mol)
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6

    #cycle_score = (-cycle_length - np.mean(cycle_scores)) / np.std(cycle_scores)
    cycle_score = -cycle_length
    #plogp = -(logp + SA_score + cycle_score)
    plogp = (logp + SA_score + cycle_score)
    return plogp

clf_model = None
def load_model():
    global clf_model
    #name = op.join(op.dirname(__file__), 'clf_py36.pkl')
    name = op.join(op.dirname(__file__), 'drd2_current.pkl')
    with open(name, "rb") as f:
        clf_model = pickle.load(f)

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp

def compute_drd2(mol):
    if clf_model is None:
        load_model()

    #print(smile)
    #mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = clf_model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0

def compute_qed(mol):
    return QED.qed(mol)

def compute_logp(mol):
    return Crippen.MolLogP(mol)

def compute_TPSA(mol):
    return rdMolDescriptors.CalcTPSA(mol)

def compute_SAS(mol):
    return sascorer.calculateScore(mol)


def check_valid_unique(smiles_list):
    total_num = len(smiles_list)
    empty_num = smiles_list.count("")

    smiles_set = set(smiles_list)
    if "" in smiles_set:
        smiles_set.remove("")
    return 1 - empty_num / float(total_num), \
        len(smiles_set) / float(total_num - empty_num)

def get_similarity(smiles1, smiles2):
    if smiles1 == "" or smiles2 == "":
        return np.nan
    sim = TanimotoSimilarity(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smiles1)),
                       FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smiles2)))
    
    return sim

def get_scaffold(smiles):
    scaffold = MurckoScaffoldSmiles(smiles)
    return scaffold