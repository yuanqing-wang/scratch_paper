#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import dgl
from openeye import oechem, oedepict, oegrapheme
import espaloma as esp
import math


# In[2]:


# # layer
# layer = esp.nn.layers.dgl_legacy.gn("SAGEConv")

# # representation
# representation = esp.nn.Sequential(layer, config=[128, "relu", 128, "relu", 128, "relu"])
# janossy_config = [128, "relu", 128, "relu", 128, "relu", 128, "relu"]
# readout = esp.nn.readout.janossy.JanossyPooling(
#     in_features=128, config=janossy_config,
#     out_features={
#             1: {"e": 1, "s": 1},
#             2: {'log_coefficients': 2},
#             3: {'log_coefficients': 2},
#             4: {'k': 6},
#     },
# )

# readout_improper = esp.nn.readout.janossy.JanossyPoolingImproper(
#     in_features=128, config=janossy_config
# )

# class ExpCoeff(torch.nn.Module):
#     def forward(self, g):
#         g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
#         g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
#         g.nodes['n2'].data['k'], g.nodes['n2'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
#             g.nodes['n2'].data['coefficients'][:, 0][:, None],
#             g.nodes['n2'].data['coefficients'][:, 1][:, None],
#             1.5, 6.0,
#         )

#         g.nodes['n3'].data['k'], g.nodes['n3'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
#             g.nodes['n3'].data['coefficients'][:, 0][:, None],
#             g.nodes['n3'].data['coefficients'][:, 1][:, None],
#             0.0, math.pi
#         )
        
#         return g

# class GetLoss(torch.nn.Module):
#     def forward(self, g):
#         return torch.nn.MSELoss()(
#             g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(),
#             g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].mean(),
#         )

# net = torch.nn.Sequential(
#         representation,
#         readout,
#         readout_improper,
#         ExpCoeff(),
#         esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
#         # esp.mm.geometry.GeometryInGraph(),
#         # esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
# )


# In[3]:


# net.load_state_dict(
#     torch.load(
#         "everything_joint.th",
#         map_location="cpu",
#     )
# )


# In[4]:


net = torch.load("espaloma-0.2.2.pt")


# In[5]:


def get_mol(n=1):
    g = esp.Graph(
        "CC(=O)" + n * "NC(C)C(=O)" + "NC",
    )
    
    # g.mol.compute_partial_charges_am1bcc()
    # g.nodes['n1'].data['q'] = torch.tensor(g.mol.partial_charges.flatten())[:, None]
    return g


# In[6]:


class AtomPartialChargeArcFxn(oegrapheme.OESurfaceArcFxnBase):
    def __init__(self, colorg, g):
        oegrapheme.OESurfaceArcFxnBase.__init__(self)
        self.colorg = colorg
        self.g = g

    def __call__(self, image, arc):
        adisp = arc.GetAtomDisplay()
        if adisp is None or not adisp.IsVisible():
            return False

        atom = adisp.GetAtom()
        if atom is None:
            return False

        idx = atom.GetIdx()
        charge = self.g.nodes['n1'].data['q'][idx].item()
        if charge == 0.0:
            return True
        color = self.colorg.GetColorAt(charge)

        pen = oedepict.OEPen()
        pen.SetForeColor(color)
        pen.SetLineWidth(2.0)

        center = arc.GetCenter()
        radius = arc.GetRadius()
        bAngle = arc.GetBgnAngle()
        eAngle = arc.GetEndAngle()

        edgeAngle = 5.0
        dir = oegrapheme.OEPatternDirection_Outside
        patternAngle = 10.0
        oegrapheme.OEDrawBrickRoadSurfaceArc(image, center, bAngle, eAngle, radius, pen,
                                             edgeAngle, dir, patternAngle)
        return True

    def CreateCopy(self):
        return AtomPartialChargeArcFxn(self.colorg, self.g).__disown__()


# In[7]:


class GetAtomLabel(oedepict.OEDisplayAtomPropBase):
    def __init__(self, g):
        oedepict.OEDisplayAtomPropBase.__init__(self)
        self.g = g

    def __call__(self, atom):
        idx = atom.GetIdx()
        charge = self.g.nodes['n1'].data['q'][idx].item()
        return "%.2f" % charge

    def CreateCopy(self):
        copy = GetAtomLabel(g=self.g)
        return copy.__disown__()


# In[8]:


class GetBondEq(oedepict.OEDisplayBondPropBase):
    def __init__(self, g):
        oedepict.OEDisplayBondPropBase.__init__(self)
        self.g = g

    def __call__(self, bond):
        bond_idx = bond.GetIdx()
        eq = 0.529177 * g.heterograph.nodes['n2'].data['eq'].flatten()[bond_idx]
        return "%.2f" % eq

    def CreateCopy(self):
        copy = GetBondEq(g=self.g)
        return copy.__disown__()


# In[ ]:


for idx in [1, 3]:
    g = get_mol(idx)
    net(g.heterograph)
    opts = oedepict.OE2DMolDisplayOptions(idx*300, idx*300, oedepict.OEScale_AutoScale)
    mol = g.mol.to_openeye()
    # oechem.OEAddExplicitHydrogens(mol)
    oechem.OEMMFFAtomTypes(mol)
    oechem.OEMMFF94PartialCharges(mol)
    oedepict.OEPrepareDepiction(mol, True, False)
    oechem.OETriposAtomTypeNames(mol)
    opts.SetAtomPropertyFunctor(GetAtomLabel(g=g))
    # opts.SetBondPropertyFunctor(GetBondEq(g=g))
    # opts.SetBondPropLabelFont(oedepict.OEFont(oechem.OEDarkBlue))
    opts.SetHydrogenStyle(oedepict.OEHydrogenStyle_Hidden)
    disp = oedepict.OE2DMolDisplay(mol, opts)
    coloranion = oechem.OEColorStop(-1.0, oechem.OEColor(oechem.OEDarkRed))
    colorcation = oechem.OEColorStop(+1.0, oechem.OEColor(oechem.OEDarkBlue))
    colorg = oechem.OELinearColorGradient(coloranion, colorcation)
    colorg.AddStop(oechem.OEColorStop(0.0, oechem.OEColor(oechem.OEWhite)))

    arcfxn = AtomPartialChargeArcFxn(colorg, g=g)

    for atom in mol.GetAtoms():
        oegrapheme.OESetSurfaceArcFxn(mol, atom, arcfxn)
    oegrapheme.OEDraw2DSurface(disp)
    oedepict.OERenderMolecule("charge_%s.png" % idx, disp)


# In[35]:


dir(oedepict)


# In[36]:


for idx in [1, 3]:
    g = get_mol(idx)
    net(g.heterograph)
    opts = oedepict.OE2DMolDisplayOptions(idx*300, idx*300, oedepict.OEScale_AutoScale)
    mol = g.mol.to_openeye()
    oechem.OEMMFFAtomTypes(mol)
    oechem.OEMMFF94PartialCharges(mol)
    oedepict.OEPrepareDepiction(mol, True, False)
    oechem.OETriposAtomTypeNames(mol)
    # opts.SetAtomPropertyFunctor(GetAtomLabel(g=g))
    opts.SetHydrogenStyle(oedepict.OEHydrogenStyle_Hidden)
    opts.SetBondPropertyFunctor(GetBondEq(g=g))
    opts.SetBondPropLabelFont(oedepict.OEFont(oechem.OEDarkBlue))
    disp = oedepict.OE2DMolDisplay(mol, opts)
    coloranion = oechem.OEColorStop(-1.0, oechem.OEColor(oechem.OEDarkRed))
    colorcation = oechem.OEColorStop(+1.0, oechem.OEColor(oechem.OEDarkBlue))
    colorg = oechem.OELinearColorGradient(coloranion, colorcation)
    colorg.AddStop(oechem.OEColorStop(0.0, oechem.OEColor(oechem.OEWhite)))

    #     arcfxn = AtomPartialChargeArcFxn(colorg)

    #     for atom in mol.GetAtoms():
    #         oegrapheme.OESetSurfaceArcFxn(mol, atom, arcfxn)
    # oegrapheme.OEDraw2DSurface(disp)
    oedepict.OERenderMolecule("bond_%s.png" % idx, disp)


# In[ ]:




