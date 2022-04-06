# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import os

import numpy as np
import torch

import espaloma as esp


def run():
    # layer
    layer = esp.nn.layers.dgl_legacy.gn("SAGEConv")

    # representation
    representation = esp.nn.Sequential(layer, config=[128, "relu", 128, "relu", 128, "relu"])
    janossy_config = [128, "relu", 128, "relu", 128, "relu", 128, "relu"]
    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=128, config=janossy_config,
        out_features={
                2: {'log_coefficients': 2},
                3: {'log_coefficients': 2},
                4: {'k': 6},
        },
    )

    readout_improper = esp.nn.readout.janossy.JanossyPoolingImproper(
        in_features=128, config=janossy_config
    )

    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            import math
            g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()


            g.nodes['n2'].data['k'], g.nodes['n2'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
                g.nodes['n2'].data['coefficients'][:, 0][:, None],
                g.nodes['n2'].data['coefficients'][:, 1][:, None],
                1.5, 6.0,
            )

            g.nodes['n3'].data['k'], g.nodes['n3'].data['eq'] = esp.mm.functional.linear_mixture_to_original(
                g.nodes['n3'].data['coefficients'][:, 0][:, None],
                g.nodes['n3'].data['coefficients'][:, 1][:, None],
                0.0, math.pi
            )
 


            return g

    class GetLoss(torch.nn.Module):
        def forward(self, g):
            return torch.nn.MSELoss()(
                g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
                g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].mean(dim=-1, keepdims=True),
            )

    net = torch.nn.Sequential(
            representation,
            readout,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4"]),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4"], suffix="_ref")
    )

    rmse = [] 
    
    state_dict = torch.load(
            "128_SAGEConv_relu_1.0_1e-3_1___single_gpu_janossy_first_distributed_slow_decay/net4970.th",
            map_location="cpu",
    )

    # net = net.cuda()
    net.load_state_dict(state_dict)
 
    g = esp.Graph("[H]C1(C(OOC1(C([H])([H])[H])C([H])([H])[H])([H])[H])[H]")

    esp.data.md.MoleculeVacuumSimulation(n_conformers=1, n_samples=1000).run(g)
    esp.graphs.legacy_force_field.LegacyForceField().parametrize(g)

    net(g.heterograph)

    print(g.nodes['g'].data['u_ref'])
    print(g.nodes['g'].data['u'])

    print("rmse of total energy, centered",
        esp.metrics.center(esp.metrics.rmse)(g.nodes['g'].data['u_ref'], g.nodes['g'].data['u']).item() * 625,
        "kcal/mol"
    )

    print("rmse of bond energy, centered",
        esp.metrics.center(esp.metrics.rmse)(g.nodes['n2'].data['u_ref'], g.nodes['n2'].data['u']).item() * 625,
        "kcal/mol"
    )

    print("rmse of angle energy, centered",
        esp.metrics.center(esp.metrics.rmse)(g.nodes['n3'].data['u'], g.nodes['n3'].data['u_ref']).item() * 625,
        "kcal / mol"
    )

    print("rmse of proper torsion energy, centered: ",
        esp.metrics.center(esp.metrics.rmse)(g.nodes['n4'].data['u'], g.nodes['n4'].data['u_ref']).item() * 625,
        "kcal / mol"
    )


    print("rmse of total energy, uncentered",
        esp.metrics.rmse(g.nodes['g'].data['u_ref'], g.nodes['g'].data['u']).item() * 625,
        "kcal/mol"
    )

    print("rmse of bond energy, uncentered",
        esp.metrics.rmse(g.nodes['n2'].data['u_ref'], g.nodes['n2'].data['u']).item() * 625,
        "kcal/mol"
    )

    print("rmse of angle energy, uncentered",
        esp.metrics.rmse(g.nodes['n3'].data['u'], g.nodes['n3'].data['u_ref']).item() * 625,
        "kcal / mol"
    )

    print("rmse of proper torsion energy, uncentered: ",
        esp.metrics.rmse(g.nodes['n4'].data['u'], g.nodes['n4'].data['u_ref']).item() * 625,
        "kcal / mol"
    )




    '''
    print("average deviation from reference angle equilibrium: ",
        (g.nodes['n3'].data['x'] - g.nodes['n3'].data['eq_ref']).abs().mean().item()
    )

    print("rmse of angle equilibrium: ",
        esp.metrics.rmse(g.nodes['n3'].data['eq'], g.nodes['n3'].data['eq_ref']).item(),
        "rad"
    )

    print("rmse of angle force constant: ",
        esp.metrics.rmse(g.nodes['n3'].data['k'], g.nodes['n3'].data['k_ref']).item() * 625,
        "kcal / (mol * rad ** 2)"
    )
    '''




if __name__ == "__main__":
    import sys
    run()
