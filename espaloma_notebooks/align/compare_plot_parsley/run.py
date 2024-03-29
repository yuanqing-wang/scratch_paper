import math
import torch
import espaloma as esp
import numpy as np
from simtk import unit

def run(idx):
    ds = esp.data.dataset.GraphDataset.load("/data/chodera/wangyq/espaloma/scripts/data_with_breakdown/parsley/merged_parsley")
    ds.shuffle(2666)
    _, ds, __ = ds.split([8, 1, 1])
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
                g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(),
                g.nodes['g'].data['u_ref'] - g.nodes['g'].data['u_ref'].mean(),
            )

    net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            ExpCoeff(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
    ) 

    net.load_state_dict(
        torch.load(
            "/data/chodera/wangyq/espaloma/scripts/parsley_deep/128_SAGEConv_relu_1.0_1e-5_1___mean_/net2030.th",
            map_location="cpu",
        )
    )

    def compare(g, forcefields=["gaff-1.81", "gaff-2.11", "openff-1.2.0"]):
        net(g.heterograph)
        system = esp.graphs.deploy.openmm_system_from_graph(g)
        
        from simtk.unit.quantity import Quantity
        from simtk.openmm.app import Simulation
        from simtk import openmm, unit
        from openmmforcefields.generators import SystemGenerator

        # simulation specs
        TEMPERATURE = 350 * unit.kelvin
        STEP_SIZE = 1.0 * unit.femtosecond
        COLLISION_RATE = 1.0 / unit.picosecond
        EPSILON_MIN = 0.05 * unit.kilojoules_per_mole

        # use langevin integrator, although it's not super useful here
        integrator = openmm.LangevinIntegrator(
            TEMPERATURE, COLLISION_RATE, STEP_SIZE
        )

        # create simulation
        simulation = Simulation(
            topology=g.mol.to_topology().to_openmm(), system=system, integrator=integrator
        )

        idx_lowest = g.nodes['g'].data['u_ref'].flatten().argmin()
        x_qm = (
            Quantity(
                g.nodes["n1"].data["xyz"][:, idx_lowest, :].detach().numpy(),
                esp.units.DISTANCE_UNIT,
            )
            # .value_in_unit(unit.nanometer)
        )

        simulation.context.setPositions(x_qm)
        simulation.minimizeEnergy(tolerance=0.1)

        x_esp = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        xs = {}
        
        for forcefield in forcefields:
            
            # define a system generator
            system_generator = SystemGenerator(
                small_molecule_forcefield=forcefield,
            )

            mol = g.mol
            # mol.assign_partial_charges("formal_charge")
            # create system
            integrator = openmm.LangevinIntegrator(
                TEMPERATURE, COLLISION_RATE, STEP_SIZE
            )
            system = system_generator.create_system(
                topology=mol.to_topology().to_openmm(),
                molecules=mol,
            )
        
            # create simulation
            simulation = Simulation(
                topology=g.mol.to_topology().to_openmm(), system=system, integrator=integrator
            )
            
            simulation.context.setPositions(x_qm)
            simulation.minimizeEnergy(tolerance=0.1)
            
            xs["x_%s" % (forcefield.replace("-", "_"))] = simulation.context\
                .getState(getPositions=True).getPositions(asNumpy=True)
            
        return (
            g, x_qm, x_esp, *[xs["x_%s" % (forcefield.replace("-", "_"))] for forcefield in forcefields]
        )
    g, x_qm, x_esp, x_gaff1, x_gaff2, x_openff = compare(ds[idx])
    np.save("x_qm_%s" % idx, x_qm.value_in_unit(unit.angstrom))
    np.save("x_esp_%s" % idx, x_esp.value_in_unit(unit.angstrom))
    np.save("x_gaff1_%s" % idx, x_gaff1.value_in_unit(unit.angstrom))
    np.save("x_gaff2_%s" % idx, x_gaff2.value_in_unit(unit.angstrom))
    np.save("x_openff_%s" % idx, x_openff.value_in_unit(unit.angstrom))

    rmsd = (((x_esp - x_qm) ** 2).mean() ** 0.5).value_in_unit(unit.angstrom)
    print(rmsd)
    print(g.mol)
    g.mol.add_conformer(x_qm)
    g.mol.to_file("qm_%s_%.3f.sdf" % (idx, rmsd), "sdf")
    g.mol._conformers = []
    g.mol.add_conformer(x_esp)
    g.mol.to_file("esp_%s_%.3f.sdf" % (idx, rmsd), "sdf")

if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]))
    
