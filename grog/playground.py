import torch
import dgl
import espaloma as esp
from torchdiffeq import odeint
from rdkit.Chem import AllChem

def run():

    ds = esp.data.dataset.GraphDataset(
        [esp.Graph("C"*idx) for idx in range(1, 5)]
    )

    def get_2d_conformer(g):
        mol = g.mol.to_rdkit()
        AllChem.Compute2DCoords(mol)
        x = next(iter(mol.GetConformers())).GetPositions() / 0.529177
        x = torch.tensor(x)
        g.nodes['n1'].data['xyz_init'] = x[:, None, :]
       
        return g

    ds.apply(get_2d_conformer, in_place=True)
    ds.apply(esp.graphs.legacy_force_field.LegacyForceField("gaff-1.81").parametrize, in_place=True)

    g = next(iter(ds.view(batch_size=len(ds)))).to("cuda")

    D = 16
    BATCH_SIZE=16
    layer = esp.nn.layers.dgl_legacy.gn()
    representation = esp.nn.Sequential(layer, config=[16, "relu", 16, "relu"])
    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=16, config=[16, "elu"],
        out_features={
            2: {"W1": D, "B1": D, "W2": D * 2, "B2": 2},
            3: {"W1": D, "B1": D, "W2": D * 2, "B2": 2},
        },
    )

    net = torch.nn.Sequential(
        representation,
        readout
    )

    def get_force(x, t, g):
        g.nodes['n1'].data['xyz'] = x
        
        # grab neural network weights
        w1_n2 = g.nodes["n2"].data["W1"]
        b1_n2 = g.nodes["n2"].data["B1"]
        w2_n2 = g.nodes["n2"].data["W2"]
        b2_n2 = g.nodes["n2"].data["B2"]

        w1_n3 = g.nodes["n3"].data["W1"]
        b1_n3 = g.nodes["n3"].data["B1"]
        w2_n3 = g.nodes["n3"].data["W2"]
        b2_n3 = g.nodes["n3"].data["B2"]

        # compose coefficients
        g.nodes["n2"].data["k"], g.nodes["n2"].data["eq"] = torch.split(
                torch.matmul(
                    torch.nn.functional.relu(
                        t * w1_n2 + b1_n2
                    ).reshape([-1, 1, D]),
                    w2_n2.reshape([-1, D, 2])
                ) + b2_n2[:, None, :],
                1,
                dim=-1)

        g.nodes["n3"].data["k"], g.nodes["n3"].data["eq"] = torch.split(
                torch.matmul(
                    torch.nn.functional.relu(
                        t * w1_n3 + b1_n3
                ).reshape([-1, 1, D]),
                w2_n3.reshape([-1, D, 2])
                ) + b2_n3[:, None, :],
                1,
                dim=-1)

        g.nodes["n2"].data["k"] = g.nodes["n2"].data["k"].squeeze(1) # + g.nodes["n2"].data["k_ref"] # * t 
        g.nodes["n2"].data["eq"] = g.nodes["n2"].data["eq"].squeeze(1) # + g.nodes["n2"].data["eq_ref"]
        g.nodes["n3"].data["k"] = g.nodes["n3"].data["k"].squeeze(1) # + g.nodes["n3"].data["k_ref"] # * t
        g.nodes["n3"].data["eq"] = g.nodes["n3"].data["eq"].squeeze(1) # + g.nodes["n3"].data["eq_ref"]

        esp.mm.geometry.geometry_in_graph(g)

        esp.mm.energy.energy_in_graph(g, terms=["n2", "n3"])
        
        force = torch.autograd.grad(
            g.nodes['g'].data['u'].sum(),
            g.nodes['n1'].data['xyz'],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        return force

    class ForwardKernel(torch.nn.Module):
        def __init__(self):
            super(ForwardKernel, self).__init__()
            self.net = net
            #torch.nn.init.normal_(net[1].f_out_2_to_W2.weight, std=1e-5)
            # torch.nn.init.normal_(net[1].f_out_3_to_W2.weight, std=1e-5)
            # torch.nn.init.zeros_(net[1].f_out_2_to_W2.bias)
            # torch.nn.init.zeros_(net[1].f_out_3_to_W2.bias)

            # torch.nn.init.normal_(net[1].f_out_2_to_B2.weight, std=1e-5)
            # torch.nn.init.normal_(net[1].f_out_3_to_B2.weight, std=1e-5)
            # torch.nn.init.zeros_(net[1].f_out_2_to_B2.bias)
            # torch.nn.init.zeros_(net[1].f_out_3_to_B2.bias)


        def forward(self, t, state, g=g):
            # x.shape = v.shape = (n_atoms, N_WINDOWS, 3)
            x, v = state

            dx_dt = v
            dv_dt = get_force(x, t, g)
            return dx_dt, dv_dt

        def get_initial_state(self, g=g):
            v0 = torch.zeros([g.number_of_nodes('n1'), BATCH_SIZE, 3], device="cuda")
            x0 = torch.distributions.Normal(
                torch.tensor(0.0, device="cuda"),
                torch.tensor(0.1, device="cuda")
            ).rsample([g.number_of_nodes('n1'), BATCH_SIZE, 3]) + g.nodes['n1'].data['xyz_init']

            x0.requires_grad = True
            v0.requires_grad = True
            return x0, v0

        def simulate(self):
            self.net(g)
            state = self.get_initial_state()
            solution = odeint(self, state, torch.tensor([0.0, 1.0], device="cuda"))
            return solution

    forward_kernel = ForwardKernel().cuda()

    optimizer = torch.optim.Adam(forward_kernel.parameters(), 1e-3)

    for _ in range(1000):
        optimizer.zero_grad()
        x, v = forward_kernel.simulate()
        g.nodes['n1'].data['xyz'] = x[-1]
        esp.mm.geometry.geometry_in_graph(g)
        esp.mm.energy.energy_in_graph(g, terms=["n2", "n3"], suffix="_ref")
        loss = g.nodes['g'].data['u_ref']
        loss.sum().backward()
        print(loss.sum() * 625.0 / (BATCH_SIZE*5), flush=True)
        optimizer.step()

if __name__ == "__main__":
    run()



