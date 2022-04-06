import torch
import espaloma as esp
from openff.toolkit.topology import Molecule
from simtk import openmm

molecule = Molecule.from_smiles("CC")
print(molecule.find_rotatable_bonds())

mol_graph = esp.Graph(molecule)

esp_model = torch.load("espaloma_model.pt")
esp_model(mol_graph.heterograph)

system = esp.graphs.deploy.openmm_system_from_graph(mol_graph)
# write to file 
xml = openmm.XmlSerializer.serialize(system)

with open("ethane.xml", "w") as output:
    output.write(xml)

molecule.generate_conformers()
molecule.to_file("ethane.pdb", "pdb")

