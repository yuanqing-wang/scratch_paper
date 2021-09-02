import torch
import espaloma as esp

def run():
    # grab dataset
    # esol = esp.data.esol(first=20)
    ds = esp.data.dataset.GraphDataset.load(
        "/data/chodera/wangyq/espaloma/scripts/mm_sampling_data/_zinc"
    )

    _, __, ds = ds.split([8, 1, 1]) 

    # do some typing
    typing = esp.graphs.legacy_force_field.LegacyForceField('gaff-1.81')
    ds.apply(typing, in_place=True) # this modify the original data
    ds.save("zinc_param")


if __name__ == "__main__":
    run()
