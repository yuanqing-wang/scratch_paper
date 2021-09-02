import torch

import espaloma as esp


def run():
    # grab dataset
    # esol = esp.data.esol(first=20)
    ds = esp.data.dataset.GraphDataset.load(
        "/data/chodera/wangyq/espaloma/scripts/mm_sampling_data/_zinc"
    )


    # do some typing
    typing = esp.graphs.legacy_force_field.LegacyForceField('gaff-1.81')
    ds.apply(typing, in_place=True) # this modify the original data

    # split
    # NOTE:
    # I don't like torch-generic splitting function as it requires
    # specifically the volume of each partition and it is inconsistent
    # with the specification of __getitem__ method
    ds_tr, ds_te, ds_vl = ds.split([8, 1, 1])

    # get a loader object that views this dataset in some way
    # using this specific flag the dataset turns into an iterator
    # that outputs loss function, per John's suggestion
    ds_tr = ds_tr.view('graph', batch_size=100, shuffle=True)
    ds_te = ds_te.view('graph', batch_size=100)
    ds_vl = ds_vl.view('graph', batch_size=100)

    # define a layer
    layer = esp.nn.layers.dgl_legacy.gn("SAGEConv")

    # define a representation
    representation = esp.nn.Sequential(
            layer,
            [128, "relu", 128, "relu", 128, "relu"],
    )

    # define a readout
    readout = esp.nn.readout.node_typing.NodeTyping(
            in_features=128,
            n_classes=100
    ) # not too many elements here I think?

    net = torch.nn.Sequential(
        representation,
        readout
    )

    exp = esp.TrainAndTest(
        ds_tr=ds_tr,
        ds_vl=ds_vl,
        ds_te=ds_te,
        net=net,
        metrics_te=[esp.metrics.TypingAccuracy()],
        n_epochs=3000,
        record_interval=100,
    )

    results = exp.run()
    curves = esp.app.report.curve(results)
    import os
    os.mkdir("results")
    for spec, curve in curves.items():
        import numpy as np
        np.save("results" + "/" + "_".join(spec) + ".npy", curve)



    for idx, state in exp.states.items():
        torch.save(state, "results/net%s.th" % idx)



if __name__ == '__main__':
    import sys
    run()
