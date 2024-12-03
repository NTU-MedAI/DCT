import torch
import torch.nn as nn
import torch.nn.functional as F
from module.MDAM_implementations import Graph_encoder
from module.spherenet import SphereNet
from torch_geometric.nn.acts import swish

class SphereNetModel(nn.Module):
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        act=swish, output_init='GlorotOrthogonal', use_node_features=True):
        super(SphereNetModel, self).__init__()
        self.model = SphereNet(
            energy_and_force=energy_and_force, cutoff=cutoff, num_layers=num_layers,
            hidden_channels=hidden_channels, out_channels=out_channels, int_emb_size=int_emb_size,
            basis_emb_size_dist=basis_emb_size_dist, basis_emb_size_angle=basis_emb_size_angle, basis_emb_size_torsion=basis_emb_size_torsion, out_emb_channels=out_emb_channels,
            num_spherical=num_spherical, num_radial=num_radial, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip, num_output_layers=num_output_layers,
            act=act, output_init=output_init, use_node_features=use_node_features
        )
        self.name = 'SphereNetModel'
        self.energy_and_force = energy_and_force
    def forward(self,batch_data):
        return self.model(batch_data)

class MDAMModel(nn.Module):
    def __init__(
        self,
            node_features_1=75, edge_features_1=4,
            message_size=25, message_passes=2, out_features=1,
            msg_depth=4, msg_hidden_dim=200, att_depth=3, att_hidden_dim=200,
            gather_width=100, gather_att_depth=3, gather_att_hidden_dim=100,
            gather_emb_depth=3, gather_emb_hidden_dim=100,
            out_depth=2, out_hidden_dim=100, out_layer_shrinkage=1.0
    ):
        super(MDAMModel, self).__init__()
        self.model = Graph_encoder(
            node_features_1=node_features_1, edge_features_1=edge_features_1, message_size=message_size, message_passes=message_passes, out_features=out_features,
            msg_depth=msg_depth, msg_hidden_dim=msg_hidden_dim, att_depth=att_depth, att_hidden_dim=att_hidden_dim,
            gather_width=gather_width, gather_att_depth=gather_att_depth, gather_att_hidden_dim=gather_att_hidden_dim,
            gather_emb_depth=gather_emb_depth, gather_emb_hidden_dim=gather_emb_hidden_dim,
            out_depth=out_depth, out_hidden_dim=out_hidden_dim, out_layer_shrinkage=out_layer_shrinkage
        )
        self.name = 'MDAMModel'
        self.energy_and_force = False
    def forward(self,batch_data):
        adj_1, nd_1, ed_1, d1, mask_1 = batch_data.adj, batch_data.nd,\
                                        batch_data.ed, batch_data.d, batch_data.mask
        return self.model(adj_1, nd_1, ed_1, d1, mask_1)

class CombineModel(nn.Module):
    def __init__(
        self, batch_size,
        energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        act=swish, output_init='GlorotOrthogonal', use_node_features=True,
        node_features_1=75, edge_features_1=4,
        message_size=25, message_passes=2, out_features=1,
        msg_depth=4, msg_hidden_dim=200, att_depth=3, att_hidden_dim=200,
        gather_width=100, gather_att_depth=3, gather_att_hidden_dim=100,
        gather_emb_depth=3, gather_emb_hidden_dim=100,
        out_depth=2, out_hidden_dim=100, out_layer_shrinkage=1.0,
    ):
        super(CombineModel, self).__init__()
        self.sphereNet = SphereNet(
            energy_and_force=energy_and_force, cutoff=cutoff, num_layers=num_layers,
            hidden_channels=hidden_channels, out_channels=out_channels, int_emb_size=int_emb_size,
            basis_emb_size_dist=basis_emb_size_dist, basis_emb_size_angle=basis_emb_size_angle, basis_emb_size_torsion=basis_emb_size_torsion, out_emb_channels=out_emb_channels,
            num_spherical=num_spherical, num_radial=num_radial, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip, num_output_layers=num_output_layers,
            act=act, output_init=output_init, use_node_features=use_node_features
        )
        self.MDAMNet = Graph_encoder(
            node_features_1=node_features_1, edge_features_1=edge_features_1, message_size=message_size, message_passes=message_passes, out_features=out_features,
            msg_depth=msg_depth, msg_hidden_dim=msg_hidden_dim, att_depth=att_depth, att_hidden_dim=att_hidden_dim,
            gather_width=gather_width, gather_att_depth=gather_att_depth, gather_att_hidden_dim=gather_att_hidden_dim,
            gather_emb_depth=gather_emb_depth, gather_emb_hidden_dim=gather_emb_hidden_dim,
            out_depth=out_depth, out_hidden_dim=out_hidden_dim, out_layer_shrinkage=out_layer_shrinkage
        )
        self.batch_size = batch_size
        self.name = 'CombineModel'
        self.energy_and_force = energy_and_force
        self.linear = nn.Linear(gather_width+128+out_channels,1)

    def forward(self,batch_data):
        adj,nd,ed,d, mask = batch_data.adj,batch_data.nd,batch_data.ed,batch_data.d, batch_data.mask
        adj = adj.reshape(-1,9,9)
        nd = nd.reshape(-1,9,75)
        ed = ed.reshape(-1,9,9,4)
        d = d.reshape(-1,50)
        mask = mask.reshape(-1,50)
        (e, v, u) = self.sphereNet(batch_data)
        out = self.MDAMNet(adj,nd,ed,d, mask)
        result = torch.concat([out,u],dim=1)
        return self.linear(result)


