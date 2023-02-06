import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset

from .conv import GNN_node, GNN_node_Virtualnode

class GraphEnvAug(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 600, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, add_fp=None):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GraphEnvAug, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.add_fp = add_fp
        self.gamma  = gamma
        self.env_w = 0.5
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        rationale_emb_dim = emb_dim
        rationale_dropout = drop_ratio
        rationale_jk = 'last'
        if 'virtual' in gnn_type: 
            rationale_gnn_node = GNN_node_Virtualnode(2, rationale_emb_dim, JK = rationale_jk, drop_ratio = rationale_dropout, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node_Virtualnode(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        else:
            rationale_gnn_node = GNN_node(2, rationale_emb_dim, JK = rationale_jk, drop_ratio = rationale_dropout, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        self.separator = separator(
            rationale_gnn_node=rationale_gnn_node, 
            gate_nn = torch.nn.Sequential(torch.nn.Linear(rationale_emb_dim, 2*rationale_emb_dim), torch.nn.BatchNorm1d(2*rationale_emb_dim), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*rationale_emb_dim, 1)),
            nn=None
            )
        rep_dim = emb_dim
        if 'ECFP' in self.add_fp:
            rep_dim += 2048
        if 'MACCS' in self.add_fp:
            rep_dim += 167
        if 'onlyECFP' in self.add_fp:
            rep_dim = 2048
        if 'onlyMACCS' in self.add_fp:
            rep_dim = 167      
        self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))
        # self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        self.classifier = torch.nn.Linear(rep_dim, self.num_tasks)

    def forward(self, batched_data, cls=False, return_rationale=False, graph_node_prob_dict=None, outer_step=None):
        h_node = self.graph_encoder(batched_data)[0]
        h_r, h_env, r_node_num, env_node_num = self.separator(batched_data, h_node)
        h_rep = (h_r.unsqueeze(1) + self.env_w * h_env.unsqueeze(0)).view(-1, self.emb_dim)
        if 'ECFP' in self.add_fp:
            h_r = torch.cat((h_r, torch.cuda.LongTensor(batched_data['mgf']).to(torch.float32)),axis=1)
            ecfps = torch.cuda.LongTensor(batched_data['mgf']).to(torch.float32).repeat_interleave(batched_data.batch[-1]+1,dim=0)
            h_rep = torch.cat((h_rep, ecfps),axis=1)
        if 'MACCS' in self.add_fp:
            h_r = torch.cat((h_r, torch.cuda.LongTensor(batched_data['maccs']).to(torch.float32)),axis=1)
            maccses = torch.cuda.LongTensor(batched_data['maccs']).to(torch.float32).repeat_interleave(batched_data.batch[-1]+1,dim=0)
            h_rep = torch.cat((h_rep, maccses),axis=1)
        if cls:
            pred_rem = self.classifier(h_r)
            pred_rep = self.classifier(h_rep)
        else:
            pred_rem = self.predictor(h_r)
            pred_rep = self.predictor(h_rep)
        loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()
        output = {'pred_rep': pred_rep, 'pred_rem': pred_rem, 'loss_reg':loss_reg, 'reps': h_r}
        if return_rationale:
            if graph_node_prob_dict is None:
                graph_node_prob_dict = {}
            node_prob_batch = self.separator.get_rationale_node_probability().view(-1).cpu()
            batch = batched_data.batch.cpu() 
            for graph_idx in range(batch[-1]+1):
                # print(f'step: {step}, graph_idx: {graph_idx}, total: {int(step*batch_size + graph_idx)}')
                mask = batch == graph_idx
                node_prob_batch[mask]
                if outer_step is not None:
                    graph_node_prob_dict[int(outer_step*(batch[-1]+1) + graph_idx)] = node_prob_batch[mask].numpy()
                else:
                    graph_node_prob_dict[int(graph_idx)] = node_prob_batch[mask].numpy()
            output['dict_node_prob'] = graph_node_prob_dict
        return output
    
    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False
        model.apply(fn)
    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True
        model.apply(fn)

class separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)
    def get_rationale_node_probability(self,):
        return self.rationale_node_probability

    def forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data)[0]
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)                
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)
        self.rationale_node_probability = gate

        h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)
        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)
        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8 