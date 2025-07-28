from model import ModelQuantizer, top_k_sparsification, reconstruct_from_sparse
import torch

class CommunicationEfficientFedHB:
    def __init__(self, compression_ratio=0.1, quant_bits=8, sampling_ratio=0.3):
        self.comp = compression_ratio
        self.samp = sampling_ratio
        self.quantizer = ModelQuantizer(bits=quant_bits)

    def efficient_client_update(self, client_model, global_model, loader,
                                criterion, round_idx, total_rounds, device):
        from model import alt_client_update
        updated, loss, epochs, sim = alt_client_update(
            client_model, global_model, loader, criterion, round_idx, total_rounds, device
        )
        sd = updated.state_dict()
        sparse, idxs = top_k_sparsification(sd, self.comp)
        quantized, scales = self.quantizer.quantize_model(sparse)
        return {'quant': quantized, 'idxs': idxs, 'scales': scales,
                'epochs': epochs, 'sim': sim}

    def server_aggregate(self, global_model, updates):
        # 중요도: (1 - sim) * epochs
        imps = [(i, (1 - u['sim']) * u['epochs']) for i, u in enumerate(updates)]
        imps.sort(key=lambda x: x[1], reverse=True)
        selected = [i for i, _ in imps[:max(1, int(len(updates) * self.samp))]]
        agg, total_w = {}, 0.0
        for idx in selected:
            u = updates[idx]
            sparse = self.quantizer.dequantize_model(u['quant'], u['scales'])
            full = reconstruct_from_sparse(sparse, u['idxs'])
            w = u['epochs']; total_w += w
            for name, param in full.items():
                agg.setdefault(name, torch.zeros_like(param))
                agg[name] += param * w
        for name in agg:
            agg[name] /= total_w
        global_model.load_state_dict(agg)
        return global_model

    def server_aggregate_full(self, global_model, updates, use_momentum=False, momentum_beta=0.9, use_pruning=False):
        # updates: [{'state_dict': ..., 'num_samples': ...}, ...]
        total_samples = sum(u['num_samples'] for u in updates)
        new_state = {}
        prev_state = global_model.state_dict()
        for key in prev_state.keys():
            agg_param = sum(u['state_dict'][key] * u['num_samples'] for u in updates) / total_samples
            # Pruning
            if use_pruning:
                mask = agg_param.abs() > 0.15
                agg_param = agg_param * mask
            # Momentum
            if use_momentum:
                agg_param = momentum_beta * prev_state[key] + (1 - momentum_beta) * agg_param
            new_state[key] = agg_param
        global_model.load_state_dict(new_state)
        return global_model 