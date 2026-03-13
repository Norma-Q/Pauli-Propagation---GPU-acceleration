"""Tensor DAG evaluator (GPU-first)."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any, cast

torch: Any
try:
    import torch as _torch
    torch = _torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = cast(Any, None)
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from torch import Tensor
else:  # pragma: no cover - typing only
    Tensor = Any

torch = cast(Any, torch)

from .tensor_types import TensorDAGGraph, TensorPauliSum, TensorSparseStep


class TensorDAGEvaluator:
    """GPU-first evaluator using layered edge tensors."""

    def __init__(self, graph: TensorDAGGraph):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor backend.")
        if graph.layers is None:
            raise ValueError("graph.layers is required for TensorDAGEvaluator")
        self.graph = graph

    def evaluate_tensor(self, thetas) -> Tensor:
        """Evaluate DAG and return a torch scalar (autograd-friendly)."""
        g = self.graph
        thetas_t = torch.as_tensor(thetas, dtype=g.node_value_init.dtype, device=g.node_value_init.device)
        x = g.node_value_init.clone().unsqueeze(1)  # (N, 1)

        if g.layers is None:
            raise ValueError("graph.layers is required for TensorDAGEvaluator")
        for layer in g.layers:
            theta_vals = thetas_t[layer.edge_param]
            trig_vals = torch.where(
                layer.edge_trig > 0,
                torch.cos(theta_vals),
                torch.where(layer.edge_trig < 0, torch.sin(theta_vals), torch.ones_like(theta_vals)),
            )
            values = layer.edge_sign * trig_vals
            contrib = values.unsqueeze(1) * x[layer.parent_idx]
            x.index_add_(0, layer.child_idx, contrib)

        return torch.sum(x[g.final_nodes] * g.final_coeff.unsqueeze(1))

    def evaluate(self, thetas) -> float:
        """Evaluate DAG and return a Python float (non-differentiable)."""
        result = self.evaluate_tensor(thetas)
        return float(result.detach().cpu().item())


__all__ = ["TensorDAGEvaluator"]


def _step_to_device(step: TensorSparseStep, device: str) -> TensorSparseStep:
    return TensorSparseStep(
        mat_const=step.mat_const.to(device),
        mat_cos=step.mat_cos.to(device),
        mat_sin=step.mat_sin.to(device),
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
        shape=step.shape,
        same_cols=None if step.same_cols is None else step.same_cols.to(device),
        anti_same_pos=None if step.anti_same_pos is None else step.anti_same_pos.to(device),
    )


class TensorSparseEvaluator:
    """Evaluator for sparse-step tensor propagation."""

    def __init__(self, psum: TensorPauliSum, compute_device: str = "cuda", chunk_size: Optional[int] = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor backend.")
        self.psum = psum
        self.compute_device = compute_device
        self.chunk_size = chunk_size if chunk_size is not None else 1_000_000

    def evaluate_coeffs(self, thetas, embedding=None, coeff_init: Optional[Tensor] = None) -> Tensor:
        """Return coefficient vector after applying all sparse steps with gradient control."""
        psum = self.psum
        v = psum.coeff_init if coeff_init is None else coeff_init
        
        # 벡터는 연산 장치에 상주
        v = v.to(self.compute_device)
        if v.dim() == 1:
            v = v.unsqueeze(1)

        # PyTorch 텐서 변환 및 장치 설정
        thetas_t = torch.as_tensor(thetas, dtype=v.dtype, device=v.device)
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=v.dtype, device=v.device)
        
        if embedding is not None:
            embedding_t = torch.as_tensor(embedding, dtype=v.dtype, device=v.device)
            if embedding_t.dim() == 1:
                embedding_t = embedding_t.unsqueeze(0)
        else:
            embedding_t = None

        # Chunked Sparse MM Helper (y = M @ v)
        # M is on Storage (CPU), v is on Calc (GPU)
        def _chunked_mm(mat_storage: Tensor, vec_calc: Tensor) -> Tensor:
            if mat_storage._nnz() == 0:
                return torch.zeros((mat_storage.shape[0], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
            
            # If matrix is small or already on calc device, run directly
            if mat_storage.device.type == vec_calc.device.type or mat_storage.shape[0] <= self.chunk_size:
                mat_calc = mat_storage.to(vec_calc.device)
                try:
                    return torch.sparse.mm(mat_calc, vec_calc)
                except RuntimeError:
                    return torch.sparse.mm(mat_calc.coalesce(), vec_calc)

            # Chunking loop
            n_rows = mat_storage.shape[0]
            result_parts = []
            for i in range(0, n_rows, self.chunk_size):
                end = min(i + self.chunk_size, n_rows)
                # Slice rows from storage tensor
                # Note: PyTorch sparse slicing creates a view or copy of indices, relatively cheap
                mat_chunk = mat_storage[i:end].to(vec_calc.device)
                # Result chunk
                res_chunk = torch.sparse.mm(mat_chunk, vec_calc)
                result_parts.append(res_chunk)
            
            return torch.cat(result_parts, dim=0)

        def _apply_implicit_same(step: TensorSparseStep, vec_calc: Tensor, cos_t: Tensor) -> Optional[Tensor]:
            if step.same_cols is None:
                return None
            same_cols = step.same_cols.to(vec_calc.device)
            out = torch.zeros((step.shape[0], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
            n_same = int(same_cols.numel())
            if n_same == 0:
                return out
            same_vals = vec_calc.index_select(0, same_cols)
            if step.anti_same_pos is not None and step.anti_same_pos.numel() > 0:
                anti_pos = step.anti_same_pos.to(vec_calc.device)
                same_vals = same_vals.clone()
                same_vals[anti_pos] = cos_t * same_vals.index_select(0, anti_pos)
            out[:n_same] = same_vals
            return out

        for i, step in enumerate(psum.steps):
            if step.emb_idx >= 0 and embedding_t is not None:
                if int(embedding_t.shape[0]) != 1:
                    raise ValueError(
                        "evaluate_coeffs() does not support batched embedding; "
                        "use a single embedding vector or call expvals() for batched evaluation."
                    )
                theta = embedding_t[0, step.emb_idx].detach()
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
            elif step.param_idx >= 0:
                theta = thetas_t[step.param_idx]
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
            else:
                cos_t = torch.ones((), dtype=v.dtype, device=v.device)
                sin_t = torch.zeros((), dtype=v.dtype, device=v.device)

            implicit_same = _apply_implicit_same(step, v, cos_t)
            if implicit_same is not None:
                v_next = implicit_same
            else:
                v_next = _chunked_mm(step.mat_const, v)

            if implicit_same is None and step.mat_cos._nnz() > 0:
                v_next = v_next + cos_t * _chunked_mm(step.mat_cos, v)
            if step.mat_sin._nnz() > 0:
                v_next = v_next + sin_t * _chunked_mm(step.mat_sin, v)
            v = v_next

        return v.squeeze(1)

    def evaluate_adjoint(self, thetas, out_vec: Tensor, embedding=None) -> Tensor:
        """Apply adjoint with explicit batch-wise scaling."""
        psum = self.psum
        
        # 1. 초기 w 설정 (Num_Paulis, Batch)
        if embedding is not None and embedding.ndim > 1:
            batch_size = embedding.shape[0]
            w = out_vec.unsqueeze(1).expand(-1, batch_size).clone()
        else:
            batch_size = 1
            w = out_vec.unsqueeze(1)

        w = w.to(self.compute_device)

        thetas_t = torch.as_tensor(thetas, dtype=w.dtype, device=w.device)
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=w.dtype, device=w.device)

        if embedding is not None:
            embedding_t = torch.as_tensor(embedding, dtype=w.dtype, device=w.device)
        else:
            embedding_t = None

        # Chunked Adjoint MM Helper (y = M^T @ w)
        # M is on Storage (CPU), w is on Calc (GPU)
        # y = sum_i (M_i^T @ w_i) where M_i is row-chunk of M, w_i is row-chunk of w
        def _chunked_mm_T(mat_storage: Tensor, vec_calc: Tensor) -> Tensor:
            if mat_storage._nnz() == 0:
                return torch.zeros((mat_storage.shape[1], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)

            # Fast path
            if mat_storage.device.type == vec_calc.device.type or mat_storage.shape[0] <= self.chunk_size:
                mat_calc = mat_storage.to(vec_calc.device)
                try:
                    return torch.sparse.mm(mat_calc.transpose(0, 1), vec_calc)
                except RuntimeError:
                    return torch.sparse.mm(mat_calc.coalesce().transpose(0, 1), vec_calc)

            # Chunking loop (Reduce Sum)
            n_rows = mat_storage.shape[0]
            y_acc = None
            
            for i in range(0, n_rows, self.chunk_size):
                end = min(i + self.chunk_size, n_rows)
                
                # Slice M rows (CPU) -> GPU
                mat_chunk = mat_storage[i:end].to(vec_calc.device)
                # Slice w rows (GPU)
                vec_chunk = vec_calc[i:end]
                
                # M_chunk^T @ vec_chunk
                term = torch.sparse.mm(mat_chunk.transpose(0, 1), vec_chunk)
                
                if y_acc is None:
                    y_acc = term
                else:
                    y_acc += term
            if y_acc is None:
                return torch.zeros((mat_storage.shape[1], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
            return y_acc

        def _apply_implicit_same_T(step: TensorSparseStep, vec_calc: Tensor, cos_t: Tensor) -> Optional[Tensor]:
            if step.same_cols is None:
                return None
            same_cols = step.same_cols.to(vec_calc.device)
            out = torch.zeros((step.shape[1], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
            n_same = int(same_cols.numel())
            if n_same == 0:
                return out
            same_block = vec_calc[:n_same]
            if step.anti_same_pos is not None and step.anti_same_pos.numel() > 0:
                anti_pos = step.anti_same_pos.to(vec_calc.device)
                same_block = same_block.clone()
                same_block[anti_pos] = cos_t.view(1, -1) * same_block.index_select(0, anti_pos)
            out.index_add_(0, same_cols, same_block)
            return out

        # 2. 역순 전파 루프
        for rev_i, step in enumerate(reversed(psum.steps)):
            # 배치별 각도 계산 (Batch,)
            if step.emb_idx >= 0 and embedding_t is not None:
                theta = embedding_t[:, step.emb_idx].detach()
                cos_t = torch.cos(theta) # (Batch,)
                sin_t = torch.sin(theta) # (Batch,)
            elif step.param_idx >= 0:
                theta = thetas_t[step.param_idx]
                cos_t = torch.cos(theta).expand(batch_size) # 모든 배치에 동일 적용
                sin_t = torch.sin(theta).expand(batch_size)
            else:
                cos_t = torch.ones(batch_size, dtype=w.dtype, device=w.device)
                sin_t = torch.zeros(batch_size, dtype=w.dtype, device=w.device)

            implicit_same = _apply_implicit_same_T(step, w, cos_t)
            if implicit_same is not None:
                w_next = implicit_same
            else:
                w_next = _chunked_mm_T(step.mat_const, w)

            if implicit_same is None and step.mat_cos._nnz() > 0:
                w_next = w_next + cos_t.view(1, -1) * _chunked_mm_T(step.mat_cos, w)
            if step.mat_sin._nnz() > 0:
                w_next = w_next + sin_t.view(1, -1) * _chunked_mm_T(step.mat_sin, w)
            w = w_next

        # 3. 결과 반환 (Batch, Num_Paulis)
        res = w.transpose(0, 1)
        if batch_size == 1:
            return res.squeeze(0)
        return res

    def evaluate(self, thetas) -> float:
        """Return sum of coefficients as a scalar."""
        coeffs = self.evaluate_coeffs(thetas)
        return float(torch.sum(coeffs).detach().cpu().item())


__all__.append("TensorSparseEvaluator")
