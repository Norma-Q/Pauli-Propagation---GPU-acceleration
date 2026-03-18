# """Tensor DAG evaluator (GPU-first)."""

# from __future__ import annotations

# from typing import Optional, TYPE_CHECKING, Any, cast

# torch: Any
# try:
#     import torch as _torch
#     torch = _torch
#     _TORCH_AVAILABLE = True
# except Exception:  # pragma: no cover - optional dependency
#     torch = cast(Any, None)
#     _TORCH_AVAILABLE = False

# if TYPE_CHECKING:
#     from torch import Tensor
# else:  # pragma: no cover - typing only
#     Tensor = Any

# torch = cast(Any, torch)

# from .tensor_types import TensorDAGGraph, TensorPauliSum, TensorSparseStep
# from torch.utils.checkpoint import checkpoint


# class TensorDAGEvaluator:
#     """GPU-first evaluator using layered edge tensors."""

#     def __init__(self, graph: TensorDAGGraph):
#         if not _TORCH_AVAILABLE:
#             raise RuntimeError("PyTorch is required for tensor backend.")
#         if graph.layers is None:
#             raise ValueError("graph.layers is required for TensorDAGEvaluator")
#         self.graph = graph

#     def evaluate_tensor(self, thetas) -> Tensor:
#         """Evaluate DAG and return a torch scalar (autograd-friendly)."""
#         g = self.graph
#         thetas_t = torch.as_tensor(thetas, dtype=g.node_value_init.dtype, device=g.node_value_init.device)
#         x = g.node_value_init.clone().unsqueeze(1)  # (N, 1)

#         if g.layers is None:
#             raise ValueError("graph.layers is required for TensorDAGEvaluator")
#         for layer in g.layers:
#             theta_vals = thetas_t[layer.edge_param]
#             trig_vals = torch.where(
#                 layer.edge_trig > 0,
#                 torch.cos(theta_vals),
#                 torch.where(layer.edge_trig < 0, torch.sin(theta_vals), torch.ones_like(theta_vals)),
#             )
#             values = layer.edge_sign * trig_vals
#             contrib = values.unsqueeze(1) * x[layer.parent_idx]
#             x.index_add_(0, layer.child_idx, contrib)

#         return torch.sum(x[g.final_nodes] * g.final_coeff.unsqueeze(1))

#     def evaluate(self, thetas) -> float:
#         """Evaluate DAG and return a Python float (non-differentiable)."""
#         result = self.evaluate_tensor(thetas)
#         return float(result.detach().cpu().item())


# __all__ = ["TensorDAGEvaluator"]


# def _step_to_device(step: TensorSparseStep, device: str) -> TensorSparseStep:
#     return TensorSparseStep(
#         mat_const=step.mat_const.to(device),
#         mat_cos=step.mat_cos.to(device),
#         mat_sin=step.mat_sin.to(device),
#         param_idx=step.param_idx,
#         emb_idx=step.emb_idx,
#         shape=step.shape,
#         same_cols=None if step.same_cols is None else step.same_cols.to(device),
#         anti_same_pos=None if step.anti_same_pos is None else step.anti_same_pos.to(device),
#     )


# class TensorSparseEvaluator:
#     """Evaluator for sparse-step tensor propagation."""

#     def __init__(self, psum: TensorPauliSum, compute_device: str = "cuda", chunk_size: Optional[int] = None):
#         if not _TORCH_AVAILABLE:
#             raise RuntimeError("PyTorch is required for tensor backend.")
#         self.psum = psum
#         self.compute_device = compute_device
#         self.chunk_size = chunk_size if chunk_size is not None else 1_000_000
#         self.checkpoint_target_fraction = 0.72
#         self.checkpoint_target_bytes_cap = int(11.0 * 1024 ** 3)
#         self.checkpoint_activation_multiplier = 6.0
#         self.checkpoint_max_steps_per_block = 1024  # Hard cap: each checkpoint block recomputes at most this many steps

#     def _get_checkpoint_budget_bytes(self) -> Optional[int]:
#         if (not torch.cuda.is_available()) or (not str(self.compute_device).startswith("cuda")):
#             return None
#         try:
#             _free_bytes, total_bytes = torch.cuda.mem_get_info(self.compute_device)
#         except Exception:
#             return None
#         return min(int(total_bytes * self.checkpoint_target_fraction), int(self.checkpoint_target_bytes_cap))

#     def _estimate_step_activation_bytes(self, step: TensorSparseStep, vec_cols: int, elem_size: int) -> int:
#         # One step can materialize multiple tensors with the same output shape during autograd.
#         return int(step.shape[0]) * int(vec_cols) * int(elem_size) * int(self.checkpoint_activation_multiplier)

#     def _adaptive_same_chunk_size(self, vec_calc: Tensor, n_same: int) -> int:
#         chunk = max(1, int(self.chunk_size))
#         if (n_same <= 0) or (not torch.cuda.is_available()) or (vec_calc.device.type != "cuda"):
#             return min(chunk, max(1, int(n_same)))

#         try:
#             free_bytes, _total_bytes = torch.cuda.mem_get_info(vec_calc.device)
#         except Exception:
#             return min(chunk, max(1, int(n_same)))

#         # Keep a healthy margin because forward/autograd tensors are already live.
#         safe_bytes = max(1, int(free_bytes * 0.25))
#         per_row_bytes = max(1, int(vec_calc.shape[1]) * int(vec_calc.element_size()) * 3)
#         adaptive_chunk = max(1, safe_bytes // per_row_bytes)
#         return min(chunk, adaptive_chunk, max(1, int(n_same)))

#     def _make_checkpoint_blocks(self, steps_subset, vec_cols: int, elem_size: int) -> list[list[TensorSparseStep]]:
#         budget_bytes = self._get_checkpoint_budget_bytes()
#         max_steps = max(1, int(self.checkpoint_max_steps_per_block))

#         blocks: list[list[TensorSparseStep]] = []
#         current_block: list[TensorSparseStep] = []
#         current_bytes = 0
#         for step in steps_subset:
#             step_bytes = self._estimate_step_activation_bytes(step, vec_cols, elem_size)
#             # Split on max_steps hard cap OR byte budget (whichever triggers first).
#             # The max_steps cap prevents single-block recompute OOM during checkpoint backward.
#             over_budget = (budget_bytes is not None) and (budget_bytes > 0) and (current_bytes + step_bytes > budget_bytes)
#             over_step_cap = len(current_block) >= max_steps
#             if current_block and (over_budget or over_step_cap):
#                 blocks.append(current_block)
#                 current_block = []
#                 current_bytes = 0
#             current_block.append(step)
#             current_bytes += step_bytes

#         if current_block:
#             blocks.append(current_block)
#         return blocks

#     def _should_use_checkpoint(self, steps_subset, vec_cols: int, elem_size: int, requires_grad: bool) -> bool:
#         if not requires_grad:
#             return False
#         total_steps = len(steps_subset)
#         total_output_rows = sum(int(step.shape[0]) for step in steps_subset)
#         max_step_rows = max((int(step.shape[0]) for step in steps_subset), default=0)

#         # Long training circuits blow up because autograd keeps many step outputs alive.
#         # Force checkpointing for clearly large cases instead of relying on a fragile byte estimate.
#         if total_steps >= 1024:
#             return True
#         if total_output_rows >= 1_000_000:
#             return True
#         if max_step_rows >= 250_000:
#             return True

#         budget_bytes = self._get_checkpoint_budget_bytes()
#         if budget_bytes is None:
#             return False

#         est_activation_bytes = sum(
#             self._estimate_step_activation_bytes(step, vec_cols, elem_size) for step in steps_subset
#         )
#         return est_activation_bytes > budget_bytes

#     def evaluate_coeffs(self, thetas, embedding=None, coeff_init: Optional[Tensor] = None) -> Tensor:
#         """Return coefficient vector after applying all sparse steps with gradient control."""
#         psum = self.psum
#         v = psum.coeff_init if coeff_init is None else coeff_init
        
#         # 벡터는 연산 장치에 상주
#         v = v.to(self.compute_device)
#         if v.dim() == 1:
#             v = v.unsqueeze(1)

#         # PyTorch 텐서 변환 및 장치 설정
#         thetas_t = torch.as_tensor(thetas, dtype=v.dtype, device=v.device)
#         if thetas_t.numel() == 0:
#             thetas_t = torch.zeros(1, dtype=v.dtype, device=v.device)
        
#         if embedding is not None:
#             embedding_t = torch.as_tensor(embedding, dtype=v.dtype, device=v.device)
#             if embedding_t.dim() == 1:
#                 embedding_t = embedding_t.unsqueeze(0)
#         else:
#             embedding_t = None

#         # Chunked Sparse MM Helper (y = M @ v)
#         # M is on Storage (CPU), v is on Calc (GPU)
#         def _chunked_mm(mat_storage: Tensor, vec_calc: Tensor) -> Tensor:
#             if mat_storage._nnz() == 0:
#                 return torch.zeros((mat_storage.shape[0], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
            
#             # If matrix is small or already on calc device, run directly
#             if mat_storage.device.type == vec_calc.device.type or mat_storage.shape[0] <= self.chunk_size:
#                 mat_calc = mat_storage.to(vec_calc.device)
#                 try:
#                     return torch.sparse.mm(mat_calc, vec_calc)
#                 except RuntimeError:
#                     return torch.sparse.mm(mat_calc.coalesce(), vec_calc)
#             raise RuntimeError(
#                 "Chunked sparse.mm on SparseCPU row slices is not currently supported in this evaluator. "
#                 "Lower-memory training should rely on checkpointed forward blocks instead of SparseCPU row slicing."
#             )

#         def _chunk_bounds(length: int, chunk_override: Optional[int] = None):
#             chunk = max(1, int(self.chunk_size if chunk_override is None else chunk_override))
#             for start in range(0, int(length), chunk):
#                 end = min(start + chunk, int(length))
#                 yield start, end

#         def _apply_implicit_same(step: TensorSparseStep, vec_calc: Tensor, cos_t: Tensor) -> Optional[Tensor]:
#             if step.same_cols is None:
#                 return None
#             same_cols_src = step.same_cols
#             n_same = int(same_cols_src.numel())
#             if n_same == int(step.shape[0]):
#                 out = torch.empty((step.shape[0], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
#             else:
#                 out = torch.zeros((step.shape[0], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
#             if n_same == 0:
#                 return out
#             anti_pos_src = step.anti_same_pos
#             same_chunk_size = self._adaptive_same_chunk_size(vec_calc, n_same)

#             for start, end in _chunk_bounds(n_same, same_chunk_size):
#                 cols_chunk = same_cols_src[start:end].to(vec_calc.device, non_blocking=True)
#                 out_chunk = out[start:end]
#                 out_chunk.copy_(vec_calc.index_select(0, cols_chunk))

#                 if anti_pos_src is not None and anti_pos_src.numel() > 0:
#                     anti_mask = (anti_pos_src >= int(start)) & (anti_pos_src < int(end))
#                     if bool(anti_mask.any().item()):
#                         anti_local = (anti_pos_src[anti_mask] - int(start)).to(vec_calc.device, non_blocking=True)
#                         out_chunk[anti_local] = cos_t * out_chunk.index_select(0, anti_local)

#             return out

#         def _apply_step_sequence(v_in: Tensor, steps_subset, thetas_local: Tensor, embedding_local: Optional[Tensor]) -> Tensor:
#             v_local = v_in
#             for step in steps_subset:
#                 if step.emb_idx >= 0 and embedding_local is not None:
#                     if int(embedding_local.shape[0]) != 1:
#                         raise ValueError(
#                             "evaluate_coeffs() does not support batched embedding; "
#                             "use a single embedding vector or call expvals() for batched evaluation."
#                         )
#                     theta = embedding_local[0, step.emb_idx].detach()
#                     cos_t = torch.cos(theta)
#                     sin_t = torch.sin(theta)
#                 elif step.param_idx >= 0:
#                     theta = thetas_local[step.param_idx]
#                     cos_t = torch.cos(theta)
#                     sin_t = torch.sin(theta)
#                 else:
#                     cos_t = torch.ones((), dtype=v_local.dtype, device=v_local.device)
#                     sin_t = torch.zeros((), dtype=v_local.dtype, device=v_local.device)

#                 implicit_same = _apply_implicit_same(step, v_local, cos_t)
#                 if implicit_same is not None:
#                     v_next = implicit_same
#                 else:
#                     v_next = _chunked_mm(step.mat_const, v_local)

#                 if implicit_same is None and step.mat_cos._nnz() > 0:
#                     v_next = v_next + cos_t * _chunked_mm(step.mat_cos, v_local)
#                 if step.mat_sin._nnz() > 0:
#                     v_next = v_next + sin_t * _chunked_mm(step.mat_sin, v_local)
#                 v_local = v_next
#             return v_local

#         steps_all = list(psum.steps)
#         input_requires_grad = bool(getattr(thetas, "requires_grad", False) or thetas_t.requires_grad)
#         use_checkpoint = self._should_use_checkpoint(
#             steps_all,
#             vec_cols=int(v.shape[1]),
#             elem_size=int(v.element_size()),
#             requires_grad=input_requires_grad,
#         )

#         # Fail-safe: very long CUDA step chains should not run as one giant autograd graph.
#         if (not use_checkpoint) and str(v.device).startswith("cuda") and len(steps_all) >= 1024:
#             use_checkpoint = True

#         def _run_checkpoint_blocks(v_in: Tensor) -> Tensor:
#             checkpoint_blocks = self._make_checkpoint_blocks(
#                 steps_all,
#                 vec_cols=int(v_in.shape[1]),
#                 elem_size=int(v_in.element_size()),
#             )
#             v_out = v_in
#             for steps_subset in checkpoint_blocks:
#                 if embedding_t is not None:
#                     def _block_fn_with_embedding(v_block: Tensor, thetas_block: Tensor, embedding_block: Tensor, *, _steps=steps_subset) -> Tensor:
#                         return _apply_step_sequence(v_block, _steps, thetas_block, embedding_block)

#                     v_out = checkpoint(_block_fn_with_embedding, v_out, thetas_t, embedding_t, use_reentrant=False)
#                 else:
#                     def _block_fn_no_embedding(v_block: Tensor, thetas_block: Tensor, *, _steps=steps_subset) -> Tensor:
#                         return _apply_step_sequence(v_block, _steps, thetas_block, None)

#                     v_out = checkpoint(_block_fn_no_embedding, v_out, thetas_t, use_reentrant=False)
#             return cast(Tensor, v_out)

#         if use_checkpoint:
#             v = _run_checkpoint_blocks(v)
#         else:
#             v = _apply_step_sequence(v, steps_all, thetas_t, embedding_t)

#         return v.squeeze(1)

#     def evaluate_adjoint(self, thetas, out_vec: Tensor, embedding=None) -> Tensor:
#         """Apply adjoint with explicit batch-wise scaling."""
#         psum = self.psum
        
#         # 1. 초기 w 설정 (Num_Paulis, Batch)
#         if embedding is not None and embedding.ndim > 1:
#             batch_size = embedding.shape[0]
#             w = out_vec.unsqueeze(1).expand(-1, batch_size).clone()
#         else:
#             batch_size = 1
#             w = out_vec.unsqueeze(1)

#         w = w.to(self.compute_device)

#         thetas_t = torch.as_tensor(thetas, dtype=w.dtype, device=w.device)
#         if thetas_t.numel() == 0:
#             thetas_t = torch.zeros(1, dtype=w.dtype, device=w.device)

#         if embedding is not None:
#             embedding_t = torch.as_tensor(embedding, dtype=w.dtype, device=w.device)
#         else:
#             embedding_t = None

#         # Chunked Adjoint MM Helper (y = M^T @ w)
#         # M is on Storage (CPU), w is on Calc (GPU)
#         # y = sum_i (M_i^T @ w_i) where M_i is row-chunk of M, w_i is row-chunk of w
#         def _chunked_mm_T(mat_storage: Tensor, vec_calc: Tensor) -> Tensor:
#             if mat_storage._nnz() == 0:
#                 return torch.zeros((mat_storage.shape[1], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)

#             # Fast path
#             if mat_storage.device.type == vec_calc.device.type or mat_storage.shape[0] <= self.chunk_size:
#                 mat_calc = mat_storage.to(vec_calc.device)
#                 try:
#                     return torch.sparse.mm(mat_calc.transpose(0, 1), vec_calc)
#                 except RuntimeError:
#                     return torch.sparse.mm(mat_calc.coalesce().transpose(0, 1), vec_calc)

#             raise RuntimeError(
#                 "Chunked sparse.mm_T on SparseCPU row slices is not currently supported in this evaluator."
#             )

#         def _chunk_bounds(length: int, chunk_override: Optional[int] = None):
#             chunk = max(1, int(self.chunk_size if chunk_override is None else chunk_override))
#             for start in range(0, int(length), chunk):
#                 end = min(start + chunk, int(length))
#                 yield start, end

#         def _apply_implicit_same_T(step: TensorSparseStep, vec_calc: Tensor, cos_t: Tensor) -> Optional[Tensor]:
#             if step.same_cols is None:
#                 return None
#             out = torch.zeros((step.shape[1], vec_calc.shape[1]), dtype=vec_calc.dtype, device=vec_calc.device)
#             same_cols_src = step.same_cols
#             n_same = int(same_cols_src.numel())
#             if n_same == 0:
#                 return out
#             anti_pos_src = step.anti_same_pos

#             for start, end in _chunk_bounds(n_same):
#                 cols_chunk = same_cols_src[start:end].to(vec_calc.device, non_blocking=True)
#                 same_block = vec_calc[start:end]

#                 if anti_pos_src is not None and anti_pos_src.numel() > 0:
#                     anti_mask = (anti_pos_src >= int(start)) & (anti_pos_src < int(end))
#                     if bool(anti_mask.any().item()):
#                         anti_local = (anti_pos_src[anti_mask] - int(start)).to(vec_calc.device, non_blocking=True)
#                         same_block = same_block.clone()
#                         same_block[anti_local] = cos_t.view(1, -1) * same_block.index_select(0, anti_local)

#                 out.index_add_(0, cols_chunk, same_block)
#             return out

#         # 2. 역순 전파 루프
#         for rev_i, step in enumerate(reversed(psum.steps)):
#             # 배치별 각도 계산 (Batch,)
#             if step.emb_idx >= 0 and embedding_t is not None:
#                 theta = embedding_t[:, step.emb_idx].detach()
#                 cos_t = torch.cos(theta) # (Batch,)
#                 sin_t = torch.sin(theta) # (Batch,)
#             elif step.param_idx >= 0:
#                 theta = thetas_t[step.param_idx]
#                 cos_t = torch.cos(theta).expand(batch_size) # 모든 배치에 동일 적용
#                 sin_t = torch.sin(theta).expand(batch_size)
#             else:
#                 cos_t = torch.ones(batch_size, dtype=w.dtype, device=w.device)
#                 sin_t = torch.zeros(batch_size, dtype=w.dtype, device=w.device)

#             implicit_same = _apply_implicit_same_T(step, w, cos_t)
#             if implicit_same is not None:
#                 w_next = implicit_same
#             else:
#                 w_next = _chunked_mm_T(step.mat_const, w)

#             if implicit_same is None and step.mat_cos._nnz() > 0:
#                 w_next = w_next + cos_t.view(1, -1) * _chunked_mm_T(step.mat_cos, w)
#             if step.mat_sin._nnz() > 0:
#                 w_next = w_next + sin_t.view(1, -1) * _chunked_mm_T(step.mat_sin, w)
#             w = w_next

#         # 3. 결과 반환 (Batch, Num_Paulis)
#         res = w.transpose(0, 1)
#         if batch_size == 1:
#             return res.squeeze(0)
#         return res

#     def evaluate(self, thetas) -> float:
#         """Return sum of coefficients as a scalar."""
#         coeffs = self.evaluate_coeffs(thetas)
#         return float(torch.sum(coeffs).detach().cpu().item())


# __all__.append("TensorSparseEvaluator")


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