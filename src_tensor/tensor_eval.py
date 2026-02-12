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
    )


class TensorSparseEvaluator:
    """Evaluator for sparse-step tensor propagation."""

    def __init__(self, psum: TensorPauliSum, stream_device: Optional[str] = None, offload_back: bool = False):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor backend.")
        self.psum = psum
        self.stream_device = stream_device
        self.offload_back = offload_back

    def evaluate_coeffs(self, thetas, priors=None) -> Tensor:
        """Return coefficient vector after applying all sparse steps with gradient control."""
        psum = self.psum
        v = psum.coeff_init
        if self.stream_device is not None:
            v = v.to(self.stream_device)
        if v.dim() == 1:
            v = v.unsqueeze(1)

        # PyTorch 텐서 변환 및 장치 설정
        thetas_t = torch.as_tensor(thetas, dtype=v.dtype, device=v.device)
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=v.dtype, device=v.device)
        
        # priors 처리 [추가]
        if priors is not None:
            priors_t = torch.as_tensor(priors, dtype=v.dtype, device=v.device)
        else:
            priors_t = None

        def _mm_coalesce_once(mat: Tensor, vec: Tensor) -> Tensor:
            if mat._nnz() == 0:
                return torch.zeros((mat.shape[0], vec.shape[1]), dtype=vec.dtype, device=vec.device)
            try:
                return torch.sparse.mm(mat, vec)
            except Exception:
                mat2 = mat.coalesce()
                return torch.sparse.mm(mat2, vec)

        for i, step in enumerate(psum.steps):
            if step.mat_const.device != v.device:
                step = _step_to_device(step, v.device)
            
            # [수정] 파라미터 결정 로직: emb_idx가 우선순위를 가짐
            if step.emb_idx >= 0 and priors_t is not None:
                # 임베딩 층: priors에서 값을 가져오고 detach()로 기울기 차단
                theta = priors_t[step.emb_idx].detach()
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
            elif step.param_idx >= 0:
                # 학습 층: thetas에서 값을 가져옴 (기울기 유지)
                theta = thetas_t[step.param_idx]
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
            else:
                # 파라미터 없는 스텝 (Clifford 등)
                cos_t = torch.ones((), dtype=v.dtype, device=v.device)
                sin_t = torch.zeros((), dtype=v.dtype, device=v.device)

            v_const = _mm_coalesce_once(step.mat_const, v)
            v_cos = _mm_coalesce_once(step.mat_cos, v)
            v_sin = _mm_coalesce_once(step.mat_sin, v)
            v = v_const + cos_t * v_cos + sin_t * v_sin

            if self.stream_device is not None and self.offload_back:
                psum.steps[i] = _step_to_device(step, "cpu")

        return v.squeeze(1)

    def evaluate_adjoint(self, thetas, out_vec: Tensor, priors=None) -> Tensor:
        """Apply adjoint with explicit batch-wise scaling."""
        psum = self.psum
        
        # 1. 초기 w 설정 (Num_Paulis, Batch)
        if priors is not None and priors.ndim > 1:
            batch_size = priors.shape[0]
            w = out_vec.unsqueeze(1).expand(-1, batch_size).clone()
        else:
            batch_size = 1
            w = out_vec.unsqueeze(1)

        if self.stream_device is not None:
            w = w.to(self.stream_device)

        thetas_t = torch.as_tensor(thetas, dtype=w.dtype, device=w.device)
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=w.dtype, device=w.device)

        if priors is not None:
            priors_t = torch.as_tensor(priors, dtype=w.dtype, device=w.device)
        else:
            priors_t = None

        def _mmT_coalesce_once(mat: Tensor, vec: Tensor) -> Tensor:
            if mat._nnz() == 0:
                return torch.zeros((mat.shape[1], vec.shape[1]), dtype=vec.dtype, device=vec.device)
            try:
                return torch.sparse.mm(mat.transpose(0, 1), vec)
            except Exception:
                mat2 = mat.coalesce()
                return torch.sparse.mm(mat2.transpose(0, 1), vec)

        # 2. 역순 전파 루프
        for rev_i, step in enumerate(reversed(psum.steps)):
            if step.mat_const.device != w.device:
                step = _step_to_device(step, str(w.device))

            # 배치별 각도 계산 (Batch,)
            if step.emb_idx >= 0 and priors_t is not None:
                theta = priors_t[:, step.emb_idx].detach()
                cos_t = torch.cos(theta) # (Batch,)
                sin_t = torch.sin(theta) # (Batch,)
            elif step.param_idx >= 0:
                theta = thetas_t[step.param_idx]
                cos_t = torch.cos(theta).expand(batch_size) # 모든 배치에 동일 적용
                sin_t = torch.sin(theta).expand(batch_size)
            else:
                cos_t = torch.ones(batch_size, dtype=w.dtype, device=w.device)
                sin_t = torch.zeros(batch_size, dtype=w.dtype, device=w.device)

            w_const = _mmT_coalesce_once(step.mat_const, w)
            w_cos = _mmT_coalesce_once(step.mat_cos, w)
            w_sin = _mmT_coalesce_once(step.mat_sin, w)

            # [핵심] Element-wise Multiplication 차원 명시
            # w_cos: (Num_Paulis, Batch) * cos_t: (Batch,)
            # 파이토치 브로드캐스팅은 뒤에서부터 맞추므로 (Num_Paulis, Batch) * (Batch,) 가 
            # 올바르게 동작하려면 cos_t를 (1, Batch)로 만들어야 합니다.
            w = w_const + cos_t.view(1, -1) * w_cos + sin_t.view(1, -1) * w_sin

            if self.stream_device is not None and self.offload_back:
                orig_i = len(psum.steps) - 1 - rev_i
                psum.steps[orig_i] = _step_to_device(step, "cpu")

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
