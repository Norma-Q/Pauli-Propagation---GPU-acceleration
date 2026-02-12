"""
Pauli Propagation Surrogate - Python Implementation
GPU-accelerated version of PauliPropagation.jl Surrogate

Phase 1: Build Propagation Tree & Zero Filtering
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional, Set
from collections import defaultdict
import numpy as np


# ============================================================================
# 1. Pauli Algebra - Symplectic Representation
# ============================================================================
@dataclass(frozen=True)
class PauliString:
    """
    Pauli string represented in symplectic form.
    - x_mask: bitmask for X component (bit i = 1 if qubit i has X)
    - z_mask: bitmask for Z component (bit i = 1 if qubit i has Z)
    - I: (0, 0), X: (1, 0), Z: (0, 1), Y: (1, 1) for each qubit
    """
    x_mask: int
    z_mask: int
    
    def __repr__(self):
        return f"P({self.x_mask:04x},{self.z_mask:04x})"


def get_local_pauli(x_mask: int, z_mask: int, q: int) -> Tuple[int, int]:
    """Get local Pauli at qubit q: returns (x_bit, z_bit)"""
    bit = 1 << q
    return (1 if (x_mask & bit) else 0), (1 if (z_mask & bit) else 0)


def count_weight(x_mask: int, z_mask: int) -> int:
    """Count number of non-identity Paulis (weight)"""
    return bin(x_mask | z_mask).count('1')


def count_xy(x_mask: int, z_mask: int) -> int:
    """Count number of X/Y Paulis (exclude I and Z)"""
    return bin(x_mask).count('1')


def popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint64 arrays."""
    x = arr.astype(np.uint64, copy=False)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x


def build_mask_blocks(pstrs: List["PauliString"], n_blocks: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pack masks into uint64 blocks of shape (n_terms, n_blocks)."""
    n_terms = len(pstrs)
    x_blocks = np.zeros((n_terms, n_blocks), dtype=np.uint64)
    z_blocks = np.zeros((n_terms, n_blocks), dtype=np.uint64)
    mask64 = (1 << 64) - 1
    for b in range(n_blocks):
        shift = 64 * b
        x_blocks[:, b] = np.fromiter(((p.x_mask >> shift) & mask64 for p in pstrs), dtype=np.uint64, count=n_terms)
        z_blocks[:, b] = np.fromiter(((p.z_mask >> shift) & mask64 for p in pstrs), dtype=np.uint64, count=n_terms)
    return x_blocks, z_blocks


def blocks_to_int(block_row: np.ndarray) -> int:
    """Reconstruct Python int mask from uint64 blocks."""
    out = 0
    for b, val in enumerate(block_row):
        out |= int(val) << (64 * b)
    return out


def get_bit_from_blocks(blocks: np.ndarray, q: int) -> np.ndarray:
    """Get bit array for qubit q from block matrix."""
    b = q // 64
    shift = q % 64
    return (blocks[:, b] >> np.uint64(shift)) & np.uint64(1)


def set_bit_in_blocks(blocks: np.ndarray, q: int, bit_values: np.ndarray) -> None:
    """Set bit for qubit q in block matrix using bit_values (0/1)."""
    b = q // 64
    shift = q % 64
    bit = np.uint64(1 << shift)
    blocks[:, b] = (blocks[:, b] & ~bit) | (bit_values.astype(np.uint64) << np.uint64(shift))


def make_pauli_string(pauli: str, qubits: Optional[List[int]] = None) -> PauliString:
    """Create PauliString from a string like 'XYIZ'.

    If qubits is provided, pauli length must match len(qubits) and letters map
    to those qubits. If qubits is None, indices are 0..len(pauli)-1.
    """
    pauli = pauli.upper()
    if qubits is None:
        qubits = list(range(len(pauli)))
    if len(pauli) != len(qubits):
        raise ValueError("Length of pauli string must match qubits length")

    x_mask = 0
    z_mask = 0
    for p, q in zip(pauli, qubits):
        bit = 1 << q
        if p == "X":
            x_mask |= bit
        elif p == "Z":
            z_mask |= bit
        elif p == "Y":
            x_mask |= bit
            z_mask |= bit
        elif p == "I":
            pass
        else:
            raise ValueError(f"Unsupported Pauli char: {p}")
    return PauliString(x_mask, z_mask)


def pauli_rotation_masks(gate: 'PauliRotation') -> Tuple[int, int]:
    """Return (x_mask, z_mask) for PauliRotation gate."""
    x_mask = 0
    z_mask = 0
    for q, p in zip(gate.qubits, gate.pauli):
        if p in ("X", "Y"):
            x_mask |= 1 << q
        if p in ("Z", "Y"):
            z_mask |= 1 << q
    return x_mask, z_mask


def gate_support(gate: 'Gate') -> Set[int]:
    if isinstance(gate, (PauliRotation, CliffordGate)):
        return set(gate.qubits)
    return set()


def commutes_gate(g1: 'Gate', g2: 'Gate') -> bool:
    s1 = gate_support(g1)
    s2 = gate_support(g2)
    if not (s1 & s2):
        return True  # disjoint supports commute

    # PauliRotation vs PauliRotation: symplectic commutation
    if isinstance(g1, PauliRotation) and isinstance(g2, PauliRotation):
        x1, z1 = pauli_rotation_masks(g1)
        x2, z2 = pauli_rotation_masks(g2)
        symp = bin((x1 & z2) ^ (z1 & x2)).count('1') % 2
        return symp == 0

    # Conservative: if overlapping and either is Clifford, stop block
    return False


def split_commuting_blocks(circuit: List['Gate']) -> List[List['Gate']]:
    blocks: List[List[Gate]] = []
    current: List[Gate] = []
    for gate in circuit:
        if not current:
            current = [gate]
            continue
        if all(commutes_gate(gate, g) for g in current):
            current.append(gate)
        else:
            blocks.append(current)
            current = [gate]
    if current:
        blocks.append(current)
    return blocks


def build_commuting_depths(circuit: List['Gate']) -> List[List['Gate']]:
    """Group gates into commuting depth buckets.

    Each depth contains gates that mutually commute (by commutes_gate).
    Gates are placed in the earliest depth where they commute with all
    gates already in that depth; otherwise a new depth is created.
    """
    depths: List[List[Gate]] = []
    gate_depths: List[int] = []

    for i, gate in enumerate(circuit):
        max_block = -1
        for j in range(i):
            if not commutes_gate(gate, circuit[j]):
                max_block = max(max_block, gate_depths[j])

        depth = max_block + 1
        while depth >= len(depths):
            depths.append([])
        depths[depth].append(gate)
        gate_depths.append(depth)

    return depths


def build_param_depths(circuit: List['Gate']) -> Dict[int, int]:
    """Map PauliRotation param_idx to commuting depth index."""
    depths = build_commuting_depths(circuit)
    param_depths: Dict[int, int] = {}
    for depth_idx, depth in enumerate(depths):
        for gate in depth:
            if isinstance(gate, PauliRotation) and gate.param_idx >= 0:
                param_depths[gate.param_idx] = depth_idx
    return param_depths


def pauli_product(p1: PauliString, p2: PauliString) -> Tuple[PauliString, int]:
    """
    Multiply two Pauli strings: P1 * P2
    Returns (new_pauli, phase) where phase ∈ {0,1,2,3} for {+1,+i,-1,-i}
    """
    # Local Pauli multiplication table (Pennylane convention: Y = i X Z)
    # phase: 0=+1, 1=+i, 2=-1, 3=-i
    local_product = {
        ('I', 'I'): ('I', 0),
        ('I', 'X'): ('X', 0),
        ('I', 'Y'): ('Y', 0),
        ('I', 'Z'): ('Z', 0),
        ('X', 'I'): ('X', 0),
        ('Y', 'I'): ('Y', 0),
        ('Z', 'I'): ('Z', 0),
        ('X', 'X'): ('I', 0),
        ('Y', 'Y'): ('I', 0),
        ('Z', 'Z'): ('I', 0),
        ('X', 'Y'): ('Z', 1),
        ('Y', 'X'): ('Z', 3),
        ('X', 'Z'): ('Y', 3),
        ('Z', 'X'): ('Y', 1),
        ('Y', 'Z'): ('X', 1),
        ('Z', 'Y'): ('X', 3),
    }

    to_pauli = {(0, 0): 'I', (1, 0): 'X', (1, 1): 'Y', (0, 1): 'Z'}
    to_xz = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}

    x_new = 0
    z_new = 0
    phase = 0

    for q in range(64):  # max 64 qubits
        bit = 1 << q
        x1 = 1 if (p1.x_mask & bit) else 0
        z1 = 1 if (p1.z_mask & bit) else 0
        x2 = 1 if (p2.x_mask & bit) else 0
        z2 = 1 if (p2.z_mask & bit) else 0

        p1_local = to_pauli[(x1, z1)]
        p2_local = to_pauli[(x2, z2)]

        p_out, p_phase = local_product[(p1_local, p2_local)]
        phase = (phase + p_phase) % 4

        x_out, z_out = to_xz[p_out]
        if x_out:
            x_new |= bit
        if z_out:
            z_new |= bit

    return PauliString(x_new, z_new), phase


def pauli_product_phase_on_support(pstr: PauliString, gate: "PauliRotation") -> int:
    """Compute multiplication phase for P * G using only gate support."""
    local_product = {
        ('I', 'I'): ('I', 0),
        ('I', 'X'): ('X', 0),
        ('I', 'Y'): ('Y', 0),
        ('I', 'Z'): ('Z', 0),
        ('X', 'I'): ('X', 0),
        ('Y', 'I'): ('Y', 0),
        ('Z', 'I'): ('Z', 0),
        ('X', 'X'): ('I', 0),
        ('Y', 'Y'): ('I', 0),
        ('Z', 'Z'): ('I', 0),
        ('X', 'Y'): ('Z', 1),
        ('Y', 'X'): ('Z', 3),
        ('X', 'Z'): ('Y', 3),
        ('Z', 'X'): ('Y', 1),
        ('Y', 'Z'): ('X', 1),
        ('Z', 'Y'): ('X', 3),
    }
    to_pauli = {(0, 0): 'I', (1, 0): 'X', (1, 1): 'Y', (0, 1): 'Z'}

    phase = 0
    for q, g in zip(gate.qubits, gate.pauli):
        x1, z1 = get_local_pauli(pstr.x_mask, pstr.z_mask, q)
        p1_local = to_pauli[(x1, z1)]
        p_out, p_phase = local_product[(p1_local, g)]
        phase = (phase + p_phase) % 4
    return phase


def commutes(p1: PauliString, p2: PauliString) -> bool:
    """Check if two Pauli strings commute"""
    # Symplectic inner product
    count = 0
    for q in range(64):
        bit = 1 << q
        x1 = 1 if (p1.x_mask & bit) else 0
        z1 = 1 if (p1.z_mask & bit) else 0
        x2 = 1 if (p2.x_mask & bit) else 0
        z2 = 1 if (p2.z_mask & bit) else 0
        count += x1 * z2 + z1 * x2
    return (count % 2) == 0


# ============================================================================
# 2. Surrogate Circuit Nodes (Computation Graph)
# ============================================================================

class CircuitNode:
    """Base class for nodes in the surrogate computation graph"""
    def __init__(self):
        self.is_evaluated = False
        self.cumulative_value = 0.0


@dataclass
class EvalEndNode(CircuitNode):
    """
    Terminal node representing the starting Pauli string from the observable.
    This is where backward evaluation starts.
    """
    pstr: PauliString
    coefficient: float = 1.0
    
    def __post_init__(self):
        super().__init__()
    
    def __repr__(self):
        return f"EndNode({self.pstr}, coeff={self.coefficient})"


@dataclass
class PauliRotationNode(CircuitNode):
    """
    Node representing a Pauli rotation gate effect.
    - parents: list of parent nodes in the DAG
    - trig_inds: +1 for cos, -1 for sin
    - signs: ±1 for coefficient signs
    - param_idx: parameter index in the theta array
    """
    parents: List[Union[EvalEndNode, 'PauliRotationNode']] = field(default_factory=list)
    trig_inds: List[int] = field(default_factory=list)  # +1 for cos, -1 for sin
    signs: List[int] = field(default_factory=list)       # ±1
    param_idx: int = -1
    
    def __post_init__(self):
        super().__init__()
    
    def __repr__(self):
        return f"RotNode(θ[{self.param_idx}], {len(self.parents)} parents)"


@dataclass
class NodePathProperties:
    """
    Wrapper carrying a CircuitNode instead of a numerical coefficient.
    This is the "surrogate" coefficient type.
    """
    node: Union[EvalEndNode, PauliRotationNode]
    nsins: int = 0   # for truncation tracking
    ncos: int = 0
    freq: int = 0
    
    def __repr__(self):
        return f"Path({type(self.node).__name__}, freq={self.freq})"


# ============================================================================
# 3. Gates (Structure-only for Surrogate)
# ============================================================================

@dataclass
class Gate:
    """Base gate class"""
    pass


@dataclass
class CliffordGate(Gate):
    """
    Clifford gate (CNOT, H, S, etc.)
    - symbol: gate name
    - qubits: list of qubit indices
    """
    symbol: str
    qubits: List[int]
    
    def __repr__(self):
        return f"{self.symbol}{self.qubits}"


@dataclass
class PauliRotation(Gate):
    """
    Pauli rotation gate: exp(-i θ/2 P)
    - pauli: Pauli string generator
    - qubits: qubit indices
    - param_idx: parameter index for surrogate
    """
    pauli: str  # e.g., 'X', 'Y', 'Z', 'XX', 'YZ', etc.
    qubits: List[int]
    param_idx: int = -1
    
    def __repr__(self):
        return f"R{self.pauli}{self.qubits}(θ[{self.param_idx}])"
    
    def to_pauli_string(self) -> PauliString:
        """Convert gate generator to PauliString"""
        x_mask = 0
        z_mask = 0
        
        for q, p in zip(self.qubits, self.pauli):
            bit = 1 << q
            if p == 'X':
                x_mask |= bit
            elif p == 'Z':
                z_mask |= bit
            elif p == 'Y':
                x_mask |= bit
                z_mask |= bit
        
        return PauliString(x_mask, z_mask)


# ============================================================================
# 4. Clifford Gate Maps (Pre-computed lookup tables)
# ============================================================================

# Single-qubit Clifford maps: input Pauli -> (output Pauli, sign)
# Format: {gate: [(x,z,sign) for input I,X,Z,Y]}
CLIFFORD_MAPS = {
    'H': [  # Hadamard: H X H = Z, H Z H = X, H Y H = -Y
        ((0, 0), 1),   # I -> I
        ((0, 1), 1),   # X -> Z
        ((1, 0), 1),   # Z -> X
        ((1, 1), -1),  # Y -> -Y
    ],
    'S': [  # Phase gate: S X S† = Y, S Y S† = -X, S Z S† = Z
        ((0, 0), 1),   # I -> I
        ((1, 1), 1),   # X -> Y
        ((0, 1), 1),   # Z -> Z
        ((1, 0), -1),  # Y -> -X
    ],
    'SX': [  # √X gate
        ((0, 0), 1),   # I -> I
        ((1, 0), 1),   # X -> X
        ((1, 1), -1),  # Z -> -Y
        ((0, 1), 1),   # Y -> Z
    ],
    'CNOT': [  # CNOT gate (control, target) - Verified with Pennylane
        # Indexing: idx = control_pauli + 4 * target_pauli where I=0, X=1, Y=2, Z=3
        # Format: ((new_control_x, new_control_z), (new_target_x, new_target_z)), sign
        (((0, 0), (0, 0)), 1),   # 0x00: II -> II
        (((0, 0), (1, 0)), 1),   # 0x01: IX -> IX
        (((0, 1), (1, 1)), 1),   # 0x02: IY -> ZY
        (((0, 1), (0, 1)), 1),   # 0x03: IZ -> ZZ
        (((1, 0), (1, 0)), 1),   # 0x04: XI -> XX
        (((1, 0), (0, 0)), 1),   # 0x05: XX -> XI
        (((1, 1), (0, 1)), 1),   # 0x06: XY -> YZ
        (((1, 1), (1, 1)), -1),  # 0x07: XZ -> -YY
        (((1, 1), (1, 0)), 1),   # 0x08: YI -> YX
        (((1, 1), (0, 0)), 1),   # 0x09: YX -> YI
        (((1, 0), (0, 1)), -1),  # 0x0a: YY -> -XZ
        (((1, 0), (1, 1)), 1),   # 0x0b: YZ -> XY
        (((0, 1), (0, 0)), 1),   # 0x0c: ZI -> ZI
        (((0, 1), (1, 0)), 1),   # 0x0d: ZX -> ZX
        (((0, 0), (1, 1)), 1),   # 0x0e: ZY -> IY
        (((0, 0), (0, 1)), 1),   # 0x0f: ZZ -> IZ
    ],
}

# Precomputed single-qubit Clifford maps for vectorized updates
_SINGLE_Q_CLIFFORD_ARRAYS = {
    sym: (
        np.array([m[0][0] for m in CLIFFORD_MAPS[sym]], dtype=np.uint64),
        np.array([m[0][1] for m in CLIFFORD_MAPS[sym]], dtype=np.uint64),
        np.array([m[1] for m in CLIFFORD_MAPS[sym]], dtype=np.int8),
    )
    for sym in ("H", "S", "SX")
}

_CNOT_CLIFFORD_ARRAYS = (
    np.array([m[0][0][0] for m in CLIFFORD_MAPS["CNOT"]], dtype=np.uint64),
    np.array([m[0][0][1] for m in CLIFFORD_MAPS["CNOT"]], dtype=np.uint64),
    np.array([m[0][1][0] for m in CLIFFORD_MAPS["CNOT"]], dtype=np.uint64),
    np.array([m[0][1][1] for m in CLIFFORD_MAPS["CNOT"]], dtype=np.uint64),
    np.array([m[1] for m in CLIFFORD_MAPS["CNOT"]], dtype=np.int8),
)

def apply_clifford(gate: CliffordGate, pstr: PauliString) -> Tuple[PauliString, int]:
    """Apply a Clifford gate to a Pauli string using lookup table"""
    
    if gate.symbol == 'CNOT':
        # Two-qubit CNOT: use full lookup table
        assert len(gate.qubits) == 2
        control, target = gate.qubits
        
        # Extract local Paulis at control and target qubits
        xc, zc = get_local_pauli(pstr.x_mask, pstr.z_mask, control)
        xt, zt = get_local_pauli(pstr.x_mask, pstr.z_mask, target)
        
        # Lookup index: Pennylane convention is idx = target_pauli + 4 * control_pauli
        # where I=0, X=1, Y=2, Z=3
        # Convert (x,z) to pauli index: I=(0,0)→0, X=(1,0)→1, Y=(1,1)→2, Z=(0,1)→3
        def xz_to_pauli_idx(x, z):
            if x == 0 and z == 0:
                return 0  # I
            elif x == 1 and z == 0:
                return 1  # X
            elif x == 1 and z == 1:
                return 2  # Y
            else:  # x == 0 and z == 1
                return 3  # Z
        
        control_pauli = xz_to_pauli_idx(xc, zc)
        target_pauli = xz_to_pauli_idx(xt, zt)
        lookup_idx = target_pauli + 4 * control_pauli
        
        # Get new Paulis and sign from lookup table
        ((new_xc, new_zc), (new_xt, new_zt)), sign = CLIFFORD_MAPS['CNOT'][lookup_idx]
        
        # Build new Pauli string
        x_new = pstr.x_mask
        z_new = pstr.z_mask
        
        # Update control qubit
        bc = 1 << control
        x_new = (x_new & ~bc) | (new_xc << control)
        z_new = (z_new & ~bc) | (new_zc << control)
        
        # Update target qubit
        bt = 1 << target
        x_new = (x_new & ~bt) | (new_xt << target)
        z_new = (z_new & ~bt) | (new_zt << target)
        
        return PauliString(x_new, z_new), sign
    
    # Single-qubit Clifford
    assert len(gate.qubits) == 1
    q = gate.qubits[0]
    
    # Get local Pauli at qubit q
    xq, zq = get_local_pauli(pstr.x_mask, pstr.z_mask, q)
    lookup_idx = xq + 2 * zq  # I=0, X=1, Z=2, Y=3
    
    (new_xq, new_zq), sign = CLIFFORD_MAPS[gate.symbol][lookup_idx]
    
    # Update the Pauli string
    bit = 1 << q
    x_new = (pstr.x_mask & ~bit) | (new_xq << q)
    z_new = (pstr.z_mask & ~bit) | (new_zq << q)
    
    return PauliString(x_new, z_new), sign


def apply_single_qubit_clifford_batch(
    gate: CliffordGate,
    psum: "PauliSum",
    thetas: Optional[np.ndarray] = None,
    min_abs: Optional[float] = None,
) -> "PauliSum":
    """Apply a single-qubit Clifford to all terms in psum using vectorized bit ops."""
    new_psum = PauliSum(psum.n_qubits)

    if gate.symbol not in _SINGLE_Q_CLIFFORD_ARRAYS:
        # Fallback to scalar path if unknown
        for pstr, coeff in psum.terms.items():
            new_pstr, sign = apply_clifford(gate, pstr)
            if isinstance(coeff, NodePathProperties):
                new_node = multiply_sign(coeff.node, sign)
                new_coeff = NodePathProperties(new_node, coeff.nsins, coeff.ncos, coeff.freq)
            else:
                new_coeff = coeff * sign
            if min_abs is not None and thetas is not None and isinstance(new_coeff, NodePathProperties):
                reset_node(new_coeff.node)
                if abs(eval_node(new_coeff.node, thetas)) < min_abs:
                    continue
            new_psum.add(new_pstr, new_coeff)
        return new_psum

    q = gate.qubits[0]
    bit = np.uint64(1 << q)
    nx, nz, sgn = _SINGLE_Q_CLIFFORD_ARRAYS[gate.symbol]

    pstrs = list(psum.terms.keys())
    if not pstrs:
        return new_psum

    x_arr = np.fromiter((p.x_mask for p in pstrs), dtype=np.uint64, count=len(pstrs))
    z_arr = np.fromiter((p.z_mask for p in pstrs), dtype=np.uint64, count=len(pstrs))

    xq = (x_arr >> q) & 1
    zq = (z_arr >> q) & 1
    idx = xq + (zq << 1)

    new_x = (x_arr & ~bit) | (nx[idx] << q)
    new_z = (z_arr & ~bit) | (nz[idx] << q)
    sign_arr = sgn[idx]

    for i, pstr in enumerate(pstrs):
        new_pstr = PauliString(int(new_x[i]), int(new_z[i]))
        coeff = psum.terms[pstr]
        sign = int(sign_arr[i])

        if isinstance(coeff, NodePathProperties):
            new_node = multiply_sign(coeff.node, sign)
            new_coeff = NodePathProperties(new_node, coeff.nsins, coeff.ncos, coeff.freq)
        else:
            new_coeff = coeff * sign

        if min_abs is not None and thetas is not None and isinstance(new_coeff, NodePathProperties):
            reset_node(new_coeff.node)
            if abs(eval_node(new_coeff.node, thetas)) < min_abs:
                continue

        new_psum.add(new_pstr, new_coeff)

    return new_psum


def apply_single_qubit_clifford_batch_blocks(
    gate: CliffordGate,
    psum: "PauliSum",
    n_blocks: int,
    thetas: Optional[np.ndarray] = None,
    min_abs: Optional[float] = None,
) -> "PauliSum":
    """Apply a single-qubit Clifford using block masks (supports >64 qubits)."""
    new_psum = PauliSum(psum.n_qubits)

    if gate.symbol not in _SINGLE_Q_CLIFFORD_ARRAYS:
        for pstr, coeff in psum.terms.items():
            new_pstr, sign = apply_clifford(gate, pstr)
            if isinstance(coeff, NodePathProperties):
                new_node = multiply_sign(coeff.node, sign)
                new_coeff = NodePathProperties(new_node, coeff.nsins, coeff.ncos, coeff.freq)
            else:
                new_coeff = coeff * sign
            if min_abs is not None and thetas is not None and isinstance(new_coeff, NodePathProperties):
                reset_node(new_coeff.node)
                if abs(eval_node(new_coeff.node, thetas)) < min_abs:
                    continue
            new_psum.add(new_pstr, new_coeff)
        return new_psum

    q = gate.qubits[0]
    nx, nz, sgn = _SINGLE_Q_CLIFFORD_ARRAYS[gate.symbol]

    pstrs = list(psum.terms.keys())
    if not pstrs:
        return new_psum

    x_blocks, z_blocks = build_mask_blocks(pstrs, n_blocks)
    xq = get_bit_from_blocks(x_blocks, q)
    zq = get_bit_from_blocks(z_blocks, q)
    idx = xq + (zq << np.uint64(1))

    new_xq = nx[idx]
    new_zq = nz[idx]
    sign_arr = sgn[idx]

    set_bit_in_blocks(x_blocks, q, new_xq)
    set_bit_in_blocks(z_blocks, q, new_zq)

    for i, pstr in enumerate(pstrs):
        new_pstr = PauliString(blocks_to_int(x_blocks[i]), blocks_to_int(z_blocks[i]))
        coeff = psum.terms[pstr]
        sign = int(sign_arr[i])

        if isinstance(coeff, NodePathProperties):
            new_node = multiply_sign(coeff.node, sign)
            new_coeff = NodePathProperties(new_node, coeff.nsins, coeff.ncos, coeff.freq)
        else:
            new_coeff = coeff * sign

        if min_abs is not None and thetas is not None and isinstance(new_coeff, NodePathProperties):
            reset_node(new_coeff.node)
            if abs(eval_node(new_coeff.node, thetas)) < min_abs:
                continue

        new_psum.add(new_pstr, new_coeff)

    return new_psum


def apply_cnot_batch_blocks(
    gate: CliffordGate,
    psum: "PauliSum",
    n_blocks: int,
    thetas: Optional[np.ndarray] = None,
    min_abs: Optional[float] = None,
) -> "PauliSum":
    """Apply a CNOT gate using block masks (supports >64 qubits)."""
    new_psum = PauliSum(psum.n_qubits)
    assert gate.symbol == "CNOT" and len(gate.qubits) == 2

    control, target = gate.qubits
    nxc, nzc, nxt, nzt, sgn = _CNOT_CLIFFORD_ARRAYS

    pstrs = list(psum.terms.keys())
    if not pstrs:
        return new_psum

    x_blocks, z_blocks = build_mask_blocks(pstrs, n_blocks)
    xc = get_bit_from_blocks(x_blocks, control)
    zc = get_bit_from_blocks(z_blocks, control)
    xt = get_bit_from_blocks(x_blocks, target)
    zt = get_bit_from_blocks(z_blocks, target)

    # Pennylane CNOT table index order: I=0, X=1, Y=2, Z=3
    # Map (x,z) -> idx by swapping Y/Z relative to x + 2z
    c_idx = (xc + (zc << np.uint64(1))) ^ zc
    t_idx = (xt + (zt << np.uint64(1))) ^ zt
    lookup_idx = t_idx + (c_idx << np.uint64(2))

    new_xc = nxc[lookup_idx]
    new_zc = nzc[lookup_idx]
    new_xt = nxt[lookup_idx]
    new_zt = nzt[lookup_idx]
    sign_arr = sgn[lookup_idx]

    set_bit_in_blocks(x_blocks, control, new_xc)
    set_bit_in_blocks(z_blocks, control, new_zc)
    set_bit_in_blocks(x_blocks, target, new_xt)
    set_bit_in_blocks(z_blocks, target, new_zt)

    for i, pstr in enumerate(pstrs):
        new_pstr = PauliString(blocks_to_int(x_blocks[i]), blocks_to_int(z_blocks[i]))
        coeff = psum.terms[pstr]
        sign = int(sign_arr[i])

        if isinstance(coeff, NodePathProperties):
            new_node = multiply_sign(coeff.node, sign)
            new_coeff = NodePathProperties(new_node, coeff.nsins, coeff.ncos, coeff.freq)
        else:
            new_coeff = coeff * sign

        if min_abs is not None and thetas is not None and isinstance(new_coeff, NodePathProperties):
            reset_node(new_coeff.node)
            if abs(eval_node(new_coeff.node, thetas)) < min_abs:
                continue

        new_psum.add(new_pstr, new_coeff)

    return new_psum


# ============================================================================
# 5. Pauli Rotation Application (Creates Branching)
# ============================================================================

def apply_pauli_rotation_surrogate(
    gate: PauliRotation,
    pstr: PauliString,
    coeff: NodePathProperties,
) -> List[Tuple[PauliString, NodePathProperties]]:
    """
    Apply a Pauli rotation gate in Heisenberg picture.
    If gate commutes with pstr: returns [(pstr, coeff)]
    If not: returns [(pstr, cos_coeff), (new_pstr, sin_coeff)] where new_pstr = pstr * generator
    
    Formula: R_G(θ)[P] = cos(θ)P + sin(θ)P' where P' = i[G,P]/2
    
    Since [G,P] = GP - PG and for anticommuting Paulis: GP = -PG = phase * (result Pauli)
    We have: P' = i[G,P]/2 = i(GP - PG)/2 = i·GP - i·PG = i·GP - i·(-GP) = i·2·GP/2 = i·GP
    
    So P' = i · (phase_coeff · result_pauli) where phase_coeff ∈ {1, i, -1, -i}
    Multiplying by i: i·1=i, i·i=-1, i·(-1)=-i, i·(-i)=1
    This gives: phase 0→1, 1→2, 2→3, 3→0, so new_phase = (phase + 1) % 4
    
    But we want real coefficients only, so we extract the real part:
    - phase 0 (1):  i·1 = i     → imaginary, filtered
    - phase 1 (i):  i·i = -1    → real part: -1
    - phase 2 (-1): i·(-1) = -i → imaginary, filtered  
    - phase 3 (-i): i·(-i) = 1  → real part: +1
    
    Therefore: sign = 1 if phase==3 else (-1 if phase==1 else 0)
    Simplified: sign = -1 if phase==1 else (+1 if phase==3 else 0)
    Or using bit operations: sign = ((phase + 1) & 2) - 1
    """
    generator = gate.to_pauli_string()
    
    if commutes(pstr, generator):
        # No change
        return [(pstr, coeff)]
    
    # Gate does not commute -> split into cos and sin branches
    # Compute P' = P * G (Pauli product)
    new_pstr, phase = pauli_product(pstr, generator)
    
    # Compute sign from phase according to P' = i[G,P]/2 formula
    # Using P*G from pauli_product (Pennylane convention): P' = -i (P*G)
    # phase: 0=+1, 1=+i, 2=-1, 3=-i
    # -i * (+i) = +1  -> sign +1 when phase == 1
    # -i * (-i) = -1  -> sign -1 when phase == 3
    sign = 1 if phase == 1 else (-1 if phase == 3 else 0)
    
    # Create new nodes
    cos_node = PauliRotationNode(
        parents=[coeff.node],
        trig_inds=[+1],  # cos
        signs=[+1],
        param_idx=gate.param_idx,
    )
    
    sin_node = PauliRotationNode(
        parents=[coeff.node],
        trig_inds=[-1],  # sin
        signs=[sign],
        param_idx=gate.param_idx,
    )
    
    cos_coeff = NodePathProperties(cos_node, coeff.nsins, coeff.ncos + 1, coeff.freq + 1)
    sin_coeff = NodePathProperties(sin_node, coeff.nsins + 1, coeff.ncos, coeff.freq + 1)
    
    return [(pstr, cos_coeff), (new_pstr, sin_coeff)]


# ============================================================================
# 6. Propagate Tree Construction (Phase 1)
# ============================================================================

@dataclass
class PauliSum:
    """
    Collection of Pauli strings with coefficients (or NodePathProperties for surrogate)
    terms: Dict[PauliString, coefficient]
    """
    n_qubits: int
    terms: Dict[PauliString, Union[float, NodePathProperties]] = field(default_factory=dict)
    
    def add(self, pstr: PauliString, coeff: Union[float, NodePathProperties]):
        """Add or merge a term"""
        if pstr in self.terms:
            # Merge nodes if both are NodePathProperties
            if isinstance(coeff, NodePathProperties) and isinstance(self.terms[pstr], NodePathProperties):
                existing = self.terms[pstr]
                
                # Both are PauliRotationNodes - merge parent lists
                if isinstance(coeff.node, PauliRotationNode) and isinstance(existing.node, PauliRotationNode):
                    existing.node.parents.extend(coeff.node.parents)
                    existing.node.trig_inds.extend(coeff.node.trig_inds)
                    existing.node.signs.extend(coeff.node.signs)
                    # Update tracking
                    existing.nsins = min(existing.nsins, coeff.nsins)
                    existing.ncos = min(existing.ncos, coeff.ncos)
                    existing.freq = min(existing.freq, coeff.freq)
                
                # Both are EvalEndNodes - sum coefficients
                elif isinstance(coeff.node, EvalEndNode) and isinstance(existing.node, EvalEndNode):
                    existing.node.coefficient += coeff.node.coefficient
                
                # Mixed types - create a new RotationNode with both as parents
                else:
                    # This shouldn't happen in normal operation but handle it
                    if isinstance(coeff.node, PauliRotationNode):
                        self.terms[pstr] = coeff
            else:
                # Numerical coefficients
                self.terms[pstr] = self.terms[pstr] + coeff
        else:
            self.terms[pstr] = coeff

    def add_from_str(self, pauli: str, coeff: Union[float, NodePathProperties], qubits: Optional[List[int]] = None):
        """Add a term using a Pauli string like 'XYIZ'."""
        self.add(make_pauli_string(pauli, qubits=qubits), coeff)
    
    def __len__(self):
        return len(self.terms)
    
    def __repr__(self):
        return f"PauliSum({len(self.terms)} terms)"


def propagate_surrogate(
    circuit: List[Gate],
    observable: PauliSum,
    max_weight: int = 1000,
    max_freq: int = 1000,
    max_nsins: int = 1000,
    max_xy: int = 1000,
    use_commuting_blocks: bool = False,
    thetas: Optional[np.ndarray] = None,
    min_abs: Optional[float] = None,
) -> PauliSum:
    """
    Phase 1: Build propagation tree structure (without parameter values)
    
    Args:
        circuit: List of gates (applied in Schrödinger picture order)
        observable: Starting observable as PauliSum
        max_weight: Maximum Pauli weight for truncation
        max_freq: Maximum frequency (number of parameters) for truncation
        max_nsins: Maximum number of sine factors for truncation
        max_xy: Maximum number of X/Y terms for truncation
        use_commuting_blocks: Split circuit into commuting blocks before propagation
        thetas: Parameter values for min_abs truncation (optional)
        min_abs: Truncate paths with |value| < min_abs when thetas is provided
    
    Returns:
        Surrogate PauliSum with NodePathProperties as coefficients
    """
    # Wrap observable coefficients into NodePathProperties
    psum = PauliSum(observable.n_qubits)
    for pstr, coeff in observable.terms.items():
        end_node = EvalEndNode(pstr, coefficient=float(coeff))
        node_coeff = NodePathProperties(end_node)
        psum.add(pstr, node_coeff)
    
    # Propagate backward through circuit (Heisenberg picture)
    if use_commuting_blocks:
        depths = build_commuting_depths(circuit)
        for depth in reversed(depths):
            print(f"Processing commuting depth with {len(depth)} gates")
            for gate in reversed(depth):
                psum = propagate_gate(gate, psum, max_weight, max_freq, max_nsins, max_xy, thetas, min_abs)
                print(f"After {gate}: {len(psum)} terms")
    else:
        for gate in reversed(circuit):
            psum = propagate_gate(gate, psum, max_weight, max_freq, max_nsins, max_xy, thetas, min_abs)
            print(f"After {gate}: {len(psum)} terms")
    
    return psum


def propagate_gate(
    gate: Gate,
    psum: PauliSum,
    max_weight: int,
    max_freq: int,
    max_nsins: int,
    max_xy: int,
    thetas: Optional[np.ndarray] = None,
    min_abs: Optional[float] = None,
) -> PauliSum:
    """Apply one gate to all Pauli strings in psum"""
    if not psum.terms:
        return PauliSum(psum.n_qubits)

    n_blocks = (psum.n_qubits + 63) // 64

    if isinstance(gate, CliffordGate):
        if gate.symbol in _SINGLE_Q_CLIFFORD_ARRAYS and len(gate.qubits) == 1:
            if n_blocks == 1:
                return apply_single_qubit_clifford_batch(gate, psum, thetas=thetas, min_abs=min_abs)
            return apply_single_qubit_clifford_batch_blocks(gate, psum, n_blocks, thetas=thetas, min_abs=min_abs)

        if gate.symbol == "CNOT" and len(gate.qubits) == 2:
            return apply_cnot_batch_blocks(gate, psum, n_blocks, thetas=thetas, min_abs=min_abs)

        # Fallback scalar path for unknown Clifford
        new_psum = PauliSum(psum.n_qubits)
        for pstr, coeff in psum.terms.items():
            new_pstr, sign = apply_clifford(gate, pstr)
            if isinstance(coeff, NodePathProperties):
                new_node = multiply_sign(coeff.node, sign)
                new_coeff = NodePathProperties(new_node, coeff.nsins, coeff.ncos, coeff.freq)
            else:
                new_coeff = coeff * sign
            if min_abs is not None and thetas is not None and isinstance(new_coeff, NodePathProperties):
                reset_node(new_coeff.node)
                if abs(eval_node(new_coeff.node, thetas)) < min_abs:
                    continue
            new_psum.add(new_pstr, new_coeff)
        return new_psum

    if isinstance(gate, PauliRotation):
        # Pauli rotation: retain (commute), mutate (cos), insert (sin)
        generator = gate.to_pauli_string()
        pstrs = list(psum.terms.keys())
        coeffs = [psum.terms[p] for p in pstrs]

        retain = PauliSum(psum.n_qubits)
        aux = PauliSum(psum.n_qubits)

        if n_blocks == 1:
            x_arr = np.fromiter((p.x_mask for p in pstrs), dtype=np.uint64, count=len(pstrs))
            z_arr = np.fromiter((p.z_mask for p in pstrs), dtype=np.uint64, count=len(pstrs))

            gx = np.uint64(generator.x_mask)
            gz = np.uint64(generator.z_mask)
            symp = popcount_u64((x_arr & gz) ^ (z_arr & gx)) & 1
            comm_idx = np.where(symp == 0)[0]
            anti_idx = np.where(symp == 1)[0]

            new_x_arr = x_arr ^ gx
            new_z_arr = z_arr ^ gz
        else:
            x_blocks, z_blocks = build_mask_blocks(pstrs, n_blocks)
            gx_blocks = np.zeros((1, n_blocks), dtype=np.uint64)
            gz_blocks = np.zeros((1, n_blocks), dtype=np.uint64)
            mask64 = (1 << 64) - 1
            for b in range(n_blocks):
                shift = 64 * b
                gx_blocks[0, b] = (generator.x_mask >> shift) & mask64
                gz_blocks[0, b] = (generator.z_mask >> shift) & mask64

            symp_blocks = popcount_u64((x_blocks & gz_blocks) ^ (z_blocks & gx_blocks))
            symp = (np.sum(symp_blocks, axis=1, dtype=np.uint64) & np.uint64(1))
            comm_idx = np.where(symp == 0)[0]
            anti_idx = np.where(symp == 1)[0]

            new_x_blocks = x_blocks ^ gx_blocks
            new_z_blocks = z_blocks ^ gz_blocks

        # Commute: retain unchanged (no cos)
        for i in comm_idx:
            pstr = pstrs[i]
            c = coeffs[i]

            if count_weight(pstr.x_mask, pstr.z_mask) > max_weight:
                continue
            if c.freq > max_freq:
                continue
            if c.nsins > max_nsins:
                continue
            if count_xy(pstr.x_mask, pstr.z_mask) > max_xy:
                continue
            if min_abs is not None and thetas is not None:
                reset_node(c.node)
                if abs(eval_node(c.node, thetas)) < min_abs:
                    continue

            retain.add(pstr, c)

        # Anticommute: mutate (cos) + insert (sin)
        for i in anti_idx:
            pstr = pstrs[i]
            c = coeffs[i]

            # mutate (cos * original)
            cos_node = PauliRotationNode(
                parents=[c.node],
                trig_inds=[+1],
                signs=[+1],
                param_idx=gate.param_idx,
            )
            cos_coeff = NodePathProperties(cos_node, c.nsins, c.ncos + 1, c.freq + 1)

            if count_weight(pstr.x_mask, pstr.z_mask) <= max_weight and \
               cos_coeff.freq <= max_freq and \
               cos_coeff.nsins <= max_nsins and \
               count_xy(pstr.x_mask, pstr.z_mask) <= max_xy:
                if min_abs is not None and thetas is not None:
                    reset_node(cos_coeff.node)
                    if abs(eval_node(cos_coeff.node, thetas)) >= min_abs:
                        retain.add(pstr, cos_coeff)
                else:
                    retain.add(pstr, cos_coeff)

            # insert (sin * new_pstr)
            if n_blocks == 1:
                new_pstr = PauliString(int(new_x_arr[i]), int(new_z_arr[i]))
            else:
                new_pstr = PauliString(blocks_to_int(new_x_blocks[i]), blocks_to_int(new_z_blocks[i]))

            phase = pauli_product_phase_on_support(pstr, gate)
            sign = 1 if phase == 1 else (-1 if phase == 3 else 0)

            sin_node = PauliRotationNode(
                parents=[c.node],
                trig_inds=[-1],
                signs=[sign],
                param_idx=gate.param_idx,
            )
            sin_coeff = NodePathProperties(sin_node, c.nsins + 1, c.ncos, c.freq + 1)

            if count_weight(new_pstr.x_mask, new_pstr.z_mask) <= max_weight and \
               sin_coeff.freq <= max_freq and \
               sin_coeff.nsins <= max_nsins and \
               count_xy(new_pstr.x_mask, new_pstr.z_mask) <= max_xy:
                if min_abs is not None and thetas is not None:
                    reset_node(sin_coeff.node)
                    if abs(eval_node(sin_coeff.node, thetas)) >= min_abs:
                        aux.add(new_pstr, sin_coeff)
                else:
                    aux.add(new_pstr, sin_coeff)

        for pstr, c in aux.terms.items():
            retain.add(pstr, c)

        return retain

    return PauliSum(psum.n_qubits)


def multiply_sign(node: Union[EvalEndNode, PauliRotationNode], sign: int) -> Union[EvalEndNode, PauliRotationNode]:
    """Multiply a sign into a circuit node (creates a new node to avoid mutation)"""
    if isinstance(node, EvalEndNode):
        # Create new node with multiplied coefficient
        new_node = EvalEndNode(node.pstr, node.coefficient * sign)
        return new_node
    elif isinstance(node, PauliRotationNode):
        # Create new node with multiplied signs
        new_node = PauliRotationNode(
            parents=node.parents.copy(),
            trig_inds=node.trig_inds.copy(),
            signs=[s * sign for s in node.signs],
            param_idx=node.param_idx
        )
        return new_node
    return node


# ============================================================================
# 7. Zero Filtering (Phase 2)
# ============================================================================

def zero_filter(psum: PauliSum) -> PauliSum:
    """
    Filter surrogate to keep only paths contributing to <0|...|0> expectation.
    This means keeping only Pauli strings with x_mask == 0 (no X or Y components).
    
    This is the key optimization: we prune the computation graph!
    """
    filtered = PauliSum(psum.n_qubits)
    
    for pstr, coeff in psum.terms.items():
        # Only keep terms where x_mask = 0 (diagonal in computational basis)
        if pstr.x_mask == 0:
            filtered.add(pstr, coeff)
    
    print(f"Zero filtering: {len(psum)} -> {len(filtered)} terms")
    return filtered


# ============================================================================
# 8. Phase 3: Evaluation with Parameter Values
# ============================================================================

def reset_node(node: CircuitNode):
    """Reset evaluation flags for a new evaluation"""
    if not node.is_evaluated:
        return
    
    node.is_evaluated = False
    node.cumulative_value = 0.0
    
    if isinstance(node, PauliRotationNode):
        for parent in node.parents:
            reset_node(parent)


def eval_node(node: CircuitNode, thetas: np.ndarray) -> float:
    """Recursively evaluate a circuit node"""
    if node.is_evaluated:
        return node.cumulative_value
    
    if isinstance(node, EvalEndNode):
        # Terminal node: just return the coefficient
        node.cumulative_value = node.coefficient
        node.is_evaluated = True
        return node.cumulative_value
    
    elif isinstance(node, PauliRotationNode):
        # Rotation node: sum over all parent branches
        value = 0.0
        theta = thetas[node.param_idx]
        
        for parent, trig_ind, sign in zip(node.parents, node.trig_inds, node.signs):
            parent_val = eval_node(parent, thetas)
            
            # trig_ind: +1 for cos, -1 for sin
            if trig_ind > 0:
                trig_val = np.cos(theta)
            else:
                trig_val = np.sin(theta)
            
            value += sign * trig_val * parent_val
        
        node.cumulative_value = value
        node.is_evaluated = True
        return value
    
    return 0.0


def compute_z_eigenvalue(z_mask: int, n_qubits: int) -> float:
    """
    Compute eigenvalue of Z Pauli string in |0⟩ state.
    
    For |0...0⟩ initial state:
    - Z_i |0⟩ = +1 |0⟩  (eigenvalue +1)
    - I_i |0⟩ = +1 |0⟩  (eigenvalue +1)
    
    So all Z and I terms give +1.
    """
    return 1.0


def evaluate_surrogate(surrogate: PauliSum, thetas: np.ndarray) -> float:
    """
    Phase 3: Evaluate surrogate with parameter values for <0|O|0>
    
    Key insight: Initial state |0⟩ means:
    - <0|X|0> = 0  (x_mask != 0 terms filtered out)
    - <0|Y|0> = 0  (x_mask != 0 terms filtered out)
    - <0|Z|0> = +1 (only these contribute!)
    - <0|I|0> = +1
    
    Args:
        surrogate: Filtered surrogate (only x_mask == 0 terms)
        thetas: Parameter values
    
    Returns:
        Expectation value <0|U†OU|0>
    """
    # Reset all nodes
    for pstr, coeff in surrogate.terms.items():
        if isinstance(coeff, NodePathProperties):
            reset_node(coeff.node)
    
    # Evaluate and sum
    result = 0.0
    
    for pstr, coeff in surrogate.terms.items():
        assert pstr.x_mask == 0, "Should be filtered to x_mask == 0"
        
        if isinstance(coeff, NodePathProperties):
            # Evaluate the computation graph
            path_value = eval_node(coeff.node, thetas)
            
            # Z eigenvalue for |0⟩ state (always +1)
            z_eigenvalue = compute_z_eigenvalue(pstr.z_mask, surrogate.n_qubits)
            
            result += z_eigenvalue * path_value
        else:
            # Numerical coefficient (shouldn't happen for surrogate)
            z_eigenvalue = compute_z_eigenvalue(pstr.z_mask, surrogate.n_qubits)
            result += z_eigenvalue * coeff
    
    return result
