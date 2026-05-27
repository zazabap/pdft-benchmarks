"""Progressive gate-unfreezing training loop, plateau detector,
Riemannian grad-norm probe, and QFT unfreeze-order helper."""
from __future__ import annotations


def qft_unfreeze_orders(m: int, n: int) -> dict[str, list[int]]:
    """Three unfreeze orderings as lists of indices into a QFTBasis(m, n).tensors.

    `basis.tensors` is stored Hadamard-first (stable within group), matching
    `qft_identity_basis`. We rebuild the emission gate list, map each emitted
    gate to its storage index, then order the storage indices three ways:

      - lr: emission order
      - rl: reverse emission order
      - bg: block-growth — group by block-stage k (highest within-axis qubit a
        gate touches); per stage do row-axis then col-axis, H_k before its CPs,
        CPs by ascending lower qubit. Stage k completes QFT(k, k).
    """
    from pdft.bases.circuit.qft import _qft_gates_1d

    gates = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    G = len(gates)

    # emission index -> storage index (Hadamard-first stable sort, == qft_identity_basis)
    emit_perm = sorted(range(G), key=lambda i: gates[i]["kind"] != "H")
    emission_to_storage = [0] * G
    for storage_pos, emission_idx in enumerate(emit_perm):
        emission_to_storage[emission_idx] = storage_pos

    def axis_of(q: int) -> int:
        return 0 if q <= m else 1

    def within(q: int) -> int:
        return q if q <= m else q - m

    keys = []  # (stage, axis, kind_rank, lower_qubit, emission_idx)
    for e, g in enumerate(gates):
        if g["kind"] == "H":
            (q,) = g["qubits"]
            keys.append((within(q), axis_of(q), 0, within(q), e))
        else:  # CP, qubits = (control, target), same axis
            c, t = g["qubits"]
            stage = max(within(c), within(t))
            keys.append((stage, axis_of(c), 1, min(within(c), within(t)), e))

    lr = [emission_to_storage[e] for e in range(G)]
    rl = list(reversed(lr))
    bg = [emission_to_storage[k[4]] for k in sorted(keys)]
    return {"bg": bg, "lr": lr, "rl": rl}
