"""Serialized SIMD-packed CKKS multi-dot products with strict role separation.

The server scores plaintext candidate embeddings against an encrypted query by
two comparable kernels:

``naive_per_candidate``
    One ciphertext-plaintext multiply, segmented reduction, and returned
    ciphertext per candidate.

``block_packed``
    Several candidates share one CKKS ciphertext.  If the embedding dimension
    is ``d``, a segment is ``P = next_power_of_two(d)`` slots and one ciphertext
    holds ``floor(slot_count / P)`` candidates.  The query is repeated in these
    segments.  After a ciphertext-plaintext multiply, rotations by
    ``1, 2, ..., P/2`` and additions make slots ``0, P, 2P, ...`` contain the
    dot products.  No masks are needed because only segment-start slots are
    read.  For d=192 and N=8192 this packs 16 scores per ciphertext, hence a
    40-candidate response uses three ciphertexts instead of forty.

``segmented_score_packed``
    The group ciphertexts are sparsely masked at their score slots, shifted by
    distinct group offsets using the same powers-of-two rotations, and added.
    Up to one slot-count of shortlist scores is returned in a single response
    ciphertext.  This adds a small packing cost after the group reductions.

Exact TenSEAL 0.3.16 API constraint
-----------------------------------
``tenseal.CKKSVector`` exposes no public ``rotate``/``rotate_`` operation.
Its public ``mm`` method can return packed scores, but local measurement shows
substantial diagonal-matmul overhead.  This module therefore uses the exposed
``tenseal.sealapi`` classes (not private ``_cpp`` bindings), specifically
``sealapi.Evaluator.rotate_vector``.  Only the powers-of-two Galois keys needed
by the configured maximum embedding dimension are generated.

The client/server boundary is real in memory as well as in the API.  The
server context is reconstructed from a byte envelope containing encryption
parameters, a public key, and limited Galois keys; the secret key is never
serialized into that envelope.  Queries and responses also cross the boundary
as serialized byte strings.  Only ``ClientCKKSContext`` owns a Decryptor.

Validated with Python 3.11 and TenSEAL 0.3.16.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tenseal as ts
from tenseal import sealapi as sa
from tenseal.sealapi import util as seal_util


TENSEAL_VALIDATED_VERSION = "0.3.16"
MICROBENCHMARK_SCHEMA = "packed_ckks_microbenchmark.v1"
SERVER_CONTEXT_SCHEMA = "packed_ckks_server_context.v1"
QUERY_SCHEMA = "packed_ckks_query.v1"
RESPONSE_SCHEMA = "packed_ckks_response.v1"
FINAL_MASK_SCALE_BITS = 19

PUBLIC_API_NOTE = (
    "TenSEAL 0.3.16 CKKSVector has no public rotate/rotate_ method. "
    "The segmented kernel uses the distributed tenseal.sealapi API "
    "(Evaluator.rotate_vector), not private _cpp bindings. Built-in "
    "CKKSVector.mm is a functional packed-score fallback but was slower in "
    "local measurement."
)

_CONTEXT_MAGIC = b"PCKCTX1\x00"
_QUERY_MAGIC = b"PCKQRY1\x00"
_RESPONSE_MAGIC = b"PCKRSP1\x00"
_UINT32 = struct.Struct("!I")


@dataclass(frozen=True)
class CKKSParameters:
    """Parameters for one ciphertext-plaintext multiplication and rescale."""

    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 60)
    scale_bits: int = 40

    def __post_init__(self) -> None:
        if self.poly_modulus_degree < 4096:
            raise ValueError("poly_modulus_degree must be at least 4096 for CKKS")
        if self.poly_modulus_degree & (self.poly_modulus_degree - 1):
            raise ValueError("poly_modulus_degree must be a power of two")
        if len(self.coeff_mod_bit_sizes) < 3:
            raise ValueError("at least three coefficient moduli are required")
        if self.scale_bits <= 0:
            raise ValueError("scale_bits must be positive")

    @property
    def slot_count(self) -> int:
        return self.poly_modulus_degree // 2

    @property
    def scale(self) -> float:
        return float(2**self.scale_bits)

    def as_json(self) -> dict[str, Any]:
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(self.coeff_mod_bit_sizes),
            "scale_bits": self.scale_bits,
            "global_scale": self.scale,
            "slot_count": self.slot_count,
        }


@dataclass
class ClientCKKSContext:
    """Private client-only SEAL objects."""

    parameters: CKKSParameters
    seal_context: Any
    encoder: Any
    encryptor: Any
    decryptor: Any
    secret_key: Any
    public_key: Any
    galois_keys: Any
    max_padded_dimension: int
    rotation_steps: tuple[int, ...]

    def has_secret_key(self) -> bool:
        return True

    def has_public_key(self) -> bool:
        return True

    def has_galois_keys(self) -> bool:
        return self.galois_keys.size() > 0

    def is_public(self) -> bool:
        return False


@dataclass
class ServerCKKSContext:
    """Public server-only SEAL objects; deliberately has no Decryptor field."""

    parameters: CKKSParameters
    seal_context: Any
    encoder: Any
    evaluator: Any
    public_key: Any
    galois_keys: Any
    max_padded_dimension: int
    rotation_steps: tuple[int, ...]

    def has_secret_key(self) -> bool:
        return False

    def has_public_key(self) -> bool:
        return True

    def has_galois_keys(self) -> bool:
        return self.galois_keys.size() > 0

    def is_public(self) -> bool:
        return True


@dataclass(frozen=True)
class ContextPair:
    """Private client plus independently deserialized public server."""

    client: ClientCKKSContext
    server: ServerCKKSContext
    serialized_server_context: bytes


@dataclass(frozen=True)
class EncryptedScoreResponse:
    """Serialized ciphertexts and locations of their meaningful score slots."""

    method: str
    payloads: tuple[bytes, ...]
    score_counts: tuple[int, ...]
    score_indices: tuple[tuple[int, ...], ...]
    seal_ciphertext_counts: tuple[int, ...]
    server_phase_ms: Mapping[str, float] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.method not in {
            "naive_per_candidate",
            "block_packed",
            "segmented_score_packed",
        }:
            raise ValueError(f"unknown response method: {self.method!r}")
        if not self.payloads:
            raise ValueError("a response must contain at least one payload")
        lengths = {
            len(self.payloads),
            len(self.score_counts),
            len(self.score_indices),
            len(self.seal_ciphertext_counts),
        }
        if len(lengths) != 1:
            raise ValueError("payload and metadata arrays must have equal lengths")
        if any(not isinstance(payload, bytes) or not payload for payload in self.payloads):
            raise ValueError("each payload must be a non-empty byte string")
        if any(count <= 0 for count in self.score_counts):
            raise ValueError("score counts must be positive")
        if any(len(indices) != count for indices, count in zip(self.score_indices, self.score_counts)):
            raise ValueError("score index counts must match score counts")
        if any(index < 0 for indices in self.score_indices for index in indices):
            raise ValueError("score indices cannot be negative")
        if any(count <= 0 for count in self.seal_ciphertext_counts):
            raise ValueError("SEAL ciphertext counts must be positive")

    @property
    def score_count(self) -> int:
        return sum(self.score_counts)

    @property
    def payload_bytes(self) -> int:
        return sum(len(payload) for payload in self.payloads)

    @property
    def seal_ciphertext_count(self) -> int:
        return sum(self.seal_ciphertext_counts)

    def to_bytes(self) -> bytes:
        header = {
            "schema": RESPONSE_SCHEMA,
            "method": self.method,
            "score_counts": list(self.score_counts),
            "score_indices": [list(indices) for indices in self.score_indices],
            "seal_ciphertext_counts": list(self.seal_ciphertext_counts),
        }
        return _encode_frame(_RESPONSE_MAGIC, header, self.payloads)

    @classmethod
    def from_bytes(cls, wire: bytes) -> "EncryptedScoreResponse":
        header, payloads = _decode_frame(_RESPONSE_MAGIC, wire)
        if header.get("schema") != RESPONSE_SCHEMA:
            raise ValueError(f"unsupported response schema: {header.get('schema')!r}")
        try:
            return cls(
                method=str(header["method"]),
                payloads=payloads,
                score_counts=tuple(int(value) for value in header["score_counts"]),
                score_indices=tuple(
                    tuple(int(value) for value in indices)
                    for indices in header["score_indices"]
                ),
                seal_ciphertext_counts=tuple(
                    int(value) for value in header["seal_ciphertext_counts"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("incomplete CKKS response metadata") from exc


def create_context_pair(
    parameters: CKKSParameters | None = None,
    *,
    max_query_dimension: int | None = None,
) -> ContextPair:
    """Create private client keys and deserialize a public-only server context.

    ``max_query_dimension`` controls the limited Galois-key set.  Benchmark
    callers should set it to their embedding dimension.  Omitting it creates
    powers-of-two rotation keys for every dimension up to the slot count.
    """

    parameters = parameters or CKKSParameters()
    if max_query_dimension is None:
        max_query_dimension = parameters.slot_count
    if not isinstance(max_query_dimension, int) or isinstance(max_query_dimension, bool):
        raise ValueError("max_query_dimension must be an integer")
    if not 1 <= max_query_dimension <= parameters.slot_count:
        raise ValueError("max_query_dimension must be between 1 and the slot count")
    max_padded_dimension = _next_power_of_two(max_query_dimension)
    rotation_steps = _reduction_steps(max_padded_dimension)
    # A single key keeps context role checks consistent for the d=1 corner
    # case; the evaluator does not use it when no reduction is necessary.
    key_steps = rotation_steps or (1,)

    encryption_parameters = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS)
    encryption_parameters.set_poly_modulus_degree(parameters.poly_modulus_degree)
    encryption_parameters.set_coeff_modulus(
        sa.CoeffModulus.Create(
            parameters.poly_modulus_degree, list(parameters.coeff_mod_bit_sizes)
        )
    )
    client_seal_context = sa.SEALContext(
        encryption_parameters, True, sa.SEC_LEVEL_TYPE.TC128
    )
    if not client_seal_context.parameters_set():
        raise ValueError(client_seal_context.parameters_error_message())

    key_generator = sa.KeyGenerator(client_seal_context)
    secret_key = key_generator.secret_key()
    public_key = sa.PublicKey()
    key_generator.create_public_key(public_key)
    galois_keys = sa.GaloisKeys()
    tool = seal_util.GaloisTool(int(math.log2(parameters.poly_modulus_degree)))
    galois_elements = tool.get_elts_from_steps(list(key_steps))
    key_generator.create_galois_keys(galois_elements, galois_keys)

    client = ClientCKKSContext(
        parameters=parameters,
        seal_context=client_seal_context,
        encoder=sa.CKKSEncoder(client_seal_context),
        encryptor=sa.Encryptor(client_seal_context, public_key),
        decryptor=sa.Decryptor(client_seal_context, secret_key),
        secret_key=secret_key,
        public_key=public_key,
        galois_keys=galois_keys,
        max_padded_dimension=max_padded_dimension,
        rotation_steps=rotation_steps,
    )
    serialized_server_context = _serialize_server_context(
        parameters,
        encryption_parameters,
        public_key,
        galois_keys,
        max_padded_dimension=max_padded_dimension,
        rotation_steps=rotation_steps,
    )
    server = deserialize_server_context(serialized_server_context)
    _require_public_server_context(server)
    return ContextPair(client, server, serialized_server_context)


def deserialize_server_context(serialized: bytes) -> ServerCKKSContext:
    """Reconstruct the server solely from public serialized material."""

    header, payloads = _decode_frame(_CONTEXT_MAGIC, serialized)
    if header.get("schema") != SERVER_CONTEXT_SCHEMA:
        raise ValueError(f"unsupported server context schema: {header.get('schema')!r}")
    if header.get("contains_secret_key") is not False:
        raise ValueError("refusing a server context envelope that may contain a secret key")
    if len(payloads) != 3:
        raise ValueError("server context must contain parameters, public key, Galois keys")
    try:
        parameters = CKKSParameters(
            poly_modulus_degree=int(header["ckks"]["poly_modulus_degree"]),
            coeff_mod_bit_sizes=tuple(
                int(value) for value in header["ckks"]["coeff_mod_bit_sizes"]
            ),
            scale_bits=int(header["ckks"]["scale_bits"]),
        )
        max_padded_dimension = int(header["max_padded_dimension"])
        rotation_steps = tuple(int(value) for value in header["rotation_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid server context metadata") from exc

    encryption_parameters = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS)
    _seal_load(encryption_parameters, payloads[0])
    seal_context = sa.SEALContext(
        encryption_parameters, True, sa.SEC_LEVEL_TYPE.TC128
    )
    if not seal_context.parameters_set():
        raise ValueError(seal_context.parameters_error_message())
    public_key = sa.PublicKey()
    _seal_load(public_key, payloads[1], context=seal_context)
    galois_keys = sa.GaloisKeys()
    _seal_load(galois_keys, payloads[2], context=seal_context)
    return ServerCKKSContext(
        parameters=parameters,
        seal_context=seal_context,
        encoder=sa.CKKSEncoder(seal_context),
        evaluator=sa.Evaluator(seal_context),
        public_key=public_key,
        galois_keys=galois_keys,
        max_padded_dimension=max_padded_dimension,
        rotation_steps=rotation_steps,
    )


def encrypt_query(
    client_context: ClientCKKSContext,
    query: Sequence[float] | np.ndarray,
) -> bytes:
    """Repeat, encrypt, and serialize a padded query on the private client."""

    _require_private_client_context(client_context)
    query_array = _as_float_vector(query, name="query")
    padded_dimension = _next_power_of_two(query_array.size)
    if padded_dimension > client_context.max_padded_dimension:
        raise ValueError(
            f"padded query dimension {padded_dimension} exceeds this context's "
            f"configured maximum {client_context.max_padded_dimension}"
        )
    if padded_dimension > client_context.parameters.slot_count:
        raise ValueError("query does not fit in one CKKS ciphertext")

    per_ciphertext = client_context.parameters.slot_count // padded_dimension
    padded_query = np.zeros(padded_dimension, dtype=np.float64)
    padded_query[: query_array.size] = query_array
    query_slots = np.tile(padded_query, per_ciphertext)
    plaintext = sa.Plaintext()
    client_context.encoder.encode(
        query_slots.tolist(), client_context.parameters.scale, plaintext
    )
    ciphertext = sa.Ciphertext()
    client_context.encryptor.encrypt(plaintext, ciphertext)
    header = {
        "schema": QUERY_SCHEMA,
        "dimension": int(query_array.size),
        "padded_dimension": padded_dimension,
        "slot_count": client_context.parameters.slot_count,
        "segments": per_ciphertext,
    }
    return _encode_frame(_QUERY_MAGIC, header, (_seal_save(ciphertext),))


def server_naive_scores(
    server_context: ServerCKKSContext,
    serialized_query: bytes,
    candidate_embeddings: Sequence[Sequence[float]] | np.ndarray,
) -> EncryptedScoreResponse:
    """Evaluate and return one encrypted score ciphertext per candidate."""

    return _server_scores(
        server_context,
        serialized_query,
        candidate_embeddings,
        method="naive_per_candidate",
        requested_block_size=1,
    )


def server_block_packed_scores(
    server_context: ServerCKKSContext,
    serialized_query: bytes,
    candidate_embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    block_size: int | None = None,
) -> EncryptedScoreResponse:
    """Evaluate segmented SIMD blocks and return packed encrypted scores."""

    return _server_scores(
        server_context,
        serialized_query,
        candidate_embeddings,
        method="block_packed",
        requested_block_size=block_size,
    )


def server_segmented_score_packed(
    server_context: ServerCKKSContext,
    serialized_query: bytes,
    candidate_embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    block_size: int | None = None,
) -> EncryptedScoreResponse:
    """Return the whole shortlist in one masked-and-shifted ciphertext.

    The method supports at most ``slot_count`` candidates, which is the
    information-theoretic capacity of one CKKS ciphertext.  ``block_size`` can
    be used for an ablation; its default is the natural segmented capacity.
    """

    return _server_scores(
        server_context,
        serialized_query,
        candidate_embeddings,
        method="segmented_score_packed",
        requested_block_size=block_size,
    )


def client_decrypt_scores(
    client_context: ClientCKKSContext,
    response: EncryptedScoreResponse | bytes,
) -> np.ndarray:
    """Load and decrypt score slots; only a private client context is accepted."""

    _require_private_client_context(client_context)
    if isinstance(response, bytes):
        response = EncryptedScoreResponse.from_bytes(response)
    if not isinstance(response, EncryptedScoreResponse):
        raise TypeError("response must be EncryptedScoreResponse or bytes")

    blocks: list[np.ndarray] = []
    for payload, count, indices_tuple in zip(
        response.payloads, response.score_counts, response.score_indices
    ):
        indices = np.asarray(indices_tuple, dtype=np.int64)
        if np.any(indices >= client_context.parameters.slot_count):
            raise ValueError("response metadata references a slot outside the ciphertext")
        ciphertext = sa.Ciphertext()
        _seal_load(ciphertext, payload, context=client_context.seal_context)
        plaintext = sa.Plaintext()
        client_context.decryptor.decrypt(ciphertext, plaintext)
        decoded = np.asarray(
            client_context.encoder.decode_double(plaintext), dtype=np.float64
        )
        blocks.append(decoded[indices])
    scores = np.concatenate(blocks)
    if scores.size != response.score_count:
        raise RuntimeError("decrypted score count does not match response metadata")
    return scores


def plaintext_scores(
    query: Sequence[float] | np.ndarray,
    candidate_embeddings: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    query_array = _as_float_vector(query, name="query")
    candidates = _as_float_matrix(candidate_embeddings)
    if candidates.shape[1] != query_array.size:
        raise ValueError(
            f"candidate dimension {candidates.shape[1]} does not match query "
            f"dimension {query_array.size}"
        )
    return candidates @ query_array


def run_microbenchmark(
    *,
    dimension: int = 192,
    candidate_count: int = 40,
    block_size: int | None = None,
    repeats: int = 5,
    warmups: int = 1,
    seed: int = 20260710,
    parameters: CKKSParameters | None = None,
) -> dict[str, Any]:
    """Benchmark serialized end-to-end server and client paths.

    Server timing includes query frame parsing/ciphertext loading, plaintext
    encoding, HE evaluation, response ciphertext serialization, and response
    framing.  The nested ``server_phase_ms`` exposes the measured components.
    Client timing includes response parsing, ciphertext loading, and decrypt.
    """

    if dimension <= 0 or candidate_count <= 0:
        raise ValueError("dimension and candidate_count must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    parameters = parameters or CKKSParameters()
    if dimension > parameters.slot_count:
        raise ValueError("dimension exceeds the configured CKKS slot count")

    rng = np.random.default_rng(seed)
    query = _unit_rows(rng.normal(size=(1, dimension)))[0]
    candidates = _unit_rows(rng.normal(size=(candidate_count, dimension)))
    expected = plaintext_scores(query, candidates)

    start = time.perf_counter_ns()
    contexts = create_context_pair(parameters, max_query_dimension=dimension)
    context_setup_ms = _elapsed_ms(start)
    start = time.perf_counter_ns()
    serialized_query = encrypt_query(contexts.client, query)
    query_encrypt_serialize_ms = _elapsed_ms(start)

    runners = (
        (
            "naive_per_candidate",
            lambda: server_naive_scores(contexts.server, serialized_query, candidates),
        ),
        (
            "block_packed",
            lambda: server_block_packed_scores(
                contexts.server,
                serialized_query,
                candidates,
                block_size=block_size,
            ),
        ),
        (
            "segmented_score_packed",
            lambda: server_segmented_score_packed(
                contexts.server,
                serialized_query,
                candidates,
                block_size=block_size,
            ),
        ),
    )
    methods: dict[str, dict[str, Any]] = {}
    method_outputs: dict[str, np.ndarray] = {}

    for method_name, runner in runners:
        for _ in range(warmups):
            runner().to_bytes()
        server_samples: list[float] = []
        decrypt_samples: list[float] = []
        phase_samples: dict[str, list[float]] = {}
        response: EncryptedScoreResponse | None = None
        wire = b""
        scores = np.empty(0)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            response = runner()
            wire = response.to_bytes()
            server_samples.append(_elapsed_ms(start))
            for phase, value in response.server_phase_ms.items():
                phase_samples.setdefault(phase, []).append(float(value))

            start = time.perf_counter_ns()
            scores = client_decrypt_scores(contexts.client, wire)
            decrypt_samples.append(_elapsed_ms(start))

        assert response is not None
        difference = scores - expected
        method_outputs[method_name] = scores
        methods[method_name] = {
            "server_ms": _timing_summary(server_samples),
            "server_phase_ms": {
                phase: _timing_summary(values)
                for phase, values in sorted(phase_samples.items())
            },
            "client_decrypt_ms": _timing_summary(decrypt_samples),
            "response": {
                "wire_bytes": len(wire),
                "ckks_payload_bytes": response.payload_bytes,
                "payload_count": len(response.payloads),
                "seal_ciphertext_count": response.seal_ciphertext_count,
                "score_counts_per_payload": list(response.score_counts),
                "score_slot_indices": [
                    list(indices) for indices in response.score_indices
                ],
            },
            "correctness": {
                "max_abs_error": float(np.max(np.abs(difference))),
                "mean_abs_error": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference * difference))),
                "allclose_rtol_1e-4_atol_1e-5": bool(
                    np.allclose(scores, expected, rtol=1e-4, atol=1e-5)
                ),
                "allclose_rtol_3e-4_atol_3e-5": bool(
                    np.allclose(scores, expected, rtol=3e-4, atol=3e-5)
                ),
            },
        }

    naive = methods["naive_per_candidate"]
    packed = methods["block_packed"]
    single = methods["segmented_score_packed"]
    padded_dimension = _next_power_of_two(dimension)
    natural_capacity = parameters.slot_count // padded_dimension
    effective_block_size = min(block_size or natural_capacity, natural_capacity)
    return {
        "schema": MICROBENCHMARK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "algorithm": "power_of_two_segmented_reduction",
            "public_api_note": PUBLIC_API_NOTE,
            "server_has_secret_key": contexts.server.has_secret_key(),
            "server_is_public": contexts.server.is_public(),
            "client_has_secret_key": contexts.client.has_secret_key(),
            "decryption_location": "client_only",
            "server_timing_scope": (
                "query parse/load + plaintext encode + HE evaluate + ciphertext "
                "serialize + response framing"
            ),
            "client_timing_scope": "response parse + ciphertext load + decrypt",
        },
        "software": {
            "python": platform.python_version(),
            "tenseal": getattr(ts, "__version__", "unknown"),
            "tenseal_validated_version": TENSEAL_VALIDATED_VERSION,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "he_backend": "Microsoft SEAL via tenseal.sealapi (CPU-only)",
        },
        "config": {
            "dimension": dimension,
            "padded_dimension": padded_dimension,
            "candidate_count": candidate_count,
            "scores_per_ciphertext_capacity": natural_capacity,
            "requested_block_size": block_size,
            "effective_block_size": effective_block_size,
            "packed_response_ciphertexts_expected": math.ceil(
                candidate_count / effective_block_size
            ),
            "rotation_steps": list(_reduction_steps(padded_dimension)),
            "galois_key_count": contexts.server.galois_keys.size(),
            "final_score_mask_scale_bits": FINAL_MASK_SCALE_BITS,
            "repeats": repeats,
            "warmups": warmups,
            "seed": seed,
            "normalized_inputs": True,
            "ckks": parameters.as_json(),
        },
        "setup": {
            "context_and_key_generation_ms": context_setup_ms,
            "serialized_public_server_context_bytes": len(
                contexts.serialized_server_context
            ),
            "query_encrypt_serialize_ms": query_encrypt_serialize_ms,
            "serialized_query_bytes": len(serialized_query),
        },
        "methods": methods,
        "comparison": {
            "server_speedup_naive_over_packed_median": _safe_ratio(
                naive["server_ms"]["median"], packed["server_ms"]["median"]
            ),
            "server_speedup_naive_over_single_response_median": _safe_ratio(
                naive["server_ms"]["median"], single["server_ms"]["median"]
            ),
            "he_core_speedup_naive_over_packed_median": _safe_ratio(
                naive["server_phase_ms"]["he_evaluate"]["median"],
                packed["server_phase_ms"]["he_evaluate"]["median"],
            ),
            "he_core_speedup_naive_over_single_response_including_pack_median": (
                _safe_ratio(
                    naive["server_phase_ms"]["he_evaluate"]["median"],
                    single["server_phase_ms"]["he_evaluate"]["median"]
                    + single["server_phase_ms"]["score_pack"]["median"],
                )
            ),
            "wire_size_reduction_naive_over_single_response_ratio": _safe_ratio(
                naive["response"]["wire_bytes"], single["response"]["wire_bytes"]
            ),
            "wire_size_reduction_ratio": _safe_ratio(
                naive["response"]["wire_bytes"], packed["response"]["wire_bytes"]
            ),
            "wire_bytes_saved": (
                naive["response"]["wire_bytes"]
                - packed["response"]["wire_bytes"]
            ),
            "seal_ciphertexts_saved": (
                naive["response"]["seal_ciphertext_count"]
                - packed["response"]["seal_ciphertext_count"]
            ),
            "seal_ciphertexts_saved_single_response": (
                naive["response"]["seal_ciphertext_count"]
                - single["response"]["seal_ciphertext_count"]
            ),
            "max_abs_difference_between_he_methods": float(
                np.max(
                    np.abs(
                        method_outputs["naive_per_candidate"]
                        - method_outputs["block_packed"]
                    )
                )
            ),
            "max_abs_difference_naive_vs_single_response": float(
                np.max(
                    np.abs(
                        method_outputs["naive_per_candidate"]
                        - method_outputs["segmented_score_packed"]
                    )
                )
            ),
        },
    }


def _server_scores(
    server_context: ServerCKKSContext,
    serialized_query: bytes,
    candidate_embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str,
    requested_block_size: int | None,
) -> EncryptedScoreResponse:
    _require_public_server_context(server_context)
    candidates = _as_float_matrix(candidate_embeddings)

    start = time.perf_counter_ns()
    encrypted_query, dimension, padded_dimension = _load_server_query(
        server_context, serialized_query, expected_dimension=candidates.shape[1]
    )
    request_load_ms = _elapsed_ms(start)
    capacity = server_context.parameters.slot_count // padded_dimension
    if method == "naive_per_candidate":
        block_size = 1
    else:
        block_size = capacity if requested_block_size is None else requested_block_size
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if block_size > capacity:
            raise ValueError(
                f"block_size {block_size} exceeds segmented capacity {capacity} "
                f"for padded dimension {padded_dimension}"
            )

    single_response = method == "segmented_score_packed"
    if single_response and candidates.shape[0] > server_context.parameters.slot_count:
        raise ValueError("one packed response cannot contain more than slot_count scores")
    group_count = math.ceil(candidates.shape[0] / block_size)
    if single_response and group_count > padded_dimension:
        raise ValueError(
            "requested block_size creates colliding score-pack offsets; increase block_size"
        )

    payloads: list[bytes] = []
    score_counts: list[int] = []
    score_indices: list[tuple[int, ...]] = []
    group_ciphertexts: list[Any] = []
    group_score_counts: list[int] = []
    encode_ms = 0.0
    evaluate_ms = 0.0
    score_pack_ms = 0.0
    serialize_ms = 0.0
    slot_count = server_context.parameters.slot_count
    steps = _reduction_steps(padded_dimension)
    if any(step not in server_context.rotation_steps for step in steps):
        raise ValueError("server context does not contain all required rotation keys")

    for offset in range(0, candidates.shape[0], block_size):
        block = candidates[offset : offset + block_size]
        slots = np.zeros(slot_count, dtype=np.float64)
        for block_index, candidate in enumerate(block):
            base = block_index * padded_dimension
            slots[base : base + dimension] = candidate

        start = time.perf_counter_ns()
        plaintext_candidates = sa.Plaintext()
        server_context.encoder.encode(
            slots.tolist(), server_context.parameters.scale, plaintext_candidates
        )
        encode_ms += _elapsed_ms(start)

        start = time.perf_counter_ns()
        encrypted_scores = sa.Ciphertext()
        server_context.evaluator.multiply_plain(
            encrypted_query, plaintext_candidates, encrypted_scores
        )
        server_context.evaluator.rescale_to_next_inplace(encrypted_scores)
        for step in steps:
            rotated = sa.Ciphertext()
            server_context.evaluator.rotate_vector(
                encrypted_scores, step, server_context.galois_keys, rotated
            )
            server_context.evaluator.add_inplace(encrypted_scores, rotated)
        evaluate_ms += _elapsed_ms(start)

        if single_response:
            group_ciphertexts.append(encrypted_scores)
            group_score_counts.append(block.shape[0])
        else:
            start = time.perf_counter_ns()
            payloads.append(_seal_save(encrypted_scores))
            serialize_ms += _elapsed_ms(start)
            score_counts.append(block.shape[0])
            score_indices.append(
                tuple(index * padded_dimension for index in range(block.shape[0]))
            )

    if single_response:
        start = time.perf_counter_ns()
        combined: Any | None = None
        combined_indices: list[int] = []
        for group_index, (group_ciphertext, group_scores) in enumerate(
            zip(group_ciphertexts, group_score_counts)
        ):
            mask_slots = np.zeros(slot_count, dtype=np.float64)
            for block_index in range(group_scores):
                mask_slots[block_index * padded_dimension] = 1.0
            plaintext_mask = sa.Plaintext()
            server_context.encoder.encode(
                mask_slots.tolist(),
                group_ciphertext.parms_id(),
                float(2**FINAL_MASK_SCALE_BITS),
                plaintext_mask,
            )
            shifted = sa.Ciphertext()
            # Deliberately do not rescale: after the first rescale the product
            # scale is about 2**40; a 2**19 mask remains below the last 60-bit
            # modulus and retains the final level for rotations and additions.
            server_context.evaluator.multiply_plain(
                group_ciphertext, plaintext_mask, shifted
            )
            for shift_step in steps:
                if group_index & shift_step:
                    rotated = sa.Ciphertext()
                    server_context.evaluator.rotate_vector(
                        shifted, shift_step, server_context.galois_keys, rotated
                    )
                    shifted = rotated
            if combined is None:
                combined = shifted
            else:
                server_context.evaluator.add_inplace(combined, shifted)
            combined_indices.extend(
                (block_index * padded_dimension - group_index) % slot_count
                for block_index in range(group_scores)
            )
        if combined is None:
            raise RuntimeError("no group ciphertexts were produced")
        score_pack_ms = _elapsed_ms(start)
        start = time.perf_counter_ns()
        payloads.append(_seal_save(combined))
        serialize_ms += _elapsed_ms(start)
        score_counts.append(candidates.shape[0])
        score_indices.append(tuple(combined_indices))

    phase_ms = {
        "request_deserialize": request_load_ms,
        "plaintext_encode": encode_ms,
        "he_evaluate": evaluate_ms,
        "score_pack": score_pack_ms,
        "ciphertext_serialize": serialize_ms,
    }
    count = len(payloads)
    return EncryptedScoreResponse(
        method=method,
        payloads=tuple(payloads),
        score_counts=tuple(score_counts),
        score_indices=tuple(score_indices),
        seal_ciphertext_counts=(1,) * count,
        server_phase_ms=phase_ms,
    )


def _serialize_server_context(
    parameters: CKKSParameters,
    encryption_parameters: Any,
    public_key: Any,
    galois_keys: Any,
    *,
    max_padded_dimension: int,
    rotation_steps: tuple[int, ...],
) -> bytes:
    header = {
        "schema": SERVER_CONTEXT_SCHEMA,
        "contains_secret_key": False,
        "payload_roles": ["encryption_parameters", "public_key", "galois_keys"],
        "ckks": parameters.as_json(),
        "max_padded_dimension": max_padded_dimension,
        "rotation_steps": list(rotation_steps),
    }
    return _encode_frame(
        _CONTEXT_MAGIC,
        header,
        (
            _seal_save(encryption_parameters),
            _seal_save(public_key),
            _seal_save(galois_keys),
        ),
    )


def _load_server_query(
    server_context: ServerCKKSContext,
    serialized_query: bytes,
    *,
    expected_dimension: int,
) -> tuple[Any, int, int]:
    header, payloads = _decode_frame(_QUERY_MAGIC, serialized_query)
    if header.get("schema") != QUERY_SCHEMA or len(payloads) != 1:
        raise ValueError("invalid serialized CKKS query")
    try:
        dimension = int(header["dimension"])
        padded_dimension = int(header["padded_dimension"])
        slot_count = int(header["slot_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid serialized CKKS query metadata") from exc
    if dimension != expected_dimension:
        raise ValueError(
            f"encrypted query dimension {dimension} does not match candidate "
            f"dimension {expected_dimension}"
        )
    if padded_dimension != _next_power_of_two(dimension):
        raise ValueError("query padded dimension is inconsistent")
    if slot_count != server_context.parameters.slot_count:
        raise ValueError("query and server CKKS slot counts differ")
    if padded_dimension > server_context.max_padded_dimension:
        raise ValueError("query requires rotations absent from the server context")
    ciphertext = sa.Ciphertext()
    _seal_load(ciphertext, payloads[0], context=server_context.seal_context)
    return ciphertext, dimension, padded_dimension


def _require_private_client_context(context: Any) -> None:
    if not isinstance(context, ClientCKKSContext) or not context.has_secret_key():
        raise ValueError("client operation requires a client context with the secret key")


def _require_public_server_context(context: Any) -> None:
    if isinstance(context, ClientCKKSContext):
        raise ValueError("server context must not contain a secret key")
    if not isinstance(context, ServerCKKSContext):
        raise TypeError("expected ServerCKKSContext")
    if context.has_secret_key():
        raise ValueError("server context must not contain a secret key")
    if not context.has_public_key() or not context.has_galois_keys():
        raise ValueError("server context lacks required public evaluation keys")
    if hasattr(context, "decryptor") or hasattr(context, "secret_key"):
        raise ValueError("server context unexpectedly exposes private material")


def _seal_save(value: Any) -> bytes:
    """Bridge SEAL's path-only Python serialization API to an in-memory blob."""

    with tempfile.TemporaryDirectory(prefix="packed_ckks_save_") as directory:
        path = Path(directory) / "object.seal"
        value.save(str(path))
        return path.read_bytes()


def _seal_load(value: Any, payload: bytes, *, context: Any | None = None) -> None:
    """Load an in-memory blob through SEAL's path-only Python API."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("SEAL payload must be non-empty bytes")
    with tempfile.TemporaryDirectory(prefix="packed_ckks_load_") as directory:
        path = Path(directory) / "object.seal"
        path.write_bytes(payload)
        if context is None:
            value.load(str(path))
        else:
            value.load(context, str(path))


def _encode_frame(magic: bytes, header: Mapping[str, Any], payloads: Sequence[bytes]) -> bytes:
    if not payloads or any(not isinstance(payload, bytes) or not payload for payload in payloads):
        raise ValueError("frame payloads must be non-empty byte strings")
    framed_header = dict(header)
    framed_header["payload_lengths"] = [len(payload) for payload in payloads]
    encoded_header = json.dumps(
        framed_header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return b"".join((magic, _UINT32.pack(len(encoded_header)), encoded_header, *payloads))


def _decode_frame(magic: bytes, wire: bytes) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    if not isinstance(wire, bytes):
        raise TypeError("serialized frame must be bytes")
    prefix_size = len(magic) + _UINT32.size
    if len(wire) < prefix_size or not wire.startswith(magic):
        raise ValueError("invalid serialized frame magic")
    header_size = _UINT32.unpack_from(wire, len(magic))[0]
    header_start = prefix_size
    header_end = header_start + header_size
    if header_end > len(wire):
        raise ValueError("truncated serialized frame header")
    try:
        header = json.loads(wire[header_start:header_end].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid serialized frame header") from exc
    lengths = header.get("payload_lengths")
    if not isinstance(lengths, list) or not lengths:
        raise ValueError("serialized frame has no payload lengths")
    if any(not isinstance(length, int) or length <= 0 for length in lengths):
        raise ValueError("invalid serialized frame payload length")
    payloads: list[bytes] = []
    offset = header_end
    for length in lengths:
        end = offset + length
        if end > len(wire):
            raise ValueError("truncated serialized frame payload")
        payloads.append(wire[offset:end])
        offset = end
    if offset != len(wire):
        raise ValueError("unexpected trailing bytes in serialized frame")
    return header, tuple(payloads)


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _reduction_steps(padded_dimension: int) -> tuple[int, ...]:
    return tuple(1 << bit for bit in range(int(math.log2(padded_dimension))))


def _as_float_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _as_float_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("candidate_embeddings must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("candidate_embeddings contains NaN or infinity")
    return np.ascontiguousarray(array)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector")
    return matrix / norms


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _percentile(samples: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in samples)
    if not ordered:
        raise ValueError("cannot compute a percentile of no samples")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    return {
        "samples": values,
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark naive and segmented SIMD-packed CKKS multi-dot kernels."
    )
    parser.add_argument("--dimension", type=int, default=192)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    result = run_microbenchmark(
        dimension=args.dimension,
        candidate_count=args.candidates,
        block_size=args.block_size,
        repeats=args.repeats,
        warmups=args.warmups,
        seed=args.seed,
    )
    rendered = json.dumps(
        result, indent=None if args.compact else 2, sort_keys=True, ensure_ascii=False
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
