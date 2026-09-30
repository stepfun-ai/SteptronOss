from __future__ import annotations


def _import_with_tilelang_hint(module_name: str, symbol_name: str):
    try:
        module = __import__(module_name, fromlist=[symbol_name])
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("tilelang"):
            raise ModuleNotFoundError(
                "GLM-5 tilelang DSA optimization requires the `tilelang` package to be installed "
                "before selecting `lighting_indexer='tilelang'` / `sparse_mla='tilelang'`."
            ) from exc
        raise
    return getattr(module, symbol_name)


def tilelang_lighting_indexer(
    index_q,
    index_k,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    topk,
    topk_indices=None,
):
    slime_lighting_indexer = _import_with_tilelang_hint(
        "steptronoss.model.optimizations.glm5_dsa.indexer",
        "lighting_indexer",
    )

    return slime_lighting_indexer(
        index_q=index_q,
        index_k=index_k,
        weights=weights,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        topk=topk,
        topk_indices=topk_indices,
    )


def tilelang_sparse_mla(
    q,
    kv,
    indices,
    scaling,
    d_v=None,
    query_chunk_size=None,
):
    del d_v, query_chunk_size

    SparseMLA = _import_with_tilelang_hint(
        "steptronoss.model.optimizations.glm5_dsa.sparse_mla",
        "SparseMLA",
    )

    return SparseMLA.apply(q, kv, indices, scaling)
