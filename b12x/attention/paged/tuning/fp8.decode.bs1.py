"""Generated decode graph policy tuning data."""

from .registry import register_decode_graph_policy

register_decode_graph_policy(
    kv_dtype="fp8",
    regime="decode",
    batch=1,
    graph_ctas_per_sm=2,
    capture_fixed_split_pages=11,
    capture_page_count=4096,
    page_size=64,
    chunk_ladder=(
        (87, 1),
        (260, 2),
        (458, 3),
        (752, 4),
        (940, 5),
        (1128, 6),
        (1316, 7),
        (1504, 8),
        (1692, 9),
        (1880, 10),
        (2068, 11),
        (2256, 12),
        (2444, 13),
        (2632, 14),
        (2820, 15),
        (3008, 16),
        (3196, 17),
        (3384, 18),
        (3572, 19),
        (3760, 20),
        (3948, 21),
        (4096, 22),
    ),
)
