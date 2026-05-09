"""
Integration tests for the per-key arbitrary-command miss tracking feature.

Each test pre-populates a known subset of keys, runs an arbitrary command
that probes a wider key range, and asserts that the resulting `mb.json`
contains the expected aggregate Hits/Misses counters and per-key buckets
under the "Per-Key Misses" section nested in "ALL STATS".
"""

import json
import os
import tempfile

from include import (
    addTLSArgs,
    add_required_env_arguments,
    debugPrintMemtierOnError,
    ensure_clean_benchmark_folder,
    get_default_memtier_config,
)
from mb import Benchmark, RunConfig


# Keys 1..3 are pre-populated; --key-maximum=10 means ~30% hit rate on average.
_KEY_PREFIX = "memtier-"
_PRELOADED_KEYS = 3
_KEY_RANGE_MAX = 10
_REQUESTS = 200


def _preload_strings(env):
    """Populate keys memtier-1, memtier-2, memtier-3 (string values)."""
    env.flush()
    conn = env.getConnection()
    for i in range(1, _PRELOADED_KEYS + 1):
        conn.set("{}{}".format(_KEY_PREFIX, i), "v{}".format(i))


def _preload_sets(env, prefix):
    """Populate sets prefix-1, prefix-2, prefix-3 (each with 3 members)."""
    env.flush()
    conn = env.getConnection()
    for i in range(1, _PRELOADED_KEYS + 1):
        conn.sadd("{}{}".format(prefix, i), "a", "b", "c")


def _run_benchmark(env, command, miss_tracking="auto", key_prefix=_KEY_PREFIX, requests=_REQUESTS):
    """Run memtier with the given --command and return the parsed mb.json."""
    test_dir = tempfile.mkdtemp()
    benchmark_specs = {
        "name": env.testName,
        "args": [
            "--command={}".format(command),
            "--command-key-pattern=R",
            "--command-stats-breakdown=line",
            "--command-miss-tracking={}".format(miss_tracking),
            "--key-prefix={}".format(key_prefix),
            "--key-minimum=1",
            "--key-maximum={}".format(_KEY_RANGE_MAX),
            "--hide-histogram",
        ],
    }
    addTLSArgs(benchmark_specs, env)

    config = get_default_memtier_config(threads=1, clients=2, requests=requests)
    master_nodes_list = env.getMasterNodesList()
    add_required_env_arguments(benchmark_specs, config, env, master_nodes_list)

    config = RunConfig(test_dir, env.testName, config, {})
    ensure_clean_benchmark_folder(config.results_dir)

    benchmark = Benchmark.from_json(config, benchmark_specs)
    memtier_ok = benchmark.run()
    debugPrintMemtierOnError(config, env)
    env.assertTrue(memtier_ok)

    json_path = "{}/mb.json".format(config.results_dir)
    env.assertTrue(os.path.isfile(json_path))
    with open(json_path) as fh:
        return json.load(fh)


def test_arbitrary_get_single_null_bulk(env):
    """GET — SingleNullBulk shape; expect 1 key bucket and ~30% hit rate."""
    env.skipOnCluster()
    _preload_strings(env)

    result = _run_benchmark(env, "GET __key__")
    per_key = result["ALL STATS"].get("Per-Key Misses", {})
    env.assertContains("GET", per_key)
    cmd_stats = per_key["GET"]

    total_ops = cmd_stats["Total Hits"] + cmd_stats["Total Misses"]
    env.assertEqual(total_ops, _REQUESTS * 2)  # 2 clients
    # Exactly one bucket for GET.
    env.assertContains("key[0] Hits", cmd_stats)
    env.assertNotContains("key[1] Hits", cmd_stats)
    # Bucket sums match overall.
    env.assertEqual(cmd_stats["key[0] Hits"], cmd_stats["Total Hits"])
    env.assertEqual(cmd_stats["key[0] Misses"], cmd_stats["Total Misses"])
    # Hit rate should be in the right ballpark for a uniform random key in
    # [1, 10] with 3 pre-populated keys; allow generous slack to avoid flakes.
    hit_rate = cmd_stats["Total Hits"] / float(total_ops)
    env.assertTrue(0.15 < hit_rate < 0.50,
                   message="GET hit_rate={} outside [0.15, 0.50]".format(hit_rate))


def test_arbitrary_mget_per_element_nulls(env):
    """MGET — ArrayPerElementNulls; expect K=3 key buckets, no phantom k[3]."""
    env.skipOnCluster()
    _preload_strings(env)

    result = _run_benchmark(env, "MGET __key__ __key__ __key__")
    per_key = result["ALL STATS"].get("Per-Key Misses", {})
    env.assertContains("MGET", per_key)
    cmd_stats = per_key["MGET"]

    total_ops = cmd_stats["Total Hits"] + cmd_stats["Total Misses"]
    env.assertEqual(total_ops, _REQUESTS * 2 * 3)  # 2 clients × 3 keys per request

    for k in (0, 1, 2):
        env.assertContains("key[{}] Hits".format(k), cmd_stats)
    # No phantom 4th bucket from off-by-one in spec evaluation.
    env.assertNotContains("key[3] Hits", cmd_stats)

    # Sum of per-key matches totals.
    sum_hits = sum(cmd_stats["key[{}] Hits".format(k)] for k in (0, 1, 2))
    sum_misses = sum(cmd_stats["key[{}] Misses".format(k)] for k in (0, 1, 2))
    env.assertEqual(sum_hits, cmd_stats["Total Hits"])
    env.assertEqual(sum_misses, cmd_stats["Total Misses"])


def test_arbitrary_smembers_empty_collection(env):
    """SMEMBERS on missing key returns empty array; empty == miss heuristic."""
    env.skipOnCluster()
    set_prefix = "memtier-set-"
    _preload_sets(env, set_prefix)

    result = _run_benchmark(env, "SMEMBERS __key__", key_prefix=set_prefix)
    per_key = result["ALL STATS"].get("Per-Key Misses", {})
    env.assertContains("SMEMBERS", per_key)
    cmd_stats = per_key["SMEMBERS"]

    total_ops = cmd_stats["Total Hits"] + cmd_stats["Total Misses"]
    env.assertEqual(total_ops, _REQUESTS * 2)
    # Exactly one bucket.
    env.assertContains("key[0] Hits", cmd_stats)
    env.assertNotContains("key[1] Hits", cmd_stats)
    # Misses must be present (we query keys 4..10 which are empty/missing).
    env.assertTrue(cmd_stats["Total Misses"] > 0,
                   message="expected SMEMBERS misses on missing keys")


def test_arbitrary_exists_integer_membership(env):
    """EXISTS — IntegerMembership; integer reply N == hit count."""
    env.skipOnCluster()
    _preload_strings(env)

    result = _run_benchmark(env, "EXISTS __key__")
    per_key = result["ALL STATS"].get("Per-Key Misses", {})
    env.assertContains("EXISTS", per_key)
    cmd_stats = per_key["EXISTS"]

    total_ops = cmd_stats["Total Hits"] + cmd_stats["Total Misses"]
    env.assertEqual(total_ops, _REQUESTS * 2)
    env.assertContains("key[0] Hits", cmd_stats)
    env.assertNotContains("key[1] Hits", cmd_stats)


def test_arbitrary_miss_tracking_off_omits_per_key_section(env):
    """--command-miss-tracking=off: backward-compatible JSON shape."""
    env.skipOnCluster()
    _preload_strings(env)

    result = _run_benchmark(env, "GET __key__", miss_tracking="off")
    all_stats = result["ALL STATS"]

    # Per-Key Misses section absent entirely.
    env.assertNotContains("Per-Key Misses", all_stats)

    # Aggregate Hits/sec and Misses/sec must be exactly zero (no bookkeeping).
    # The "Gets" entry exists when --command-stats-breakdown=command (default).
    env.assertContains("Gets", all_stats)
    env.assertEqual(all_stats["Gets"]["Hits/sec"], 0.0)
    env.assertEqual(all_stats["Gets"]["Misses/sec"], 0.0)


def test_arbitrary_set_not_missable_no_per_key_section(env):
    """SET has reply_shape NotMissable — no Per-Key entry should appear."""
    env.skipOnCluster()
    env.flush()

    result = _run_benchmark(env, "SET __key__ __data__")
    per_key = result["ALL STATS"].get("Per-Key Misses", {})
    # SET should not produce any per-key bookkeeping.
    env.assertNotContains("SET", per_key)
