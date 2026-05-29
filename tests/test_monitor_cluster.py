"""
Cluster-mode coverage for `--monitor-input` (route-then-stage).

Background
----------
In `--cluster-mode`, `--monitor-input` replays a captured command stream by
routing each command to the shard that owns its key's slot: a shard connection
reads a monitor line, computes the key slot, and pushes the command into the
slot owner's staged queue (waking it via schedule_fill), or — if that queue is
full — sends it inline and lets the resulting MOVED refresh topology. The
draining shard formats and sends the staged command.

Until now this entire route-then-stage path had NO automated cluster coverage:
every test in `tests/test_monitor_input.py` begins with `env.skipOnCluster()`.
These tests close that gap with DETERMINISTIC assertions (routing correctness,
run completion / accounting balance, and per-type stats attribution). They
deliberately avoid topology-churn-during-run (slot migration mid-replay), which
is inherently racy in this harness; that remains tracked as a separate gap.

Run:
    TEST=test_monitor_cluster.py OSS_CLUSTER=1 SHARDS=3 ./tests/run_tests.sh
"""

import json
import os
import tempfile

from redis.cluster import key_slot

from include import (
    add_required_env_arguments,
    addTLSArgs,
    debugPrintMemtierOnError,
    ensure_clean_benchmark_folder,
    get_default_memtier_config,
)
from mb import Benchmark, RunConfig


# ---------------------------------------------------------------------------
# Cluster helpers (same pattern as test_mget_cluster.py / test_cluster_transaction.py)
# ---------------------------------------------------------------------------

def _master_conns(env):
    return list(env.getOSSMasterNodesConnectionList())


def _flush_cluster(env):
    for conn in _master_conns(env):
        conn.execute_command("FLUSHALL")


def _dbsize_total_and_shards(env):
    """Return (total_keys, number_of_shards_with_data)."""
    total = 0
    shards_with_data = 0
    for conn in _master_conns(env):
        n = int(conn.execute_command("DBSIZE"))
        total += n
        if n > 0:
            shards_with_data += 1
    return total, shards_with_data


def _owning_port(env, key):
    """Master port owning the slot for *key*, via CLUSTER SLOTS (layout-agnostic)."""
    slot = key_slot(key.encode())
    any_master = _master_conns(env)[0]
    for entry in any_master.execute_command("CLUSTER", "SLOTS"):
        if int(entry[0]) <= slot <= int(entry[1]):
            return int(entry[2][1])
    raise AssertionError("no owner for slot {} of key {!r}".format(slot, key))


def _get_from_cluster(env, key):
    """GET a key from whichever master owns it; None if absent/unreachable."""
    for conn in _master_conns(env):
        try:
            val = conn.execute_command("GET", key)
        except Exception:
            continue  # non-owner replies MOVED
        if val is not None:
            return val.decode() if isinstance(val, bytes) else val
    return None


def _conn_for_port(env, port):
    for conn in _master_conns(env):
        if int(conn.connection_pool.connection_kwargs["port"]) == port:
            return conn
    return None


def _write_monitor_file(test_dir, lines):
    path = os.path.join(test_dir, "monitor.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _set_line(key, value):
    return '[ proxy ] 1764031576.604009 [0 127.0.0.1:5000] "SET" "{}" "{}"'.format(key, value)


def _run_monitor_cluster(env, monitor_file, command, monitor_pattern=None,
                         threads=2, clients=4, requests=200, extra=None):
    """Run a monitor-input cluster workload; return (ok, run_config, json_dict)."""
    args = [
        "--monitor-input={}".format(monitor_file),
        "--command={}".format(command),
        "--hide-histogram",
    ]
    if monitor_pattern is not None:
        args.append("--monitor-pattern={}".format(monitor_pattern))
    if extra:
        args.extend(extra)

    benchmark_specs = {"name": env.testName, "args": args}
    addTLSArgs(benchmark_specs, env)

    config = get_default_memtier_config(threads=threads, clients=clients, requests=requests)
    add_required_env_arguments(benchmark_specs, config, env, env.getMasterNodesList())

    test_dir = tempfile.mkdtemp()
    run_config = RunConfig(test_dir, env.testName, config, {})
    ensure_clean_benchmark_folder(run_config.results_dir)

    benchmark = Benchmark.from_json(run_config, benchmark_specs)
    ok = benchmark.run()
    debugPrintMemtierOnError(run_config, env)

    js = {}
    json_path = "{}/mb.json".format(run_config.results_dir)
    if os.path.isfile(json_path):
        with open(json_path) as fh:
            js = json.load(fh)
    return ok, run_config, js


# ---------------------------------------------------------------------------
# 1. Specific monitor line (__monitor_line1__) routes to the slot owner
# ---------------------------------------------------------------------------

def test_monitor_cluster_specific_command_routes_to_owner(env):
    """A specific SET monitor command must commit to the slot-owning shard (and
    only that shard) — the keyless-to-keyed routing of the staged command."""
    if not env.isCluster():
        env.skip()
        return

    _flush_cluster(env)
    test_dir = tempfile.mkdtemp()
    monitor_file = _write_monitor_file(test_dir, [_set_line("mon-key1", "value1")])

    ok, run_config, _js = _run_monitor_cluster(
        env, monitor_file, "__monitor_line1__", threads=1, clients=1, requests=50)

    failed = env.getNumberOfFailedAssertion()
    try:
        env.assertTrue(ok, message="memtier did not complete the specific-line cluster run")
        env.assertEqual(_get_from_cluster(env, "mon-key1"), "value1",
                        message="mon-key1 not committed via routed staged command")
        # Exactly one copy across the cluster (no MOVED leakage onto a wrong shard),
        # and it lives on the computed slot owner. EXISTS is queried only on the
        # owner connection — a non-owner plain node would reply MOVED (raise).
        total, _shards = _dbsize_total_and_shards(env)
        env.assertEqual(total, 1, message="expected exactly 1 key in cluster, got {}".format(total))
        owner_conn = _conn_for_port(env, _owning_port(env, "mon-key1"))
        env.assertTrue(owner_conn is not None and owner_conn.execute_command("EXISTS", "mon-key1"),
                       message="mon-key1 not on its slot-owning shard")
    finally:
        if env.getNumberOfFailedAssertion() > failed:
            debugPrintMemtierOnError(run_config, env)


# ---------------------------------------------------------------------------
# 2. Sequential replay: every SET is routed and committed exactly once
# ---------------------------------------------------------------------------

def test_monitor_cluster_sequential_all_keys_committed(env):
    """__monitor_line@__ sequential over N distinct SET keys spanning shards:
    every SET must be routed to the correct owner and committed (total DBSIZE
    == N, spread across >= 2 shards). Proves no command is lost to a wrong-shard
    send and the staged-queue accounting nets out (the run terminates)."""
    if not env.isCluster():
        env.skip()
        return

    _flush_cluster(env)
    test_dir = tempfile.mkdtemp()
    n = 60
    lines = [_set_line("mc:{}".format(i), "v{}".format(i)) for i in range(n)]
    monitor_file = _write_monitor_file(test_dir, lines)

    # Sequential pattern + requests == number of lines => each line runs once.
    ok, run_config, _js = _run_monitor_cluster(
        env, monitor_file, "__monitor_line@__", monitor_pattern="S",
        threads=1, clients=1, requests=n)

    failed = env.getNumberOfFailedAssertion()
    try:
        env.assertTrue(ok, message="sequential monitor-cluster run did not complete")
        total, shards = _dbsize_total_and_shards(env)
        env.assertEqual(total, n,
                        message="expected {} committed keys, got {} (routing lost commands?)".format(n, total))
        env.assertTrue(shards >= 2,
                       message="keys landed on only {} shard(s) — cross-shard staging not exercised".format(shards))
        # Spot-check a few keys live on their computed owner.
        for i in (0, n // 2, n - 1):
            key = "mc:{}".format(i)
            owner = _owning_port(env, key)
            owner_conn = _conn_for_port(env, owner)
            env.assertTrue(owner_conn is not None and owner_conn.execute_command("EXISTS", key),
                           message="{} not on its owning shard {}".format(key, owner))
    finally:
        if env.getNumberOfFailedAssertion() > failed:
            debugPrintMemtierOnError(run_config, env)


# ---------------------------------------------------------------------------
# 3. Random replay completes (no hang) and distributes across shards
# ---------------------------------------------------------------------------

def test_monitor_cluster_random_completes_and_distributes(env):
    """__monitor_line@__ random over keys spanning shards: the run must complete
    (no hang / no accounting deadlock) under --requests, with data landing on
    multiple shards."""
    if not env.isCluster():
        env.skip()
        return

    _flush_cluster(env)
    test_dir = tempfile.mkdtemp()
    lines = [_set_line("mr:{}".format(i), "v{}".format(i)) for i in range(40)]
    monitor_file = _write_monitor_file(test_dir, lines)

    ok, run_config, js = _run_monitor_cluster(
        env, monitor_file, "__monitor_line@__", monitor_pattern="R",
        threads=2, clients=4, requests=300)

    failed = env.getNumberOfFailedAssertion()
    try:
        env.assertTrue(ok, message="random monitor-cluster run did not complete (possible hang)")
        _total, shards = _dbsize_total_and_shards(env)
        env.assertTrue(shards >= 2,
                       message="random replay reached only {} shard(s)".format(shards))
        totals = js.get("ALL STATS", {}).get("Totals", {})
        env.assertTrue(totals.get("Count", 0) > 0, message="no ops recorded")
    finally:
        if env.getNumberOfFailedAssertion() > failed:
            debugPrintMemtierOnError(run_config, env)


# ---------------------------------------------------------------------------
# 4. Per-type stats attribution survives cross-shard routing
# ---------------------------------------------------------------------------

def test_monitor_cluster_stats_attribution_by_type(env):
    """A mixed-type monitor stream (__monitor_line@__) must attribute per-command
    stats correctly even though commands are staged to and drained from other
    shards: the JSON must expose multiple per-type sections with non-zero counts
    that sum to the Totals count."""
    if not env.isCluster():
        env.skip()
        return

    _flush_cluster(env)
    test_dir = tempfile.mkdtemp()
    # Distinct keys per type so they spread across shards; types: SET + GET.
    lines = []
    for i in range(20):
        lines.append(_set_line("st:{}".format(i), "v{}".format(i)))
        lines.append('[ proxy ] 1764031576.604010 [0 127.0.0.1:5000] "GET" "st:{}"'.format(i))
    monitor_file = _write_monitor_file(test_dir, lines)

    ok, run_config, js = _run_monitor_cluster(
        env, monitor_file, "__monitor_line@__", monitor_pattern="R",
        threads=2, clients=4, requests=400)

    failed = env.getNumberOfFailedAssertion()
    try:
        env.assertTrue(ok, message="mixed-type monitor-cluster run did not complete")
        all_stats = js.get("ALL STATS", {})
        # Default breakdown aggregates by type -> "Sets" and "Gets" sections.
        env.assertContains("Sets", all_stats)
        env.assertContains("Gets", all_stats)
        sets_count = all_stats["Sets"].get("Count", 0)
        gets_count = all_stats["Gets"].get("Count", 0)
        env.assertTrue(sets_count > 0, message="no SET ops attributed after cross-shard routing")
        env.assertTrue(gets_count > 0, message="no GET ops attributed after cross-shard routing")
        totals_count = all_stats.get("Totals", {}).get("Count", 0)
        env.assertEqual(sets_count + gets_count, totals_count,
                        message="per-type counts ({}+{}) != Totals ({})".format(
                            sets_count, gets_count, totals_count))
    finally:
        if env.getNumberOfFailedAssertion() > failed:
            debugPrintMemtierOnError(run_config, env)
