"""Run manifest. One JSON per Dataproc job -> gs://.../manifests/."""
import json, time, uuid, datetime

_CAPTURE = [
    "spark.app.id", "spark.app.name", "spark.master",
    "spark.executor.instances", "spark.executor.cores", "spark.executor.memory",
    "spark.driver.memory", "spark.sql.shuffle.partitions",
    "spark.dynamicAllocation.enabled",
]


def _utc():
    return datetime.datetime.now(datetime.timezone.utc)


class Run:
    def __init__(self, spark, script, git_sha="nogit", params=None):
        self.spark = spark
        self.t0 = time.time()
        self.rec = {
            "run_id": f"{script}-{_utc():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
            "script": script,
            "git_sha": git_sha,
            "utc_start": _utc().isoformat(),
            "params": params or {},
            "spark_version": spark.version,
            "cluster": self._cluster(),
            "counts": {},
            "outputs": [],
            "status": "running",
        }

    def _cluster(self):
        sc = self.spark.sparkContext
        conf = sc.getConf()
        d = {k: conf.get(k, None) for k in _CAPTURE}
        # Runtime SQL confs can be reset post-launch via spark.conf.set, so the
        # static SparkConf is stale for them. Read from the session instead.
        try:
            d["spark.sql.shuffle.partitions"] = self.spark.conf.get("spark.sql.shuffle.partitions")
        except Exception:
            pass
        try:
            d["live_executors"] = sc._jsc.sc().getExecutorMemoryStatus().size() - 1
        except Exception:
            d["live_executors"] = None
        d["default_parallelism"] = sc.defaultParallelism
        return d

    def await_executors(self, expected, timeout_s=120, poll_s=2):
        """Block until executors register, then reset the clock.
        Startup is constant regardless of data size; billing it to the
        workload would flatten the speedup curve."""
        sc = self.spark.sparkContext
        start = time.time()
        deadline = start + timeout_s
        n = 0
        while time.time() < deadline:
            try:
                n = sc._jsc.sc().getExecutorMemoryStatus().size() - 1
            except Exception:
                n = 0
            if n >= expected:
                break
            time.sleep(poll_s)
        self.rec["cluster"]["live_executors"] = n
        self.rec["cluster"]["executors_expected"] = expected
        self.rec["cluster"]["executor_wait_s"] = round(time.time() - start, 1)
        self.t0 = time.time()
        self.rec["cluster"]["default_parallelism"] = sc.defaultParallelism
        print(f"[executors] {n}/{expected} registered "
              f"in {self.rec['cluster']['executor_wait_s']}s")
        return n

    def count(self, name, value):
        v = int(value)
        self.rec["counts"][name] = v
        return v

    def param(self, k, v):
        self.rec["params"][k] = v

    def output(self, uri):
        self.rec["outputs"].append(uri)

    def finish(self, manifest_dir, status="ok", error=None):
        self.rec["wall_clock_s"] = round(time.time() - self.t0, 2)
        self.rec["utc_end"] = _utc().isoformat()
        self.rec["status"] = status
        self.rec["cluster"]["live_executors_at_finish"] = self._cluster()["live_executors"]
        if error:
            self.rec["error"] = str(error)[:2000]
        path = f"{manifest_dir}/{self.rec['run_id']}.json"
        _write_text(self.spark, path, json.dumps(self.rec, indent=2))
        print(json.dumps(self.rec, indent=2))
        print(f"[manifest] {path}")
        return path


def _write_text(spark, path, text):
    """Write a single small text file to GCS via the Hadoop FS API (no extra deps)."""
    sc = spark.sparkContext
    jvm = sc._jvm
    p = jvm.org.apache.hadoop.fs.Path(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(p.toUri(), sc._jsc.hadoopConfiguration())
    out = fs.create(p, True)
    out.write(bytearray(text.encode("utf-8")))
    out.close()