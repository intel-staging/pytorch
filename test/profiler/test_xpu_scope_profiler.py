# Owner(s): ["oncall: profiler"]

import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import (
    KernelMetricsConfig,
    PerformanceMetricsConfig,
    ProfilerActivity,
    ProfilerActivityConfig,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    preserves_unittest_skip_marks,
    run_tests,
    set_warn_always_context,
    subtest,
    TEST_XPU,
    TestCase,
)


Verbose = False


@instantiate_parametrized_tests
@preserves_unittest_skip_marks()
class XpuScopeProfilerTest(TestCase):
    def __init__(self, arg):
        super().__init__(arg)

        self.metrics_names = [
            "GpuTime",
            "GpuCoreClocks",
            "AvgGpuCoreFrequencyMHz",
            "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION",
            "XVE_ACTIVE",
            "XVE_STALL",
        ]

    def count_metrics(self, event, counter):
        self.assertTrue("args" in event)
        self.assertTrue(isinstance(event["args"], dict))

        for arg in event["args"]:
            counter[arg] += 1

    def check_metrics(self, counter):
        rev_counter = defaultdict(int)

        for metric_name_in_json in counter:
            # Kineto emits the metric either bare (counter "C" events) or with a
            # unit suffix. The scope profiler's per-kernel metadata uses
            # "<name>::<unit>" (e.g. "GpuTime::ns"); some loggers use "<name> [..]".
            metric_name_in_json_valid = sum(
                metric_name_in_json == metric_name
                or metric_name_in_json.startswith(
                    (metric_name + "::", metric_name + " [")
                )
                for metric_name in self.metrics_names
            )
            self.assertTrue(metric_name_in_json_valid <= 1)
            if metric_name_in_json_valid == 1:
                rev_counter[counter[metric_name_in_json]] += 1

        if Verbose:
            print(f"{rev_counter=}")

        # keys are amounts of certain metrics
        # only one dict element means all metrics have the same amount
        self.assertEqual(len(rev_counter), 1)
        first_value = next(iter(rev_counter.values()))
        self.assertEqual(first_value, len(self.metrics_names))

    def gen_and_check_json(self, p, json_file, expect_default_activities=True):
        p.export_chrome_trace(json_file)

        with open(json_file) as f:
            data = json.load(f)
            self.assertTrue("traceEvents" in data)

            trace_events = data["traceEvents"]
            self.assertTrue(isinstance(trace_events, list))
            self.assertTrue(len(trace_events) > 0)

            count_names = defaultdict(int)
            count_cats = defaultdict(int)
            count_x_metrics = defaultdict(int)
            count_c_metrics = defaultdict(int)
            for event in trace_events:
                self.assertTrue("ph" in event)
                self.assertTrue("name" in event)

                if event["ph"] == "X":
                    self.assertTrue("cat" in event)
                    count_names[event["name"]] += 1
                    count_cats[event["cat"]] += 1

                    if event["name"].startswith("metrics: "):
                        self.count_metrics(event, count_x_metrics)

                elif event["ph"] == "C":
                    self.count_metrics(event, count_c_metrics)

            if Verbose:
                print(f"{count_names=}")
                print(f"{count_cats=}")
                print(f"{count_x_metrics=}")
                print(f"{count_c_metrics=}")

            metric_name_in_json_valid = sum(
                name.startswith("metrics: ") for name in count_names
            )
            self.assertTrue(metric_name_in_json_valid > 0)

            self.assertTrue("xpu_runtime" in count_cats)
            self.assertTrue("kernel" in count_cats)
            # Memory copies (here: the copy back to host) are a default XPU
            # activity, dropped when the request lists metrics only.
            self.assertEqual("gpu_memcpy" in count_cats, expect_default_activities)

            self.check_metrics(count_x_metrics)
            self.check_metrics(count_c_metrics)

    def required_env_setting(self):
        zet_enable_metrics = os.environ.get("ZET_ENABLE_METRICS", "unset")
        if Verbose:
            print(f"ZET_ENABLE_METRICS={zet_enable_metrics}")
        return zet_enable_metrics == "1"

    def required_paranoid_setting(self, paranoid_path):
        try:
            paranoid_content = Path(paranoid_path).read_text().strip()
            if Verbose:
                print(f"{paranoid_path} contains: {paranoid_content}")
            return paranoid_content == "0"

        except FileNotFoundError:
            if Verbose:
                print(f"{paranoid_path} doesn't exists")
            return False

    def required_paranoid_settings(self):
        return any(
            self.required_paranoid_setting(paranoid)
            for paranoid in [
                "/proc/sys/dev/xe/observation_paranoid",
                "/proc/sys/dev/i915/perf_stream_paranoid",
            ]
        )

    def _skip_without_metrics_support(self):
        if not self.required_env_setting():
            self.skipTest("ZET_ENABLE_METRICS not set")

        if not self.required_paranoid_settings():
            self.skipTest("paranoid settings missing")

    def _profile_workload(self, activities):
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(activities=activities) as p:
            r1 = torch.matmul(a, b)
            r2 = torch.add(r1, 1.0)
            result = torch.abs(r2)
            # Copy back to host, so the trace holds a device memcpy activity.
            host_result = result.cpu()

        self.assertTrue(host_result.numel() > 0)
        return p

    def _check_trace(self, p, expect_default_activities=True):
        json_file = os.environ.get("JSON_FILE")

        if json_file:
            root, ext = os.path.splitext(json_file)
            json_file = f"{root}_{self._testMethodName}{ext}"
            self.gen_and_check_json(p, json_file, expect_default_activities)
        else:
            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
                self.gen_and_check_json(p, tmp.name, expect_default_activities)

        if Verbose:
            print(p.key_averages().table())

        count_metrics = sum(
            event.key.startswith("metrics: ") for event in p.key_averages()
        )
        self.assertTrue(count_metrics > 0)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_scope_profiler(self):
        """Metrics from KernelMetricsConfig keep the default XPU activities."""
        self._skip_without_metrics_support()

        p = self._profile_workload(
            [
                ProfilerActivity.CPU,
                {
                    ProfilerActivity.XPU: ProfilerActivityConfig(
                        profiler_configs=[
                            KernelMetricsConfig(metric_names=self.metrics_names)
                        ]
                    )
                },
            ]
        )
        self._check_trace(p)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_scope_profiler_metric_names_in_activity_list(self):
        """The deprecated form still collects metrics, but drops the defaults."""
        self._skip_without_metrics_support()

        with (
            set_warn_always_context(True),
            self.assertWarnsRegex(
                UserWarning,
                "is deprecated and will be removed.*drops the default XPU activities",
            ),
        ):
            p = self._profile_workload(
                [ProfilerActivity.CPU, {ProfilerActivity.XPU: self.metrics_names}]
            )
        self._check_trace(p, expect_default_activities=False)

    def _profile_xpu_config(self, config):
        with torch.profiler.profile(activities=[{ProfilerActivity.XPU: config}]):
            pass

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    @parametrize(
        "activity_types",
        [
            # The legacy form forwards a misspelled activity type as a metric
            # name; with KernelMetricsConfig it is rejected.
            subtest(["kernel"], name="misspelled_activity_type"),
            subtest(["CONCURRENT_KERNEL", "GpuCoreClocks"], name="metric_name"),
        ],
    )
    def test_activity_types_are_strict_with_metrics_config(self, activity_types):
        """Metric names having their own field leaves no name to guess about."""
        metrics = KernelMetricsConfig(metric_names=["GpuTime"])
        with self.assertRaisesRegex(
            RuntimeError, "belong in KernelMetricsConfig.metric_names"
        ):
            self._profile_xpu_config(
                ProfilerActivityConfig(
                    activity_types=activity_types, profiler_configs=[metrics]
                )
            )

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_time_based_metrics_not_supported(self):
        """XPU does not sample counters over time yet, so this must not pass quietly."""
        with self.assertRaisesRegex(
            RuntimeError, "not supported for ProfilerActivity.XPU"
        ):
            self._profile_xpu_config(
                ProfilerActivityConfig(
                    profiler_configs=[
                        PerformanceMetricsConfig(metric_names=["GpuTime"])
                    ]
                )
            )

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_activity_type_filter_does_not_enable_scope_profiler(self):
        """A fine-grained XPU dict that lists only activity type names (no
        hardware metric names) filters activities as in #176351 and must not
        turn on the per-kernel scope profiler. Needs no metrics hardware."""
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                {torch.profiler.ProfilerActivity.XPU: ["CONCURRENT_KERNEL"]},
            ],
        ) as p:
            result = torch.matmul(a, b)

        self.assertTrue(result.numel() > 0)
        count_metrics = sum(
            event.key.startswith("metrics: ") for event in p.key_averages()
        )
        self.assertEqual(count_metrics, 0)


if __name__ == "__main__":
    run_tests()
