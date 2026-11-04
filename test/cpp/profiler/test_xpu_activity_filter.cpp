/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_KINETO

#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <torch/csrc/profiler/kineto_shim.h>

#include <IActivityProfiler.h>
#include <libkineto.h>

namespace {

using torch::autograd::profiler::ActivityType;
using torch::profiler::impl::ExperimentalConfig;
using torch::profiler::impl::kineto::ActivityFilter;
using torch::profiler::impl::kineto::ActivitySet;
using torch::profiler::impl::kineto::prepareTrace;
using torch::profiler::impl::kineto::ProfilerExtensionMap;
using KinetoType = libkineto::ActivityType;

constexpr auto kCollectiveProfilingEnv =
    "TORCH_PROFILER_ENABLE_COLLECTIVE_PROFILING";

// What prepareTrace handed to Kineto's child profilers.
struct Captured {
  std::set<KinetoType> activities;
  std::vector<std::string> performance_metric_names;
};

class CapturingProfiler : public libkineto::IActivityProfiler {
 public:
  explicit CapturingProfiler(std::shared_ptr<Captured> captured)
      : captured_{std::move(captured)} {}

  const std::string& name() const override {
    return name_;
  }
  const std::set<KinetoType>& availableActivities() const override {
    return activities_;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<KinetoType>& activity_types,
      const libkineto::Config& config) override {
    captured_->activities = activity_types;
    captured_->performance_metric_names = config.performanceMetricNames();
    return nullptr;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      [[maybe_unused]] int64_t ts_ms,
      [[maybe_unused]] int64_t duration_ms,
      const std::set<KinetoType>& activity_types,
      const libkineto::Config& config) override {
    return configure(activity_types, config);
  }

 private:
  std::string name_{"capturing"};
  std::set<KinetoType> activities_{KinetoType::XPU_SYNC};
  std::shared_ptr<Captured> captured_;
};

class CapturingWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    messages.push_back(warning.msg());
  }

  std::vector<std::string> messages;
};

class XpuActivityFilterTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    captured_ = std::make_shared<Captured>();
    // Libkineto has no unregister API, so this factory lives for the process.
    libkineto::api().registerProfilerFactory([captured = captured_]() {
      return std::make_unique<CapturingProfiler>(captured);
    });
    // Register without device support so the test does not depend on the
    // host.
    libkineto_init(/*cpuOnly=*/true, /*logOnError=*/true);

    // A true collectivesProfilerExists() adds COLLECTIVE_COMM to every case
    // below. Pinned so expectations do not depend on the environment; the
    // assertions test membership rather than whole sets so they also hold
    // under KINETO_HAS_HCCL_PROFILER.
    c10::utils::set_env(kCollectiveProfilingEnv, "0");
  }

  static Captured prepareAndCapture(
      const ActivitySet& activities,
      const ActivityFilter& activity_filter,
      const ProfilerExtensionMap& profiler_extensions = {}) {
    *captured_ = Captured{};
    prepareTrace(
        activities,
        ExperimentalConfig{},
        /*trace_id=*/"",
        activity_filter,
        profiler_extensions);
    return *captured_;
  }

 private:
  static std::shared_ptr<Captured> captured_;
};

std::shared_ptr<Captured> XpuActivityFilterTest::captured_;

const ProfilerExtensionMap kKernelMetrics{
    {ActivityType::XPU, {{"KERNEL_METRICS", "GpuTime,XVE_ACTIVE"}}}};

bool containsAll(
    const std::string& text,
    std::initializer_list<const char*> parts) {
  return std::all_of(parts.begin(), parts.end(), [&](const char* part) {
    return text.find(part) != std::string::npos;
  });
}

} // namespace

TEST_F(XpuActivityFilterTest, SyncSelectionFollowsDefaultsAndFilters) {
  // A bare ProfilerActivity.XPU collects the group defaults, now with sync.
  const auto defaults =
      prepareAndCapture({ActivityType::CPU, ActivityType::XPU}, {}).activities;
  EXPECT_TRUE(defaults.contains(KinetoType::XPU_SYNC));

  // Naming XPU_SYNC selects it and nothing else. Before it joined the XPU
  // group the name was unrecognised and became an XPUPTI_PROFILER_METRICS
  // entry, which enabled XPU_SCOPE_PROFILER plus CONCURRENT_KERNEL.
  const auto sync_only =
      prepareAndCapture(
          {ActivityType::XPU}, {{ActivityType::XPU, {"XPU_SYNC"}}})
          .activities;
  EXPECT_TRUE(sync_only.contains(KinetoType::XPU_SYNC));
  EXPECT_FALSE(sync_only.contains(KinetoType::XPU_SCOPE_PROFILER));
  EXPECT_FALSE(sync_only.contains(KinetoType::CONCURRENT_KERNEL));
  EXPECT_FALSE(sync_only.contains(KinetoType::XPU_RUNTIME));

  // An activity name mixed with a metric name still splits: the activity is
  // collected, the metric adds the scope profiler and the kernel view.
  const auto sync_and_metrics =
      prepareAndCapture(
          {ActivityType::XPU}, {{ActivityType::XPU, {"XPU_SYNC", "GpuTime"}}})
          .activities;
  EXPECT_TRUE(sync_and_metrics.contains(KinetoType::XPU_SYNC));
  EXPECT_TRUE(sync_and_metrics.contains(KinetoType::XPU_SCOPE_PROFILER));
  EXPECT_TRUE(sync_and_metrics.contains(KinetoType::CONCURRENT_KERNEL));

  // An empty dict still opts out of the whole XPU group.
  const auto nothing =
      prepareAndCapture({ActivityType::XPU}, {{ActivityType::XPU, {}}})
          .activities;
  EXPECT_FALSE(nothing.contains(KinetoType::XPU_SYNC));

  // It needs a PTI view, so it is in the XPU group only and deliberately not
  // in the CPU group.
  const auto cpu_only = prepareAndCapture({ActivityType::CPU}, {}).activities;
  EXPECT_FALSE(cpu_only.contains(KinetoType::XPU_SYNC));

  // Unrecognised names become metric names, which PyTorch cannot validate,
  // so a misspelling collects no synchronization events.
  const auto typo =
      prepareAndCapture(
          {ActivityType::XPU}, {{ActivityType::XPU, {"XPU_SYNCH"}}})
          .activities;
  EXPECT_FALSE(typo.contains(KinetoType::XPU_SYNC));
}

TEST_F(XpuActivityFilterTest, KernelMetricsKeepDefaultsAndAddKernels) {
  const auto unfiltered =
      prepareAndCapture({ActivityType::XPU}, {}, kKernelMetrics).activities;
  EXPECT_TRUE(unfiltered.contains(KinetoType::XPU_SYNC));
  EXPECT_TRUE(unfiltered.contains(KinetoType::XPU_SCOPE_PROFILER));
  EXPECT_TRUE(unfiltered.contains(KinetoType::CONCURRENT_KERNEL));

  // The activity type list selects activities only; kernels come with the
  // metrics either way.
  const auto sync_only = prepareAndCapture(
                             {ActivityType::XPU},
                             {{ActivityType::XPU, {"XPU_SYNC"}}},
                             kKernelMetrics)
                             .activities;
  EXPECT_TRUE(sync_only.contains(KinetoType::XPU_SYNC));
  EXPECT_FALSE(sync_only.contains(KinetoType::XPU_RUNTIME));
  EXPECT_TRUE(sync_only.contains(KinetoType::XPU_SCOPE_PROFILER));
  EXPECT_TRUE(sync_only.contains(KinetoType::CONCURRENT_KERNEL));
}

TEST_F(XpuActivityFilterTest, KernelMetricsRejectWhatTheyCannotHonour) {
  // With the metrics in a field of their own, every listed name must be an
  // activity type.
  EXPECT_THROW(
      prepareAndCapture(
          {ActivityType::XPU},
          {{ActivityType::XPU, {"GpuCoreClocks"}}},
          kKernelMetrics),
      c10::Error);
  EXPECT_THROW(
      prepareAndCapture(
          {ActivityType::XPU},
          {},
          {{ActivityType::XPU, {{"KERNEL_METRICS", ""}}}}),
      c10::ValueError);
  EXPECT_THROW(
      prepareAndCapture(
          {ActivityType::XPU},
          {},
          {{ActivityType::XPU, {{"PERFORMANCE_METRICS", "GpuTime"}}}}),
      c10::NotImplementedError);
  EXPECT_THROW(
      prepareAndCapture(
          {ActivityType::CPU},
          {},
          {{ActivityType::CPU, {{"KERNEL_METRICS", "GpuTime"}}}}),
      c10::NotImplementedError);
}

TEST_F(XpuActivityFilterTest, ForwardsConfigsOfTracedGroupsOnly) {
  const ProfilerExtensionMap::mapped_type sampling{
      {"PERFORMANCE_METRICS", "metric_a"}};
  EXPECT_EQ(
      prepareAndCapture(
          {ActivityType::CPU}, {}, {{ActivityType::CPU, sampling}})
          .performance_metric_names,
      std::vector<std::string>{"metric_a"});
  EXPECT_TRUE(prepareAndCapture(
                  {ActivityType::CPU}, {}, {{ActivityType::CUDA, sampling}})
                  .performance_metric_names.empty());

  // Kineto takes one flat config, so two traced groups cannot set one entry.
  EXPECT_THROW(
      prepareAndCapture(
          {ActivityType::CPU, ActivityType::CUDA},
          {},
          {{ActivityType::CPU, sampling}, {ActivityType::CUDA, sampling}}),
      c10::ValueError);
}

TEST_F(XpuActivityFilterTest, MetricNamesInTheListWarnOfDeprecation) {
  CapturingWarningHandler handler;
  const c10::WarningUtils::WarningHandlerGuard guard(&handler);
  // The warning is raised once per process otherwise.
  const c10::WarningUtils::WarnAlways warn_always(true);

  const auto captured = prepareAndCapture(
      {ActivityType::XPU}, {{ActivityType::XPU, {"GpuTime"}}});
  EXPECT_TRUE(captured.activities.contains(KinetoType::XPU_SCOPE_PROFILER));
  EXPECT_TRUE(std::any_of(
      handler.messages.begin(),
      handler.messages.end(),
      [](const std::string& message) {
        return containsAll(
            message,
            {"is deprecated and will be removed",
             "GpuTime",
             "KernelMetricsConfig",
             "drops the default XPU activities"});
      }));
}

#endif // USE_KINETO
