#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>

#include <algorithm>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

#include <ATen/Context.h>
#include <ATen/detail/HPUHooksInterface.h>
#include <ATen/detail/MTIAHooksInterface.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/env.h>

namespace torch {

namespace profiler::impl::kineto {

// Here lies pain and `#ifdef USE_KINETO`

#ifdef USE_KINETO
namespace {

using ActivityTypeMap =
    std::unordered_map<libkineto::ActivityType, std::string>;

// clang-format off
const ActivityTypeMap kCpuTypes{
    {libkineto::ActivityType::CPU_OP,                "CPU_OP"},
    {libkineto::ActivityType::CPU_INSTANT_EVENT,     "CPU_INSTANT_EVENT"},
    {libkineto::ActivityType::USER_ANNOTATION,       "USER_ANNOTATION"},
    {libkineto::ActivityType::EXTERNAL_CORRELATION,  "EXTERNAL_CORRELATION"},
    {libkineto::ActivityType::XPU_RUNTIME,           "XPU_RUNTIME"},
    {libkineto::ActivityType::XPU_DRIVER,            "XPU_DRIVER"},
    {libkineto::ActivityType::CUDA_RUNTIME,          "CUDA_RUNTIME"},
    {libkineto::ActivityType::CUDA_DRIVER,           "CUDA_DRIVER"},
    {libkineto::ActivityType::PYTHON_FUNCTION,       "PYTHON_FUNCTION"},
    {libkineto::ActivityType::PRIVATEUSE1_RUNTIME,   "PRIVATEUSE1_RUNTIME"},
    {libkineto::ActivityType::PRIVATEUSE1_DRIVER,    "PRIVATEUSE1_DRIVER"},
};

const ActivityTypeMap kCudaTypes{
    {libkineto::ActivityType::GPU_MEMCPY,            "GPU_MEMCPY"},
    {libkineto::ActivityType::GPU_MEMSET,            "GPU_MEMSET"},
    {libkineto::ActivityType::GPU_USER_ANNOTATION,   "GPU_USER_ANNOTATION"},
    {libkineto::ActivityType::CONCURRENT_KERNEL,     "CONCURRENT_KERNEL"},
    // CUDA_RUNTIME and CUDA_DRIVER appear in both kCpuTypes and kCudaTypes.
    {libkineto::ActivityType::CUDA_RUNTIME,          "CUDA_RUNTIME"},
    {libkineto::ActivityType::CUDA_DRIVER,           "CUDA_DRIVER"},
    {libkineto::ActivityType::OVERHEAD,              "OVERHEAD"},
};

const ActivityTypeMap kXpuTypes{
    {libkineto::ActivityType::GPU_MEMCPY,            "GPU_MEMCPY"},
    {libkineto::ActivityType::GPU_MEMSET,            "GPU_MEMSET"},
    {libkineto::ActivityType::GPU_USER_ANNOTATION,   "GPU_USER_ANNOTATION"},
    {libkineto::ActivityType::CONCURRENT_KERNEL,     "CONCURRENT_KERNEL"},
    // XPU_RUNTIME and XPU_DRIVER appear in both kCpuTypes and kXpuTypes.
    {libkineto::ActivityType::XPU_RUNTIME,           "XPU_RUNTIME"},
    {libkineto::ActivityType::XPU_DRIVER,            "XPU_DRIVER"},
    {libkineto::ActivityType::XPU_SYNC,              "XPU_SYNC"},
    {libkineto::ActivityType::OVERHEAD,              "OVERHEAD"},
};

const ActivityTypeMap kMtiaTypes{
    {libkineto::ActivityType::MTIA_CCP_EVENTS,       "MTIA_CCP_EVENTS"},
    {libkineto::ActivityType::MTIA_RUNTIME,          "MTIA_RUNTIME"},
    {libkineto::ActivityType::MTIA_INSIGHT,          "MTIA_INSIGHT"},
    {libkineto::ActivityType::MTIA_COUNTERS,         "MTIA_COUNTERS"},
    {libkineto::ActivityType::COLLECTIVE_COMM,       "COLLECTIVE_COMM"},
};

const ActivityTypeMap kHpuTypes{
    {libkineto::ActivityType::HPU_OP,                "HPU_OP"},
};

const ActivityTypeMap kPrivateUse1Types{
    {libkineto::ActivityType::GPU_MEMCPY,            "GPU_MEMCPY"},
    {libkineto::ActivityType::GPU_MEMSET,            "GPU_MEMSET"},
    {libkineto::ActivityType::GPU_USER_ANNOTATION,   "GPU_USER_ANNOTATION"},
    {libkineto::ActivityType::CONCURRENT_KERNEL,     "CONCURRENT_KERNEL"},
    // PRIVATEUSE1_RUNTIME appears in both kCpuTypes and kPrivateUse1Types.
    {libkineto::ActivityType::PRIVATEUSE1_RUNTIME,   "PRIVATEUSE1_RUNTIME"},
    {libkineto::ActivityType::PRIVATEUSE1_DRIVER,    "PRIVATEUSE1_DRIVER"},
};
// clang-format on

// Given a named activity type map and a set of requested name strings,
// return the matching subset of activity types.
std::unordered_set<libkineto::ActivityType> filterActivities(
    const ActivityTypeMap& defaults,
    const std::unordered_set<std::string>& requested,
    std::string_view hint = {}) {
  std::unordered_set<libkineto::ActivityType> result;
  for (const auto& name : requested) {
    bool found = false;
    for (const auto& [type, type_name] : defaults) {
      if (type_name == name) {
        result.insert(type);
        found = true;
        break;
      }
    }
    TORCH_CHECK(
        found, "Unknown or non-member activity type name: '", name, "'", hint);
  }
  return result;
}

// The result of splitting the names requested for the XPU group into the
// activity types to collect and the hardware metric names to measure.
struct XpuActivityRequest {
  std::unordered_set<libkineto::ActivityType> activity_types;
  std::vector<std::string> metrics;
};

// For the XPU group, a requested name that matches an activity type acts as a
// fine-grained activity filter (see #176351). A name that matches none is the
// deprecated way of requesting an XPU hardware metric; such a name cannot be
// validated here - metrics are device-specific and known only to PTI/the
// driver - so it is forwarded rather than rejected.
XpuActivityRequest splitXpuRequest(
    const ActivityTypeMap& defaults,
    const std::unordered_set<std::string>& requested) {
  XpuActivityRequest out;
  for (const auto& name : requested) {
    const auto it =
        std::find_if(defaults.begin(), defaults.end(), [&](const auto& entry) {
          return entry.second == name;
        });
    if (it != defaults.end()) {
      out.activity_types.insert(it->first);
    } else {
      out.metrics.push_back(name);
    }
  }
  return out;
}

// Config keys of KernelMetricsConfig, which this shim consumes, and of
// PerformanceMetricsConfig, whose time-based sampling XPU does not implement.
constexpr const char* kKernelMetricsKey = "KERNEL_METRICS";
constexpr const char* kPerformanceMetricsKey = "PERFORMANCE_METRICS";

// Comma-separated metric names, as KernelMetricsConfig serializes them.
std::vector<std::string> splitMetricNames(const std::string& value) {
  std::vector<std::string> names;
  for (const auto name : c10::split(value, ',')) {
    if (!name.empty()) {
      names.emplace_back(name);
    }
  }
  return names;
}

// Build the Kineto config fragment that drives the XPUPTI scope profiler.
// The keys mirror those parsed by XpuptiScopeProfilerConfig::handleOption.
// Per-kernel (PTI_METRICS_SCOPE_AUTO_KERNEL) is the only mode PTI supports
// today, so it is intentionally hardcoded here; custom_profiler_config is the
// reserved channel for a future PTI_METRICS_SCOPE_USER mode.
std::string buildXpuScopeConfig(const std::vector<std::string>& metrics) {
  std::stringstream configss;
  configss << "XPUPTI_PROFILER_METRICS=";
  bool first = true;
  for (const auto& metric : metrics) {
    if (!first) {
      configss << ',';
    }
    configss << metric;
    first = false;
  }
  configss << "\nXPUPTI_PROFILER_ENABLE_PER_KERNEL=true\n";
  return std::move(configss).str();
}

// Resolve what the XPU group asked for: which activity types to collect, and
// which hardware metrics the XPUPTI scope profiler should measure. Returns the
// Kineto config fragment driving that profiler, empty without metrics.
std::string configureXpuActivities(
    const ActivityFilter& activity_filter,
    const ProfilerExtensionMap& profiler_extensions,
    std::set<libkineto::ActivityType>& k_activities) {
  // Hardware metrics requested through KernelMetricsConfig, which keeps them
  // apart from the activity types and preserves their order.
  const auto extensions_it =
      profiler_extensions.find(torch::autograd::profiler::ActivityType::XPU);
  std::vector<std::string> metrics;
  bool has_metrics_config = false;
  if (extensions_it != profiler_extensions.end()) {
    const auto& extensions = extensions_it->second;
    TORCH_CHECK_NOT_IMPLEMENTED(
        !extensions.contains(kPerformanceMetricsKey),
        "PerformanceMetricsConfig (time-based sampling) is not supported for "
        "ProfilerActivity.XPU yet; use KernelMetricsConfig for per-kernel "
        "hardware metrics");
    const auto metrics_it = extensions.find(kKernelMetricsKey);
    if (metrics_it != extensions.end()) {
      metrics = splitMetricNames(metrics_it->second);
      TORCH_CHECK_VALUE(
          !metrics.empty(),
          "KernelMetricsConfig for ProfilerActivity.XPU lists no metric name");
      has_metrics_config = true;
    }
  }
  const auto filter_it =
      activity_filter.find(torch::autograd::profiler::ActivityType::XPU);
  if (filter_it == activity_filter.end()) {
    for (const auto& entry : kXpuTypes) {
      k_activities.insert(entry.first);
    }
  } else if (has_metrics_config) {
    // The metrics have a field of their own, so every name must be an
    // activity type.
    const auto types = filterActivities(
        kXpuTypes,
        filter_it->second,
        "; hardware metric names belong in KernelMetricsConfig.metric_names");
    k_activities.insert(types.begin(), types.end());
  } else {
    const auto request = splitXpuRequest(kXpuTypes, filter_it->second);
    k_activities.insert(
        request.activity_types.begin(), request.activity_types.end());
    if (!request.metrics.empty()) {
      // TORCH_WARN_ONCE, since a DeprecationWarning is hidden unless raised
      // from __main__. A dict request collects exactly what it lists (see
      // #176351), so a list of only metric names also drops the defaults.
      TORCH_WARN_ONCE(
          "Requesting XPU hardware metrics by name in the activity type list "
          "is deprecated and will be removed in a future release. These names "
          "were taken as metric names: ",
          c10::Join(", ", request.metrics),
          ". Pass them as ProfilerActivityConfig(activity_types=<the activity "
          "type names, or None for the defaults>, profiler_configs=["
          "KernelMetricsConfig(metric_names=[...])]) instead.",
          request.activity_types.empty()
              ? " This request also drops the default XPU activities, which "
                "KernelMetricsConfig keeps."
              : "");
      metrics = request.metrics;
    }
  }
  if (metrics.empty()) {
    return {};
  }
  k_activities.insert(libkineto::ActivityType::XPU_SCOPE_PROFILER);
  // The scope profiler attaches metrics to individual kernels and needs the
  // kernel activity view to correlate them to real kernel timestamps; without
  // it the metrics would land on synthetic timestamps. Enabling metrics
  // therefore implies collecting kernels.
  k_activities.insert(libkineto::ActivityType::CONCURRENT_KERNEL);
  return buildXpuScopeConfig(metrics);
}

} // namespace
#endif // USE_KINETO

static_assert(
    std::is_trivial_v<DeviceAndResource>,
    "Kineto specific details should be in `kineto_ids`.");

const DeviceAndResource kineto_ids() {
#ifdef USE_KINETO
  return {
      /*device=*/libkineto::processId(),
      /*resource=*/libkineto::systemThreadId()};
#else
  return {};
#endif // USE_KINETO
}

void addMetadata(
    activity_t* activity,
    const std::string& key,
    const std::string& value,
    bool quote) {
#ifdef USE_KINETO
  if (quote) {
    activity->addMetadataQuoted(key, value);
  } else {
    activity->addMetadata(key, value);
  }
#endif // USE_KINETO
}

TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
#ifdef USE_KINETO
    : cpu_trace_(std::make_unique<libkineto::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gpuOpCount = -1;
  cpu_trace_->span.name = name;
}
#else
{
}
#endif // USE_KINETO

activity_t* TraceWrapper::addCPUActivity(
    const std::string& name,
    const libkineto::ActivityType type,
    const DeviceAndResource device_and_resource,
    const uint64_t correlation_id,
    const int64_t start_time,
    const int64_t end_time) {
#ifdef USE_KINETO
  TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");
  cpu_trace_->emplace_activity(cpu_trace_->span, type, name);
  auto& act = libkineto::CpuTraceBuffer::toRef(cpu_trace_->activities.back());
  act.device = device_and_resource.device;
  act.resource = device_and_resource.resource;
  act.id = static_cast<int32_t>(correlation_id);
  act.startTime = start_time;
  if (type != libkineto::ActivityType::CPU_INSTANT_EVENT) {
    act.endTime = end_time;
  }
  return cpu_trace_->activities.back().get();
#else
  return nullptr;
#endif // USE_KINETO
}

void TraceWrapper::transferCpuTrace(int64_t end_time) {
#ifdef USE_KINETO
  cpu_trace_->span.endTime = end_time;
  libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace_));
#endif // USE_KINETO
}

TraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return cpu_trace_ != nullptr;
#else
  return false;
#endif // USE_KINETO
}

ActivityTraceWrapper::ActivityTraceWrapper(
    std::unique_ptr<interface_trace_t>&& trace)
    : trace_(std::move(trace)) {}

ActivityTraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return trace_ != nullptr;
#else
  return false;
#endif // USE_KINETO
}

void ActivityTraceWrapper::save(const std::string& path) {
#ifdef USE_KINETO
  TORCH_CHECK(!saved_, "Trace is already saved.");
  TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
#else
  TORCH_CHECK(
      false,
      "Saving a trace requires using torch.profiler with Kineto support (USE_KINETO=1)");
#endif // USE_KINETO
}

ActivitySet supportedDeviceActivities() {
  using torch::autograd::profiler::ActivityType;
  ActivitySet activities;
#ifdef USE_KINETO
#if defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (at::getNumGPUs() > 0) {
    activities.insert(ActivityType::CUDA);
  }
#endif // defined(HAS_CUPTI) || defined(HAS_ROCTRACER)
  if (at::hasXPU()) {
    activities.insert(ActivityType::XPU);
  }
  // Asking for hooks that are not registered yet caches stub hooks for the
  // whole process, so an out-of-tree backend that loads later would never be
  // seen. Check the registry first.
  if (at::detail::isMTIAHooksBuilt() && at::hasMTIA()) {
    activities.insert(ActivityType::MTIA);
  }
  if (at::detail::isHPUHooksBuilt() && at::hasHPU()) {
    activities.insert(ActivityType::HPU);
  }
  if (c10::is_privateuse1_backend_registered()) {
    activities.insert(ActivityType::PrivateUse1);
  }
#endif // USE_KINETO
  return activities;
}

bool collectivesProfilerExists() {
#if defined(KINETO_HAS_HCCL_PROFILER)
  return true;
#endif
  const auto val =
      c10::utils::get_env("TORCH_PROFILER_ENABLE_COLLECTIVE_PROFILING");
  return val == "1";
}

#ifdef USE_KINETO
static const std::string setTraceID(const std::string& trace_id) {
  if (trace_id.empty()) {
    return "";
  }
  std::stringstream configss;
  configss << "REQUEST_TRACE_ID=" << trace_id << '\n';
  configss << "REQUEST_GROUP_TRACE_ID=" << trace_id << '\n';
  return std::move(configss).str();
}

static const std::string appendCustomConfig(
    const std::string& config,
    const std::string& custom_profiler_config) {
  if (custom_profiler_config.empty()) {
    return config;
  }
  std::stringstream configss;
  configss << config;
  configss << "CUSTOM_CONFIG=" << custom_profiler_config << '\n';
  return std::move(configss).str();
}
#endif

void prepareTrace(
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config,
    const std::string& trace_id,
    const ActivityFilter& activity_filter,
    const ProfilerExtensionMap& profiler_extensions) {
#ifdef USE_KINETO
  libkineto::api().resetKinetoTLS();
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(
        /*cpuOnly=*/supportedDeviceActivities().empty(), /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> k_activities;
  std::string extraConfig;

  // Helper: insert activity types for a group, applying the filter if present.
  auto insertActivities = [&](torch::autograd::profiler::ActivityType group,
                              const ActivityTypeMap& defaults) {
    auto filter_it = activity_filter.find(group);
    if (filter_it != activity_filter.end()) {
      auto filtered = filterActivities(defaults, filter_it->second);
      k_activities.insert(filtered.begin(), filtered.end());
    } else {
      for (const auto& [type, name] : defaults) {
        k_activities.insert(type);
      }
    }
  };

  const bool has_cpu_activity =
      activities.contains(torch::autograd::profiler::ActivityType::CPU);
  const bool has_mtia_activity =
      activities.contains(torch::autograd::profiler::ActivityType::MTIA);
  const bool has_mtia_activity_filter =
      activity_filter.contains(torch::autograd::profiler::ActivityType::MTIA);
  const bool has_collectives_profiler = collectivesProfilerExists();
  const bool has_device_activity =
      std::any_of(activities.begin(), activities.end(), [](auto activity) {
        return activity != torch::autograd::profiler::ActivityType::CPU;
      });

  if (has_cpu_activity) {
    insertActivities(torch::autograd::profiler::ActivityType::CPU, kCpuTypes);
  }
  if (activities.contains(torch::autograd::profiler::ActivityType::XPU)) {
    extraConfig = configureXpuActivities(
        activity_filter, profiler_extensions, k_activities);
  }
  if (has_mtia_activity) {
    TORCH_CHECK(
        !has_mtia_activity_filter || config.custom_profiler_config.empty(),
        "`custom_profiler_config` cannot be combined with an MTIA "
        "`activity_filter`; use only one to select MTIA activities.");
    if (has_mtia_activity_filter || config.custom_profiler_config.empty()) {
      insertActivities(
          torch::autograd::profiler::ActivityType::MTIA, kMtiaTypes);
    } else {
      k_activities.insert(libkineto::ActivityType::COLLECTIVE_COMM);
      if (config.custom_profiler_config.find("disable_runtime_events") ==
          std::string::npos) {
        k_activities.insert(libkineto::ActivityType::MTIA_RUNTIME);
      } else {
        LOG(INFO) << "Disabling MTIA runtime events";
      }
      if (config.custom_profiler_config.find("disable_ccp_events") ==
          std::string::npos) {
        k_activities.insert(libkineto::ActivityType::MTIA_CCP_EVENTS);
      } else {
        LOG(INFO) << "Disabling MTIA CCP events";
      }
      if (config.custom_profiler_config.find("disable_insight_events") ==
          std::string::npos) {
        k_activities.insert(libkineto::ActivityType::MTIA_INSIGHT);
      } else {
        LOG(INFO) << "Disabling MTIA insight events";
      }
      if (config.custom_profiler_config.find("disable_counter_events") ==
          std::string::npos) {
        k_activities.insert(libkineto::ActivityType::MTIA_COUNTERS);
      } else {
        LOG(INFO) << "Disabling MTIA counter events";
      }
    }
    if (!has_collectives_profiler) {
      k_activities.erase(libkineto::ActivityType::COLLECTIVE_COMM);
    }
  }
  if (activities.contains(torch::autograd::profiler::ActivityType::HPU)) {
    insertActivities(torch::autograd::profiler::ActivityType::HPU, kHpuTypes);
  }
  if (activities.contains(torch::autograd::profiler::ActivityType::CUDA)) {
    insertActivities(torch::autograd::profiler::ActivityType::CUDA, kCudaTypes);
    if (config.enable_cuda_sync_events || get_cuda_sync_enabled()) {
      LOG(INFO) << "Enabling CUDA Sync Events";
      k_activities.insert(libkineto::ActivityType::CUDA_SYNC);
    }
  }
  if (!has_mtia_activity && has_collectives_profiler) {
    k_activities.insert(libkineto::ActivityType::COLLECTIVE_COMM);
  }
  if (activities.contains(
          torch::autograd::profiler::ActivityType::PrivateUse1)) {
    insertActivities(
        torch::autograd::profiler::ActivityType::PrivateUse1,
        kPrivateUse1Types);
  }

  const std::string traceIdStr = setTraceID(trace_id);
  // Kineto takes one flat configuration, so the configs of the traced groups
  // must not set the same key; a group that is not traced passes on nothing.
  std::unordered_set<std::string> forwarded_keys;
  for (const auto& [activity, extensions] : profiler_extensions) {
    for (const auto& [name, value] : extensions) {
      if (name == kKernelMetricsKey) {
        // Consumed by configureXpuActivities; Kineto does not know the key.
        TORCH_CHECK_NOT_IMPLEMENTED(
            activity == torch::autograd::profiler::ActivityType::XPU,
            "KernelMetricsConfig is only supported for ProfilerActivity.XPU");
        if (!activities.contains(activity)) {
          TORCH_WARN(
              "KernelMetricsConfig is ignored because ProfilerActivity.XPU is "
              "not traced");
        }
        continue;
      }
      if (!activities.contains(activity)) {
        continue;
      }
      TORCH_CHECK_VALUE(
          forwarded_keys.insert(name).second,
          "More than one activity group sets the profiler config entry ",
          name,
          "; attach the config to one group only");
      extraConfig += name;
      extraConfig += '=';
      extraConfig += value;
      extraConfig += '\n';
    }
  }
  // Registration fixes only whether libkineto can trace a GPU; this key says
  // whether this trace may trace any device. A trace that requested no device
  // activity leaves device tracers alone, since another tool may own them.
  // Any device activity allows it, not only CUDA or XPU: child profilers for
  // the other devices are expected to honor the key too, and libkineto emits
  // its default device rows only when it may trace a GPU. The key goes first
  // because libkineto stops parsing at the first malformed line, and a custom
  // config can contain one.
  const std::string deviceTracingConfig = has_device_activity
      ? "DEVICE_TRACING_ENABLED=true\n"
      : "DEVICE_TRACING_ENABLED=false\n";
  const std::string configStr = deviceTracingConfig +
      appendCustomConfig(traceIdStr, config.custom_profiler_config) +
      extraConfig;

  libkineto::api().activityProfiler().prepareTrace(k_activities, configStr);
#endif // USE_KINETO
}

void toggleCollectionDynamic(const bool enable) {
#ifdef USE_KINETO
  // TODO: We may want to consider adding another input arg for this function
  // if we want to support turning off certain devices and keeping others on.
  // For now, we can keep it simple and have it turn off all tracing of "CUDA"
  // devices
  libkineto::api().activityProfiler().toggleCollectionDynamic(enable);
#endif // USE_KINETO
}

void startTrace() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().startTrace();
#endif // USE_KINETO
}

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{
#ifdef USE_KINETO
      libkineto::api().activityProfiler().stopTrace()
#else
      std::make_unique<interface_trace_t>()
#endif // USE_KINETO
  };
}

void pushCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().pushCorrelationId(correlation_id);
#endif // USE_KINETO
}

void pushUserCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().pushUserCorrelationId(correlation_id);
#endif // USE_KINETO
}

void popCorrelationId() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().popCorrelationId();
#endif // USE_KINETO
}

void popUserCorrelationId() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().popUserCorrelationId();
#endif // USE_KINETO
}

void recordThreadInfo() {
#ifdef USE_KINETO
  libkineto::api().activityProfiler().recordThreadInfo();
#endif // USE_KINETO
}

void logInvariantViolation(
    const std::string& assertion,
    const std::string& error,
    const std::string& profile_id,
    const std::string& group_profile_id) {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().logInvariantViolation(
        profile_id, assertion, error, group_profile_id);
  }
#endif // USE_KINETO
}

} // namespace profiler::impl::kineto

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type) {
  // PrivateUse1 kineto backend reuse some ActivityTypes,
  // If PrivateUse1 backend is enabled, this should return
  // c10::DeviceType::PrivateUse1.
  auto device_type_privateuse1_or = [](c10::DeviceType device_type) {
    return c10::is_privateuse1_backend_registered()
        ? c10::DeviceType::PrivateUse1
        : device_type;
  };

  switch (activity_type) {
    case libkineto::ActivityType::GPU_MEMCPY:
    case libkineto::ActivityType::GPU_MEMSET:
    case libkineto::ActivityType::CONCURRENT_KERNEL:
    case libkineto::ActivityType::GPU_USER_ANNOTATION:
#if defined(USE_XPU)
      return device_type_privateuse1_or(c10::DeviceType::XPU);
#endif
      [[fallthrough]];
    case libkineto::ActivityType::CUDA_SYNC:
    case libkineto::ActivityType::CUDA_PROFILER_RANGE:
    case libkineto::ActivityType::HARDWARE_COUNTERS:
      return device_type_privateuse1_or(c10::DeviceType::CUDA);
    // TODO: T151322015
    case libkineto::ActivityType::MTIA_CCP_EVENTS:
    case libkineto::ActivityType::MTIA_INSIGHT:
    case libkineto::ActivityType::MTIA_COUNTERS:
      return device_type_privateuse1_or(c10::DeviceType::MTIA);
    case libkineto::ActivityType::XPU_SCOPE_PROFILER:
    case libkineto::ActivityType::XPU_SYNC:
      return c10::DeviceType::XPU;
    case libkineto::ActivityType::HPU_OP:
      return c10::DeviceType::HPU;
    case libkineto::ActivityType::CPU_OP:
    case libkineto::ActivityType::USER_ANNOTATION:
    case libkineto::ActivityType::EXTERNAL_CORRELATION:
    case libkineto::ActivityType::CUDA_RUNTIME:
    case libkineto::ActivityType::XPU_RUNTIME:
    case libkineto::ActivityType::XPU_DRIVER:
    case libkineto::ActivityType::CPU_INSTANT_EVENT:
    case libkineto::ActivityType::GLOW_RUNTIME:
    case libkineto::ActivityType::MTIA_RUNTIME:
    case libkineto::ActivityType::PYTHON_FUNCTION:
    case libkineto::ActivityType::CUDA_DRIVER:
    case libkineto::ActivityType::PRIVATEUSE1_RUNTIME:
    case libkineto::ActivityType::PRIVATEUSE1_DRIVER:
    case libkineto::ActivityType::COLLECTIVE_COMM:
    case libkineto::ActivityType::OVERHEAD:
      return c10::DeviceType::CPU;
    default: {
      TORCH_WARN(
          "Unknown activity type (",
          (uint8_t)activity_type,
          "), assuming CPU device");
      return c10::DeviceType::CPU;
    }
  }
}

void addMetadataJson(const std::string& key, const std::string& value) {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
#else
  LOG(WARNING) << "Adding profiling metadata requires using "
               << "torch.profiler with Kineto support (USE_KINETO=1)";
#endif // USE_KINETO
}

void profilerStep() {
#ifdef USE_KINETO
  libkineto::api().initProfilerIfRegistered();

  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().step();
  } else {
    VLOG(1) << "Profiler is not initialized: skipping step() invocation";
  }
#endif // USE_KINETO
}

bool isKinetoStopped() {
#ifdef USE_KINETO
  if (libkineto::api().isProfilerInitialized()) {
    return libkineto::api().activityProfiler().isStopped();
  }
  return false;
#else
  return false;
#endif
}

} // namespace autograd::profiler

} // namespace torch
