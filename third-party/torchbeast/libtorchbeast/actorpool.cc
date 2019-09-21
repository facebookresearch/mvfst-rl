/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include <grpc++/grpc++.h>

#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "../nest/nest/nest.h"
#include "../nest/nest/nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<torch::Tensor> TensorNest;

TensorNest batch(std::vector<TensorNest> tensors, int64_t batch_dim) {
  return TensorNest::map(
      [batch_dim](std::vector<torch::Tensor> v) {
        return torch::cat(v, batch_dim);
      },
      tensors);
}

struct ClosedBatchingQueue : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Enable a few standard Python exceptions.
namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(runtime_error, PyExc_RuntimeError)
PYBIND11_RUNTIME_EXCEPTION(timeout_error, PyExc_TimeoutError)
PYBIND11_RUNTIME_EXCEPTION(connection_error, PyExc_ConnectionError)
}  // namespace pybind11

struct Empty {};

template <typename T = Empty>
class BatchingQueue {
 public:
  struct QueueItem {
    TensorNest tensors;
    T payload;
  };
  BatchingQueue(int64_t batch_dim, int64_t minimum_batch_size,
                int64_t maximum_batch_size,
                std::optional<int> timeout_ms = std::nullopt,
                bool check_inputs = true)
      : batch_dim_(batch_dim),
        minimum_batch_size_(
            minimum_batch_size > 0
                ? minimum_batch_size
                : throw py::value_error("Min batch size must be >= 1")),
        maximum_batch_size_(
            maximum_batch_size > 0
                ? maximum_batch_size
                : throw py::value_error("Max batch size must be >= 1")),
        timeout_(timeout_ms),
        is_closed_(false),
        check_inputs_(check_inputs) {}

  int64_t size() const {
    std::unique_lock<std::mutex> lock(mu_);
    return deque_.size();
  }

  uint64_t minimum_batch_size() const { return minimum_batch_size_; }

  uint64_t maximum_batch_size() const { return maximum_batch_size_; }

  void enqueue(QueueItem item) {
    if (check_inputs_) {
      bool is_empty = true;

      item.tensors.for_each([this, &is_empty](const torch::Tensor& tensor) {
        is_empty = false;

        if (tensor.dim() <= batch_dim_) {
          throw py::value_error(
              "Enqueued tensors must have more than batch_dim == " +
              std::to_string(batch_dim_) + " dimensions, but got " +
              std::to_string(tensor.dim()));
        }
      });

      if (is_empty) {
        throw py::value_error("Cannot enqueue empty vector of tensors");
      }
    }

    bool should_notify = false;
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (is_closed_) {
        throw ClosedBatchingQueue("Enqueue to closed queue");
      }

      // TODO: Consider a max_size at which this blocks.
      deque_.push_back(std::move(item));
      should_notify = deque_.size() >= minimum_batch_size_;
    }

    if (should_notify) {
      enough_inputs_.notify_one();
    }
  }

  std::pair<TensorNest, std::vector<T>> dequeue_many() {
    std::vector<TensorNest> tensors;
    std::vector<T> payloads;
    {
      std::unique_lock<std::mutex> lock(mu_);

      std::cv_status status = std::cv_status::no_timeout;
      while (((status == std::cv_status::timeout && deque_.empty()) ||
              (status == std::cv_status::no_timeout &&
               deque_.size() < minimum_batch_size_)) &&
             !is_closed_) {
        if (timeout_) {
          status = enough_inputs_.wait_for(lock, *timeout_);
        } else {
          enough_inputs_.wait(lock);
        }
      }
      if (is_closed_) {
        throw py::stop_iteration("Queue is closed");
      }
      const int64_t batch_size =
          std::min<int64_t>(deque_.size(), maximum_batch_size_);
      for (auto it = deque_.begin(), end = deque_.begin() + batch_size;
           it != end; ++it) {
        tensors.push_back(std::move(it->tensors));
        payloads.push_back(std::move(it->payload));
      }
      deque_.erase(deque_.begin(), deque_.begin() + batch_size);
    }
    return std::make_pair(batch(tensors, batch_dim_), std::move(payloads));
  }

  void close() {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (is_closed_) {
        throw py::runtime_error("Queue was closed already");
      }
      is_closed_ = true;
      deque_.clear();
    }
    enough_inputs_.notify_all();  // Wake up dequeues.
  }

 private:
  mutable std::mutex mu_;

  const int64_t batch_dim_;
  const uint64_t minimum_batch_size_;
  const uint64_t maximum_batch_size_;
  const std::optional<std::chrono::milliseconds> timeout_;

  std::condition_variable enough_inputs_;

  bool is_closed_ /* GUARDED_BY(mu_) */;
  std::deque<QueueItem> deque_ /* GUARDED_BY(mu_) */;

  const bool check_inputs_;
};

class DynamicBatcher {
 public:
  typedef std::promise<std::pair<std::shared_ptr<TensorNest>, int64_t>>
      BatchPromise;
  class Batch {
   public:
    Batch(int64_t batch_dim, TensorNest&& tensors,
          std::vector<BatchPromise>&& promises, bool check_outputs)
        : batch_dim_(batch_dim),
          inputs_(std::move(tensors)),
          promises_(std::move(promises)),
          check_outputs_(check_outputs) {}

    const TensorNest& get_inputs() { return inputs_; }

    void set_outputs(TensorNest outputs) {
      if (promises_.empty()) {
        // Batch has been set before.
        throw py::runtime_error("set_outputs called twice");
      }

      if (check_outputs_) {
        const int64_t expected_batch_size = promises_.size();

        outputs.for_each(
            [this, expected_batch_size](const torch::Tensor& tensor) {
              if (tensor.dim() <= batch_dim_) {
                std::stringstream ss;
                ss << "With batch dimension " << batch_dim_
                   << ", output shape must have at least " << batch_dim_ + 1
                   << " dimensions, but got " << tensor.sizes();
                throw py::value_error(ss.str());
              }
              if (tensor.sizes()[batch_dim_] != expected_batch_size) {
                throw py::value_error(
                    "Output shape must have the same batch "
                    "dimension as the input batch size. Expected: " +
                    std::to_string(expected_batch_size) + ". Observed: " +
                    std::to_string(tensor.sizes()[batch_dim_]));
              }
            });
      }

      auto shared_outputs = std::make_shared<TensorNest>(std::move(outputs));

      int64_t b = 0;
      for (auto& promise : promises_) {
        promise.set_value(std::make_pair(shared_outputs, b));
        ++b;
      }
      promises_.clear();
    }

   private:
    const int64_t batch_dim_;
    const TensorNest inputs_;
    std::vector<BatchPromise> promises_;

    const bool check_outputs_;
  };

  DynamicBatcher(int64_t batch_dim, int64_t minimum_batch_size,
                 int64_t maximum_batch_size,
                 std::optional<int> timeout_ms = std::nullopt,
                 bool check_outputs = true)
      : batching_queue_(batch_dim, minimum_batch_size, maximum_batch_size,
                        timeout_ms),
        batch_dim_(batch_dim),
        check_outputs_(check_outputs) {}

  TensorNest compute(TensorNest tensors) {
    BatchPromise promise;
    auto future = promise.get_future();

    batching_queue_.enqueue({std::move(tensors), std::move(promise)});

    std::future_status status = future.wait_for(std::chrono::seconds(10 * 60));
    if (status != std::future_status::ready) {
      throw py::timeout_error("Compute timeout reached.");
    }

    const std::pair<std::shared_ptr<TensorNest>, int64_t> pair = [&] {
      try {
        return future.get();
      } catch (const std::future_error& e) {
        if (closed_ && e.code() == std::future_errc::broken_promise) {
          throw ClosedBatchingQueue("Batching queue closed during compute");
        }
        throw;
      }
    }();

    return pair.first->map([
      batch_dim = batch_dim_, batch_entry = pair.second
    ](const torch::Tensor& t) {
      return t.slice(batch_dim, batch_entry, batch_entry + 1);
    });
  }

  std::shared_ptr<Batch> get_batch() {
    auto pair = batching_queue_.dequeue_many();
    return std::make_shared<Batch>(batch_dim_, std::move(pair.first),
                                   std::move(pair.second), check_outputs_);
  }

  int64_t size() const { return batching_queue_.size(); }
  void close() {
    closed_ = true;
    batching_queue_.close();
  }

 private:
  BatchingQueue<std::promise<std::pair<std::shared_ptr<TensorNest>, int64_t>>>
      batching_queue_;
  int64_t batch_dim_;
  std::atomic_bool closed_;

  bool check_outputs_;
};

class ActorPool {
 public:
  class ServiceImpl final : public rpcenv::ActorPoolServer::Service {
   public:
    ServiceImpl(int unroll_length,
                std::shared_ptr<BatchingQueue<>> learner_queue,
                std::shared_ptr<DynamicBatcher> inference_batcher,
                TensorNest initial_agent_state)
        : unroll_length_(unroll_length),
          learner_queue_(std::move(learner_queue)),
          inference_batcher_(std::move(inference_batcher)),
          initial_agent_state_(std::move(initial_agent_state)) {}

    uint64_t count() const { return count_; }

   private:
    virtual grpc::Status StreamingActor(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<rpcenv::Action, rpcenv::Step>* stream)
        override {
      std::cout << "StreamingActor initiated" << std::endl;
      if (learner_queue_) {
        train(stream);
      } else {
        test(stream);
      }
      return grpc::Status::OK;
    }

    void train(grpc::ServerReaderWriter<rpcenv::Action, rpcenv::Step>* stream) {
      rpcenv::Step step_pb;
      if (!stream->Read(&step_pb)) {
        throw py::connection_error("Initial read failed.");
      }

      TensorNest initial_agent_state = initial_agent_state_;

      TensorNest env_outputs = step_pb_to_nest(&step_pb);
      TensorNest compute_inputs(
          std::vector({env_outputs, initial_agent_state}));
      TensorNest all_agent_outputs =
          inference_batcher_->compute(compute_inputs);  // Copy.

      // Check this once per thread.
      if (!all_agent_outputs.is_vector()) {
        throw py::value_error("Expected agent output to be tuple");
      }
      if (all_agent_outputs.get_vector().size() != 2) {
        throw py::value_error(
            "Expected agent output to be (output, new_state) but got sequence "
            "of "
            "length " +
            std::to_string(all_agent_outputs.get_vector().size()));
      }
      TensorNest agent_state = all_agent_outputs.get_vector()[1];
      TensorNest agent_outputs = all_agent_outputs.get_vector()[0];

      TensorNest last(std::vector({env_outputs, agent_outputs}));

      // Create zero-padded env output with done==true to make a complete
      // rollout when needed.
      const bool is_batched = learner_queue_->maximum_batch_size() > 1;
      const auto& pad = is_batched ? rollout_zero_pad(last) : TensorNest();

      rpcenv::Action action_pb;
      std::vector<TensorNest> rollout;
      rollout.reserve(unroll_length_ + 1);
      bool reset = false;
      bool first = true;
      try {
        while (!reset) {
          rollout.push_back(std::move(last));

          for (int t = 1; t <= unroll_length_; ++t) {
            if (!first) {
              all_agent_outputs = inference_batcher_->compute(compute_inputs);
              agent_state = all_agent_outputs.get_vector()[1];
              agent_outputs = all_agent_outputs.get_vector()[0];
            } else {
              first = false;
            }

            int action = agent_outputs.front().item<int>();
            action_pb.set_action(action);
            stream->Write(action_pb);
            if (!stream->Read(&step_pb)) {
              // Client closed the stream. We infer this as env.reset() and put
              // whatever rollout we have into the learner queue with done set.
              auto& last_env_outputs = rollout.back().get_vector()[0];
              auto& last_done = last_env_outputs.get_vector()[2];
              last_done.front().fill_(1);
              reset = true;
              break;
            }

            env_outputs = step_pb_to_nest(&step_pb);
            compute_inputs =
                TensorNest(std::vector({env_outputs, agent_state}));

            last = TensorNest(std::vector({env_outputs, agent_outputs}));
            rollout.push_back(std::move(last));
          }

          // Add zero-padding if learner batch size > 1.
          // TODO: This requires appropriate masking in polybeast.py to avoid
          // any learning on the padded trajectory. Until that is fixed,
          // batch size 1 is recommended for environments with partial rollouts.
          while (is_batched && (rollout.size() < unroll_length_ + 1)) {
            rollout.push_back(pad);
          }
          last = rollout.back();

          // rollout.size() == 1 possible when client resets before a single
          // complete rollout. Handle this safely.
          if (rollout.size() > 1) {
            count_ += rollout.size() - 1;
            learner_queue_->enqueue({
                TensorNest(std::vector({batch(std::move(rollout), 0),
                                        std::move(initial_agent_state)})),
            });
          }
          rollout.clear();
          initial_agent_state = agent_state;  // Copy
        }
      } catch (const ClosedBatchingQueue& e) {
        // Thrown when inference_batcher_ and learner_queue_ are closed. Stop.
      }
    }

    void test(grpc::ServerReaderWriter<rpcenv::Action, rpcenv::Step>* stream) {
      rpcenv::Step step_pb;
      rpcenv::Action action_pb;
      if (!stream->Read(&step_pb)) {
        throw py::connection_error("Initial read failed.");
      }

      TensorNest env_outputs = step_pb_to_nest(&step_pb);
      TensorNest compute_inputs(
          std::vector({env_outputs, initial_agent_state_}));
      TensorNest all_agent_outputs =
          inference_batcher_->compute(compute_inputs);

      if (!env_outputs.is_vector() || env_outputs.get_vector().size() != 5) {
        throw py::value_error("Expected env output to be tuple of size 5");
      }
      if (!all_agent_outputs.is_vector() ||
          all_agent_outputs.get_vector().size() != 2) {
        throw py::value_error("Expected agent output to be tuple of size 2");
      }

      TensorNest agent_outputs = all_agent_outputs.get_vector()[0];
      TensorNest agent_state = all_agent_outputs.get_vector()[1];

      auto episode_step = env_outputs.get_vector()[3].front().item<int32_t>();
      auto episode_return = env_outputs.get_vector()[4].front().item<float>();
      float total_return = 0.0;
      uint32_t num_episodes = 0;
      try {
        while (true) {
          int action = agent_outputs.front().item<int>();
          action_pb.set_action(action);
          stream->Write(action_pb);
          if (!stream->Read(&step_pb)) {
            // Client closed the stream. We infer this as env.reset()
            // and terminate.
            total_return += episode_return;
            num_episodes += 1;
            std::cout << "Episode " << num_episodes << " ended after "
                      << episode_step << " steps. Return: " << episode_return
                      << std::endl;
            break;
          }

          env_outputs = step_pb_to_nest(&step_pb);
          bool done = env_outputs.get_vector()[2].front().item<bool>();
          episode_step = env_outputs.get_vector()[3].front().item<int32_t>();
          episode_return = env_outputs.get_vector()[4].front().item<float>();

          if (done) {
            total_return += episode_return;
            num_episodes += 1;
            std::cout << "Episode " << num_episodes << " ended after "
                      << episode_step << " steps. Return: " << episode_return
                      << std::endl;
          }

          compute_inputs = TensorNest(std::vector({env_outputs, agent_state}));
          all_agent_outputs = inference_batcher_->compute(compute_inputs);
          agent_outputs = all_agent_outputs.get_vector()[0];
          agent_state = all_agent_outputs.get_vector()[1];
        }
      } catch (const ClosedBatchingQueue& e) {
        // Thrown when inference_batcher_ is closed. Stop.
      }

      std::cout << "Average return over " << num_episodes
                << " episodes: " << total_return / num_episodes << std::endl;
    }

    static TensorNest nest_pb_to_nest(rpcenv::ArrayNest* nest_pb) {
      if (nest_pb->has_array()) {
        rpcenv::NDArray* array_pb = nest_pb->mutable_array();
        std::vector<int64_t> shape = {1, 1};  // [T=1, B=1].
        for (int i = 0, length = array_pb->shape_size(); i < length; ++i) {
          shape.push_back(array_pb->shape(i));
        }
        std::string* data = array_pb->release_data();
        at::ScalarType dtype =
            torch::utils::numpy_dtype_to_aten(array_pb->dtype());

        return TensorNest(torch::from_blob(
            data->data(), shape,
            /*deleter=*/[data](void*) { delete data; }, dtype));
      }
      if (nest_pb->vector_size() > 0) {
        std::vector<TensorNest> v;
        for (int i = 0, length = nest_pb->vector_size(); i < length; ++i) {
          v.push_back(nest_pb_to_nest(nest_pb->mutable_vector(i)));
        }
        return TensorNest(std::move(v));
      }
      if (nest_pb->map_size() > 0) {
        std::map<std::string, TensorNest> m;
        for (auto& p : *nest_pb->mutable_map()) {
          m[p.first] = nest_pb_to_nest(&p.second);
        }
        return TensorNest(std::move(m));
      }
      throw py::value_error("ArrayNest proto contained no data.");
    }

    static TensorNest step_pb_to_nest(rpcenv::Step* step_pb) {
      TensorNest done = TensorNest(
          torch::full({1, 1}, step_pb->done(), torch::dtype(torch::kUInt8)));
      TensorNest reward = TensorNest(torch::full({1, 1}, step_pb->reward()));
      TensorNest episode_step = TensorNest(torch::full(
          {1, 1}, step_pb->episode_step(), torch::dtype(torch::kInt32)));
      TensorNest episode_return =
          TensorNest(torch::full({1, 1}, step_pb->episode_return()));

      return TensorNest(
          std::vector({nest_pb_to_nest(step_pb->mutable_observation()),
                       std::move(reward), std::move(done),
                       std::move(episode_step), std::move(episode_return)}));
    }

    static TensorNest rollout_zero_pad(const TensorNest& rollout_item) {
      auto zero_filler = [](const torch::Tensor& t) {
        return torch::zeros_like(t);
      };

      TensorNest pad = rollout_item.map(zero_filler);
      if (!pad.is_vector() || pad.get_vector().size() != 2) {
        throw py::value_error("Expected rollout entry to be a tuple of size 2");
      }

      auto& env_outputs_pad = pad.get_vector()[0];
      if (!env_outputs_pad.is_vector() ||
          env_outputs_pad.get_vector().size() != 5) {
        throw py::value_error("Expected env output to be a tuple of size 5");
      }

      auto& done_pad = env_outputs_pad.get_vector()[2];
      done_pad.front().fill_(1);  // Set done to true for pad
      return pad;
    }

    std::atomic_uint64_t count_;

    const int unroll_length_;
    std::shared_ptr<BatchingQueue<>> learner_queue_;
    std::shared_ptr<DynamicBatcher> inference_batcher_;
    TensorNest initial_agent_state_;
  };

  ActorPool(int unroll_length, std::shared_ptr<BatchingQueue<>> learner_queue,
            std::shared_ptr<DynamicBatcher> inference_batcher,
            std::string server_address, TensorNest initial_agent_state)
      : server_address_(std::move(server_address)),
        service_(unroll_length, std::move(learner_queue),
                 std::move(inference_batcher), std::move(initial_agent_state)) {
  }

  uint64_t count() const { return service_.count(); }

  void run() {
    if (server_) {
      throw std::runtime_error("Server already running");
    }

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    std::cerr << "Server listening on " << server_address_ << std::endl;

    server_->Wait();
  }

  void stop() {
    if (!server_) {
      throw std::runtime_error("Server not running");
    }
    server_->Shutdown();
  }

 private:
  const std::string server_address_;
  ServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;
};

PYBIND11_MODULE(actorpool, m) {
  py::register_exception<std::future_error>(m, "AsyncError");
  py::register_exception<ClosedBatchingQueue>(m, "ClosedBatchingQueue");

  py::class_<ActorPool>(m, "ActorPool")
      .def(py::init<int, std::shared_ptr<BatchingQueue<>>,
                    std::shared_ptr<DynamicBatcher>, std::string, TensorNest>(),
           py::arg("unroll_length"), py::arg("learner_queue").none(false),
           py::arg("inference_batcher").none(false), py::arg("server_address"),
           py::arg("initial_agent_state"))
      .def("run", &ActorPool::run, py::call_guard<py::gil_scoped_release>())
      .def("stop", &ActorPool::stop)
      .def("count", &ActorPool::count);

  py::class_<DynamicBatcher::Batch, std::shared_ptr<DynamicBatcher::Batch>>(
      m, "Batch")
      .def("get_inputs", &DynamicBatcher::Batch::get_inputs)
      .def("set_outputs", &DynamicBatcher::Batch::set_outputs,
           py::arg("outputs"), py::call_guard<py::gil_scoped_release>());

  py::class_<DynamicBatcher, std::shared_ptr<DynamicBatcher>>(m,
                                                              "DynamicBatcher")
      .def(py::init<int64_t, int64_t, int64_t, std::optional<int>, bool>(),
           py::arg("batch_dim") = 1, py::arg("minimum_batch_size") = 1,
           py::arg("maximum_batch_size") = 1024, py::arg("timeout_ms") = 100,
           py::arg("check_outputs") = true)
      .def("close", &DynamicBatcher::close)
      .def("size", &DynamicBatcher::size)
      .def("__iter__",
           [](std::shared_ptr<DynamicBatcher> batcher) { return batcher; })
      .def("__next__", &DynamicBatcher::get_batch,
           py::call_guard<py::gil_scoped_release>());

  py::class_<BatchingQueue<>, std::shared_ptr<BatchingQueue<>>>(m,
                                                                "BatchingQueue")
      .def(py::init<int64_t, int64_t, int64_t, std::optional<int>, bool>(),
           py::arg("batch_dim") = 1, py::arg("minimum_batch_size") = 1,
           py::arg("maximum_batch_size") = 1024,
           py::arg("timeout_ms") = std::nullopt, py::arg("check_inputs") = true)
      .def("close", &BatchingQueue<>::close)
      .def("size", &BatchingQueue<>::size)
      .def("__iter__",
           [](std::shared_ptr<BatchingQueue<>> queue) { return queue; })
      .def("__next__", [](BatchingQueue<>& queue) {
        py::gil_scoped_release release;
        std::pair<TensorNest, std::vector<Empty>> pair = queue.dequeue_many();
        return pair.first;
      });

  m.def("front", [](const TensorNest& n) { return n.front(); });
}
