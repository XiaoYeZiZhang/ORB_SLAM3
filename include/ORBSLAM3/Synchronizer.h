//
// Created by root on 2020/10/10.
//

#ifndef ORB_SLAM3_SYNCHRONIZER_H
#define ORB_SLAM3_SYNCHRONIZER_H
#include <vector>
#include <tuple>
#include <memory>
#include <queue>

class Synchronizer {
public:
    enum ItemType {
        TYPE_INPUT_GYROSCOPE = 0,
        TYPE_INPUT_ACCELEROMETER,
        TYPE_INPUT_EMPTY,
    };

    Synchronizer() {
        latest_time = 0;
        max_delay_time = 0;
    }

    struct InputItem {
        ItemType type;
        double time;
        cv::Point3f value;

        bool operator<(const InputItem &other) const {
            if (time != other.time) {
                return time > other.time;
            } else {
                return (int)type > (int)other.type;
            }
        }
        InputItem() {
            type == TYPE_INPUT_EMPTY;
            time = -1;
            value = cv::Point3f();
        }
    };

    void consume_input(const InputItem &input) {
        latest_time = std::max(latest_time, input.time);
        if (input.time + max_delay_time >= latest_time) {
            sync_queue.push(input);
        } else {
            max_delay_time = latest_time - input.time;
            VLOG(0) << "Maximum sensor delay extended to " << max_delay_time
                    << "s.";
        }

        while (!sync_queue.empty() &&
               latest_time - sync_queue.top().time >= max_delay_time) {
            const auto &input = sync_queue.top();
            switch (input.type) {
            case TYPE_INPUT_GYROSCOPE: {
                if (last_gyroscope.time != -1) {
                    const double &t0 = last_gyroscope.time;
                    const cv::Point3f &w0 = last_gyroscope.value;
                    double dt = input.time - t0;
                    cv::Point3f dw = input.value - w0;
                    for (auto &[t, a] : pending_accelerometer) {
                        pending_imu.emplace(t, w0 + ((t - t0) / dt) * dw, a);
                    }
                    pending_accelerometer.clear();
                }
                last_gyroscope.time = input.time;
                last_gyroscope.value = input.value;
            } break;
            case TYPE_INPUT_ACCELEROMETER: {
                if (last_gyroscope.time != -1) {
                    if (input.time == last_gyroscope.time) {
                        pending_imu.emplace(
                            input.time, last_gyroscope.value, input.value);
                    } else {
                        pending_accelerometer.emplace_back(
                            input.time, input.value);
                    }
                }
            } break;
            }
            sync_queue.pop();
        }
    }

    std::priority_queue<InputItem> sync_queue;
    InputItem last_gyroscope;
    std::vector<std::tuple<double, cv::Point3f>> pending_accelerometer;
    // timestamp, gyr, acc
    std::queue<std::tuple<double, cv::Point3f, cv::Point3f>> pending_imu;
    double latest_time;
    double max_delay_time;
};
#endif // ORB_SLAM3_SYNCHRONIZER_H
