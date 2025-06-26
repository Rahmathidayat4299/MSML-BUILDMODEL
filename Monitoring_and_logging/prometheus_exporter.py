import json
import time
import logging
import os

import psutil
import requests
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency")
THROUGHPUT = Counter("http_requests_throughput", "Total number of requests per second")
ML_MODEL_PREDICTION_SUCCESS = Counter(
    "ml_model_prediction_success_total", "Total successful ML model predictions"
)
ML_MODEL_PREDICTION_FAILURE = Counter(
    "ml_model_prediction_failure_total", "Total failed ML model predictions"
)

# Metrik untuk sistem
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage_percent", "RAM Usage Percentage")
# Changed DISK_USAGE to a Gauge with a label for mount point
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk Usage Percentage by Mount Point", ["mount_point"])

# Metrik tambahan
NETWORK_BYTES_SENT = Gauge("system_network_bytes_sent_total", "Total network bytes sent")
NETWORK_BYTES_RECV = Gauge("system_network_bytes_recv_total", "Total network bytes received")
ML_MODEL_INPUT_PAYLOAD_SIZE_BYTES = Histogram(
    "ml_model_input_payload_size_bytes",
    "Size of input payload in bytes",
    buckets=(100, 500, 1000, 5000, 10000, float("inf")),
)

APP_UPTIME_SECONDS = Gauge("app_uptime_seconds", "Uptime of the Prometheus Exporter")
START_TIME = time.time()


@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        RAM_USAGE.set(psutil.virtual_memory().percent)

        # Get disk usage for all logical disks/partitions
        for part in psutil.disk_partitions(all=True):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                # Set disk usage with the mount point as a label
                DISK_USAGE.labels(mount_point=part.mountpoint).set(usage.percent)
            except Exception as e:
                # Log the error but continue with other partitions
                logging.error(f"Failed to get disk usage for path {part.mountpoint}: {e}")
                # Optionally, you might want to set a specific value for failed partitions, e.g., 0 or NaN
                DISK_USAGE.labels(mount_point=part.mountpoint).set(0.0) # Set to 0 if failed

        net_io = psutil.net_io_counters()
        NETWORK_BYTES_SENT.set(net_io.bytes_sent)
        NETWORK_BYTES_RECV.set(net_io.bytes_recv)

        APP_UPTIME_SECONDS.set(time.time() - START_TIME)

        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    except Exception as e:
        logging.exception(f"Error generating /metrics: {e}")
        return jsonify({"error": "Failed to generate metrics", "details": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    if data:
        try:
            payload_size = len(json.dumps(data).encode("utf-8"))
            ML_MODEL_INPUT_PAYLOAD_SIZE_BYTES.observe(payload_size)
        except Exception as e:
            logging.warning(f"Failed to observe input payload size: {e}")

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        if response.status_code == 200:
            ML_MODEL_PREDICTION_SUCCESS.inc()
        else:
            logging.error(f"Prediction failed with status code: {response.status_code}")
            ML_MODEL_PREDICTION_FAILURE.inc()

        return jsonify(response.json())

    except Exception as e:
        logging.exception(f"Prediction request failed: {e}")
        ML_MODEL_PREDICTION_FAILURE.inc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.info("Starting Prometheus ML Exporter...")
    app.run(host="127.0.0.1", port=8000)