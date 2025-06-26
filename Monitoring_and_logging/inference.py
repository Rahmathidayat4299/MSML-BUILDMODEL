import requests
import time

url = "http://localhost:5005/invocations"

payload = {
    "dataframe_split": {
        "columns": [
            "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
            "x10", "x11", "x12", "x13", "x14", "x17", "x18", "x19",
            "x20", "x21", "x22", "x23", "x24"
        ],
        "data": [[
            0.7105263157894737, -0.4989384288747346, 1.7933333333333337, -0.45714285714285713,
            -0.47058823529411764, -0.6, 0.6, 1.9883040935672505, 0.0,
            -0.25, 0.3333333333333333, 0.7002917882451017, 0.0, -0.25,
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
        ]]
    }
}

headers = {"Content-Type": "application/json"}

for i in range(100):
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"[{i+1}/100] Status code: {response.status_code}")
        print(f"Response: {response.text}\n")
        # Optional: Add delay if you want to avoid overwhelming the server
        # time.sleep(0.1)

    except requests.exceptions.ConnectionError as e:
        print(f"[{i+1}/100] Error connecting to server: {e}")
        print("Make sure the MLflow model is running at http://127.0.0.1:5005")
        break

    except Exception as e:
        print(f"[{i+1}/100] Unexpected error: {e}")
        break
