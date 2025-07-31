from locust import HttpUser, task, between, events
import base64
import uuid
import json
import os
import random
import time
import statistics
from datetime import datetime

class MetricsCollector:
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = time.time()
        
    def add_response_time(self, response_time):
        self.response_times.append(response_time)
        
    def add_success(self):
        self.success_count += 1
        
    def add_error(self):
        self.error_count += 1
        
    def get_qps(self):
        elapsed_time = time.time() - self.start_time
        total_requests = self.success_count + self.error_count
        return total_requests / elapsed_time if elapsed_time > 0 else 0
        
    def get_error_rate(self):
        total_requests = self.success_count + self.error_count
        return (self.error_count / total_requests * 100) if total_requests > 0 else 0
        
    def get_avg_response_time(self):
        return statistics.mean(self.response_times) if self.response_times else 0
        
    def get_p95_response_time(self):
        if self.response_times:
            sorted_times = sorted(self.response_times)
            index = int(0.95 * len(sorted_times))
            return sorted_times[index]
        return 0

metrics = MetricsCollector()

class PoseEstimationUser(HttpUser):
    wait_time = between(0.5, 2)
    
    def on_start(self):
        self.test_images = []
        self.load_test_images()
        
        if not self.test_images:
            print("Warning: No test images loaded. Creating dummy images...")
            self.create_multiple_dummy_images()
            
        print(f"User loaded {len(self.test_images)} test images")
    
    def load_test_images(self):
        image_files = ["model3-yolol/test.jpg", "model3-yolol/bus.jpg"]
        
        for image_file in image_files:
            if os.path.exists(image_file):
                try:
                    with open(image_file, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                        self.test_images.append({
                            "filename": os.path.basename(image_file),
                            "base64": image_data
                        })
                    print(f"Loaded test image: {image_file}")
                except Exception as e:
                    print(f"Failed to load image {image_file}: {e}")
        
        if os.path.exists("test_image_base64.txt"):
            try:
                with open("test_image_base64.txt", "r") as f:
                    base64_data = f.read().strip()
                    if base64_data:
                        self.test_images.append({
                            "filename": "test_image_base64.jpg",
                            "base64": base64_data
                        })
                        print("Loaded base64 test image from text file")
            except Exception as e:
                print(f"Failed to load base64 test image: {e}")
        
        for i in range(10):
            dummy_image = self.create_dummy_image()
            self.test_images.append({
                "filename": f"generated_dummy_{i}.jpg", 
                "base64": dummy_image
            })
    
    def create_multiple_dummy_images(self):
        print("Creating dummy test images...")
        
        for i in range(20):
            dummy_image = self.create_dummy_image()
            self.test_images.append({
                "filename": f"dummy_{i}.jpg",
                "base64": dummy_image
            })
    
    def create_dummy_image(self):
        try:
            import cv2
            import numpy as np
            
            size = random.choice([(64, 64), (128, 128), (256, 256)])
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            cv2.rectangle(img, (10, 10), (size[0]-10, size[1]-10), (100, 150, 200), 2)
            cv2.circle(img, (size[0]//2, size[1]//2), min(size)//4, (255, 255, 255), -1)
            
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
            
        except ImportError:
            return "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
        except Exception:
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    @task(3)
    def pose_estimation(self):
        test_image = random.choice(self.test_images)
        
        start_time = time.time()
        
        with self.client.post("/pose", json={"image": test_image["base64"], "id": f"test_{int(time.time() * 1000)}"}, catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000
            metrics.add_response_time(response_time)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    required_fields = ["count", "boxes", "keypoints"]
                    for field in required_fields:
                        if field not in result:
                            response.failure(f"Missing field: {field}")
                            metrics.add_error()
                            return
                    
                    if not isinstance(result["count"], int):
                        response.failure("count should be an integer")
                        metrics.add_error()
                        return
                    
                    if not isinstance(result["boxes"], list):
                        response.failure("boxes should be a list")
                        metrics.add_error()
                        return
                    
                    if not isinstance(result["keypoints"], list):
                        response.failure("keypoints should be a list")
                        metrics.add_error()
                        return
                    
                    response.success()
                    metrics.add_success()
                    
                    if result["count"] > 0:
                        print(f"[{response_time:.1f}ms] Detected {result['count']} person(s) in {test_image['filename']}")
                    else:
                        print(f"[{response_time:.1f}ms] No persons detected in {test_image['filename']}")
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
                    metrics.add_error()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                metrics.add_error()
    
    @task(2)
    def pose_estimation_annotated(self):
        test_image = random.choice(self.test_images)
        
        start_time = time.time()
        
        with self.client.post("/pose/annotated", json={"image": test_image["base64"], "id": f"test_ann_{int(time.time() * 1000)}"}, catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000
            metrics.add_response_time(response_time)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    required_fields = ["count", "annotated_image"]
                    for field in required_fields:
                        if field not in result:
                            response.failure(f"Missing field: {field}")
                            metrics.add_error()
                            return
                    
                    if not isinstance(result["annotated_image"], str):
                        response.failure("annotated_image should be a base64 string")
                        metrics.add_error()
                        return
                    
                    try:
                        base64.b64decode(result["annotated_image"])
                    except Exception:
                        response.failure("Invalid base64 in annotated_image")
                        metrics.add_error()
                        return
                    
                    response.success()
                    metrics.add_success()
                    
                    if result["count"] > 0:
                        print(f"[{response_time:.1f}ms] Annotated {result['count']} person(s) in {test_image['filename']}")
                    else:
                        print(f"[{response_time:.1f}ms] No persons detected for annotation in {test_image['filename']}")
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
                    metrics.add_error()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                metrics.add_error()

    @task(2)
    def health_check(self):
        start_time = time.time()
        
        with self.client.get("/health", catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000
            metrics.add_response_time(response_time)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("status") == "healthy":
                        response.success()
                        metrics.add_success()
                    else:
                        response.failure("Service not healthy")
                        metrics.add_error()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
                    metrics.add_error()
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics.add_error()
    
    @task(1)
    def root_endpoint(self):
        start_time = time.time()
        
        with self.client.get("/", catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000
            metrics.add_response_time(response_time)
            
            if response.status_code == 200:
                response.success()
                metrics.add_success()
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics.add_error()

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=== Load Test Started ===")
    print(f"Target host: {environment.host}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    print("\n=== Load Test Completed ===")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total requests: {metrics.success_count + metrics.error_count}")
    print(f"Successful requests: {metrics.success_count}")
    print(f"Failed requests: {metrics.error_count}")
    print(f"Error rate: {metrics.get_error_rate():.2f}%")
    print(f"Average response time: {metrics.get_avg_response_time():.2f}ms")
    print(f"95th percentile response time: {metrics.get_p95_response_time():.2f}ms")
    print(f"Queries per second (QPS): {metrics.get_qps():.2f}")

if __name__ == "__main__":
    import sys
    
    print("=== Enhanced Pose Estimation API Load Test ===")
    print("This script supports up to 128 test images and comprehensive performance monitoring")
    print("")
    print("Usage: locust -f locust_test.py --host=http://your-remote-ip:30001")
    print("Example: locust -f locust_test.py --host=http://203.101.225.157:30001")
    print("")
    print("Web UI will be available at: http://localhost:8089")
    print("")
    print("LOAD TESTING SCENARIOS:")
    print("1. Gradual Ramp-up Test:")
    print("   - Users: Start with 5, ramp to 30 over 10 minutes")
    print("   - Spawn rate: 1-2 users per second")
    print("   - Duration: 15-20 minutes total")
    print("")
    print("2. Stress Test (for bottleneck identification):")
    print("   - Users: 50+ concurrent users")
    print("   - Spawn rate: 5 users per second")
    print("   - Duration: 5-10 minutes")
    print("")
    print("MONITORED METRICS:")
    print("- Response Time (average and 95th percentile)")
    print("- Queries Per Second (QPS)")
    print("- Error Rate (%)")
    print("- Success/Failure counts")
    print("")
    print("RECOMMENDED TEST PARAMETERS:")
    print("- Start Users: 5-10")
    print("- Max Users: 20-50 (depending on server capacity)")
    print("- Spawn Rate: 1-3 users per second")
    print("- Test Duration: 10-30 minutes")
    print("")
    
    if len(sys.argv) == 1:
        print("Please specify the host parameter.")
        print("Example: locust -f locust_test.py --host=http://YOUR_REMOTE_IP:30001")
        sys.exit(1)
    
    print("Starting load test...")
    print("Monitor real-time metrics at: http://localhost:8089")
