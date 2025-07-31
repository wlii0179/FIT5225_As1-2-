from locust import HttpUser, task, between

class SimpleUser(HttpUser):
    wait_time = between(1, 2)
    
    @task(1)
    def test_health(self):
        self.client.get("/health")
    
    @task(1)
    def test_root(self):
        self.client.get("/")
    
    @task(1)
    def test_pose(self):
        # 简单的测试图片
        payload = {
            "id": "test123",
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
        self.client.post("/pose", json=payload) 