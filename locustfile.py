from locust import HttpUser, task, between


class EAMNetUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        self.image_file = open("test.jpg", "rb")

    def on_stop(self):
        self.image_file.close()

    @task(3)
    def test_predict(self):
        self.image_file.seek(0)

        files = {
            "file": ("test.jpg", self.image_file, "image/jpeg")
        }

        with self.client.post(
            "/predict",
            files=files,
            name="POST /predict",
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP 请求失败，状态码：{response.status_code}"
                )
                return

            try:
                data = response.json()
            except ValueError:
                response.failure("响应内容不是合法的 JSON")
                return

            if data.get("success") is not True:
                response.failure("success 不为 true")
                return

            if data.get("output_count") != 10:
                response.failure(
                    f"output_count 错误：{data.get('output_count')}"
                )
                return

            if data.get("prediction_shape") != [1, 1, 352, 352]:
                response.failure(
                    f"prediction_shape 错误：{data.get('prediction_shape')}"
                )
                return

            response.success()

    @task(1)
    def test_health(self):
        with self.client.get(
            "/",
            name="GET /"
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP 请求失败，状态码：{response.status_code}"
                )