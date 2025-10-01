import requests

BASE_URL = "http://127.0.0.1:8000"


def main():
    # GET root
    r = requests.get(f"{BASE_URL}/")
    print("GET / ->", r.status_code, r.json())

    # Example payload for inference (adjust to your API’s schema later)
    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    # POST to inference endpoint (update route/fields once your API is wired)
    r = requests.post(f"{BASE_URL}/inference", json=data)
    print("POST /inference ->", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)


if __name__ == "__main__":
    main()
