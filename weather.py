import requests

API_KEY = "24e517a06d54d224720f573713d826b7"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if "main" not in data:
        raise Exception("Invalid city")

    temp = data['main']['temp']
    humidity = data['main']['humidity']

    return temp, humidity


# 🌍 Prayagraj special boost
def adjust_for_location(city, rainfall):
    if city.lower() == "prayagraj":
        rainfall += 20   # slight boost (realistic tweak)
    return rainfall