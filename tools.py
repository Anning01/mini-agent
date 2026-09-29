"""Agent 可调用工具的定义，以及提供给 LLM 的工具 schema。"""

import json
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch_json(url: str, params: dict) -> dict:
    """向固定的天气服务地址发 GET 请求，解析返回的 JSON 对象。

    urlencode 负责对中文城市名和参数进行编码；超时让工具不会一直等待。
    请求或 JSON 解析失败时，异常会由 Agent 的工具调用分支交给模型处理。
    """

    request_url = f"{url}?{urlencode(params)}"
    with urlopen(request_url, timeout=10) as response:
        data = json.load(response)

    if not isinstance(data, dict) or data.get("error"):
        raise ValueError(f"天气服务返回错误：{data}")
    return data


WEATHER_DESCRIPTIONS = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    56: "小冻毛毛雨",
    57: "大冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小冻雨",
    67: "大冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


class Tools:
    """维护工具函数注册表，并输出 OpenAI 兼容的 function-calling 描述。"""

    def __init__(self):
        # registry 是运行时路由表：模型给出工具名后，Agent 用它找到真实 Python 函数。
        self.registry = {
            "get_weather": self.get_weather,
        }

    def get_weather(self, city: str) -> str:
        """先将城市名转换为坐标，再读取该地点的当前天气和今日预报。

        这两次 HTTP 请求均由程序执行。LLM 只负责决定调用工具、提供城市参数，
        并根据返回的天气数据回答用户。当前值来自天气模型，不代表气象站实测。
        """

        if not city.strip():
            raise ValueError("城市名称不能为空")

        # Geocoding API 把用户输入的地名换成天气 API 所需的经纬度。
        locations = fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": city.strip(), "count": 1, "language": "zh"},
        ).get("results", [])
        if not locations:
            raise ValueError(f"没有找到城市：{city}")

        location = locations[0]
        # timezone=auto 让“今天”和返回时间使用该城市当地的时区。
        weather = fetch_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": 1,
            },
        )

        current = weather["current"]
        daily = weather["daily"]
        condition = WEATHER_DESCRIPTIONS.get(
            current["weather_code"], f"天气代码 {current['weather_code']}"
        )
        # 返回匹配到的地区，避免同名城市被误认为用户想查的地点。
        place = "、".join(
            part for part in (location.get("country"), location.get("admin1"), location["name"])
            if part
        )
        return (
            f"地点：{place}（用户输入：{city}）\n"
            f"当地时间：{current['time']}（{weather['timezone']}）\n"
            f"当前模型天气：{condition}，气温 {current['temperature_2m']}°C，"
            f"相对湿度 {current['relative_humidity_2m']}%，"
            f"降水量 {current['precipitation']} mm，"
            f"风速 {current['wind_speed_10m']} km/h。\n"
            f"今日（{daily['time'][0]}）预报：最高 {daily['temperature_2m_max'][0]}°C，"
            f"最低 {daily['temperature_2m_min'][0]}°C，"
            f"最高降水概率 {daily['precipitation_probability_max'][0]}%。\n"
            "数据来源：Open-Meteo（https://open-meteo.com/）。"
        )

    def get_tools_schema(self):
        """返回发送给 LLM 的 function-calling schema。

        schema 不执行工具；它只是告诉模型可用工具、每个工具的用途及参数格式。
        真实执行发生在 Agent 通过 ``registry`` 查找 Python 函数时。
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的当前模型天气和今日预报（Open-Meteo）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
