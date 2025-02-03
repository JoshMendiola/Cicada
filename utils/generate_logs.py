import json
import random
from datetime import datetime, timedelta
import ipaddress
import os

from utils.patterns import ATTACK_PATTERNS, NORMAL_ENDPOINTS


def generate_ip():
    return str(ipaddress.IPv4Address(random.randint(0, 2 ** 32 - 1)))


def generate_user_agent(is_attack):
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
    versions = ["70.0", "80.0", "90.0", "100.0"]
    os_list = ["Windows NT 10.0", "Macintosh; Intel Mac OS X 10_15_7", "X11; Linux x86_64"]

    script_probability = 0.60 if is_attack else 0.05  # Higher chance of script for attacks

    if random.random() < script_probability:
        script_types = [
            f"Python/{random.choice(['3.7', '3.8', '3.9', '3.10'])} aiohttp/{random.choice(['3.7.4', '3.8.1', '3.9.0', '3.10.0'])}",
            f"curl/{random.choice(['7.64.1', '7.68.0', '7.72.0'])}",
            f"Wget/{random.choice(['1.20.3', '1.21.1'])}",
            f"Go-http-client/{random.choice(['1.1', '2.0'])}",
            f"node-fetch/{random.choice(['2.6.1', '3.0.0'])}"
        ]
        return random.choice(script_types)
    else:
        browser = random.choice(browsers)
        version = random.choice(versions)
        os = random.choice(os_list)
        return f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version} Safari/537.36"


def generate_log_entry(base_attack_probability=0.05):
    timestamp = datetime.now() + timedelta(seconds=random.randint(0, 86400))
    ip = generate_ip()

    # Determine if this entry is an attack based on base probability
    is_attack = random.random() < base_attack_probability

    user_agent = generate_user_agent(is_attack)

    if is_attack:
        path, method = random.choice(ATTACK_PATTERNS)
    else:
        path = random.choice(NORMAL_ENDPOINTS)
        method = random.choice(["GET", "POST"])

    headers = {
        "Host": "147.182.176.235",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": user_agent
    }

    body = None
    if method == "POST":
        if "login" in path or "admin" in path:
            body = json.dumps({"username": f"user{random.randint(1, 1000)}", "password": "password123"})
        elif is_attack:
            body = json.dumps({"malicious": "payload"})
        else:
            body = json.dumps({"key": "value"})

    if "admin" in path or "login" in path:
        is_attack = is_attack or (random.random() < 0.05)  # Slightly higher chance of being an attack
    if "script" in user_agent.lower():
        is_attack = is_attack or (
                    random.random() < 0.10)  # Higher chance for script-based user agents, but not as extreme

    log_entry = {
        "timestamp": timestamp.isoformat(),
        "type": "REQUEST",
        "ip": ip,
        "method": method,
        "path": path,
        "headers": headers,
        "body": body,
        "is_attack": is_attack
    }

    return log_entry


def generate_logs(num_logs, base_attack_percentage):
    base_attack_probability = base_attack_percentage / 100
    return [generate_log_entry(base_attack_probability) for _ in range(num_logs)]


def main(num_logs=5000, base_attack_percentage=5):
    logs = generate_logs(num_logs, base_attack_percentage)

    # Ensure the data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Write logs to file
    log_file = os.path.join(data_dir, 'logs.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

    actual_attack_percentage = (sum(1 for log in logs if log['is_attack']) / num_logs) * 100
    print(f"Generated {num_logs} log entries.")
    print(f"Base attack percentage: {base_attack_percentage}%")
    print(f"Actual attack percentage after noise: {actual_attack_percentage:.2f}%")
    print(f"Logs saved to {log_file}")


if __name__ == '__main__':
    main()