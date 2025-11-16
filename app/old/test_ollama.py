"""
Тестовый скрипт для проверки подключения к Ollama
"""
import os
import sys
import urllib.request
import urllib.error
import json

# Устанавливаем UTF-8 для вывода
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Пробуем загрузить dotenv если доступен
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Получаем URL из конфигурации или аргументов командной строки
if len(sys.argv) > 1:
    OLLAMA_API_URL = sys.argv[1]
else:
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
print(f"=" * 60)
print(f"Testing Ollama connection")
print(f"=" * 60)
print(f"OLLAMA_API_URL from .env: {OLLAMA_API_URL}")
print()

# Тест 1: Проверка доступности хоста через HTTP
print("Test 1: Direct HTTP connection to Ollama API")
print("-" * 60)
try:
    # Пробуем подключиться к API напрямую
    if OLLAMA_API_URL.startswith("http://"):
        base_url = OLLAMA_API_URL
    elif OLLAMA_API_URL.startswith("https://"):
        base_url = OLLAMA_API_URL
    else:
        base_url = f"http://{OLLAMA_API_URL}"
    
    # Убираем слэш в конце если есть
    base_url = base_url.rstrip('/')
    
    # Проверяем доступность через /api/tags
    test_url = f"{base_url}/api/tags"
    print(f"Testing URL: {test_url}")
    
    req = urllib.request.Request(test_url)
    req.add_header('Content-Type', 'application/json')
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            print(f"[OK] HTTP connection successful! Status: {response.status}")
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"  Available models: {len(data.get('models', []))}")
                for model in data.get('models', [])[:3]:
                    print(f"    - {model.get('name', 'unknown')}")
    except urllib.error.URLError as e:
        print(f"[FAIL] Connection failed: {e}")
        print(f"  Cannot reach {base_url}")
        if hasattr(e, 'reason'):
            print(f"  Reason: {e.reason}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Тест 2: Проверка через библиотеку ollama с переменной окружения
print("Test 2: Ollama library with OLLAMA_HOST environment variable")
print("-" * 60)
try:
    # Устанавливаем переменную окружения
    if OLLAMA_API_URL.startswith("http://"):
        ollama_host = OLLAMA_API_URL[7:]
    elif OLLAMA_API_URL.startswith("https://"):
        ollama_host = OLLAMA_API_URL[8:]
    else:
        ollama_host = OLLAMA_API_URL
    
    os.environ["OLLAMA_HOST"] = ollama_host
    print(f"Setting OLLAMA_HOST to: {ollama_host}")
    
    import ollama
    
    # Пробуем получить список моделей
    response = ollama.list()
    print(f"[OK] Ollama library connection successful!")
    print(f"  Available models: {len(response.get('models', []))}")
    for model in response.get('models', [])[:3]:
        print(f"    - {model.get('name', 'unknown')}")
except Exception as e:
    print(f"[FAIL] Ollama library connection failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Тест 3: Проверка через клиент ollama с явным указанием хоста
print("Test 3: Ollama Client with explicit host parameter")
print("-" * 60)
try:
    import ollama
    
    # Пробуем создать клиент с полным URL
    try:
        client = ollama.Client(host=OLLAMA_API_URL)
        response = client.list()
        print(f"[OK] Client with full URL successful!")
        print(f"  Available models: {len(response.get('models', []))}")
    except Exception as e1:
        print(f"  Failed with full URL: {e1}")
        
        # Пробуем с host:port
        if OLLAMA_API_URL.startswith("http://"):
            host_port = OLLAMA_API_URL[7:]
        elif OLLAMA_API_URL.startswith("https://"):
            host_port = OLLAMA_API_URL[8:]
        else:
            host_port = OLLAMA_API_URL
        
        try:
            client = ollama.Client(host=host_port)
            response = client.list()
            print(f"[OK] Client with host:port successful!")
            print(f"  Available models: {len(response.get('models', []))}")
        except Exception as e2:
            print(f"[FAIL] Client with host:port also failed: {e2}")
            raise e2
except Exception as e:
    print(f"[FAIL] Ollama Client creation failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Тест 4: Тестовый запрос к модели
print("Test 4: Test chat request to Ollama")
print("-" * 60)
try:
    import ollama
    
    # Пробуем отправить простой запрос
    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one word"
            }
        ],
        options={
            "temperature": 0.0,
            "num_predict": 10
        }
    )
    
    print(f"[OK] Chat request successful!")
    if "message" in response and "content" in response["message"]:
        print(f"  Response: {response['message']['content']}")
    else:
        print(f"  Response: {response}")
except Exception as e:
    print(f"[FAIL] Chat request failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Тест 5: Проверка сетевой доступности
print("Test 5: Network connectivity check")
print("-" * 60)
import socket

if OLLAMA_API_URL.startswith("http://"):
    host = OLLAMA_API_URL[7:].split(':')[0]
    port = int(OLLAMA_API_URL[7:].split(':')[1]) if ':' in OLLAMA_API_URL[7:] else 11434
elif OLLAMA_API_URL.startswith("https://"):
    host = OLLAMA_API_URL[8:].split(':')[0]
    port = int(OLLAMA_API_URL[8:].split(':')[1]) if ':' in OLLAMA_API_URL[8:] else 11434
else:
    parts = OLLAMA_API_URL.split(':')
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 11434

print(f"Testing socket connection to {host}:{port}")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((host, port))
    sock.close()
    
    if result == 0:
        print(f"[OK] Socket connection successful!")
    else:
        print(f"[FAIL] Socket connection failed (error code: {result})")
except socket.gaierror as e:
    print(f"[FAIL] DNS resolution failed: {e}")
    print(f"  Cannot resolve hostname '{host}'")
except Exception as e:
    print(f"[FAIL] Socket connection error: {e}")

print()
print("=" * 60)
print("Summary:")
print("=" * 60)
print("If all tests fail, check:")
print("1. Is Ollama running on the target machine?")
print("2. Is the hostname/IP address correct?")
print("3. Is port 11434 accessible from this machine?")
print("4. Are there firewall rules blocking the connection?")
print("5. Try using IP address instead of hostname")
print("=" * 60)

