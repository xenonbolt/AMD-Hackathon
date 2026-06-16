import requests
with open('/home/dwijo/Desktop/AMD/frontend/src/App.tsx', 'r') as f:
    code = f.read()

res = requests.post("http://localhost:8000/api/scan", json={"files": [{"path": "App.tsx", "content": code}]})
print(res.json())
