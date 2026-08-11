import re

with open('main.py', 'r') as f:
    content = f.read()

# Add import json_repair if not there
if 'import json_repair' not in content:
    content = content.replace('import json\n', 'import json\nimport json_repair\n')

# Replace json.loads with json_repair.loads in the endpoints
content = re.sub(r'return json\.loads\(content\)', 'return json_repair.loads(content)', content)

with open('main.py', 'w') as f:
    f.write(content)

print("Fixed JSON loading")
