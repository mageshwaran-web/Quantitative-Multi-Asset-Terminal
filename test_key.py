import os, requests
key = os.environ.get('FEATHERLESS_API_KEY')
print('Key length:', len(key) if key else 'NOT SET')
print('Key preview:', repr(key[:8]) + '...' if key else None)
resp = requests.post('https://api.featherless.ai/v1/chat/completions',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json={'model':'mistralai/Mistral-7B-Instruct-v0.3','messages':[{'role':'user','content':'hi'}],'max_tokens':10})
print('Status:', resp.status_code)
print('Body:', resp.text[:400])
