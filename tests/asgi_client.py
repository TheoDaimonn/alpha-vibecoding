"""Minimal in-process ASGI client without optional HTTP client dependencies."""
import asyncio


async def call_asgi(app, body=b'', path='/process', headers=(), method='POST'):
    complete = asyncio.Event()
    delivered = False
    messages = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await complete.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.body' and not message.get('more_body', False):
            complete.set()

    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
             'http_version': '1.1', 'method': method, 'scheme': 'http',
             'path': path, 'raw_path': path.encode(), 'query_string': b'',
             'root_path': '', 'headers': [(b'content-type', b'application/json'), *headers],
             'client': ('127.0.0.1', 1234), 'server': ('test', 80)}
    await asyncio.wait_for(app(scope, receive, send), 5)
    start = next(message for message in messages if message['type'] == 'http.response.start')
    result = b''.join(message.get('body', b'') for message in messages if message['type'] == 'http.response.body')
    return start['status'], dict(start['headers']), result
