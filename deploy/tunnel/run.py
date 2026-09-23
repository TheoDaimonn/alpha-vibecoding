"""Keep an anonymous localhost.run tunnel connected and publish its current URL."""
import os
import re
import signal
import subprocess
import threading
from pathlib import Path

URL_FILE = Path('/run/tunnel/public-url')
STATE = Path('/state')
STOP = threading.Event()
process = None


def tunnel_url(line):
    # Ignore documentation links and QR codes in SSH's welcome banner.
    if 'tunneled with' not in line:
        return None
    match = re.search(r'https://[a-zA-Z0-9.-]+', line)
    return match.group(0) if match else None


def shutdown(signum, frame):
    STOP.set()
    URL_FILE.unlink(missing_ok=True)
    if process is not None and process.poll() is None:
        process.terminate()


def main():
    global process
    STATE.mkdir(parents=True, exist_ok=True)
    URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    URL_FILE.unlink(missing_ok=True)
    key = STATE / 'id_ed25519'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key)], check=True)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    command = [
        'ssh', '-F', '/dev/null', '-T', '-i', str(key),
        '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=accept-new', '-o', f'UserKnownHostsFile={STATE}/known_hosts',
        '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
        '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes',
        '-R', f"80:{os.environ.get('TUNNEL_TARGET', 'api:8000')}", 'nokey@localhost.run',
    ]
    while not STOP.is_set():
        URL_FILE.unlink(missing_ok=True)
        print('Connecting to localhost.run...', flush=True)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                url = tunnel_url(line)
                if url and not STOP.is_set():
                    tmp = URL_FILE.with_suffix('.tmp')
                    tmp.write_text(url + '\n')
                    tmp.replace(URL_FILE)
                    print(f'PUBLIC_URL={url}\nDOCS_URL={url}/docs\nPROCESS_URL={url}/process', flush=True)
                elif any(word in line.lower() for word in ('denied', 'failed', 'closed', 'error', 'timed out')):
                    print(line.strip(), flush=True)
            process.wait()
        finally:
            URL_FILE.unlink(missing_ok=True)
        if not STOP.is_set():
            print('Tunnel disconnected. URL cleared; reconnecting in 5 seconds.', flush=True)
            STOP.wait(5)


if __name__ == '__main__':
    main()
