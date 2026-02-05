
import socket, struct, hashlib, os, time, threading

BUF = 64 * 1024

# ============ Server: Run in background thread ============
class BackgroundFileServer:
    def __init__(self, host="0.0.0.0", port=9001, save_dir="./received"):
        self.host = host
        self.port = port
        self.save_dir = save_dir
        self._stop = threading.Event()
        self._thread = None
        self._sock = None

    def _recv_exact(self, conn, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("socket closed")
            data += chunk
        return data

    def _handle_conn(self, conn, addr):
        try:
            # header: filename_len(uint16) | filename | file_size(uint64) | sha256(32B)
            fn_len = struct.unpack(">H", self._recv_exact(conn, 2))[0]
            filename = self._recv_exact(conn, fn_len).decode("utf-8")
            file_size = struct.unpack(">Q", self._recv_exact(conn, 8))[0]
            want_sha = self._recv_exact(conn, 32)

            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, filename)

            # body
            h = hashlib.sha256()
            remaining = file_size
            with open(path, "wb") as f:
                while remaining:
                    chunk = conn.recv(min(BUF, remaining))
                    if not chunk:
                        raise ConnectionError("socket closed during transfer")
                    f.write(chunk)
                    h.update(chunk)
                    remaining -= len(chunk)

            ok = (h.digest() == want_sha)
            conn.sendall(b"OK" if ok else b"ER")
            print(f"[FileServer] {filename} {file_size}B from {addr} -> {'OK' if ok else 'SHA MISMATCH'}")
        except Exception as e:
            print("[FileServer] Error:", e)
            try: conn.sendall(b"ER")
            except: pass
        finally:
            conn.close()

    def _loop(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.listen(128)
            self._sock.settimeout(1.0)  # allow accept to return periodically to check stop
            print(f"[FileServer] Listening on {self.host}:{self.port}, saving to {os.path.abspath(self.save_dir)}")
            while not self._stop.is_set():
                try:
                    conn, addr = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._handle_conn, args=(conn, addr), daemon=True).start()
        finally:
            if self._sock:
                try: self._sock.close()
                except: pass
                self._sock = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._sock:
            try: self._sock.shutdown(socket.SHUT_RDWR)  # wake up accept
            except: pass
            try: self._sock.close()
            except: pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("[FileServer] Stopped")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(BUF), b""):
            h.update(chunk)
    return h.digest()

def send_checkpoint(ip: str = "10.1.100.233",
                    port: int = 9001,
                    file_path: str = "model.pt",
                    timeout: float = 10.0) -> bool:
    """Send local file_path over TCP to ip:port receiver; return True on success."""
    filename = os.path.basename(file_path)
    fn_bytes = filename.encode("utf-8")
    size = os.path.getsize(file_path)
    digest = _sha256(file_path)
    try:

        with socket.create_connection((ip, port), timeout=timeout) as sock:
            # header: filename_len(uint16) | filename | file_size(uint64) | SHA256(32B)
            sock.sendall(struct.pack(">H", len(fn_bytes)))
            sock.sendall(fn_bytes)
            sock.sendall(struct.pack(">Q", size))
            sock.sendall(digest)

            # body: send file content in chunks
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(BUF), b""):
                    sock.sendall(chunk)

            # ACK
            ack = sock.recv(2)
            ok = (ack == b"OK")
            print(f"[Client] Sent {filename} ({size}B) -> {ip}:{port}, result: {'OK' if ok else 'ERROR'}")
            return ok
    except Exception as e:
        # let the higher-level retry handle it
        print(f"[Client] Exception: {e}")
        return False

def send_checkpoint_until_success(
    ip: str,
    port: int,
    file_path: str,
    timeout: float = 10.0,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """
    Retry forever until success or stop_event is set.
    - base_delay: initial backoff seconds (1.0 -> 1s, 2s, 4s, ...)
    - max_delay: backoff cap
    - jitter: backoff jitter ratio (0.2 means +-20%)
    - stop_event: optional threading.Event for external termination
    """
    attempt = 0
    while True:
        ok = send_checkpoint(ip, port, file_path, timeout=timeout)
        if ok:
            print("[Client] Transfer succeeded.")
            return True

        # compute backoff time
        delay = 10

        attempt += 1

        print(f"[Client] Retry #{attempt} in {delay:.2f}s ...")
        # sleep that can be interrupted by stop_event
        t_end = time.time() + delay
        while time.time() < t_end:
            time.sleep(0.1)
