"""Exercise outbound peer identity and origin boundaries over real TLS."""

import contextlib
import json
import shutil
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from plugins.platforms.a2a import protocol, tools


@pytest.fixture
def certificates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "*")
    if not shutil.which("openssl"):
        pytest.skip("openssl binary not available")
    certs = {}
    for name in ("localhost", "alice", "bob"):
        cert, key = tmp_path / f"{name}.crt", tmp_path / f"{name}.key"
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", f"/CN={name}", "-addext", f"subjectAltName=DNS:{name}",
        ], check=True, capture_output=True)
        certs[name] = {"cert_file": str(cert), "key_file": str(key)}
    for name in ("alice", "bob"):
        certs[name]["ca_file"] = certs["localhost"]["cert_file"]
    return certs


@contextlib.contextmanager
def _server(certs, *, redirect=None, rpc_url=None):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _identity(self):
            cert = self.connection.getpeercert()
            name = cert["subject"][0][0][1] if cert else None
            received.append((self.command, self.path, name))
            return name

        def _json(self, body):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            name = self._identity()
            if redirect and self.path.endswith("agent-card.json"):
                self.send_response(302)
                self.send_header("Location", redirect)
                self.end_headers()
            elif self.path.endswith("agent-card.json"):
                # Verify that legacy discovery keeps the same identity too.
                self.send_error(404)
            else:
                self._json(protocol.build_agent_card(
                    name=name or "anonymous", description="TLS peer",
                    url=rpc_url or f"https://localhost:{self.server.server_port}/rpc",
                ))

        def do_POST(self):
            name = self._identity()
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._json(protocol.jsonrpc_result(body["id"], protocol.build_task(
                "task", body["params"]["message"]["contextId"],
                protocol.STATE_COMPLETED, f"authenticated as {name}",
            )))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certs["localhost"]["cert_file"], certs["localhost"]["key_file"])
    for name in ("alice", "bob"):
        ctx.load_verify_locations(cafile=certs[name]["cert_file"])
    ctx.verify_mode = ssl.CERT_REQUIRED
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("operation", ["call", "url", "discover", "orchestrate"])
def test_selected_peer_identity_survives_discovery_and_rpc(certificates, tmp_path, operation):
    with _server(certificates) as (url, received):
        peers = {name: {"url": f"{url}/{name}", "tls": certificates[name],
                        "capabilities": [name]} for name in ("alice", "bob")}
        (tmp_path / "config.yaml").write_text(json.dumps({"a2a_agents": peers}))
        if operation == "discover":
            result = tools.a2a_discover({"url": f"{url}/bob"})
            assert "Agent: bob" in result
        elif operation == "orchestrate":
            result = tools.a2a_orchestrate({"capability": "bob", "message": "hello"})
            assert "authenticated as bob" in result
        else:
            agent = f"{url}/bob" if operation == "url" else "bob"
            result = tools.a2a_call({"agent": agent, "message": "hello"})
            assert "authenticated as bob" in result
        assert received and all(name == "bob" for _, _, name in received)
        assert any(path.endswith("agent.json") for _, path, _ in received)
        if operation != "discover":
            assert ("POST", "/rpc", "bob") in received


@pytest.mark.parametrize("route", ["redirect", "rpc", "same_origin_redirect"])
def test_tls_identity_stays_within_peer_origin(certificates, tmp_path, route):
    with _server(certificates) as (destination, destination_requests):
        options = {"redirect": destination} if route == "redirect" else {"rpc_url": destination}
        if route == "same_origin_redirect":
            options = {"redirect": "/card"}
        with _server(certificates, **options) as (source, source_requests):
            (tmp_path / "config.yaml").write_text(json.dumps({"a2a_agents": {
                "peer": {"url": source, "tls": certificates["alice"]},
            }}))
            if route == "rpc":
                result = tools.a2a_call({"agent": "peer", "message": "hello"})
            else:
                result = tools.a2a_discover({"url": source})
            if route == "same_origin_redirect":
                assert "Agent: alice" in result
                assert ("GET", "/card", "alice") in source_requests
            else:
                assert "Error:" in result
            assert destination_requests == []


def test_url_ambiguity_requires_named_peer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    url = "https://peer.example"
    peers = {
        "alice": {"url": url, "tls": {"cert_file": "/alice.pem"}},
        "bob": {"url": url, "tls": {"cert_file": "/bob.pem"}},
    }
    (tmp_path / "config.yaml").write_text(json.dumps({"a2a_agents": peers}))
    assert "ambiguous TLS" in tools.a2a_call({"agent": url, "message": "hello"})
    assert "ambiguous TLS" in tools.a2a_discover({"url": url})
    for name, peer in peers.items():
        assert tools._resolve_peer(name)["tls"] == peer["tls"]
