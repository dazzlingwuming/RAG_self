import os
import socket
import subprocess
import time

from hybrid_graph_rag_app import settings


class Neo4jRuntime:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((host, port)) == 0

    def is_ready(self) -> bool:
        return self._port_open("127.0.0.1", 8687) and self._port_open("127.0.0.1", 8474)

    def ensure_started(self, timeout: int = 45) -> None:
        if self.is_ready():
            return

        env = os.environ.copy()
        env["NEO4J_ACCEPT_LICENSE_AGREEMENT"] = "yes"
        env["JAVA_HOME"] = str(settings.NEO4J_JAVA_HOME)
        env["PATH"] = f"{settings.NEO4J_JAVA_HOME / 'bin'};{env['PATH']}"

        self.process = subprocess.Popen(
            [str(settings.NEO4J_HOME / "bin" / "neo4j.bat"), "console"],
            cwd=str(settings.NEO4J_HOME),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

        started = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_ready():
                started = True
                break
            if self.process.poll() is not None:
                break
            time.sleep(1)

        if not started:
            raise RuntimeError("Neo4j 图谱实例未能在预期时间内启动。")

