"""Open-Puffer HTTP client and server manager.

Provides a Python interface to the Open-Puffer vector database server.
https://github.com/harishsg993010/open-puffer
"""

import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


class OpenPufferClient:
    """HTTP client wrapper for Open-Puffer vector database."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: float = 30.0,
    ):
        """Initialize the Open-Puffer client.

        Args:
            host: Server hostname
            port: Server port
            timeout: Request timeout in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            # Try common health endpoints
            for endpoint in ["/health", "/", "/v1/collections"]:
                try:
                    resp = self.session.get(
                        f"{self.base_url}{endpoint}",
                        timeout=5.0,
                    )
                    if resp.status_code in (200, 404):  # 404 on /v1/collections is OK (empty)
                        return True
                except requests.exceptions.RequestException:
                    continue
            return False
        except Exception:
            return False

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str | None = None,
    ) -> dict[str, Any]:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            metric: Distance metric ("cosine", "l2", "dot")
            index_type: Index type (optional, server may not support)

        Returns:
            Response from server
        """
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
        }
        # Only include index_type if specified (some servers may not support it)
        if index_type:
            payload["index_type"] = index_type
        resp = self.session.post(
            f"{self.base_url}/v1/collections",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name
        """
        resp = self.session.delete(
            f"{self.base_url}/v1/collections/{name}",
            timeout=self.timeout,
        )
        # 404 is OK - collection doesn't exist
        if resp.status_code != 404:
            resp.raise_for_status()

    def insert_points(
        self,
        collection_name: str,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert points into a collection.

        Args:
            collection_name: Target collection
            ids: Point IDs
            vectors: Vector data as numpy array (N, dim)
            payloads: Metadata for each point

        Returns:
            Response from server
        """
        # Convert numpy to list for JSON serialization
        vectors_list = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors

        payload = {
            "points": [
                {"id": id_, "vector": vec, "payload": meta}
                for id_, vec, meta in zip(ids, vectors_list, payloads, strict=True)
            ]
        }
        resp = self.session.post(
            f"{self.base_url}/v1/collections/{collection_name}/points",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int,
        include_vectors: bool = True,
        include_payload: bool = True,
        nprobe: int = 16,
    ) -> dict[str, Any]:
        """Search for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            top_k: Number of results to return
            include_vectors: Include vectors in response
            include_payload: Include payloads in response
            nprobe: Number of clusters to probe (for IVF)

        Returns:
            Search results
        """
        vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        payload = {
            "vector": vector_list,
            "top_k": top_k,
            "include_vectors": include_vectors,
            "include_payload": include_payload,
            "nprobe": nprobe,
        }
        resp = self.session.post(
            f"{self.base_url}/v1/collections/{collection_name}/search",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_collection_stats(self, name: str) -> dict[str, Any]:
        """Get collection statistics.

        Args:
            name: Collection name

        Returns:
            Collection stats including point_count
        """
        resp = self.session.get(
            f"{self.base_url}/v1/collections/{name}/stats",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name

        Returns:
            True if collection exists
        """
        try:
            self.get_collection_stats(name)
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return False
            raise


class OpenPufferServerManager:
    """Manages Open-Puffer server lifecycle via subprocess."""

    def __init__(
        self,
        binary_path: str,
        data_dir: str,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        """Initialize the server manager.

        Args:
            binary_path: Path to puffer-server binary
            data_dir: Directory for data storage
            host: Bind address
            port: Server port
        """
        self.binary_path = binary_path
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.process: subprocess.Popen | None = None
        self._started_by_us = False

    def start(self, timeout: float = 30.0) -> bool:
        """Start the server and wait for it to be ready.

        Args:
            timeout: Maximum time to wait for server to start

        Returns:
            True if server started successfully
        """
        # Check if server is already running
        client = OpenPufferClient(host="localhost", port=self.port)
        if client.health_check():
            print(f"Open-Puffer server already running on port {self.port}")
            return True

        # Validate binary path
        if not os.path.isfile(self.binary_path):
            raise FileNotFoundError(f"Open-Puffer binary not found: {self.binary_path}")

        # Create data directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # Start the server
        cmd = [
            self.binary_path,
            "--bind-addr", f"{self.host}:{self.port}",
            "--data-dir", self.data_dir,
        ]

        print(f"Starting Open-Puffer server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._started_by_us = True

        # Register cleanup on exit
        atexit.register(self.stop)

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                # Process exited
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"Open-Puffer server exited unexpectedly.\n"
                    f"stdout: {stdout.decode()}\n"
                    f"stderr: {stderr.decode()}"
                )

            if client.health_check():
                print(f"Open-Puffer server started on port {self.port}")
                return True

            time.sleep(0.5)

        # Timeout - stop the server
        self.stop()
        raise TimeoutError(
            f"Open-Puffer server failed to start within {timeout} seconds"
        )

    def stop(self) -> None:
        """Stop the server if we started it."""
        if self.process and self._started_by_us:
            print("Stopping Open-Puffer server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            self._started_by_us = False

    def is_running(self) -> bool:
        """Check if the server process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def __del__(self):
        """Cleanup on garbage collection."""
        self.stop()


def get_openpuffer_client(
    host: str = "localhost",
    port: int = 8080,
    binary_path: str | None = None,
    data_dir: str | None = None,
    auto_start: bool = True,
) -> tuple[OpenPufferClient, OpenPufferServerManager | None]:
    """Get an Open-Puffer client, optionally starting the server.

    Args:
        host: Server hostname
        port: Server port
        binary_path: Path to puffer-server binary (for auto-start)
        data_dir: Data directory (for auto-start)
        auto_start: Whether to auto-start the server

    Returns:
        Tuple of (client, server_manager or None)
    """
    client = OpenPufferClient(host=host, port=port)
    server_manager = None

    # Check if server is already running
    if client.health_check():
        return client, None

    # Try to start the server
    if auto_start and binary_path and data_dir:
        server_manager = OpenPufferServerManager(
            binary_path=binary_path,
            data_dir=data_dir,
            port=port,
        )
        server_manager.start()
        return client, server_manager

    # Server not running and can't auto-start
    raise RuntimeError(
        f"Open-Puffer server not available at {host}:{port}. "
        f"Either start the server manually or provide binary_path and data_dir for auto-start."
    )
