"""Bounded Bluetooth-serial transport for Petoi speech and actions."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any

import serial
from dotenv import load_dotenv
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from serial.tools import list_ports


class _SerialCommandError(RuntimeError):
    """Raised when a Petoi serial command cannot be delivered."""


def _port_sort_key(port: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.*?)(\d+)", port.strip(), re.IGNORECASE)
    if match is None:
        return (port.casefold(), -1, port.casefold())
    return (match.group(1).casefold(), int(match.group(2)), port.casefold())


class _PetoiWorkerThread(QThread):
    connection_changed = pyqtSignal(bool, str)
    connection_attempt_finished = pyqtSignal(bool, str, str)
    status_changed = pyqtSignal(str)
    command_sent = pyqtSignal(dict)
    sequence_finished = pyqtSignal(dict)
    sequence_failed = pyqtSignal(str)

    def __init__(
        self,
        baud_rate: int,
        connect_rounds: int,
        round_retry_delay_s: float,
        serial_timeout_s: float,
        write_timeout_s: float,
        connect_settle_s: float,
        audio_command: str,
        audio_start_buffer_s: float,
        command_gap_s: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.baud_rate = baud_rate
        self.connect_rounds = connect_rounds
        self.round_retry_delay_s = round_retry_delay_s
        self.serial_timeout_s = serial_timeout_s
        self.write_timeout_s = write_timeout_s
        self.connect_settle_s = connect_settle_s
        self.audio_command = audio_command
        self.audio_start_buffer_s = audio_start_buffer_s
        self.command_gap_s = command_gap_s
        self._requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stop_requested = threading.Event()
        self._serial: serial.Serial | None = None
        self._connected_port = ""

    def enqueue(self, request: dict[str, Any]) -> None:
        self._requests.put(request)

    def request_stop(self) -> None:
        self._stop_requested.set()
        self.requestInterruption()
        self._requests.put(None)

    def run(self) -> None:
        while not self._should_stop():
            request = self._requests.get()
            if request is None or self._should_stop():
                break

            request_kind = str(request.get("kind", ""))

            try:
                if request_kind == "connect":
                    self._connect_request(request)
                elif request_kind == "sequence":
                    self._execute_sequence(request)
            except _SerialCommandError as error:
                self.sequence_failed.emit(str(error))
                self._disconnect()
            except Exception as error:
                message = f"Petoi request failed: {type(error).__name__}: {error}"
                if request_kind == "connect":
                    reason = str(request.get("reason", "manual"))
                    self._disconnect()
                    self.status_changed.emit(message)
                    self.connection_attempt_finished.emit(False, "", reason)
                else:
                    self.sequence_failed.emit(message)

        self._disconnect()

    def _should_stop(self) -> bool:
        return self._stop_requested.is_set() or self.isInterruptionRequested()

    def _connection_is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def _connect_request(self, request: dict[str, Any]) -> None:
        ports = list(request.get("ports", []))
        reason = str(request.get("reason", "manual"))
        force = bool(request.get("force", False))

        if self._connection_is_open() and not force:
            self.status_changed.emit(f"Already connected to {self._connected_port}")
            self.connection_attempt_finished.emit(True, self._connected_port, reason)
            return

        if force:
            self._disconnect()

        if not ports:
            self.status_changed.emit("Disconnected; no COM ports are available")
            self.connection_changed.emit(False, "")
            self.connection_attempt_finished.emit(False, "", reason)
            return

        for round_number in range(1, self.connect_rounds + 1):
            for port in ports:
                if self._should_stop():
                    self.connection_attempt_finished.emit(False, "", reason)
                    return

                self.status_changed.emit(f"Connecting to {port} (round {round_number}/{self.connect_rounds})")
                connection: serial.Serial | None = None

                try:
                    connection = serial.Serial(
                        port=port,
                        baudrate=self.baud_rate,
                        timeout=self.serial_timeout_s,
                        write_timeout=self.write_timeout_s,
                    )

                    if self._stop_requested.wait(self.connect_settle_s):
                        connection.close()
                        self.connection_attempt_finished.emit(False, "", reason)
                        return

                    connection.reset_input_buffer()
                    self._serial = connection
                    self._connected_port = port
                    self.connection_changed.emit(True, port)
                    self.status_changed.emit(f"Connected to {port}")
                    self.connection_attempt_finished.emit(True, port, reason)
                    return

                except (serial.SerialException, OSError, ValueError) as error:
                    if connection is not None:
                        try:
                            connection.close()
                        except Exception:
                            pass
                    self.status_changed.emit(f"Could not open {port}: {type(error).__name__}: {error}")

            if round_number < self.connect_rounds:
                self.status_changed.emit(f"All ports failed; starting round {round_number + 1}/{self.connect_rounds}")
                if self._stop_requested.wait(self.round_retry_delay_s):
                    self.connection_attempt_finished.emit(False, "", reason)
                    return

        self.connection_changed.emit(False, "")
        self.status_changed.emit(f"Disconnected after {self.connect_rounds} connection rounds")
        self.connection_attempt_finished.emit(False, "", reason)

    def _execute_sequence(self, request: dict[str, Any]) -> None:
        turn_id = str(request["turn_id"])
        revision = int(request["revision"])

        if not self._connection_is_open():
            raise _SerialCommandError("Petoi is disconnected; the speech/action sequence was skipped.")

        self._send_command(self.audio_command, turn_id, revision, "audio", "tts_audio")

        audio_wait_s = max(0.0, float(request["audio_duration_s"])) + self.audio_start_buffer_s
        if self._stop_requested.wait(audio_wait_s):
            return

        for action in request["action_steps"]:
            if self._should_stop():
                return

            self._send_command(action["command"], turn_id, revision, "action", action["action_key"])
            duration_s = max(0.0, float(action.get("duration_s", 0.0)))
            if duration_s and self._stop_requested.wait(duration_s):
                return

            stop_command = action.get("stop_command")
            if stop_command:
                self._send_command(stop_command, turn_id, revision, "action_stop", action["action_key"])

            if self.command_gap_s and self._stop_requested.wait(self.command_gap_s):
                return

        self.sequence_finished.emit({
            "turn_id": turn_id,
            "revision": revision,
            "actions": [step["action_key"] for step in request["action_steps"]],
        })

    def _send_command(self, command: str, turn_id: str, revision: int, kind: str, action_key: str) -> None:
        if not self._connection_is_open():
            raise _SerialCommandError("Petoi serial connection is unavailable.")

        payload = f"{command.strip()}\n".encode("utf-8")

        try:
            bytes_written = self._serial.write(payload)
        except (serial.SerialException, serial.SerialTimeoutException, OSError) as error:
            raise _SerialCommandError(f"Serial write to {self._connected_port} failed: {error}") from error

        if bytes_written != len(payload):
            raise _SerialCommandError(
                f"Serial write to {self._connected_port} was incomplete ({bytes_written}/{len(payload)} bytes)."
            )

        self.command_sent.emit({
            "turn_id": turn_id,
            "revision": revision,
            "port": self._connected_port,
            "kind": kind,
            "action_key": action_key,
            "command": command,
        })

    def _disconnect(self) -> None:
        connection = self._serial
        previous_port = self._connected_port
        self._serial = None
        self._connected_port = ""

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

        if previous_port:
            self.connection_changed.emit(False, "")


class PetoiService(QObject):
    ports_changed = pyqtSignal(list, str)
    connection_changed = pyqtSignal(bool, str)
    connection_busy_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    command_sent = pyqtSignal(dict)
    sequence_finished = pyqtSignal(dict)
    sequence_failed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, action_catalog_path: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")

        configured_ports = os.getenv("PETOI_SERIAL_PORTS", "")
        self.configured_ports = self._unique_ports(configured_ports.replace(";", ",").split(","))
        self.preferred_port = self.configured_ports[0] if self.configured_ports else ""
        self.baud_rate = int(os.getenv("PETOI_SERIAL_BAUD", "115200"))
        self.connect_rounds = max(1, int(os.getenv("PETOI_CONNECT_ROUNDS", "2")))
        self.round_retry_delay_s = max(0.0, float(os.getenv("PETOI_PORT_RETRY_DELAY_SECONDS", "1")))
        self.serial_timeout_s = max(0.05, float(os.getenv("PETOI_SERIAL_TIMEOUT_SECONDS", "0.25")))
        self.write_timeout_s = max(0.1, float(os.getenv("PETOI_WRITE_TIMEOUT_SECONDS", "3")))
        self.connect_settle_s = max(0.0, float(os.getenv("PETOI_CONNECT_SETTLE_SECONDS", "1")))
        self.audio_command = os.getenv("PETOI_AUDIO_COMMAND", "v").strip() or "v"
        self.audio_start_buffer_s = max(0.0, float(os.getenv("PETOI_AUDIO_START_BUFFER_SECONDS", "0.5")))
        self.command_gap_s = max(0.0, float(os.getenv("PETOI_COMMAND_GAP_SECONDS", "0.15")))

        catalog_path = action_catalog_path or root / "config" / "actions.json"
        self.action_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        self._worker: _PetoiWorkerThread | None = None
        self._connected = False
        self._connected_port = ""
        self._connection_requests = 0
        self._busy = False
        self._stopping = False

    @staticmethod
    def _unique_ports(ports: Any) -> list[str]:
        unique: dict[str, str] = {}
        for raw_port in ports:
            port = str(raw_port).strip()
            if port:
                unique.setdefault(port.casefold(), port)
        return list(unique.values())

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def connected_port(self) -> str:
        return self._connected_port

    @property
    def is_connecting(self) -> bool:
        return self._connection_requests > 0

    @property
    def is_busy(self) -> bool:
        return self._busy

    def available_ports(self) -> list[str]:
        try:
            detected_ports = sorted(
                [item.device for item in list_ports.comports()],
                key=_port_sort_key,
            )
        except Exception as error:
            self.error.emit(f"COM port discovery failed: {type(error).__name__}: {error}")
            detected_ports = []
        return self._unique_ports([*self.configured_ports, *detected_ports])

    @pyqtSlot()
    def refresh_ports(self) -> None:
        ports = self.available_ports()
        selected = self._connected_port or self.preferred_port
        if selected not in ports:
            selected = ports[0] if ports else ""
        self.ports_changed.emit(ports, selected)

    @pyqtSlot()
    def start(self) -> None:
        if self._worker is not None or self._stopping:
            return

        self._worker = _PetoiWorkerThread(
            baud_rate=self.baud_rate,
            connect_rounds=self.connect_rounds,
            round_retry_delay_s=self.round_retry_delay_s,
            serial_timeout_s=self.serial_timeout_s,
            write_timeout_s=self.write_timeout_s,
            connect_settle_s=self.connect_settle_s,
            audio_command=self.audio_command,
            audio_start_buffer_s=self.audio_start_buffer_s,
            command_gap_s=self.command_gap_s,
            parent=self,
        )
        self._worker.connection_changed.connect(self._on_connection_changed)
        self._worker.connection_attempt_finished.connect(self._on_connection_attempt_finished)
        self._worker.status_changed.connect(self.status_changed.emit)
        self._worker.command_sent.connect(self.command_sent.emit)
        self._worker.sequence_finished.connect(self._on_sequence_finished)
        self._worker.sequence_failed.connect(self._on_sequence_failed)
        self._worker.start()
        self.refresh_ports()
        self.ensure_connected(reason="startup")

    def _queue_connection(self, ports: list[str], reason: str, force: bool) -> bool:
        if self._stopping or self._worker is None:
            return False

        ports = self._unique_ports(ports)
        if not ports:
            self.connection_changed.emit(False, "")
            self.status_changed.emit("Disconnected; no COM ports are available")
            return False

        self._connection_requests += 1
        if self._connection_requests == 1:
            self.connection_busy_changed.emit(True)

        self._worker.enqueue({"kind": "connect", "ports": ports, "reason": reason, "force": force})
        return True

    def ensure_connected(self, preferred_port: str = "", reason: str = "llm_dispatch") -> bool:
        if self._connected:
            return True

        preferred_port = preferred_port.strip()
        if preferred_port:
            self.preferred_port = preferred_port

        # Automatic scans are deliberately limited to the .env list plus
        # a port explicitly selected by the user. Other detected devices are
        # displayed in the UI but are never opened automatically.
        ports = self._unique_ports([self.preferred_port, *self.configured_ports])

        return self._queue_connection(ports, reason, force=False)

    @pyqtSlot(str)
    def connect_selected_port(self, port: str) -> None:
        selected_port = port.strip()
        if not selected_port:
            self.error.emit("Select an available COM port before connecting to Petoi.")
            return

        self.preferred_port = selected_port
        self.ports_changed.emit(self.available_ports(), selected_port)
        self._queue_connection([selected_port], "manual", force=True)

    def play_audio_then_actions(
        self,
        turn_id: str,
        revision: int,
        audio_duration_s: float,
        actions: list[str],
    ) -> bool:
        if self._stopping or self._worker is None:
            return False

        if not self._connected and not self.is_connecting:
            self.sequence_failed.emit("Petoi is disconnected; the speech/action sequence was skipped.")
            return False

        if self._busy:
            self.sequence_failed.emit("Petoi is already running a speech/action sequence.")
            return False

        action_steps = self._build_action_steps(actions)
        if action_steps is None:
            return False

        self._set_busy(True)
        self._worker.enqueue({
            "kind": "sequence",
            "turn_id": turn_id,
            "revision": revision,
            "audio_duration_s": audio_duration_s,
            "action_steps": action_steps,
        })
        return True

    def _build_action_steps(self, actions: list[str]) -> list[dict[str, Any]] | None:
        steps: list[dict[str, Any]] = []

        for action_key in actions:
            details = self.action_catalog.get(action_key)
            if details is None:
                self.error.emit(f"No action catalog entry exists for '{action_key}'.")
                return None

            mapping = details.get("petoi")
            if mapping is None:
                continue

            command = str(mapping.get("command", "")).strip()
            if not command:
                self.error.emit(f"Petoi action '{action_key}' has no serial command.")
                return None

            step = {
                "action_key": action_key,
                "command": command,
                "duration_s": max(0.0, float(mapping.get("duration_s", 0.0))),
            }
            stop_command = str(mapping.get("stop_command", "")).strip()
            if stop_command:
                step["stop_command"] = stop_command
            steps.append(step)

        return steps

    @pyqtSlot(bool, str)
    def _on_connection_changed(self, connected: bool, port: str) -> None:
        self._connected = connected
        self._connected_port = port if connected else ""
        if connected:
            self.preferred_port = port
        self.connection_changed.emit(connected, self._connected_port)
        self.refresh_ports()

    @pyqtSlot(bool, str, str)
    def _on_connection_attempt_finished(self, success: bool, port: str, reason: str) -> None:
        self._connection_requests = max(0, self._connection_requests - 1)
        if self._connection_requests == 0:
            self.connection_busy_changed.emit(False)
        if not success and not self._stopping:
            self.status_changed.emit(f"Connection request '{reason}' finished without a Petoi connection")

    @pyqtSlot(dict)
    def _on_sequence_finished(self, result: dict) -> None:
        self._set_busy(False)
        self.sequence_finished.emit(result)

    @pyqtSlot(str)
    def _on_sequence_failed(self, message: str) -> None:
        self._set_busy(False)
        self.sequence_failed.emit(message)

    def _set_busy(self, busy: bool) -> None:
        if self._busy == busy:
            return
        self._busy = busy
        self.busy_changed.emit(busy)

    @pyqtSlot()
    def stop(self) -> None:
        if self._stopping:
            return

        self._stopping = True
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            worker.wait()

        self._worker = None
        self._connected = False
        self._connected_port = ""
        self._connection_requests = 0
        self.connection_busy_changed.emit(False)
        self._set_busy(False)
