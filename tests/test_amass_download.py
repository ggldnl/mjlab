import io
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from mjlab.datasets.amass import download


def test_download_posts_credentials_and_maps_babel_subset(
  tmp_path: Path, monkeypatch: Any
) -> None:
  captured = {}

  class Response(io.BytesIO):
    headers = {"Content-Type": "application/x-bzip2", "Content-Length": "6"}

  class Opener:
    def open(self, request):
      captured["request"] = request
      return Response(b"BZh123")

  def build_opener(*handlers):
    captured["handlers"] = handlers
    return Opener()

  monkeypatch.setattr(download.urllib.request, "build_opener", build_opener)
  output = tmp_path / "MPI_HDM05.tar.bz2"

  download._download_subset("person@example.com", "secret", "MPI_HDM05", output)

  request = captured["request"]
  assert "HDM05.tar.bz2" in request.full_url
  assert "resume=1" in request.full_url
  assert urllib.parse.parse_qs(request.data.decode()) == {
    "username": ["person@example.com"],
    "password": ["secret"],
  }
  assert isinstance(captured["handlers"][0], urllib.request.HTTPCookieProcessor)
  assert output.read_bytes() == b"BZh123"


def test_download_reports_portal_error(tmp_path: Path, monkeypatch: Any) -> None:
  class Opener:
    def open(self, request):
      raise urllib.error.HTTPError(
        request.full_url,
        401,
        "Unauthorized",
        Message(),
        io.BytesIO(b"Error: Username/Password wrong.<!DOCTYPE html>"),
      )

  monkeypatch.setattr(
    download.urllib.request, "build_opener", lambda *handlers: Opener()
  )

  with pytest.raises(SystemExit, match="Username/Password wrong"):
    download._download_subset("person@example.com", "secret", "ACCAD", tmp_path / "x")
