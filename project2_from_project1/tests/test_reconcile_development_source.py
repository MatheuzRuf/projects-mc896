"""Exercise the source reconciliation command through its public interface."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def invoke(source, adjudication, partitions, output):
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-m", "project2_from_project1.scripts.reconcile_development_source",
         "--corpus", str(source), "--adjudication", str(adjudication),
         "--partitions", str(partitions), "--output-dir", str(output)],
        cwd=root, capture_output=True, text=True,
    )


def test_reconciles_only_line_endings_and_records_source_bindings(tmp_path):
    original = b'{"id":1}\r\n{"id":2}\r\n'
    expected = original.replace(b'\r\n', b'\n')
    source = tmp_path / 'corpus.jsonl'
    source.write_bytes(original)
    adjudication = tmp_path / 'adjudication.jsonl'
    adjudication.write_text(json.dumps({'binding': {'source_sha256': digest(expected)}}) + '\n', encoding='utf-8')
    partitions = tmp_path / 'partitions.json'
    partitions.write_text(json.dumps({'source_jsonl_sha256': digest(original)}), encoding='utf-8')
    output = tmp_path / 'out'
    result = invoke(source, adjudication, partitions, output)
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == original
    assert (output / 'corpus.jsonl').read_bytes() == expected
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['input_sha256']['corpus'] == digest(original)
    assert manifest['output_sha256']['corpus.jsonl'] == digest(expected)
    assert manifest['adjudication_source_sha256'] == digest(expected)
    assert invoke(source, adjudication, partitions, output).returncode != 0


def test_rejects_inconsistent_bindings_without_writing_output(tmp_path):
    source = tmp_path / 'corpus.jsonl'
    source.write_bytes(b'{}\r\n')
    adjudication = tmp_path / 'adjudication.jsonl'
    adjudication.write_text(json.dumps({'binding': {'source_sha256': '0' * 64}}) + '\n', encoding='utf-8')
    partitions = tmp_path / 'partitions.json'
    partitions.write_text(json.dumps({'source_jsonl_sha256': digest(source.read_bytes())}), encoding='utf-8')
    output = tmp_path / 'out'
    assert invoke(source, adjudication, partitions, output).returncode != 0
    assert not output.exists()
