import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / 'g1_handover_20261009_followup/evaluate_handover.py'
    spec = importlib.util.spec_from_file_location('recorded_handover_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source'
    (source / 'meta').mkdir(parents=True)
    (source / 'meta/info.json').write_text(json.dumps({
        'video_path': 'episode_{episode_index}.mp4', 'chunks_size': 1000, 'fps': 50,
    }))
    (source / 'episode_15.mp4').write_bytes(b'test-video-identity')
    monkeypatch.setattr(module, 'SOURCE', source)

    def extract(video, index, target):
        buffer = io.BytesIO()
        Image.new('RGB', (4, 4), (index, 0, 0)).save(buffer, format='JPEG')
        target.write_bytes(buffer.getvalue())
        return buffer.getvalue()

    monkeypatch.setattr(module, 'extract', extract)
    monkeypatch.delenv('GENON_API_KEY', raising=False)
    return module


def test_default_prepare_preserves_genon_settings(evaluator, tmp_path, monkeypatch):
    output = tmp_path / 'result'
    monkeypatch.setattr('sys.argv', ['eval', '--output', str(output), '--prepare-only',
                                   '--case', 'desk-bottle'])
    assert evaluator.main() == 0
    report = json.loads((output / 'result.json').read_text())
    assert report['model'] == 'openai/gpt-5.6-sol'
    assert report['base_url'] == 'https://api.genon.ai/v1'
    assert report['request_options'] == {}
    assert report['robot_connection'] is False


def test_local_qwen_settings_reach_provider_and_monitor(evaluator, tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv('QWEN_LOCAL_API_KEY', 'test-local-key')

    def client(**kwargs):
        captured['client'] = kwargs
        return object()

    def chat(client, **kwargs):
        captured['request'] = kwargs
        return SimpleNamespace(model=kwargs['model'], usage=None, choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content='{"outcome":"yes","reason":"Bottle on desk"}'))])

    monkeypatch.setattr(evaluator, 'OpenAI', client)
    monkeypatch.setattr(evaluator, 'chat_create', chat)
    output = tmp_path / 'result'
    monkeypatch.setattr('sys.argv', ['eval', '--output', str(output),
        '--case', 'desk-bottle', '--model', 'Qwen/Qwen3.6-27B',
        '--base-url', 'http://127.0.0.1:13402/v1', '--api-key-env', 'QWEN_LOCAL_API_KEY',
        '--max-tokens', '256', '--disable-thinking'])
    assert evaluator.main() == 0
    assert captured['client']['base_url'] == 'http://127.0.0.1:13402/v1'
    assert captured['client']['api_key'] == 'test-local-key'
    assert captured['request']['model'] == 'Qwen/Qwen3.6-27B'
    assert captured['request']['max_tokens'] == 256
    assert captured['request']['extra_body'] == {
        'chat_template_kwargs': {'enable_thinking': False}}
    report = json.loads((output / 'result.json').read_text())
    assert report['cases'][0]['frame_outcome'] == 'yes'
    assert report['cases'][0]['confirmation_streak'] == 1
    assert report['cases'][0]['provider_call']['finish_reason'] == 'stop'
    assert report['cases'][0]['provider_call']['content_present'] is True
    assert report['cases'][0]['provider_call']['reasoning_present'] is False
    assert 'test-local-key' not in (output / 'result.json').read_text()


def test_default_evaluation_still_requires_genon_key(evaluator, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['eval', '--output', str(tmp_path / 'result')])
    with pytest.raises(SystemExit) as exc:
        evaluator.main()
    assert exc.value.code == 2


def test_nonpositive_token_cap_rejected(evaluator, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['eval', '--output', str(tmp_path / 'result'),
                                   '--max-tokens', '0', '--prepare-only'])
    with pytest.raises(SystemExit) as exc:
        evaluator.main()
    assert exc.value.code == 2


def test_reasoning_only_response_is_structured_unavailable(evaluator, tmp_path, monkeypatch):
    monkeypatch.setenv('QWEN_LOCAL_API_KEY', 'test-local-key')
    monkeypatch.setattr(evaluator, 'OpenAI', lambda **kwargs: object())
    monkeypatch.setattr(evaluator, 'chat_create', lambda client, **kwargs:
        SimpleNamespace(model=kwargs['model'], usage=None, choices=[
            SimpleNamespace(finish_reason='length', message=SimpleNamespace(
                content=None, reasoning_content='test reasoning'))]))
    output = tmp_path / 'result'
    monkeypatch.setattr('sys.argv', ['eval', '--output', str(output),
        '--case', 'desk-bottle', '--model', 'Qwen/Qwen3.6-27B',
        '--api-key-env', 'QWEN_LOCAL_API_KEY', '--max-tokens', '256'])
    assert evaluator.main() == 1
    report = json.loads((output / 'result.json').read_text())
    case = report['cases'][0]
    assert case['decision']['outcome'] == 'unavailable'
    assert case['confirmation_streak'] == 0
    assert case['provider_call']['finish_reason'] == 'length'
    assert case['provider_call']['content_present'] is False
    assert case['provider_call']['reasoning_present'] is True
    assert report['summary']['unavailable'] == 1
