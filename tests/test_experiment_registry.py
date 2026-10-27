import json
from modern.experiment_tracker import log_experiment, write_model_card
from modern.model_registry import register_model, verify_model


def test_tracker_and_model_card(tmp_path):
    log=tmp_path/'runs.jsonl'
    record=log_experiment('trial-1', {'loss':0.3}, {'lr':0.001}, path=str(log))
    assert json.loads(log.read_text().splitlines()[0])['name']=='trial-1'
    card=tmp_path/'CARD.md'
    write_model_card(str(card),model_name='demo',intended_use='learning',data_description='toy data',evaluation='not run',limitations='not production')
    assert 'Model Card' in card.read_text()


def test_registry_hash_verification(tmp_path):
    artifact=tmp_path/'model.txt'; artifact.write_text('demo model bytes')
    register_model(str(artifact),'demo','v1',str(tmp_path/'registry'))
    assert verify_model('demo','v1',str(tmp_path/'registry'))['valid'] is True
    (tmp_path/'registry'/'demo'/'v1'/'model.txt').write_text('tampered')
    assert verify_model('demo','v1',str(tmp_path/'registry'))['valid'] is False
