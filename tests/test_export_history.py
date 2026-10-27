import csv, json
from modern.export_history import export

def test_export_history_json_to_csv(tmp_path):
    source = tmp_path / 'history.json'
    target = tmp_path / 'history.csv'
    source.write_text(json.dumps({'loss':[1.0,0.5], 'accuracy':[0.4,0.8]}), encoding='utf-8')
    export(str(source), str(target))
    rows = list(csv.reader(target.open(encoding='utf-8')))
    assert rows[0] == ['epoch', 'loss', 'accuracy']
    assert rows[1] == ['1', '1.0', '0.4']
    assert rows[2] == ['2', '0.5', '0.8']
