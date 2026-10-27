import csv
from modern.data_quality import profile_csv, compare_numeric_drift


def _write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['x','label']); w.writeheader(); w.writerows(rows)


def test_profile_and_drift(tmp_path):
    ref=tmp_path/'ref.csv'; cur=tmp_path/'cur.csv'
    _write(ref,[{'x':i,'label':'a'} for i in range(10)])
    _write(cur,[{'x':i+100,'label':'b'} for i in range(10)])
    p=profile_csv(str(ref))
    assert p['row_count']==10 and p['columns']['x']['inferred_type']=='numeric'
    d=compare_numeric_drift(str(ref),str(cur))
    assert d['columns']['x']['screening_flag'] is True
