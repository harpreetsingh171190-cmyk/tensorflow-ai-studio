"""CSV data profiling and lightweight numeric drift checks (stdlib only)."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from statistics import mean, median, pstdev


def load_csv(path: str):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError('CSV must contain a header row')
        rows = list(reader)
    return list(reader.fieldnames), rows


def profile_csv(path: str) -> dict:
    columns, rows = load_csv(path)
    result = {'file': Path(path).name, 'row_count': len(rows), 'column_count': len(columns), 'columns': {}}
    for col in columns:
        vals = [r.get(col) for r in rows]
        present = [v.strip() for v in vals if v is not None and v.strip() != '']
        nums = []
        for v in present:
            try:
                x = float(v)
                if math.isfinite(x): nums.append(x)
            except (ValueError, TypeError):
                pass
        info = {'missing_count': len(rows)-len(present), 'missing_fraction': (len(rows)-len(present))/len(rows) if rows else 0.0,
                'unique_count': len(set(present)), 'inferred_type': 'numeric' if present and len(nums)==len(present) else 'text'}
        if nums:
            info.update({'min': min(nums), 'max': max(nums), 'mean': mean(nums), 'median': median(nums), 'std_population': pstdev(nums)})
        result['columns'][col] = info
    return result


def _numeric_values(rows, column):
    out=[]
    for row in rows:
        try:
            x=float(row.get(column, ''))
            if math.isfinite(x): out.append(x)
        except (ValueError, TypeError):
            continue
    return out


def compare_numeric_drift(reference_csv: str, current_csv: str, bins: int = 10) -> dict:
    """Compare numeric columns using normalized histogram total-variation distance.

    This is a screening signal, not a statistical test or proof of model degradation.
    Reference data determines bin edges; out-of-range current values enter edge bins.
    """
    if bins < 2: raise ValueError('bins must be at least 2')
    ref_cols, ref_rows = load_csv(reference_csv); cur_cols, cur_rows = load_csv(current_csv)
    common = [c for c in ref_cols if c in cur_cols]
    result={'reference_rows':len(ref_rows),'current_rows':len(cur_rows),'columns':{}}
    for col in common:
        a=_numeric_values(ref_rows,col); b=_numeric_values(cur_rows,col)
        if len(a)<2 or len(b)<2:
            continue
        lo, hi=min(a),max(a)
        if hi == lo:
            hist_a=[1.0]+[0.0]*(bins-1)
            hist_b=[0.0]*bins
            for x in b: hist_b[0 if x == lo else (bins-1 if x > lo else 0)] += 1/len(b)
        else:
            def hist(vals):
                counts=[0]*bins
                for x in vals:
                    idx=min(bins-1,max(0,int((x-lo)/(hi-lo)*bins)))
                    counts[idx]+=1
                return [n/len(vals) for n in counts]
            hist_a=hist(a); hist_b=hist(b)
        tvd=0.5*sum(abs(x-y) for x,y in zip(hist_a,hist_b))
        result['columns'][col]={'reference_numeric_count':len(a),'current_numeric_count':len(b),'total_variation_distance':round(tvd,6),'screening_flag':tvd>=0.2}
    result['note']='Heuristic only: thresholds are configurable in your process; review sample sizes and context.'
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    prof=sub.add_parser('profile'); prof.add_argument('csv')
    drift=sub.add_parser('drift'); drift.add_argument('reference_csv'); drift.add_argument('current_csv'); drift.add_argument('--bins',type=int,default=10)
    a=p.parse_args(); out=profile_csv(a.csv) if a.command=='profile' else compare_numeric_drift(a.reference_csv,a.current_csv,a.bins)
    print(json.dumps(out,indent=2))
