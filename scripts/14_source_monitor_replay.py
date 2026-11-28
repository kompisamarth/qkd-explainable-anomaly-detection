"""Replay every saved regime with the same evaluator; do not modify the run."""
import argparse
import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.source_artifacts import validate_run,load_model,load_sessions
from src.evaluation.source_side import event_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    manifest=validate_run(args.run)
    expected=json.loads((args.run/'events.json').read_text())
    actual=[]
    models={}
    for item in manifest['evaluations']:
        filename=item['model']
        if filename not in models:
            models[filename]=load_model(args.run,filename,manifest)
        model=models[filename]
        for session in load_sessions(args.run/'data',item['sessions']):
            scores=model.raw_scores(session)
            for detector,threshold in model.thresholds.items():
                alarm,_=model.policy(scores[detector],threshold)
                actual.append(dict(seed=item['seed'],variant=item['variant'],regime=item['regime'],detector=detector,
                                   **event_metrics(session,alarm)))
    assert actual==expected,'Replay differs from saved evaluation'
    print(f'PASS: {len(actual)} event rows reproduced exactly; all checksums, schemas, versions verified.')


if __name__=='__main__':
    main()
