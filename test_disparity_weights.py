#!/usr/bin/env python
"""
Diagnostic test to verify that different valueWeights produce different disparity scores.
Run this after the backend is running.
"""

import requests
import json
import time

base = 'http://127.0.0.1:5050'

def test_disparity_changes_with_weights():
    print('\n' + '='*60)
    print('DISPARITY DIAGNOSTIC TEST')
    print('='*60)
    
    # Step 1: Get default applicants
    print('\n[1/6] Fetching default applicants...')
    r1 = requests.get(base + '/api/init/default-applicants')
    apps = r1.json()['applicants']
    print(f'    ✓ Got {len(apps)} applicants')
    
    # Step 2: Submit with initial decisions
    print('\n[2/6] Submitting 12-applicant form...')
    candidates = []
    for i, a in enumerate(apps):
        decision = 'admit' if i < 6 else 'reject'
        candidates.append({
            'candidateLabel': a['candidateLabel'],
            'candidateName': a['candidateName'],
            'decision': decision,
            'notes': 'Test decision note',
            'profile': a['profile']
        })
    
    body = {
        'candidateEvaluations': candidates,
        'overallRationale': 'I balance merit with context and resilience factors.'
    }
    r2 = requests.post(base + '/api/init/start', json=body)
    session = r2.json()
    session_id = session['sessionId']
    print(f'    ✓ Created session: {session_id[:16]}...')
    print(f'    Initial race sensitivity: {session.get("raceSensitivity", "N/A")}')
    
    # Step 3: Submit pairwise answers
    print('\n[3/6] Submitting pairwise answers...')
    questions = session.get('clarificationQuestions', [])
    answers = {q['id']: 'balanced' for q in questions}
    r3 = requests.post(base + '/api/init/answers', json={'sessionId': session_id, 'answers': answers})
    final = r3.json()
    print(f'    ✓ Finalized seed and answers')
    print(f'    Final race sensitivity: {final.get("raceSensitivity", "N/A")}')
    
    # Step 4-6: Test with different value weights
    print('\n[4/6] Testing dialogue with EQUAL weights (25/25/25/25)...')
    vw1 = {'merit': 0.25, 'family': 0.25, 'school': 0.25, 'community': 0.25}
    r4 = requests.post(base + '/api/analysis/dialogue', json={'sessionId': session_id, 'valueWeights': vw1})
    d1 = r4.json()
    sens1 = d1.get('sensitivity', None)
    print(f'    ✓ Sensitivity: {sens1}')
    print(f'    Top pairs: {[p.get("pairKey") for p in d1.get("topDisparityPairs", [])[:2]]}')
    
    time.sleep(0.3)
    
    print('\n[5/6] Testing dialogue with MERIT-HEAVY weights (50/16.7/16.7/16.7)...')
    vw2 = {'merit': 0.5, 'family': 0.167, 'school': 0.167, 'community': 0.167}
    r5 = requests.post(base + '/api/analysis/dialogue', json={'sessionId': session_id, 'valueWeights': vw2})
    d2 = r5.json()
    sens2 = d2.get('sensitivity', None)
    print(f'    ✓ Sensitivity: {sens2}')
    print(f'    Top pairs: {[p.get("pairKey") for p in d2.get("topDisparityPairs", [])[:2]]}')
    
    time.sleep(0.3)
    
    print('\n[6/6] Testing dialogue with CONTEXT-HEAVY weights (10/30/30/30)...')
    vw3 = {'merit': 0.1, 'family': 0.3, 'school': 0.3, 'community': 0.3}
    r6 = requests.post(base + '/api/analysis/dialogue', json={'sessionId': session_id, 'valueWeights': vw3})
    d3 = r6.json()
    sens3 = d3.get('sensitivity', None)
    print(f'    ✓ Sensitivity: {sens3}')
    print(f'    Top pairs: {[p.get("pairKey") for p in d3.get("topDisparityPairs", [])[:2]]}')
    
    # Analysis
    print('\n' + '='*60)
    print('RESULTS')
    print('='*60)
    print(f'\nSensitivity(25/25/25/25):       {sens1}')
    print(f'Sensitivity(50/16.7/16.7/16.7): {sens2}')
    print(f'Sensitivity(10/30/30/30):       {sens3}')
    
    if sens1 is not None and sens2 is not None and sens3 is not None:
        varies = (sens1 != sens2) or (sens2 != sens3)
        status = '✓ PASS: Scores vary with weights' if varies else '✗ FAIL: Scores do NOT change'
        print(f'\nTest Result: {status}')
        if varies:
            delta12 = abs(sens1 - sens2)
            delta23 = abs(sens2 - sens3)
            print(f'  Delta (equal vs merit-heavy):    {delta12:.2f} points')
            print(f'  Delta (merit-heavy vs context):  {delta23:.2f} points')
    else:
        print('\n✗ FAIL: Could not compute sensitivity scores')

if __name__ == '__main__':
    try:
        test_disparity_changes_with_weights()
    except Exception as e:
        print(f'\n✗ ERROR: {e}')
        import traceback
        traceback.print_exc()
