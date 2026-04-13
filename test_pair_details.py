#!/usr/bin/env python
"""Detailed test showing top 3 pairs with individual gap values"""

import requests
import json

base = 'http://127.0.0.1:5050'

print('\n' + '='*70)
print('DETAILED PAIR DISPARITY TEST (with individual gap values)')
print('='*70)

# Get default applicants
r1 = requests.get(base + '/api/init/default-applicants')
apps = r1.json()['applicants']

# Submit form
candidates = []
for i, a in enumerate(apps):
    decision = 'admit' if i < 6 else 'reject'
    candidates.append({
        'candidateLabel': a['candidateLabel'],
        'candidateName': a['candidateName'],
        'decision': decision,
        'notes': 'Test note',
        'profile': a['profile']
    })

body = {
    'candidateEvaluations': candidates,
    'overallRationale': 'I balance merit with context and resilience.'
}
r2 = requests.post(base + '/api/init/start', json=body)
session = r2.json()
session_id = session['sessionId']

# Submit pairwise
questions = session.get('clarificationQuestions', [])
answers = {q['id']: 'balanced' for q in questions}
r3 = requests.post(base + '/api/init/answers', json={'sessionId': session_id, 'answers': answers})

# Query dialogue with equal weights
print('\n[Query] Sending equal weights (25/25/25/25)...\n')
vw = {'merit': 0.25, 'family': 0.25, 'school': 0.25, 'community': 0.25}
r4 = requests.post(base + '/api/analysis/dialogue', json={'sessionId': session_id, 'valueWeights': vw})
data = r4.json()

print(f'Overall Race Disparity Score: {data.get("sensitivity", "N/A")} / 100\n')

top_pairs = data.get('topDisparityPairs', [])
all_pairs = data.get('pairDisparities', [])

print(f'Top 3 Disparity Pairs:')
print('─' * 70)
print(f"{'Pair':<20} {'Mean Gap':<15} {'Favored':<12} {'# Samples':<10}")
print('─' * 70)

if top_pairs:
    for i, pair in enumerate(top_pairs[:3], 1):
        pair_key = pair.get('pairKey', 'Unknown')
        mean_gap_raw = pair.get('meanGap', 0.0)
        mean_gap_pct = mean_gap_raw * 100
        favored = pair.get('favoredRace', 'tie')
        count = pair.get('count', 0)
        print(f"{pair_key:<20} {mean_gap_pct:>6.2f} pts    {favored:<12} {count:<10}")
else:
    print("No pairs found")

print('─' * 70)
print(f'\nTotal pairs computed: {len(all_pairs)}\n')

print('All Pairs Ranked by Disparity (Top 6):')
print('─' * 70)
print(f"{'Pair':<20} {'Mean Gap':<15} {'Favored':<12}")
print('─' * 70)

for i, pair in enumerate(all_pairs[:6]):
    pair_key = pair.get('pairKey', 'Unknown')
    mean_gap_raw = pair.get('meanGap', 0.0) 
    mean_gap_pct = mean_gap_raw * 100
    favored = pair.get('favoredRace', 'tie')
    print(f"{pair_key:<20} {mean_gap_pct:>6.2f} pts    {favored:<12}")

print('─' * 70)
print('\n✓ Test complete. If top 3 pairs show DIFFERENT values, fix is successful!')
