"""Per-session model forensics: compare QR vs Hawkes-mult sim quality against
real data across all 62 sessions.  Output: which model fits each session best
and by how much.

Quality score = weighted sum of absolute metric errors (lower is better):
  spread mean %  +  events/pair %  +  §29 price-impact sum-|err|
"""
import subprocess, re, os

def parse(path):
    if not os.path.exists(path) or os.path.getsize(path) < 400:
        return None
    out = subprocess.run(['./metrics', f'{path}:even', f'{path}:odd'],
                         capture_output=True, text=True).stdout
    d = {'impact': []}
    for line in out.split('\n'):
        s = line.strip()
        if s.startswith('events/pair ') and 'B/A' not in s and '+' not in s:
            nums = re.findall(r'[\d.]+', s)
            if len(nums) >= 2: d['ev'] = (float(nums[0]), float(nums[1]))
        elif s.startswith('spread mean') and 'B/A' not in s and '+' not in s:
            nums = re.findall(r'[\d.]+', s)
            if len(nums) >= 2: d['sp'] = (float(nums[0]), float(nums[1]))
        elif s.startswith('spread std'):
            nums = re.findall(r'[\d.]+', s)
            if len(nums) >= 2: d['sp_std'] = (float(nums[0]), float(nums[1]))
        elif 'nA=' in s:
            nums = re.findall(r'[+-]?\d+\.\d+', s)
            if len(nums) >= 2: d['impact'].append((float(nums[0]), float(nums[1])))
    return d

def score(d):
    if d is None or 'ev' not in d or 'sp' not in d: return float('inf')
    ev_err = abs(d['ev'][1] / d['ev'][0] - 1) * 100
    sp_err = abs(d['sp'][1] / d['sp'][0] - 1) * 100
    # Sum |price impact error| (half-ticks)
    imp_err = sum(abs(b - a) for a, b in d['impact'])
    return ev_err + sp_err + imp_err * 2    # weight impact moderately


print(f'{"ses":>3} {"regime":>6}  {"QR events":>10} {"QR sp":>7} {"QR imp":>7} {"QR score":>9}  '
      f'{"HK events":>10} {"HK sp":>7} {"HK imp":>7} {"HK score":>9}  winner')
print('-' * 130)

results = []
for s in range(62):
    regime = 'HOT' if s >= 52 or s == 44 else 'calm'
    qr = parse(f'/tmp/neonka/sim/t55_qr_{s}.raw')
    hk = parse(f'/tmp/neonka/sim/t55_h8_{s}.raw')
    sq, sh = score(qr), score(hk)
    winner = 'QR' if sq < sh else 'HK' if sh < sq else '~'
    results.append((s, regime, qr, hk, sq, sh, winner))

    def fmt(d, key):
        if d is None or key not in d: return '    n/a'
        a, b = d[key]
        return f'{100*(b/a-1):>+7.1f}%'
    def fmt_imp(d):
        if d is None or not d['impact']: return '    n/a'
        s_err = sum(abs(b-a) for a, b in d['impact'])
        return f'{s_err:>6.2f}'

    print(f'{s:>3} {regime:>6}  {fmt(qr,"ev"):>10} {fmt(qr,"sp"):>7} {fmt_imp(qr):>7} {sq:>8.1f}  '
          f'{fmt(hk,"ev"):>10} {fmt(hk,"sp"):>7} {fmt_imp(hk):>7} {sh:>8.1f}  {winner}')

# Summary
print()
qr_wins = [r for r in results if r[6] == 'QR']
hk_wins = [r for r in results if r[6] == 'HK']
print(f'QR wins: {len(qr_wins)}/62  (calm: {sum(1 for r in qr_wins if r[1]=="calm")}, hot: {sum(1 for r in qr_wins if r[1]=="HOT")})')
print(f'HK wins: {len(hk_wins)}/62  (calm: {sum(1 for r in hk_wins if r[1]=="calm")}, hot: {sum(1 for r in hk_wins if r[1]=="HOT")})')
print()
print(f'Median score — QR: {sorted(r[4] for r in results)[31]:.1f}')
print(f'Median score — HK: {sorted(r[5] for r in results)[31]:.1f}')
