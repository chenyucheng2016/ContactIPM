import json
for name in ['pendulum', 'quadrotor', 'chain_mass']:
    print('===', name, '===')
    for tag, f in [('ContactIPM', f'contactipm_{name}.json'),
                   ('acados', f'acados_{name}.json')]:
        d = json.load(open(f))
        st = d['stages']
        N = d['N']
        xN = st[N]['x']
        x0 = st[0]['x']
        u0 = st[0]['u']
        xs = [round(v, 4) for v in xN]
        print('  %-11s u0=%s  xN=%s  iters=%d' % (tag, [round(v,3) for v in u0], xs, d['iterations']))
