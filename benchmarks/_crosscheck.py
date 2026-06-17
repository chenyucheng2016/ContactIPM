"""
Independent cross-evaluation: apply the EXACT SAME cost + violation formulas
to both solvers' dumped trajectories. Removes any ambiguity about whether
"cost" is computed consistently.

Uses the unscaled formulas defined in the acados main_*.c compute_total_cost()
(which ContactIPM also uses for its reported cost).
"""
import json, math

PI = math.pi

def load(path):
    return json.load(open(path))

def pendulum_cost(stages, N):
    WX, WT, WV, WW, WR = 1.0, 10.0, 0.1, 0.5, 0.01
    total = 0.0
    for k in range(N):
        x, u = stages[k]['x'], stages[k]['u']
        dx, dt = x[0], x[1]-PI
        dv, dw = x[2], x[3]
        total += 0.5*(WX*dx*dx + WT*dt*dt + WV*dv*dv + WW*dw*dw + WR*u[0]*u[0])
    x = stages[N]['x']
    dx, dt = x[0], x[1]-PI
    dv, dw = x[2], x[3]
    total += 5.0*(WX*dx*dx + WT*dt*dt + WV*dv*dv + WW*dw*dw)
    return total

def quadrotor_cost(stages, N):
    WY,WZ,WPHI,WVY,WVZ,WDPHI,WU = 5.0,10.0,2.0,0.5,1.0,0.3,0.001
    YDES, ZDES = 0.0, 2.0
    total = 0.0
    for k in range(N):
        x, u = stages[k]['x'], stages[k]['u']
        dy,dz,dp = x[0]-YDES, x[1]-ZDES, x[2]
        vy,vz,dphi = x[3],x[4],x[5]
        total += 0.5*(WY*dy*dy+WZ*dz*dz+WPHI*dp*dp+WVY*vy*vy+WVZ*vz*vz+WDPHI*dphi*dphi
                      +WU*(u[0]*u[0]+u[1]*u[1]))
    x = stages[N]['x']
    dy,dz,dp = x[0]-YDES, x[1]-ZDES, x[2]
    vy,vz,dphi = x[3],x[4],x[5]
    total += 10.0*(WY*dy*dy+WZ*dz*dz+WPHI*dp*dp+WVY*vy*vy+WVZ*vz*vz+WDPHI*dphi*dphi)
    return total

def chain_cost(stages, N, NM=5, NU=3):
    WPOS,WVEL,WCTRL = 10.0,1.0,0.01
    total = 0.0
    for k in range(N):
        x, u = stages[k]['x'], stages[k]['u']
        c = 0.0
        for i in range(NM):
            c += WPOS*x[i]*x[i] + WVEL*x[NM+i]*x[NM+i]
        for i in range(NU):
            c += WCTRL*u[i]*u[i]
        total += 0.5*c
    x = stages[N]['x']
    c = 0.0
    for i in range(NM):
        c += 20.0*WPOS*x[i]*x[i] + 20.0*WVEL*x[NM+i]*x[NM+i]
    total += 0.5*c
    return total

def pendulum_viol(stages, N):
    mv = 0.0
    for k in range(N+1):
        mv = max(mv, abs(stages[k]['x'][0]) - 2.0)
    for k in range(N):
        mv = max(mv, abs(stages[k]['u'][0]) - 30.0)
    return max(mv, 0.0)

def quadrotor_viol(stages, N):
    HOVER = 0.5*9.81; U1MAX = 2.0*HOVER; U2MAX = 0.5*HOVER*0.15
    mv = 0.0
    for k in range(N+1):
        mv = max(mv, -stages[k]['x'][1])
    for k in range(N):
        u1,u2 = stages[k]['u'][0], stages[k]['u'][1]
        mv = max(mv, -u1, u1-U1MAX, -u2-U2MAX, u2-U2MAX)
    return max(mv, 0.0)

def chain_viol(stages, N, NM=5, NU=3):
    mv = 0.0
    for k in range(N):
        for i in range(NU):
            mv = max(mv, abs(stages[k]['u'][i]) - 5.0)
    for k in range(N+1):
        for i in range(NM):
            mv = max(mv, abs(stages[k]['x'][i]) - 2.0)
    return max(mv, 0.0)

PROBLEMS = {
    'pendulum':  ('contactipm_pendulum.json', 'acados_pendulum.json', pendulum_cost, pendulum_viol),
    'quadrotor': ('contactipm_quadrotor.json','acados_quadrotor.json',quadrotor_cost,quadrotor_viol),
    'chain_mass':('contactipm_chain_mass.json','acados_chain_mass.json',chain_cost,  chain_viol),
}

print(f"{'problem':<12} {'solver':<11} {'cost (re-eval)':>14} {'cost (reported)':>16} {'max viol':>10}")
print('-'*70)
for name,(cf,af,costfn,violfn) in PROBLEMS.items():
    for label, fn in [('ContactIPM', cf), ('acados', af)]:
        try:
            d = load(fn)
            N = d['N']; stages = d['stages']
            re = costfn(stages, N)
            rep = d.get('cost_total', float('nan'))
            viol = violfn(stages, N)
            print(f"{name:<12} {label:<11} {re:14.4f} {rep:16.4f} {viol:10.2e}")
        except Exception as e:
            print(f"{name:<12} {label:<11} ERROR: {e}")
