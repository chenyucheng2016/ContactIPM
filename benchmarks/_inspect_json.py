import json
for f in ['pendulum_ocp.json', 'quadrotor_2d_ocp.json', 'chain_mass_ocp.json']:
    d = json.load(open(f))
    so = d.get('solver_options', {})
    dims = d.get('dims', {})
    cost = d.get('cost', {})
    print('===', f, '===')
    print('  dims:', {k: dims.get(k) for k in ['nx','nu','nz','N','ny','ny_e','np','nbx','nbu','ng','nh'] if k in dims})
    print('  tf:', so.get('tf'), 'N_horizon:', so.get('N_horizon'))
    print('  integrator:', so.get('integrator_type'),
          'stages:', so.get('sim_method_num_stages'),
          'steps:', so.get('sim_method_num_steps'))
    print('  qp_solver:', so.get('qp_solver'))
    print('  nlp_solver:', so.get('nlp_solver_type'), 'step_len:', so.get('nlp_solver_step_length'),
          'glob:', so.get('globalization'))
    print('  collocation:', so.get('collocation_type'))
    print('  cost_type:    ', cost.get('cost_type'), '0:', cost.get('cost_type_0'), 'e:', cost.get('cost_type_e'))
    print('  cost_scaling: ', cost.get('cost_scaling'))
    # tolerances stored anywhere?
    print('  tol keys:', {k: so[k] for k in so if 'tol' in k.lower()})
    print()
