## Task 1

Genetic Algorithm.

```python
def evaluate(individual):
    if len(set(individual)) != ROSTER_SIZE:
        return -1e15, 

    selected_df = df.iloc[individual]
    total_salary = selected_df['AAV'].sum()
    if total_salary > SALARY_CAP:
        return -1 * (total_salary - SALARY_CAP), 

    counts = selected_df['Pos'].value_counts()
    for pos, min_count in MIN_POS.items():
        if counts.get(pos, 0) < min_count:
            return -1e15, 

    # Objective Calculation
    sorted_ws = selected_df['WS'].sort_values(ascending=False).values
    top3_sum = np.sum(sorted_ws[:3])
    rest_sum = np.sum(sorted_ws[3:])
    T_perf = a * top3_sum + b * rest_sum - c * (top3_sum ** 2)
    ln_T_perf = np.log(T_perf) if T_perf > 0 else -100
    
    total_mvp_b = selected_df['MVP_B_Score'].sum() 
    total_risk = selected_df['Risk_Score'].sum()

    # W: Profit, V: Valuation
    W = alpha * ln_T_perf + beta * total_mvp_b
    V = alpha * ln_T_perf + beta * total_mvp_b - gamma * total_risk
    
    Final_A = weight_x * W + weight_y * V
    return Final_A,

def cxSetBased(ind1, ind2):
    pool = list(set(ind1) | set(ind2))
    if len(pool) < ROSTER_SIZE:
        all_players = set(range(len(df)))
        needed = ROSTER_SIZE - len(pool)
        pool.extend(random.sample(list(all_players - set(pool)), needed))
    
    ind1[:] = random.sample(pool, ROSTER_SIZE)
    ind2[:] = random.sample(pool, ROSTER_SIZE)
    return ind1, ind2

def mutate_roster(individual, indpb):
    if random.random() < indpb:
        idx_to_remove = random.randrange(len(individual))
        current_set = set(individual)
        all_indices = set(range(len(df)))
        available = list(all_indices - current_set)
        
        if available:
            individual[idx_to_remove] = random.choice(available)
    return individual,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 lambda: random.sample(range(len(df)), ROSTER_SIZE))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", cxSetBased)      # Use custom set-based crossover
toolbox.register("mutate", mutate_roster, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, 
                                   verbose=False, halloffame=hof)
    
    return hof[0]
```

## Task 2

The Massey-Thaler curve and synergy model mentioned in the paper.

```python
def calculate_lambda(roster):
    lambdas = []
    for _, p in roster.iterrows():
        if p['High_Usage'] and p['High_Def']:
            lambdas.append(lambda_high_usage_high_def) 
        elif p['High_Usage'] and not p['High_Def']:
            lambdas.append(lambda_high_usage_low_def) 
        else:
            lambdas.append(lambda_base) 
    return np.mean(lambdas) if lambdas else lambda_base

def calculate_draft_risk(pick_number):
    expected_val = alpha_draft * (pick_number ** -beta_draft)
    variance = (expected_val ** 2) * (np.exp(sigma_sq) - 1)
    return variance

def evaluate_roster(roster_df):
    sum_Sj = roster_df['WS'].sum()
    lam = calculate_lambda(roster_df)
    T_perf = sum_Sj - lam * (sum_Sj ** 2)
    
    total_MVP_B = roster_df['MVP_B'].sum()
    
    base_risk = (roster_df['Age'] * 0.01).sum() 
    draft_risk_penalty = 0
    if 'Draft_Rank' in roster_df.columns:
        draft_picks = roster_df[roster_df['Source'] == 'Draft']
        for _, p in draft_picks.iterrows():
            draft_risk_penalty += calculate_draft_risk(p['Draft_Rank'])
    
    total_risk = base_risk + theta_risk * draft_risk_penalty
    
    total_salary = roster_df['AAV'].sum()
    W = alpha_profit * np.log(T_perf) + beta_profit * total_MVP_B - total_salary
    V = alpha_profit * np.log(T_perf) + beta_profit * total_MVP_B - gamma_risk * total_risk
    Final_Score = x_weight * W + y_weight * V
    return Final_Score, W, V
```

## Task 3

Dynamic decision-making under different scenarios.

```python
def run_scenario_analysis(market_df, budget, target_city_coords):
    # ind_lat, ind_lon = 39.7684, -86.1581
    # target_lat, target_lon = target_city_coords
    # dist = haversine(ind_lat, ind_lon, target_lat, target_lon)
    candidates = market_df.copy()
    candidates['Adj_Risk'] = candidates.apply(
        lambda row: calculate_adjusted_risk(row, dist), axis=1
    )

    # ROI = (Performance / Cost) * (1 - Risk_Probability)
    candidates['ROI_Score'] = (
        candidates['WS'] / (candidates['AAV'] + 1e-5)
    ) * (1 - candidates['Adj_Risk'])
    
    affordable = candidates[candidates['AAV'] <= budget]
    if affordable.empty:
        return None
        
    best_target = affordable.sort_values(by='ROI_Score', ascending=False).iloc[0]
    return best_target
```

## Task 4

Solving.

```python
def solve_optimization(team_df):
    n_players = len(team_df)
    initial_eta = np.zeros(n_players) 
    
    constraints = []
    
    constraints.append({
        'type': 'ineq', 
        'fun': lambda x: params['Eta_max'] - np.sum(x)
    })
    
    P_const = params['Base_Revenue']
    for i in range(n_players):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: params['Kappa'] - x[i] * team_df['Risk'].iloc[i]
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (team_df['AAV'].iloc[i] * params['Comp_max_factor']) - 
                                  (team_df['AAV'].iloc[i] + x[i] * P_const)
        })

    bounds = [(0, 1) for _ in range(n_players)]

    result = minimize(
        objective_function, 
        initial_eta, 
        args=(team_df,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result
```

## Task 5

Risk modeling and the loss function.

```python

def pinball_loss(y_true, y_pred, tau=0.90):
    u = y_true - y_pred
    return np.mean(np.where(u >= 0, tau * u, (tau - 1) * u))

def calculate_risk_logistic(age, is_injured=False):
    if is_injured:
        return 0.95
    
    # R(t) = R_max / (1 + exp(-k * (Age - Age_mid)))
    risk = PARAMS['R_max'] / (1 + np.exp(-PARAMS['k'] * (age - PARAMS['A_mid'])))
    return risk

def calculate_mvp_b_stochastic(row):
    ig = row['IG_Followers']
    interaction = row['Engagement']
    search = row['Google']
    
    val = (PARAMS['d'] * np.log(ig + 1) + 
           PARAMS['e'] * interaction / 10000 + 
           PARAMS['f'] * search / 10000 + 
           PARAMS['g'])
    return val
```

```python
def run_simulation(team_df, pool_df):
    core_idx = team_df['WS'].idxmax()
    core_player = team_df.loc[core_idx]
    
    team_hold = team_df.copy()
    team_hold.at[core_idx, 'WS'] *= INJURY_RATE 
    team_hold.at[core_idx, 'Risk'] = calculate_risk_logistic(0, is_injured=True) # Risk spikes
    metrics_hold = calculate_team_metrics(team_hold)
    
    team_minus_core = team_df.drop(core_idx)
    budget = PARAMS['salary_cap'] - team_minus_core['AAV'].sum()
    
    candidates = pool_df[
        (pool_df['Pos'] == core_player['Pos']) & 
        (pool_df['AAV'] <= budget)
    ]
    
    best_v_replace = -np.inf
    
    for _, cand in candidates.iterrows():
        temp_team = pd.concat([team_minus_core, pd.DataFrame([cand])], ignore_index=True)
        m = calculate_team_metrics(temp_team)
        if m['V'] > best_v_replace:
            best_v_replace = m['V']
            
    return metrics_hold, best_v_replace
```

